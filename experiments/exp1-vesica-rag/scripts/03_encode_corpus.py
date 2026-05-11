"""Encode the HotpotQA Wikipedia corpus with contriever on Apple MPS.

Each of the 5,233,329 passages is encoded to a 768-D float32 vector by
`facebook/contriever-msmarco`. We mean-pool the BERT last hidden state over
non-pad tokens (the original contriever pooling, matching Izacard et al.
arXiv:2112.09118 §3.2).

Outputs are sharded to disk in `index/contriever_shards/`:
  shard_NNNNN.npz  — fp16 embeddings, shape (S, 768), and parallel chunk_ids
                     of length S. Default shard size = 200,000 passages.

We use fp16 storage to halve disk usage (~8 GB → ~4 GB total). Computation
is fp32 because MPS has known stability issues for some fp16 ops.

Usage:
  uv run python scripts/03_encode_corpus.py                # full corpus
  uv run python scripts/03_encode_corpus.py --limit 10000  # benchmark
  uv run python scripts/03_encode_corpus.py --resume       # resume from
                                                           # last completed
                                                           # shard
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "index" / "contriever_shards"
META_PATH = REPO_ROOT / "index" / "contriever_meta.json"

MODEL_NAME = "facebook/contriever-msmarco"
EMBED_DIM = 768
SHARD_SIZE = 200_000      # passages per shard
BATCH_SIZE = 256          # passages per forward pass (M4 Pro MPS sweet spot)
MAX_LEN = 128             # tokens. Median passage is ~63 words (~100 tokens),
                          # so most are not truncated. Halving from 256 nearly
                          # doubles throughput (140 p/s vs 75 p/s on M4 Pro)
                          # at negligible quality cost — both baseline and
                          # Vesica-RAG use the same encoder with the same
                          # truncation, so the *comparison* metric (which is
                          # what the slice measures) is invariant. Documented
                          # in BENCHMARKS.md.


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Contriever's mean pooling: average non-pad token embeddings."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1)
    return summed / count


def encode_batch(model, tokenizer, texts: list[str], device: torch.device) -> np.ndarray:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    return emb.detach().to(torch.float32).cpu().numpy()


def find_resume_point() -> int:
    """Return the global passage index from which to resume.

    Reads existing shard files in INDEX_DIR; resumes after the last
    fully-complete shard. Returns 0 if none exist.
    """
    if not INDEX_DIR.exists():
        return 0
    shards = sorted(INDEX_DIR.glob("shard_*.npz"))
    if not shards:
        return 0
    last = shards[-1]
    # Inspect to confirm it loads cleanly; if corrupt, drop it.
    try:
        d = np.load(last)
        if "embeddings" not in d or "chunk_ids" not in d:
            last.unlink()
            return find_resume_point()
        n = d["embeddings"].shape[0]
    except Exception:
        last.unlink()
        return find_resume_point()
    # shard_NNNNN.npz means we wrote indices [NNNNN*SHARD_SIZE, +n)
    shard_idx = int(last.stem.split("_")[1])
    return shard_idx * SHARD_SIZE + n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many passages (for benchmarking).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last completed shard.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}", flush=True)

    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    print("Loading BeIR/hotpotqa corpus...", flush=True)
    corpus = load_dataset("BeIR/hotpotqa", "corpus", split="corpus")
    total = len(corpus)
    if args.limit is not None:
        total = min(total, args.limit)
        print(f"  --limit set; will process {total} passages")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    start_global = find_resume_point() if args.resume else 0
    if start_global > 0:
        print(f"Resuming from passage {start_global}")
    if start_global >= total:
        print("Nothing to do — already complete up to limit/total.")
        return 0

    # Stream the corpus in shards.
    t_start = time.time()
    n_done_session = 0
    shard_start = (start_global // SHARD_SIZE) * SHARD_SIZE

    while shard_start < total:
        shard_idx = shard_start // SHARD_SIZE
        shard_end = min(shard_start + SHARD_SIZE, total)
        # If we're resuming mid-shard, skip the already-encoded prefix.
        skip = max(0, start_global - shard_start)
        shard_passages = corpus.select(range(shard_start + skip, shard_end))
        shard_buf: list[np.ndarray] = []
        shard_ids: list[str] = []

        for batch_start in range(0, len(shard_passages), args.batch_size):
            batch = shard_passages.select(
                range(batch_start, min(batch_start + args.batch_size, len(shard_passages)))
            )
            # Concatenate title + text per BeIR convention; matches contriever
            # training format better than text alone for Wikipedia abstracts.
            texts = [f"{r['title']} | {r['text']}" for r in batch]
            embs = encode_batch(model, tokenizer, texts, device)
            shard_buf.append(embs.astype(np.float16))
            shard_ids.extend(r["_id"] for r in batch)
            n_done_session += len(texts)
            if batch_start % (args.batch_size * 20) == 0:
                elapsed = time.time() - t_start
                rate = n_done_session / max(elapsed, 1e-6)
                done_global = shard_start + skip + batch_start + len(texts)
                eta_full = (total - done_global) / max(rate, 1e-6)
                print(
                    f"  shard {shard_idx} batch {batch_start // args.batch_size}: "
                    f"{done_global}/{total} ({100 * done_global / total:.2f}%) "
                    f"rate={rate:.0f} p/s eta={eta_full/3600:.2f}h",
                    flush=True,
                )

        # Write shard. If we resumed mid-shard, append to existing.
        out_path = INDEX_DIR / f"shard_{shard_idx:05d}.npz"
        new_embs = np.concatenate(shard_buf, axis=0).astype(np.float16)
        new_ids = np.array(shard_ids, dtype=object)
        if skip > 0 and out_path.exists():
            existing = np.load(out_path, allow_pickle=True)
            new_embs = np.concatenate([existing["embeddings"], new_embs], axis=0)
            new_ids = np.concatenate([existing["chunk_ids"], new_ids], axis=0)
        np.savez(out_path, embeddings=new_embs, chunk_ids=new_ids)
        print(f"  wrote {out_path} ({new_embs.shape[0]} passages)")

        shard_start = shard_end
        start_global = shard_start  # subsequent shards start at boundary

    elapsed = time.time() - t_start
    rate = n_done_session / max(elapsed, 1e-6)
    print(f"\nDone. encoded {n_done_session} passages in {elapsed/60:.1f} min ({rate:.0f} p/s)")

    # Write/update meta.
    meta = {
        "model": MODEL_NAME,
        "embed_dim": EMBED_DIM,
        "max_len": MAX_LEN,
        "shard_size": SHARD_SIZE,
        "passages_encoded": int(total) if args.limit is None else int(n_done_session + (start_global - n_done_session)),
        "device": str(device),
        "passage_format": "title | text",
        "pool": "mean(last_hidden_state * attention_mask)",
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {META_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
