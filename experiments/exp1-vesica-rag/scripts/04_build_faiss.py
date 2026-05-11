"""Build a FAISS index over the contriever shards.

We use IndexFlatIP (inner product, exact search) on L2-normalized vectors
to compute cosine similarity. Memory: 5.2M × 768 × 4 bytes = ~16 GB; fits
comfortably in 48 GB RAM and gives exact retrieval semantics with no
approximation tradeoffs. For the slice we prefer exact over approximate
because (a) retrieval quality is the thing being measured — we don't want
the baseline weakened by ANN approximation, and (b) the cost is one-shot.

Outputs:
  index/contriever.faiss    — serialized IndexFlatIP
  index/chunk_ids.npy       — parallel chunk_id array (object dtype)
  index/contriever.meta.json — record of build parameters

Re-runnable against the encoded shards directory. Picks up however many
shards are currently on disk, which is useful for testing the pipeline
against a partial encode (e.g. shard 0 of 27 during development).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the project root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from src.data import (
    CONTRIEVER_SHARD_DIR,
    INDEX_DIR,
    iter_shards,
    load_shard,
)

OUT_FAISS = INDEX_DIR / "contriever.faiss"
OUT_IDS = INDEX_DIR / "chunk_ids.npy"
OUT_META = INDEX_DIR / "contriever.meta.json"


def main() -> int:
    if not CONTRIEVER_SHARD_DIR.exists():
        print(f"ERROR: {CONTRIEVER_SHARD_DIR} not found. Run scripts/03 first.")
        return 1
    shards = list(iter_shards())
    if not shards:
        print("ERROR: no shards found. Run scripts/03 first.")
        return 1

    print(f"Found {len(shards)} shard(s).")

    # Pass 1: count total passages to pre-allocate.
    total = 0
    for _, path in shards:
        e, _ = load_shard(path)
        total += e.shape[0]
    print(f"Total passages across shards: {total}")

    # Pass 2: build the FAISS index, normalizing on the fly.
    print("Building IndexFlatIP (768-D) over normalized vectors...")
    index = faiss.IndexFlatIP(768)
    all_ids: list[str] = []
    t0 = time.time()
    n_added = 0
    for shard_idx, path in shards:
        e_fp16, ids = load_shard(path)
        e = e_fp16.astype(np.float32)
        # L2-normalize for cosine via IP
        norms = np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-12)
        e = e / norms
        index.add(e.astype(np.float32))
        all_ids.extend(ids)
        n_added += e.shape[0]
        print(f"  added shard {shard_idx} ({e.shape[0]} passages, "
              f"cumulative {n_added}/{total}, "
              f"{time.time() - t0:.1f}s)")

    OUT_FAISS.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting FAISS index → {OUT_FAISS}")
    faiss.write_index(index, str(OUT_FAISS))

    print(f"Writing chunk_ids → {OUT_IDS}")
    np.save(OUT_IDS, np.array(all_ids, dtype=object), allow_pickle=True)

    meta = {
        "ntotal": index.ntotal,
        "dim": 768,
        "type": "IndexFlatIP (cosine via L2-normalized vectors)",
        "shards_consumed": len(shards),
        "build_seconds": round(time.time() - t0, 1),
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"Writing meta → {OUT_META}")
    print(f"\nDone. ntotal={index.ntotal} in {meta['build_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
