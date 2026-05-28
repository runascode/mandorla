"""Calibrate τ_v — the minimum expected-intersection log-volume below which
a candidate Vesica is discarded during retrieval (PRECOMMIT.md §"Architecture
spec for the slice").

Procedure (frozen in PRECOMMIT.md): dry-run the Vesica-search step on a 1k-
question sample of HotpotQA **train**, collect the expected-intersection
log-volume for every pairwise candidate over each question's contriever
top-k descent, and set τ_v to the **50th percentile** of all those values.
Calibrated once, then frozen for the dev eval.

Note this is a *calibration* step, not training: it tunes one scalar on the
train split and never touches dev. The sample of 1k questions is drawn
deterministically with seed=1337.

Requires scripts/04 (FAISS index) and scripts/05 (box index + projection)
to have run against the full corpus first.

Output: index/tau_v.json — {tau_v_log_volume, percentile, n_questions,
n_pairs_observed, k_descent, beta, seed, sample_question_ids_hash}
"""

from __future__ import annotations

import hashlib
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.box import DEFAULT_BETA, expected_intersection_log_volume
from src.data import load_hotpotqa
from src.index_io import (
    QueryEncoder,
    load_box_store,
    load_faiss_retriever,
    load_projection,
    save_tau_v,
)

N_QUESTIONS = 1000
K_DESCENT = 20
PERCENTILE = 50.0
SEED = 1337


def main() -> int:
    print("Loading FAISS retriever, box store, projection, query encoder...", flush=True)
    retriever = load_faiss_retriever()
    boxes = load_box_store()
    _ = load_projection()  # not used in τ_v calibration (no cosine factor here)
    encoder = QueryEncoder()
    box_idx_of = {cid: i for i, cid in enumerate(boxes.chunk_ids)}

    # Deterministic 1k sample of train questions.
    print("Loading HotpotQA train and sampling 1k questions (seed=1337)...", flush=True)
    train = list(load_hotpotqa("train"))
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(train), size=min(N_QUESTIONS, len(train)), replace=False)
    sample = [train[int(i)] for i in sorted(sample_idx)]
    sample_ids = [q.id for q in sample]
    ids_hash = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()[:16]
    print(f"  sampled {len(sample)} questions; ids hash={ids_hash}")

    # Dry-run vesica search; collect all pairwise log-volumes.
    print(f"Dry-running vesica search (k_descent={K_DESCENT}, "
          f"C({K_DESCENT},2)={K_DESCENT * (K_DESCENT - 1) // 2} pairs/question)...", flush=True)
    all_log_vols: list[float] = []
    t0 = time.time()
    for i, q in enumerate(sample):
        q_vec = encoder.encode(q.question)
        descent = retriever.top_k(q_vec, K_DESCENT)
        for a, b in combinations(descent, 2):
            if a.chunk_id not in box_idx_of or b.chunk_id not in box_idx_of:
                continue
            ba = boxes.box_for_index(box_idx_of[a.chunk_id])
            bb = boxes.box_for_index(box_idx_of[b.chunk_id])
            all_log_vols.append(
                expected_intersection_log_volume(ba, bb, beta=DEFAULT_BETA)
            )
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(sample)} questions "
                  f"({len(all_log_vols)} pairs, {time.time() - t0:.1f}s)", flush=True)

    arr = np.array(all_log_vols, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        print("ERROR: no finite pairwise log-volumes observed. Cannot calibrate τ_v.")
        return 1

    tau_v = float(np.percentile(arr, PERCENTILE))
    print(f"\nτ_v calibration:")
    print(f"  n questions       = {len(sample)}")
    print(f"  n pairs observed  = {arr.size}")
    print(f"  log-volume stats  = min={arr.min():.4g} "
          f"p10={np.percentile(arr, 10):.4g} "
          f"p50={np.percentile(arr, 50):.4g} "
          f"p90={np.percentile(arr, 90):.4g} "
          f"max={arr.max():.4g}")
    print(f"  τ_v (p{PERCENTILE:g} log-volume) = {tau_v:.6g}")
    print(f"  (≈ {np.exp(tau_v):.3g} in linear volume)")

    meta = {
        "percentile": PERCENTILE,
        "n_questions": len(sample),
        "n_pairs_observed": int(arr.size),
        "k_descent": K_DESCENT,
        "beta": DEFAULT_BETA,
        "seed": SEED,
        "sample_question_ids_hash": ids_hash,
        "log_volume_stats": {
            "min": float(arr.min()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
        },
    }
    path = save_tau_v(tau_v, meta)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
