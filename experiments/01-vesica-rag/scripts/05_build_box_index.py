"""Build the 64-D box index for Vesica-RAG.

Pipeline:
  1. Load all contriever shards and L2-normalize the 768-D embeddings.
  2. Project to 64-D using RandomProjection (seed=1337, materialized).
  3. Compute mean-of-k=10-NN distance per chunk in the 64-D space using a
     FAISS HNSW index (fast, approximate; PRECOMMIT.md picks random
     projection over PCA for the same kind of speed/repro trade).
  4. Calibrate α on a random 10k-chunk sample so the median pairwise
     intersection volume is ~5% of the median single-box volume.
  5. Build per-chunk half_widths = α × mean_knn_distance (isotropic across
     dimensions) and save the box index as parallel numpy arrays.

Outputs into index/:
  box_centers.npy        — (N, 64) float32
  box_half_widths.npy    — (N, 64) float32  (isotropic per chunk)
  box_chunk_ids.npy      — (N,)    object  (parallel chunk_ids)
  projection.npz         — RandomProjection matrix + meta
  box.meta.json          — α, achieved ratio, grid, build params
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from src.calibration import calibrate_alpha
from src.data import INDEX_DIR, iter_shards, load_shard
from src.projection import DEFAULT_SEED, RandomProjection

K_NN = 10
SAMPLE_SIZE_FOR_ALPHA = 10_000
# Wide grid spanning ~3 orders of magnitude so we never bottom or top out.
# Empirically (200k-passage smoke), the right α is in the high single digits,
# so the grid is biased toward that region without losing coverage.
ALPHA_GRID = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 17.0, 22.0, 30.0, 50.0)
TARGET_RATIO = 0.05
# 1000 random pairs gives a tight CI on the median (~half a decade of noise
# floor instead of the ~1-decade noise we saw at 200 pairs).
N_CALIBRATION_PAIRS = 1000
D_BOX = 64
D_CONTRIEVER = 768


def load_and_project_all() -> tuple[np.ndarray, list[str], RandomProjection]:
    """Load all shards, L2-normalize 768-D, project to 64-D float32."""
    shards = list(iter_shards())
    if not shards:
        raise RuntimeError("No contriever shards found. Run scripts/03 first.")
    print(f"Found {len(shards)} shard(s).", flush=True)

    proj = RandomProjection(d_in=D_CONTRIEVER, d_out=D_BOX, seed=DEFAULT_SEED)

    all_centers: list[np.ndarray] = []
    all_ids: list[str] = []
    t0 = time.time()
    for shard_idx, path in shards:
        e_fp16, ids = load_shard(path)
        e = e_fp16.astype(np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-12)
        e_hat = e / norms
        centers_64 = proj.project(e_hat)
        all_centers.append(centers_64)
        all_ids.extend(ids)
        print(f"  projected shard {shard_idx} ({e.shape[0]} passages, "
              f"cumulative {len(all_ids)}, {time.time() - t0:.1f}s)", flush=True)
    return np.concatenate(all_centers, axis=0), all_ids, proj


def compute_knn_distances(centers_64: np.ndarray, k: int = K_NN) -> np.ndarray:
    """Mean of top-k L2 distances per chunk (excluding self).

    Uses FAISS HNSW (fast approximate kNN). For 5M+ vectors brute force is
    infeasible; HNSW is the right trade-off for the slice (small recall
    errors don't materially affect a global calibration statistic).
    """
    n = centers_64.shape[0]
    print(f"Building HNSW index over {n} 64-D centers...", flush=True)
    t0 = time.time()
    index = faiss.IndexHNSWFlat(D_BOX, 32)   # M=32; reasonable default
    index.hnsw.efConstruction = 80
    index.add(centers_64.astype(np.float32))
    print(f"  built in {time.time() - t0:.1f}s", flush=True)

    print(f"Querying k+1={k + 1} nearest (k+1 because index returns self)...", flush=True)
    index.hnsw.efSearch = 64
    t1 = time.time()
    # Search in chunks to keep memory bounded
    chunk = 50_000
    means = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        D, _ = index.search(centers_64[start:end].astype(np.float32), k + 1)
        # FAISS HNSW returns squared L2 distances; take sqrt
        D = np.sqrt(np.maximum(D, 0))
        # Exclude self (column 0 of each row, since self is closest)
        knn = D[:, 1:k + 1]
        means[start:end] = knn.mean(axis=1)
        if start % (chunk * 5) == 0:
            print(f"  ...{end}/{n} ({100 * end / n:.1f}%) "
                  f"elapsed={time.time() - t1:.1f}s", flush=True)
    print(f"  kNN query complete in {time.time() - t1:.1f}s", flush=True)
    return means


def main() -> int:
    centers_64, chunk_ids, proj = load_and_project_all()
    n = centers_64.shape[0]
    print(f"\nTotal chunks in box space: {n}\n", flush=True)

    knn_distances = compute_knn_distances(centers_64, k=K_NN)
    print(f"kNN distance stats: "
          f"min={knn_distances.min():.4f} "
          f"median={np.median(knn_distances):.4f} "
          f"mean={knn_distances.mean():.4f} "
          f"max={knn_distances.max():.4f}\n", flush=True)

    # --- α calibration ---
    rng = np.random.default_rng(DEFAULT_SEED)
    sample_size = min(SAMPLE_SIZE_FOR_ALPHA, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sample_centers = centers_64[sample_idx]
    sample_knn = knn_distances[sample_idx]

    print(f"Calibrating α on {sample_size} sampled chunks...", flush=True)
    cal = calibrate_alpha(
        sample_centers,
        sample_knn,
        target_ratio=TARGET_RATIO,
        alpha_grid=ALPHA_GRID,
        n_pairs=N_CALIBRATION_PAIRS,
        seed=DEFAULT_SEED,
    )
    print(f"\nα calibration result:")
    print(f"  chosen α  = {cal.alpha}")
    print(f"  target    = {cal.target_ratio}")
    print(f"  achieved  = {cal.achieved_ratio:.4g}")
    print(f"  grid (α, ratio):")
    for a, r in cal.grid:
        marker = "  ←" if a == cal.alpha else ""
        print(f"    {a:6.3g}  {r:.4g}{marker}")
    print(flush=True)

    # --- Build full box index ---
    half_widths_iso = (cal.alpha * knn_distances).astype(np.float32)
    # (N, 64) isotropic
    half_widths = np.broadcast_to(
        half_widths_iso[:, None], (n, D_BOX)
    ).astype(np.float32).copy()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing box index to {INDEX_DIR}/...")
    np.save(INDEX_DIR / "box_centers.npy", centers_64.astype(np.float32))
    np.save(INDEX_DIR / "box_half_widths.npy", half_widths)
    np.save(INDEX_DIR / "box_chunk_ids.npy", np.array(chunk_ids, dtype=object), allow_pickle=True)
    proj.save(INDEX_DIR / "projection.npz")

    meta = {
        "d_box": D_BOX,
        "d_contriever": D_CONTRIEVER,
        "n": int(n),
        "k_nn": K_NN,
        "alpha": float(cal.alpha),
        "target_ratio": cal.target_ratio,
        "achieved_ratio": float(cal.achieved_ratio),
        "alpha_grid": [(float(a), float(r)) for a, r in cal.grid],
        "calibration_sample_size": int(sample_size),
        "calibration_pairs": N_CALIBRATION_PAIRS,
        "seed": DEFAULT_SEED,
        "knn_stats": {
            "min": float(knn_distances.min()),
            "median": float(np.median(knn_distances)),
            "mean": float(knn_distances.mean()),
            "max": float(knn_distances.max()),
        },
        "projection": {"seed": DEFAULT_SEED, "d_in": D_CONTRIEVER, "d_out": D_BOX},
    }
    (INDEX_DIR / "box.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {INDEX_DIR / 'box.meta.json'}")
    print(f"\nDone. Box index: n={n} d={D_BOX} α={cal.alpha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
