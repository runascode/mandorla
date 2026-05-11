"""α and τ_v calibration helpers (PRECOMMIT.md B + section "τ_v").

Two pieces of geometry-level calibration the slice runs once, before the eval
loop:

1. **α** — scalar that controls per-chunk box half-widths
       half_width_i = α · mean_knn_distance(chunk_i, k=10)
   Chosen so the *median pairwise expected intersection volume* on a random
   10k-chunk sample is ~5% of the *median single-box volume*. This is a
   calibration step, not training: it picks α once and freezes it. We pin
   the calibration sample seed so the chosen α is deterministic.

2. **τ_v** — minimum expected intersection log-volume to consider a candidate
   Vesica during retrieval. Set to the 50th percentile of observed
   E[intersection log-volume] across a 1k-question dry-run on HotpotQA train
   (in scripts/06). Vesicas below τ_v are dropped at search time.

Both functions are pure (numpy-only) so they're unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .box import (
    DEFAULT_BETA,
    box_log_volume,
    expected_intersection_log_volume,
)
from .regions import BoxExtent


# ─── α calibration ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AlphaCalibrationResult:
    """Outcome of the α grid search.

    Attributes:
      alpha: the chosen alpha (closest to the target ratio).
      target_ratio: the median(intersection_volume) / median(box_volume) ratio
                    we were aiming for (default 0.05).
      achieved_ratio: the ratio achieved at the chosen alpha.
      grid: list of (alpha, ratio) tuples explored.
    """

    alpha: float
    target_ratio: float
    achieved_ratio: float
    grid: tuple[tuple[float, float], ...]


def _boxes_from_centers_half(
    centers: NDArray[np.float32],
    half: NDArray[np.float32],
) -> list[BoxExtent]:
    return [
        BoxExtent(
            min_corner=(centers[i] - half[i]).astype(np.float32),
            max_corner=(centers[i] + half[i]).astype(np.float32),
        )
        for i in range(centers.shape[0])
    ]


def median_ratio_at_alpha(
    centers: NDArray[np.float32],
    knn_distances: NDArray[np.float32],
    alpha: float,
    n_pairs: int = 200,
    rng: np.random.Generator | None = None,
    beta: float = DEFAULT_BETA,
) -> tuple[float, float, float]:
    """For a given alpha, compute median(box volume), median(pairwise expected
    intersection volume), and their ratio over `n_pairs` random pairs from
    the centers array.

    All sides are isotropic: half_widths[i, :] = alpha * knn_distances[i].
    Volumes are computed in log space, then exponentiated for the median; we
    rank by log-volume to avoid underflow then exponentiate only the median.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = centers.shape[0]
    half = (alpha * knn_distances).astype(np.float32)
    # Broadcast half to (n, d) — isotropic across dimensions.
    d = centers.shape[1]
    half_full = np.broadcast_to(half[:, None], (n, d)).astype(np.float32)

    boxes = _boxes_from_centers_half(centers, half_full)
    log_volumes = np.array([box_log_volume(b) for b in boxes], dtype=np.float64)
    finite = log_volumes[np.isfinite(log_volumes)]
    median_log_vol = float(np.median(finite)) if finite.size else float("-inf")

    # Sample random pairs
    pair_log_vols = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        lv = expected_intersection_log_volume(boxes[int(i)], boxes[int(j)], beta=beta)
        pair_log_vols.append(lv)
    pair_log_vols_arr = np.array(pair_log_vols, dtype=np.float64)
    median_log_inter = (
        float(np.median(pair_log_vols_arr)) if pair_log_vols_arr.size else float("-inf")
    )

    # Ratio in linear space (median of ratios is approximated by ratio of medians here).
    log_ratio = median_log_inter - median_log_vol
    ratio = float(np.exp(log_ratio)) if np.isfinite(log_ratio) else 0.0
    median_box_vol = float(np.exp(median_log_vol)) if np.isfinite(median_log_vol) else 0.0
    median_inter_vol = float(np.exp(median_log_inter)) if np.isfinite(median_log_inter) else 0.0
    return median_box_vol, median_inter_vol, ratio


def calibrate_alpha(
    centers: NDArray[np.float32],
    knn_distances: NDArray[np.float32],
    target_ratio: float = 0.05,
    alpha_grid: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0),
    n_pairs: int = 200,
    seed: int = 1337,
    beta: float = DEFAULT_BETA,
) -> AlphaCalibrationResult:
    """Grid-search α so that median pairwise intersection / median box volume
    ≈ target_ratio.

    The relationship is monotonic in α for isotropic scaling: bigger α =
    more overlap. We pick the grid value whose achieved ratio is closest to
    target in log-ratio space (so 0.01 and 0.25 are equally far from 0.05).
    """
    rng = np.random.default_rng(seed)
    grid_results: list[tuple[float, float]] = []
    for alpha in alpha_grid:
        _, _, ratio = median_ratio_at_alpha(
            centers, knn_distances, alpha, n_pairs=n_pairs, rng=rng, beta=beta
        )
        grid_results.append((alpha, ratio))

    log_target = float(np.log(max(target_ratio, 1e-12)))

    def distance(item: tuple[float, float]) -> float:
        _, r = item
        return abs(float(np.log(max(r, 1e-30))) - log_target)

    best_alpha, best_ratio = min(grid_results, key=distance)
    return AlphaCalibrationResult(
        alpha=float(best_alpha),
        target_ratio=target_ratio,
        achieved_ratio=float(best_ratio),
        grid=tuple(grid_results),
    )
