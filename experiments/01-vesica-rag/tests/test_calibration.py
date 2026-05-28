"""Unit tests for α calibration.

Verify the grid-search picks a sensible alpha on a synthetic corpus, and
that ratios scale monotonically with alpha (the assumption that justifies
grid search rather than something fancier).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    AlphaCalibrationResult,
    calibrate_alpha,
    median_ratio_at_alpha,
)


def synthetic_corpus(n: int = 200, d: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n, d)).astype(np.float32)
    # Simulate plausible knn distances (positive, varied)
    knn = rng.uniform(0.3, 1.0, size=n).astype(np.float32)
    return centers, knn


def test_median_ratio_monotonic_in_alpha() -> None:
    """Bigger α → boxes wider → more overlap → bigger ratio."""
    centers, knn = synthetic_corpus()
    rng = np.random.default_rng(0)
    _, _, r_small = median_ratio_at_alpha(centers, knn, alpha=0.1, n_pairs=100, rng=rng)
    rng = np.random.default_rng(0)
    _, _, r_med = median_ratio_at_alpha(centers, knn, alpha=1.0, n_pairs=100, rng=rng)
    rng = np.random.default_rng(0)
    _, _, r_big = median_ratio_at_alpha(centers, knn, alpha=5.0, n_pairs=100, rng=rng)
    assert r_small <= r_med <= r_big


def test_calibrate_alpha_returns_result() -> None:
    centers, knn = synthetic_corpus(n=100, d=8)
    result = calibrate_alpha(
        centers, knn, target_ratio=0.05, n_pairs=50,
        alpha_grid=(0.5, 1.0, 2.0, 5.0),
    )
    assert isinstance(result, AlphaCalibrationResult)
    assert result.alpha in {0.5, 1.0, 2.0, 5.0}
    assert len(result.grid) == 4


def test_calibrate_alpha_deterministic_under_seed() -> None:
    centers, knn = synthetic_corpus()
    r1 = calibrate_alpha(centers, knn, seed=1337, n_pairs=100)
    r2 = calibrate_alpha(centers, knn, seed=1337, n_pairs=100)
    assert r1.alpha == r2.alpha
    assert r1.achieved_ratio == pytest.approx(r2.achieved_ratio)


def test_calibrate_alpha_picks_closest_to_target() -> None:
    """If the grid spans the target ratio, the chosen alpha should be the one
    whose ratio is closest to the target (in log space)."""
    centers, knn = synthetic_corpus(n=300, d=8, seed=42)
    result = calibrate_alpha(
        centers, knn, target_ratio=0.05, n_pairs=200,
        alpha_grid=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    # The "best" entry must have the smallest log-distance to log(0.05)
    log_target = float(np.log(0.05))
    distances = [abs(float(np.log(max(r, 1e-30))) - log_target) for _, r in result.grid]
    best_idx = int(np.argmin(distances))
    assert result.alpha == result.grid[best_idx][0]
