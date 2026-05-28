"""Unit tests for GumbelBox math.

Covers: hard-volume limit as β → 0, symmetry, disjoint-box near-zero volume,
threshold short-circuit, β smoothing, and the identical-box identity.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.box import (
    box_log_volume,
    box_volume,
    expected_intersection_log_volume,
    expected_intersection_sides,
    expected_intersection_volume,
    intersect_boxes,
)
from src.regions import BoxExtent


def cube(lo: float, hi: float, dim: int = 4) -> BoxExtent:
    return BoxExtent(
        min_corner=np.full(dim, lo, dtype=np.float32),
        max_corner=np.full(dim, hi, dtype=np.float32),
    )


def test_box_volume_unit_cube() -> None:
    assert box_volume(cube(0.0, 1.0, dim=3)) == pytest.approx(1.0)


def test_box_volume_scaled() -> None:
    assert box_volume(cube(0.0, 2.0, dim=4)) == pytest.approx(16.0)


def test_box_volume_degenerate_returns_zero() -> None:
    box = BoxExtent(
        min_corner=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        max_corner=np.array([1.0, 0.5, 1.0], dtype=np.float32),
    )
    assert box_volume(box) == 0.0
    assert box_log_volume(box) == float("-inf")


def test_identical_box_intersection_approaches_self_volume() -> None:
    """At small β, intersect(B, B) ≈ volume(B)."""
    box = cube(0.0, 1.0, dim=3)
    assert expected_intersection_volume(box, box, beta=1e-4) == pytest.approx(1.0, rel=1e-2)


def test_intersection_symmetry() -> None:
    a = cube(0.0, 1.0, dim=3)
    b = cube(0.5, 1.5, dim=3)
    assert expected_intersection_volume(a, b) == pytest.approx(
        expected_intersection_volume(b, a), rel=1e-6
    )


def test_disjoint_boxes_underflow_at_small_beta() -> None:
    """Disjoint boxes with a large gap should have near-zero expected volume."""
    a = cube(0.0, 1.0, dim=3)
    b = cube(10.0, 11.0, dim=3)
    assert expected_intersection_volume(a, b, beta=0.01) < 1e-20


def test_hard_intersection_limit_2d() -> None:
    """As β → 0, the intersection of [0,1]² and [0.5,1.5]² has volume 0.25."""
    a = cube(0.0, 1.0, dim=2)
    b = cube(0.5, 1.5, dim=2)
    assert expected_intersection_volume(a, b, beta=1e-5) == pytest.approx(0.25, rel=1e-2)


def test_intersect_boxes_returns_centered_box() -> None:
    a = cube(0.0, 1.0, dim=3)
    b = cube(0.5, 1.5, dim=3)
    result = intersect_boxes(a, b)
    assert result is not None
    np.testing.assert_allclose(result.center, 0.75, atol=1e-3)


def test_intersect_boxes_respects_log_volume_threshold() -> None:
    a = cube(0.0, 1.0, dim=3)
    b = cube(10.0, 11.0, dim=3)
    assert intersect_boxes(a, b, min_log_volume=float(np.log(1e-5))) is None


def test_larger_beta_smooths_disjoint_intersection() -> None:
    """Disjoint boxes get higher expected volume at larger β."""
    a = cube(0.0, 1.0, dim=2)
    b = cube(1.1, 2.1, dim=2)
    assert expected_intersection_volume(a, b, beta=1.0) > expected_intersection_volume(
        a, b, beta=0.01
    )


def test_expected_sides_match_hard_at_small_beta() -> None:
    a = cube(0.0, 1.0, dim=2)
    b = cube(0.25, 0.75, dim=2)
    sides = expected_intersection_sides(a, b, beta=1e-5)
    np.testing.assert_allclose(sides, 0.5, atol=1e-3)


def test_log_volume_consistent_with_volume() -> None:
    a = cube(0.0, 1.0, dim=4)
    b = cube(0.5, 1.5, dim=4)
    log_v = expected_intersection_log_volume(a, b)
    v = expected_intersection_volume(a, b)
    assert log_v == pytest.approx(float(np.log(v)), rel=1e-6)


def test_64d_intersection_stable() -> None:
    """Sanity: 64-D random boxes intersect without overflow/underflow in log space."""
    rng = np.random.default_rng(0)
    a_min = rng.standard_normal(64).astype(np.float32) - 0.5
    a_max = a_min + 1.0
    b_min = rng.standard_normal(64).astype(np.float32) - 0.5
    b_max = b_min + 1.0
    a = BoxExtent(min_corner=a_min.astype(np.float32), max_corner=a_max.astype(np.float32))
    b = BoxExtent(min_corner=b_min.astype(np.float32), max_corner=b_max.astype(np.float32))
    log_vol = expected_intersection_log_volume(a, b)
    assert np.isfinite(log_vol)
