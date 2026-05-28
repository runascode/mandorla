"""GumbelBox math per Dasgupta et al. NeurIPS 2020 (arXiv:2010.04831).

The slice uses the closed-form softplus expression from Lemma 1 of that paper
for expected per-dimension intersection side length, then a product across
dimensions for expected volume. β is a small positive temperature; β → 0
recovers hard-box intersection, β → ∞ over-smooths.

PRECOMMIT.md fixes β = 0.01 for the slice: small enough that the geometry is
near-hard (the boxes really do represent regions, not blurry shells), non-zero
so almost-disjoint boxes still get well-defined positive expected volume
(useful for scoring borderline candidates without zero-gradient pathologies).

All computations are in float64 internally for numerical stability in high
dimensions; outputs are cast back to float32 for storage consistency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .regions import BoxExtent, Vec

DEFAULT_BETA: float = 0.01

# Floor for log-volume computations. Well below any meaningful intersection
# in 64-D under our calibration; prevents log(0) in degenerate cases.
_LOG_FLOOR: float = 1e-30


def _softplus(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softplus: ln(1 + exp(x)). For large positive x,
    returns x (avoiding overflow). For very negative x, returns ~0."""
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -50.0, 30.0))))


def expected_intersection_sides(
    box_a: BoxExtent,
    box_b: BoxExtent,
    beta: float = DEFAULT_BETA,
) -> Vec:
    """Per-dimension expected intersection side length under the GumbelBox
    model (Dasgupta 2020, Lemma 1):

        E[side_j] = β · softplus((min(max_a_j, max_b_j) - max(min_a_j, min_b_j)) / β)

    Returns a length-d float32 vector. As β → 0, this recovers the hard
    intersection side length max(0, upper - lower).
    """
    if box_a.dim != box_b.dim:
        raise ValueError(f"Box dimension mismatch: {box_a.dim} vs {box_b.dim}")
    if beta <= 0:
        raise ValueError(f"beta must be positive; got {beta}")

    upper = np.minimum(box_a.max_corner, box_b.max_corner).astype(np.float64)
    lower = np.maximum(box_a.min_corner, box_b.min_corner).astype(np.float64)
    sides = beta * _softplus((upper - lower) / beta)
    return sides.astype(np.float32)


def expected_intersection_log_volume(
    box_a: BoxExtent,
    box_b: BoxExtent,
    beta: float = DEFAULT_BETA,
) -> float:
    """Log of expected intersection volume = sum of log per-dimension expected
    sides. Use this for scoring; expected_intersection_volume can underflow to
    zero in 64-D for non-overlapping inputs."""
    sides = expected_intersection_sides(box_a, box_b, beta=beta)
    return float(np.sum(np.log(np.maximum(sides.astype(np.float64), _LOG_FLOOR))))


def expected_intersection_volume(
    box_a: BoxExtent,
    box_b: BoxExtent,
    beta: float = DEFAULT_BETA,
) -> float:
    """Expected intersection volume = product of per-dimension expected sides.

    Computed in log space then exponentiated to handle underflow in high
    dimensions. Returns a non-negative float; may be ~0 for truly disjoint
    boxes with small β.
    """
    return float(np.exp(expected_intersection_log_volume(box_a, box_b, beta=beta)))


def intersect_boxes(
    box_a: BoxExtent,
    box_b: BoxExtent,
    beta: float = DEFAULT_BETA,
    min_log_volume: Optional[float] = None,
) -> Optional[BoxExtent]:
    """Compute the GumbelBox intersection of two boxes as a new BoxExtent.

    The intersection box is centered at the midpoint of the hard corner-
    intersection and has per-dimension width equal to the expected side under
    the softplus formulation. As β → 0 this is the standard axis-aligned
    intersection.

    If `min_log_volume` is provided and the expected log-volume falls below it,
    returns None (interpreted as "no meaningful Vesica spawned").
    """
    sides = expected_intersection_sides(box_a, box_b, beta=beta)
    log_vol = float(np.sum(np.log(np.maximum(sides.astype(np.float64), _LOG_FLOOR))))
    if min_log_volume is not None and log_vol < min_log_volume:
        return None

    lower_hard = np.maximum(box_a.min_corner, box_b.min_corner).astype(np.float64)
    upper_hard = np.minimum(box_a.max_corner, box_b.max_corner).astype(np.float64)
    center = 0.5 * (upper_hard + lower_hard)
    half = 0.5 * sides.astype(np.float64)
    return BoxExtent(
        min_corner=(center - half).astype(np.float32),
        max_corner=(center + half).astype(np.float32),
    )


def box_volume(box: BoxExtent) -> float:
    """Volume of a single box (product of side lengths). Returns 0 for
    degenerate (non-positive-sided) boxes."""
    sides = (box.max_corner - box.min_corner).astype(np.float64)
    if np.any(sides <= 0):
        return 0.0
    return float(np.exp(np.sum(np.log(sides))))


def box_log_volume(box: BoxExtent) -> float:
    """Log of box_volume. Returns -inf for degenerate boxes."""
    sides = (box.max_corner - box.min_corner).astype(np.float64)
    if np.any(sides <= 0):
        return float("-inf")
    return float(np.sum(np.log(sides)))
