"""Differentiable soft axis-aligned boxes for the curriculum pilot.

A box is (center, log_half): half-extent = softplus(log_half) + MIN_HALF,
corners are center ± half. Intersection takes the elementwise
max-of-mins / min-of-maxes. The *side length* of the intersection is
computed with a softplus, not a relu:

    side = beta * softplus( (max_i - min_i) / beta )

so that **disjoint boxes still receive gradient** (a hard relu gives
exactly zero gradient once two boxes separate, which is the classic
box-embedding training failure — and the slice already hit a related
underflow with a static index). As beta → 0 this recovers the hard
volume. log-volume sums log(side + eps) over dims.

Everything here is pure tensor logic with explicit numerical floors so
the analytic limits (identical / disjoint / contained boxes) are
testable without a model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MIN_HALF = 1e-3        # boxes never collapse to a point
DEFAULT_BETA = 0.1     # intersection softness
LOGV_EPS = 1e-6        # floor inside log(side + eps)


def half_extent(log_half: torch.Tensor) -> torch.Tensor:
    """Strictly-positive half-extent from an unconstrained parameter."""
    return F.softplus(log_half) + MIN_HALF


def corners(center: torch.Tensor, log_half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = half_extent(log_half)
    return center - h, center + h


def intersect(
    min_a: torch.Tensor, max_a: torch.Tensor,
    min_b: torch.Tensor, max_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise box intersection corners. If the boxes are disjoint
    in some dim then min_i > max_i there; `soft_side` handles that
    smoothly rather than clamping to zero."""
    return torch.maximum(min_a, min_b), torch.minimum(max_a, max_b)


def soft_side(
    min_i: torch.Tensor, max_i: torch.Tensor, beta: float = DEFAULT_BETA
) -> torch.Tensor:
    """Per-dim soft side length. Always > 0; smooth through the
    disjoint→overlapping transition so gradients never vanish."""
    return beta * F.softplus((max_i - min_i) / beta)


def log_volume(
    min_i: torch.Tensor, max_i: torch.Tensor, beta: float = DEFAULT_BETA
) -> torch.Tensor:
    """log soft-volume = sum over the last dim of log(soft_side + eps)."""
    side = soft_side(min_i, max_i, beta)
    return torch.log(side + LOGV_EPS).sum(dim=-1)


def box_log_volume(
    center: torch.Tensor, log_half: torch.Tensor, beta: float = DEFAULT_BETA
) -> torch.Tensor:
    lo, hi = corners(center, log_half)
    return log_volume(lo, hi, beta)


def intersection_log_volume(
    c_a: torch.Tensor, lh_a: torch.Tensor,
    c_b: torch.Tensor, lh_b: torch.Tensor,
    beta: float = DEFAULT_BETA,
) -> torch.Tensor:
    """log soft-volume of the intersection of two parameterized boxes."""
    lo_a, hi_a = corners(c_a, lh_a)
    lo_b, hi_b = corners(c_b, lh_b)
    lo_i, hi_i = intersect(lo_a, hi_a, lo_b, hi_b)
    return log_volume(lo_i, hi_i, beta)


def intersection_embedding(
    c_a: torch.Tensor, lh_a: torch.Tensor,
    c_b: torch.Tensor, lh_b: torch.Tensor,
) -> torch.Tensor:
    """A dense vector summarizing the intersection region: the midpoint
    of the (possibly empty) intersection box concatenated with its
    per-dim soft side. Used as the input to the Vesica-prediction and
    parent-reconstruction heads. Differentiable everywhere."""
    lo_a, hi_a = corners(c_a, lh_a)
    lo_b, hi_b = corners(c_b, lh_b)
    lo_i, hi_i = intersect(lo_a, hi_a, lo_b, hi_b)
    mid = 0.5 * (lo_i + hi_i)
    side = soft_side(lo_i, hi_i)
    return torch.cat([mid, side], dim=-1)
