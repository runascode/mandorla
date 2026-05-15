"""Analytic-limit tests for the soft box module.

Each test pins behavior at a named geometric limit so a regression in
the intersection / volume math surfaces here rather than silently
during training.
"""

from __future__ import annotations

import torch

from src.box import (
    MIN_HALF,
    box_log_volume,
    corners,
    intersect,
    intersection_embedding,
    intersection_log_volume,
    soft_side,
)


def test_identical_boxes_intersection_is_the_box():
    c = torch.tensor([[0.0, 1.0, -2.0]])
    lh = torch.tensor([[0.5, 0.5, 0.5]])
    self_iv = intersection_log_volume(c, lh, c, lh, beta=1e-3)
    box_v = box_log_volume(c, lh, beta=1e-3)
    torch.testing.assert_close(self_iv, box_v, rtol=1e-3, atol=1e-3)


def test_contained_box_intersection_equals_inner():
    c = torch.tensor([[0.0, 0.0]])
    inner_lh = torch.tensor([[-3.0, -3.0]])   # small softplus → small half
    outer_lh = torch.tensor([[3.0, 3.0]])     # large half, same center
    iv = intersection_log_volume(c, inner_lh, c, outer_lh, beta=1e-3)
    inner_v = box_log_volume(c, inner_lh, beta=1e-3)
    torch.testing.assert_close(iv, inner_v, rtol=1e-3, atol=1e-3)


def test_disjoint_boxes_have_smaller_logvolume_than_overlapping():
    c_a = torch.tensor([[0.0]])
    c_far = torch.tensor([[50.0]])
    c_near = torch.tensor([[0.1]])
    lh = torch.tensor([[0.0]])
    far = intersection_log_volume(c_a, lh, c_far, lh)
    near = intersection_log_volume(c_a, lh, c_near, lh)
    assert far.item() < near.item()


def test_disjoint_boxes_in_training_regime_receive_gradient():
    """The point of the softplus side: a hard relu gives exactly zero
    gradient once boxes separate. The soft side keeps gradient alive
    through the disjoint→overlap transition and at *moderate*
    separation — the regime the training loop actually operates in
    (embeddings are small-init and lightly regularized, so box centers
    don't drift tens of units apart). Boxes here are disjoint by ~1.1
    units with the default half-extent; gradient must still flow."""
    c_a = torch.zeros(1, 1, requires_grad=True)
    c_b = torch.full((1, 1), 2.5, requires_grad=True)   # half≈0.69 each → gap≈1.1
    lh = torch.zeros(1, 1)
    lv = intersection_log_volume(c_a, lh, c_b, lh, beta=0.5)
    lv.sum().backward()
    assert c_a.grad is not None and c_b.grad is not None
    assert torch.isfinite(c_a.grad).all()
    assert c_a.grad.abs().sum() > 0.0


def test_extreme_separation_saturates_gradient_known_limitation():
    """Documented limitation, not a bug: no fixed-beta softplus keeps
    gradient alive at *arbitrary* separation — the tail goes flat. At
    40 units apart with beta=0.1 the gradient underflows to ~0. This is
    why the train loop must keep box centers bounded (small init + mild
    weight decay on the box head); it is recorded here so the behavior
    is a tracked property, not a surprise during Q1 stability analysis."""
    c_a = torch.zeros(1, 1, requires_grad=True)
    c_b = torch.full((1, 1), 40.0, requires_grad=True)
    lh = torch.zeros(1, 1)
    intersection_log_volume(c_a, lh, c_b, lh, beta=0.1).sum().backward()
    # Gradient is finite (no NaN/Inf) but has saturated to zero.
    assert torch.isfinite(c_a.grad).all()
    assert c_a.grad.abs().sum() == 0.0


def test_larger_box_has_larger_logvolume():
    c = torch.zeros(1, 4)
    small = torch.full((1, 4), -2.0)
    big = torch.full((1, 4), 2.0)
    assert box_log_volume(c, big).item() > box_log_volume(c, small).item()


def test_half_extent_floor_enforced():
    lo, hi = corners(torch.zeros(1, 3), torch.full((1, 3), -1e9))
    # softplus(-1e9) ≈ 0, so half ≈ MIN_HALF
    side = (hi - lo) / 2.0
    assert torch.all(side >= MIN_HALF - 1e-6)


def test_intersection_embedding_shape_and_finiteness():
    c_a = torch.randn(5, 8)
    lh_a = torch.randn(5, 8)
    c_b = torch.randn(5, 8)
    lh_b = torch.randn(5, 8)
    emb = intersection_embedding(c_a, lh_a, c_b, lh_b)
    assert emb.shape == (5, 16)         # midpoint(8) ++ side(8)
    assert torch.isfinite(emb).all()


def test_soft_side_strictly_positive_even_when_disjoint():
    lo_i = torch.tensor([[5.0]])
    hi_i = torch.tensor([[-5.0]])       # max_i < min_i  → disjoint
    s = soft_side(lo_i, hi_i)
    assert torch.all(s > 0.0)
    assert torch.isfinite(s).all()


def test_intersection_is_symmetric():
    c_a, lh_a = torch.randn(3, 6), torch.randn(3, 6)
    c_b, lh_b = torch.randn(3, 6), torch.randn(3, 6)
    ab = intersection_log_volume(c_a, lh_a, c_b, lh_b)
    ba = intersection_log_volume(c_b, lh_b, c_a, lh_a)
    torch.testing.assert_close(ab, ba)


def test_intersect_corners_are_max_of_mins_min_of_maxes():
    min_a = torch.tensor([[0.0, 0.0]])
    max_a = torch.tensor([[2.0, 2.0]])
    min_b = torch.tensor([[1.0, -1.0]])
    max_b = torch.tensor([[3.0, 1.0]])
    lo, hi = intersect(min_a, max_a, min_b, max_b)
    torch.testing.assert_close(lo, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(hi, torch.tensor([[2.0, 1.0]]))
