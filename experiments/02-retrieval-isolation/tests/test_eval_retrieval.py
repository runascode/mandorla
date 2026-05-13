"""Unit tests for retrieval-isolation metrics.

Deterministic, no network, no real data. Each test exercises the
contract at named edge cases (empty, partial, full, position-sensitive)
so a regression that changes the metric's behavior at one of those
points will surface here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval_retrieval import (
    _bootstrap_mean,
    _bootstrap_paired_diff,
    any_gold_recall_at_k,
    pair_recall_at_k,
    reciprocal_rank_first_pair,
    vesica_covered_pair_titles,
)


# ─── Pair-Recall@k ──────────────────────────────────────────────────────────


def test_pair_recall_perfect_at_smallest_k():
    assert pair_recall_at_k(["A", "B", "C"], gold_titles=["A", "B"], k=2) == 1.0


def test_pair_recall_misses_when_one_gold_outside_topk():
    assert pair_recall_at_k(["A", "X", "Y", "Z", "B"], gold_titles=["A", "B"], k=4) == 0.0


def test_pair_recall_with_three_gold_titles():
    assert pair_recall_at_k(["A", "B", "C", "D"], gold_titles=["A", "B", "C"], k=3) == 1.0
    assert pair_recall_at_k(["A", "B", "X", "D"], gold_titles=["A", "B", "C"], k=4) == 0.0


def test_pair_recall_nan_on_no_gold():
    assert math.isnan(pair_recall_at_k(["A", "B"], gold_titles=[], k=2))


def test_pair_recall_order_irrelevant_inside_topk():
    """Pair-recall is a set test inside top-k. Order within k doesn't matter."""
    assert pair_recall_at_k(["B", "A", "C"], gold_titles=["A", "B"], k=2) == 1.0


# ─── Any-Gold-Recall@k ──────────────────────────────────────────────────────


def test_any_gold_recall_hits_on_single_match():
    assert any_gold_recall_at_k(["X", "Y", "A", "Z"], gold_titles=["A", "B"], k=4) == 1.0


def test_any_gold_recall_misses_when_no_overlap():
    assert any_gold_recall_at_k(["X", "Y"], gold_titles=["A", "B"], k=2) == 0.0


def test_any_gold_recall_with_empty_gold_is_nan():
    assert math.isnan(any_gold_recall_at_k(["A"], gold_titles=[], k=1))


# ─── Reciprocal Rank of First Gold-Pair Completion ──────────────────────────


def test_rr_first_pair_at_rank_2_when_immediate():
    """gold = {A, B}; retrieved = [A, B, C, ...] → both completed at rank 2 → 1/2."""
    assert reciprocal_rank_first_pair(["A", "B", "C"], gold_titles=["A", "B"], max_k=25) == pytest.approx(0.5)


def test_rr_first_pair_at_rank_n_with_distractors_between():
    """gold = {A, B}; retrieved = [A, X, X, B] → completed at rank 4 → 1/4."""
    assert reciprocal_rank_first_pair(["A", "X", "X", "B"], gold_titles=["A", "B"], max_k=25) == pytest.approx(0.25)


def test_rr_first_pair_zero_when_never_completed_within_maxk():
    assert reciprocal_rank_first_pair(["A", "X", "Y"], gold_titles=["A", "B"], max_k=3) == 0.0


def test_rr_first_pair_three_gold_titles():
    """gold = {A, B, C}; retrieved = [A, B, X, C, ...] → completed at rank 4 → 1/4."""
    assert reciprocal_rank_first_pair(["A", "B", "X", "C", "D"], gold_titles=["A", "B", "C"], max_k=25) == pytest.approx(0.25)


def test_rr_first_pair_nan_on_no_gold():
    assert math.isnan(reciprocal_rank_first_pair(["A"], gold_titles=[], max_k=10))


# ─── Vesica-coverage (title-level) ──────────────────────────────────────────


def test_vesica_covered_pair_titles_hit():
    assert vesica_covered_pair_titles([["X", "Y"], ["A", "B"]], gold_titles=["A", "B"]) == 1.0


def test_vesica_covered_pair_titles_miss_when_no_parent_pair_subset():
    assert vesica_covered_pair_titles([["X", "Y"], ["A", "X"]], gold_titles=["A", "B"]) == 0.0


def test_vesica_covered_pair_titles_hit_when_3_gold():
    """One Vesica parent-pair {A, B} ⊆ gold {A, B, C} → covered."""
    assert vesica_covered_pair_titles([["A", "B"]], gold_titles=["A", "B", "C"]) == 1.0


def test_vesica_covered_pair_titles_nan_no_gold():
    assert math.isnan(vesica_covered_pair_titles([["A", "B"]], gold_titles=[]))


# ─── Bootstrap utilities (numerical correctness at known limits) ────────────


def test_bootstrap_mean_point_estimate_matches_sample_mean():
    xs = np.array([0.2, 0.4, 0.6, 0.8])
    br = _bootstrap_mean(xs, n_resamples=200, seed=1337)
    assert br.point_estimate == pytest.approx(0.5, rel=1e-9)
    assert br.ci_low < br.point_estimate <= br.ci_high


def test_bootstrap_mean_empty_returns_nan():
    br = _bootstrap_mean(np.array([]), n_resamples=10, seed=1337)
    assert math.isnan(br.point_estimate)
    assert math.isnan(br.ci_low)
    assert math.isnan(br.ci_high)


def test_bootstrap_paired_diff_point_estimate():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    br = _bootstrap_paired_diff(a, b, n_resamples=100, seed=1337)
    assert br.point_estimate == pytest.approx(1.0)
    # all-equal diffs → bootstrap CI is degenerate at 1.0
    assert br.ci_low == pytest.approx(1.0)
    assert br.ci_high == pytest.approx(1.0)


def test_bootstrap_paired_diff_length_mismatch_raises():
    with pytest.raises(ValueError):
        _bootstrap_paired_diff(np.array([0.0, 1.0]), np.array([0.0]), n_resamples=10, seed=1337)
