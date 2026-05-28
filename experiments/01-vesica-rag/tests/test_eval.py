"""Unit tests for HotpotQA F1/EM normalization, vesica-coverage, and bootstrap.

F1/EM cases mirror the canonical HotpotQA scorer behavior. Bootstrap tests
verify CI determinism under fixed seed and sanity of CI bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval import (
    bootstrap_mean,
    bootstrap_paired_difference,
    exact_match,
    f1_score,
    normalize_answer,
    vesica_covered,
)


# ─── normalize_answer ───────────────────────────────────────────────────────


def test_normalize_lowercases() -> None:
    assert normalize_answer("Apollo") == "apollo"


def test_normalize_strips_articles() -> None:
    assert normalize_answer("The apple") == "apple"
    assert normalize_answer("a banana") == "banana"
    assert normalize_answer("an orange") == "orange"


def test_normalize_strips_punctuation() -> None:
    assert normalize_answer("Hello, world!") == "hello world"


def test_normalize_collapses_whitespace() -> None:
    assert normalize_answer("  hello   world  ") == "hello world"


def test_normalize_empty() -> None:
    assert normalize_answer("") == ""


# ─── exact_match ────────────────────────────────────────────────────────────


def test_em_identical() -> None:
    assert exact_match("yes", "yes") == 1.0


def test_em_different() -> None:
    assert exact_match("yes", "no") == 0.0


def test_em_normalized_match() -> None:
    """EM is insensitive to articles, case, punctuation."""
    assert exact_match("The Apollo Program!", "apollo program") == 1.0


# ─── f1_score ───────────────────────────────────────────────────────────────


def test_f1_exact() -> None:
    assert f1_score("apollo program", "apollo program") == pytest.approx(1.0)


def test_f1_partial_overlap() -> None:
    # pred="the apollo program" → tokens ["apollo", "program"]
    # gold="apollo 11 program" → tokens ["apollo", "11", "program"]
    # common = {"apollo", "program"} → n_same = 2
    # precision = 2/2 = 1, recall = 2/3 → F1 = 0.8
    assert f1_score("the apollo program", "apollo 11 program") == pytest.approx(0.8, rel=1e-3)


def test_f1_no_overlap() -> None:
    assert f1_score("apollo", "banana") == 0.0


def test_f1_yes_no() -> None:
    """Yes/no answers behave like EM."""
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("no", "yes") == 0.0
    assert f1_score("yes please", "yes") == 0.0  # special-case behavior


def test_f1_empty_both() -> None:
    assert f1_score("", "") == 1.0


def test_f1_empty_one_side() -> None:
    assert f1_score("", "apollo") == 0.0
    assert f1_score("apollo", "") == 0.0


# ─── vesica_covered ─────────────────────────────────────────────────────────


def test_vesica_covered_exact_pair() -> None:
    """Standard 2-supporting-fact case: covered iff parents == golds."""
    gold = {"chunk_a", "chunk_b"}
    assert vesica_covered(gold, [("chunk_a", "chunk_b")]) is True
    assert vesica_covered(gold, [("chunk_b", "chunk_a")]) is True   # order-invariant


def test_vesica_covered_missing_one_parent() -> None:
    gold = {"chunk_a", "chunk_b"}
    assert vesica_covered(gold, [("chunk_a", "chunk_x")]) is False


def test_vesica_covered_no_candidates() -> None:
    assert vesica_covered({"a", "b"}, []) is False


def test_vesica_covered_among_many_candidates() -> None:
    gold = {"a", "b"}
    cands = [("x", "y"), ("a", "z"), ("b", "z"), ("a", "b"), ("c", "d")]
    assert vesica_covered(gold, cands) is True


def test_vesica_covered_three_golds_any_subset_matches() -> None:
    """For 3+ gold paragraphs (rare): any 2-subset match counts."""
    gold = {"a", "b", "c"}
    assert vesica_covered(gold, [("a", "c")]) is True
    assert vesica_covered(gold, [("a", "d")]) is False


def test_vesica_covered_empty_gold() -> None:
    assert vesica_covered(set(), [("a", "b")]) is False


# ─── bootstrap_mean ─────────────────────────────────────────────────────────


def test_bootstrap_mean_point_estimate_is_sample_mean() -> None:
    values = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    result = bootstrap_mean(values, n_resamples=1000, seed=0)
    assert result.point_estimate == pytest.approx(0.4)


def test_bootstrap_mean_deterministic_under_seed() -> None:
    values = np.random.RandomState(0).random(50)
    r1 = bootstrap_mean(values, n_resamples=500, seed=1337)
    r2 = bootstrap_mean(values, n_resamples=500, seed=1337)
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high


def test_bootstrap_mean_ci_brackets_point() -> None:
    """CI should always bracket the point estimate."""
    values = np.array([0.0, 0.5, 1.0] * 50, dtype=np.float64)
    r = bootstrap_mean(values, n_resamples=500)
    assert r.ci_low <= r.point_estimate <= r.ci_high


def test_bootstrap_mean_narrow_ci_for_uniform_values() -> None:
    """All-equal values → near-zero CI width."""
    values = np.full(100, 0.7, dtype=np.float64)
    r = bootstrap_mean(values, n_resamples=500)
    assert r.ci_high - r.ci_low == pytest.approx(0.0, abs=1e-9)


# ─── bootstrap_paired_difference ────────────────────────────────────────────


def test_paired_difference_point_estimate() -> None:
    baseline = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    treatment = np.array([0.6, 0.8, 0.7], dtype=np.float64)
    # diffs = [0.1, 0.2, 0.0]; mean diff = 0.1
    r = bootstrap_paired_difference(baseline, treatment, n_resamples=500)
    assert r.point_estimate == pytest.approx(0.1)


def test_paired_difference_shape_mismatch_raises() -> None:
    baseline = np.zeros(10)
    treatment = np.zeros(11)
    with pytest.raises(ValueError):
        bootstrap_paired_difference(baseline, treatment)


def test_paired_difference_deterministic_under_seed() -> None:
    rng = np.random.RandomState(0)
    baseline = rng.random(40)
    treatment = baseline + rng.normal(0, 0.05, 40)
    r1 = bootstrap_paired_difference(baseline, treatment, n_resamples=500, seed=42)
    r2 = bootstrap_paired_difference(baseline, treatment, n_resamples=500, seed=42)
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high
