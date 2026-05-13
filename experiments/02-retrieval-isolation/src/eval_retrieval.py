"""Pure-retrieval metrics for Experiment 02.

All metrics operate over **titles**, not chunk_ids, because:

- HotpotQA's gold supporting facts are titles + sentence ids; the slice's
  chunk_ids index the corpus by passage, not by sentence. A title-level
  match is the correct cross-dataset operationalization, also matching
  the standard HotpotQA retrieval-eval convention.
- 2Wiki and MuSiQue's gold passages come from a different Wikipedia dump
  than the slice's corpus, so any chunk-id-level matching would
  introduce spurious mismatches that title-level matching avoids.

Bootstrap CIs are computed with 10,000 resamples by default; the master
seed is 1337 to match the slice's discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int


def _bootstrap_mean(
    xs: np.ndarray, n_resamples: int, seed: int
) -> BootstrapResult:
    if len(xs) == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), n_resamples)
    rng = np.random.default_rng(seed)
    n = len(xs)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idxs = rng.integers(0, n, size=n)
        means[i] = float(xs[idxs].mean())
    return BootstrapResult(
        point_estimate=float(xs.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        n_resamples=n_resamples,
    )


def _bootstrap_paired_diff(
    a: np.ndarray, b: np.ndarray, n_resamples: int, seed: int
) -> BootstrapResult:
    """CI on b - a, paired over questions."""
    if len(a) != len(b):
        raise ValueError("paired arrays must have the same length")
    if len(a) == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), n_resamples)
    rng = np.random.default_rng(seed)
    diff = b - a
    n = len(diff)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idxs = rng.integers(0, n, size=n)
        means[i] = float(diff[idxs].mean())
    return BootstrapResult(
        point_estimate=float(diff.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        n_resamples=n_resamples,
    )


def pair_recall_at_k(
    retrieved_titles_in_order: Sequence[str],
    gold_titles: Sequence[str],
    k: int,
) -> float:
    """1.0 iff every gold title appears in the top-k retrieved set."""
    if not gold_titles:
        return float("nan")
    top = set(retrieved_titles_in_order[:k])
    return 1.0 if set(gold_titles).issubset(top) else 0.0


def any_gold_recall_at_k(
    retrieved_titles_in_order: Sequence[str],
    gold_titles: Sequence[str],
    k: int,
) -> float:
    """1.0 iff at least one gold title appears in the top-k retrieved set."""
    if not gold_titles:
        return float("nan")
    top = set(retrieved_titles_in_order[:k])
    return 1.0 if (top & set(gold_titles)) else 0.0


def reciprocal_rank_first_pair(
    retrieved_titles_in_order: Sequence[str],
    gold_titles: Sequence[str],
    max_k: int = 25,
) -> float:
    """1/k where k is the smallest prefix containing all gold titles, or
    0 if no such k <= max_k exists. NaN if there are no gold titles."""
    if not gold_titles:
        return float("nan")
    gold = set(gold_titles)
    seen: set[str] = set()
    for i, t in enumerate(retrieved_titles_in_order[:max_k], start=1):
        if t in gold:
            seen.add(t)
            if seen == gold:
                return 1.0 / i
    return 0.0


def vesica_covered_pair_titles(
    candidate_vesica_parent_titles: Sequence[Sequence[str]],
    gold_titles: Sequence[str],
) -> float:
    """Slice-compatible Vesica-coverage at the title level: 1.0 iff some
    candidate Vesica's parent-pair titles is a subset of the gold titles."""
    if not gold_titles:
        return float("nan")
    gold = set(gold_titles)
    for pair_titles in candidate_vesica_parent_titles:
        if set(pair_titles).issubset(gold):
            return 1.0
    return 0.0
