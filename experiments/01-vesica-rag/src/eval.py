"""HotpotQA evaluation metrics + vesica-coverage diagnostic + bootstrap CIs.

The F1/EM normalization here matches the official HotpotQA eval script
(`hotpot_evaluate_v1.py` released alongside Yang et al. 2018): lowercase,
strip articles, strip punctuation, collapse whitespace, then token-level
precision/recall/F1.

The vesica-coverage diagnostic is the slice's primary go/no-go number per
PRECOMMIT.md: fraction of dev questions where the gold supporting-paragraph
pair was identified by the retriever as a single candidate Vesica before
generation. "Identified as a single Vesica" means: the Vesica's two parent
chunk_ids equal (or are a subset of) the gold-supporting-paragraph chunk_id
set.

Bootstrap CIs use paired resampling — for each resample, we draw the same
question indices for both baseline and Vesica-RAG, so the difference's CI
reflects the within-question paired difference rather than between-condition
variance from independent samples.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ─── HotpotQA F1/EM normalization (matches official eval) ───────────────────

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """HotpotQA-official answer normalization."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split() if s else []


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 on HotpotQA-normalized strings."""
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    # For yes/no/noanswer "exact-match-style" answers, fall back to EM-like.
    if gold_toks in (["yes"], ["no"], ["noanswer"]) or pred_toks in (
        ["yes"],
        ["no"],
        ["noanswer"],
    ):
        return 1.0 if pred_toks == gold_toks else 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalized strings are identical, else 0.0."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


# ─── Vesica-coverage diagnostic ─────────────────────────────────────────────

def vesica_covered(
    gold_chunk_ids: set[str],
    candidate_vesica_parents: list[tuple[str, str]],
) -> bool:
    """A candidate Vesica covers the gold pair iff its two parent chunk_ids
    are a subset of the gold supporting-paragraph chunk_id set.

    For the typical 2-supporting-fact case this is equivalent to set equality
    between the Vesica's parents and the gold pair. For rarer cases with >2
    gold paragraphs, any 2-subset match counts as covered.
    """
    if not gold_chunk_ids:
        return False
    for a, b in candidate_vesica_parents:
        if {a, b}.issubset(gold_chunk_ids):
            return True
    return False


# ─── Bootstrap CIs ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap CI on a scalar metric."""

    point_estimate: float       # metric on the actual sample
    ci_low: float               # 2.5th percentile of resampled values
    ci_high: float              # 97.5th percentile
    n_resamples: int


def bootstrap_mean(
    values: NDArray[np.float64],
    n_resamples: int = 10_000,
    seed: int = 1337,
) -> BootstrapResult:
    """Bootstrap 95% CI on the mean of a 1-D array of per-question scores."""
    if values.ndim != 1:
        raise ValueError(f"Expected 1-D array; got {values.ndim}-D")
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    point = float(values.mean())
    return BootstrapResult(
        point_estimate=point,
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        n_resamples=n_resamples,
    )


def bootstrap_paired_difference(
    baseline: NDArray[np.float64],
    treatment: NDArray[np.float64],
    n_resamples: int = 10_000,
    seed: int = 1337,
) -> BootstrapResult:
    """Bootstrap 95% CI on the per-question paired difference treatment - baseline.

    Resamples question indices once per bootstrap iteration and draws the
    paired difference at those indices. This is the slice's headline-lift CI.
    """
    if baseline.shape != treatment.shape:
        raise ValueError(
            f"Shape mismatch: baseline {baseline.shape} vs treatment {treatment.shape}"
        )
    if baseline.ndim != 1:
        raise ValueError(f"Expected 1-D arrays; got {baseline.ndim}-D")
    diffs = treatment - baseline
    rng = np.random.default_rng(seed)
    n = diffs.shape[0]
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(diffs[idx].mean())
    return BootstrapResult(
        point_estimate=float(diffs.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        n_resamples=n_resamples,
    )
