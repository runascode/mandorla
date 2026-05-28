# Slice Diagnostic (post-hoc, exploratory)

**This is exploratory analysis on the existing slice JSONLs. It does not change the binding NO-GO decision in [`PRECOMMIT.md`](../PRECOMMIT.md). It exists to inform the next pre-commit.**

## H1 — is coverage the bottleneck?

If the gold pair *is* surfaced as a candidate Vesica, does Vesica-RAG outperform the contriever baseline?

| Slice | n | Baseline F1 | Vesica-RAG F1 | F1 lift |
|---|---:|---|---|---|
| vesica_covered=True | 323 | 0.64 (CI 0.60, 0.69) | 0.63 (CI 0.58, 0.68) | **-0.01 (CI -0.04, 0.02)** |
| vesica_covered=False | 7082 | 0.45 (CI 0.44, 0.46) | 0.43 (CI 0.42, 0.44) | **-0.02 (CI -0.02, -0.01)** |
| all | 7405 | 0.46 (CI 0.45, 0.47) | 0.44 (CI 0.43, 0.45) | **-0.02 (CI -0.02, -0.01)** |

## By question type × coverage

| Slice | n | Baseline F1 | Vesica-RAG F1 | F1 lift |
|---|---:|---|---|---|
| bridge & covered | 190 | 0.68 (CI 0.62, 0.74) | 0.66 (CI 0.60, 0.72) | **-0.02 (CI -0.06, 0.02)** |
| bridge & not covered | 5728 | 0.42 (CI 0.41, 0.43) | 0.40 (CI 0.39, 0.41) | **-0.02 (CI -0.03, -0.01)** |
| comparison & covered | 133 | 0.59 (CI 0.51, 0.67) | 0.59 (CI 0.51, 0.67) | **0.00 (CI -0.06, 0.06)** |
| comparison & not covered | 1354 | 0.58 (CI 0.55, 0.60) | 0.58 (CI 0.56, 0.60) | **0.00 (CI -0.02, 0.02)** |

## H3 — does Vesica context displace useful baseline context?

- Mean baseline context size: **25.0** chunks
- Mean Vesica-RAG context size: **23.8** chunks
- Mean overlap between the two contexts: **8.3** chunks (median 8, p10–p90 6–10)
- Mean **baseline-only** chunks (dropped by Vesica-RAG): **16.7**

Gold-chunk recall in the final LLM context:

| Recall metric | Baseline | Vesica-RAG |
|---|---|---|
| Any gold chunk present | 92.05% | 87.64% |
| Full gold pair present | 41.39% | 30.74% |

## Interpretation

Three hypotheses were on the table for why a +4.36% vesica-coverage signal didn't move F1. Reading the tables above:

**H1 — coverage is the bottleneck. Falsified.** On the 323 questions where the gold pair *was* surfaced as a candidate Vesica, the F1 lift is **−0.01 (CI −0.04, +0.02)**. The intersection primitive doing its best — exactly catching the right pair — produces no detectable F1 improvement. The primitive is not selecting evidence the LLM uses differently. Stratifying further: bridge & covered (n=190) gives −0.02 (CI −0.06, +0.02); comparison & covered (n=133) gives 0.00 (CI −0.06, +0.06). At these sample sizes a real ≥+1 F1 lift would have been detectable. It is not present.

**H2 — Vesicas don't help the LLM. Supported.** Same evidence: even on the coverage-hit subset, the gain is null. The LLM is not differentially extracting from Vesica-augmented context vs. dense top-25.

**H3 — Vesicas displace useful baseline context. Quantified and confirmed (but not the bottleneck).** Vesica-RAG's mean context contains **23.8** chunks (vs. baseline's 25.0); the overlap between the two contexts is **8.3** chunks; an average of **16.7** baseline chunks are dropped per question. The cost shows up cleanly in gold-recall: the baseline puts the full gold pair in the LLM's context **41.39%** of the time; Vesica-RAG does **30.74%** — an **11 pp drop** caused by displacement. Any-gold-chunk recall also drops, from 92.05% to 87.64%. So Vesica-RAG is *materially worse at delivering gold evidence to the model*, and yet the coverage-hit subset shows no F1 penalty when the gold pair IS preserved. Conclusion: displacement is real, but it's not the primary loss mechanism — the LLM is saturated enough at top-25 dense retrieval that whether the gold pair is in context or not doesn't move the dial as much as you'd expect.

### What this implies for the next experiment

The screening slice's RAG operationalization confounded two questions:

1. *Does the intersection primitive retrieve different (better) evidence than dense?*
2. *Does a frontier-class extractive QA model convert that into better answers on HotpotQA?*

The slice answered (2) with a clear no, and used that to answer (1) negatively too — but the diagnostic above shows the LLM-saturation effect is plausibly hiding any (1)-level signal that might exist. The next experiment must **isolate (1)**: evaluate the primitive on pure retrieval-side metrics, with no generation in the loop, on multiple multi-hop datasets. If even there the primitive does not move retrieval metrics, it is dead in this projection and we move to a different experiment. If it does, the retrieval-as-cognition claim survives the slice, and B3 (learned box head) becomes the natural follow-up with a sharper target.

The design for that next experiment is locked at [`experiments/02-retrieval-isolation/PRECOMMIT.md`](../../02-retrieval-isolation/PRECOMMIT.md).

