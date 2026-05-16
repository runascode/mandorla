# Reader saturation makes HotpotQA answer-metrics a weak instrument for evaluating retrieval

**Jacob Patterson** · runascode@protonmail.com · 2026-05-15
Draft research note. Thesis-independent (see "Provenance and independence").

## Summary

On the HotpotQA distractor-dev set, a frontier-class 8B reader
(`llama3.1:8b-instruct-q5_K_M`, greedy) reconstructs answers well
enough from partial context that the answer metrics (F1/EM) become
**insensitive to large differences in retrieval quality**. Two
independent measurements:

1. Changing whether the *gold supporting-paragraph pair* is even
   present in the reader's context by **~11 percentage points**
   (41.39% → 30.74%) moved answer **F1 by only ~1.6 points**.
2. On the subset of questions where the correct supporting pair was
   *definitively* surfaced (n=323), the F1 difference between two
   retrieval methods was **−0.01 (95% CI −0.04, +0.02)** — i.e.,
   indistinguishable from zero.

Over the same conditions, *retrieval-side* metrics differed by **7–11
points** (Pair-Recall@25). The retrieval differences are real and
large; the answer metric is nearly blind to them. The implication: a
HotpotQA F1/EM improvement attributed to a retrieval method, measured
with a strong reader at a typical context budget, is a **low-power test
of retrieval quality** — it largely measures reader robustness
(parametric knowledge + single-supporting-fact sufficiency), not how
good the retrieval was.

This is a statement about an **evaluation regime**, not a universal
law. But it is the regime a large fraction of the RAG-improvement
literature uses.

## Setup

All numbers are from a controlled comparison run for an unrelated
project (see "Provenance"). Two retrieval conditions over the **same**
full-Wikipedia corpus (5,233,329 BeIR/HotpotQA abstract chunks), the
**same** reader, the **same** prompt and decoding
(`temperature=0, seed=1337, num_ctx=8192, num_predict=128`), the
**same** ~25-chunk context budget:

- **Dense baseline:** contriever (`facebook/contriever-msmarco`)
  top-25 by cosine.
- **Alternative retrieval:** a structurally different retriever (an
  intersection-indexed method) producing a deduplicated ≤25-chunk
  context from the same corpus.

The only thing that varies is *which chunks reach the reader*. Answer
F1/EM and a retrieval-side diagnostic were computed per question over
all 7,405 dev questions, with 10,000-resample bootstrap CIs.

## Evidence

### 1. Answer metric barely moves when the gold pair's presence moves a lot

| Quantity | Dense baseline | Alternative | Δ |
|---|---|---|---|
| Gold supporting **pair** present in reader context | **41.39%** | **30.74%** | **−10.65 pp** |
| Any gold chunk present | 92.05% | 87.64% | −4.40 pp |
| Answer **F1** | 45.69 | 44.05 | **−1.64** (CI −2.34, −0.93) |
| Answer **EM** | 34.53 | 32.96 | −1.57 |

An 11-point swing in whether the *correct evidence pair* is even in
front of the reader produced a ~1.6-point F1 change. The reader is
absorbing most of the retrieval degradation — answering from the
single supporting fact that *is* present, or from parametric
knowledge, on a large fraction of questions.

### 2. When the right pair is definitively surfaced, the answer metric does not respond

Restricting to the n=323 questions where the alternative method
*provably* surfaced the gold pair as a retrieval candidate:

| Subset | n | F1 lift (alt − dense) |
|---|---:|---|
| gold pair surfaced | 323 | **−0.01 (CI −0.04, +0.02)** |
| bridge & surfaced | 190 | −0.02 (CI −0.06, +0.02) |
| comparison & surfaced | 133 | 0.00 (CI −0.06, +0.06) |

On these questions the dense baseline already scored F1 ≈ 0.64 (vs.
0.456 overall): they are questions the reader *already answers well*.
Handing it the exact correct evidence pair through a different
mechanism changed nothing, because the reader was not
retrieval-limited on them in the first place. That is the saturation
signature: better retrieval has no answer-metric headroom to convert.

### 3. The retrieval signal is large where the answer metric is flat

The same two conditions, scored on a no-reader retrieval metric
(Pair-Recall@25 — gold supporting pair present in the top-25), across
three multi-hop datasets over the same corpus:

| Dataset | Dense Pair-Recall@25 | Alternative | Δ |
|---|---|---|---|
| HotpotQA | 41.39% | 30.74% | **−10.65 pp** |
| 2WikiMultiHop | 25.85% | 18.86% | −6.99 pp |
| MuSiQue-Ans | 6.79% | 3.56% | −3.23 pp |

The retrieval methods differ by 7–11 points on HotpotQA — an effect
size an order of magnitude larger than the ~1.6-point F1 difference
the same methods produce through the reader. The retrieval signal is
not small; the *answer metric's sensitivity to it* is.

(Internal-consistency check: HotpotQA dense Pair-Recall@25 measured by
the retrieval-only pipeline, 41.39%, equals the "gold-pair-in-context"
figure measured by the separate answer-pipeline diagnostic, 41.39%, to
the digit — two independently written pipelines agree, so the numbers
are a property of the method, not a harness artifact.)

## Why this happens

HotpotQA bridge/comparison questions are frequently answerable from
*one* of the two supporting paragraphs plus a strong reader's
parametric knowledge; the second hop is often inferable rather than
strictly required from context. A top-25 dense context over
full-Wikipedia already clears the threshold where the reader's answer
stops depending on retrieval quality. Past that threshold, the
answer metric measures the reader, not the retriever. Smaller context
budgets, weaker/extractive readers, or harder datasets (MuSiQue, where
even dense Pair-Recall@25 is only 6.79%) push back below the
threshold; HotpotQA + a strong reader + a generous budget does not.

## Implications for evaluating retrieval methods

1. **Report a no-reader retrieval metric alongside answer metrics.**
   Recall / Pair-Recall of the gold supporting set at the actual
   context budget. If a method improves answers but not retrieval, the
   gain is reader-side; if it improves retrieval but not answers, the
   benchmark is saturated — both are findings the answer metric alone
   hides.
2. **Demonstrate the reader is not saturated in the reported regime.**
   A one-line control: does answer F1 respond when the gold pair's
   presence is varied? If an 11-point presence swing moves F1 by ~1
   point, the benchmark cannot adjudicate retrieval methods at that
   budget.
3. **Prefer unsaturated regimes for retrieval claims.** Lower-k
   budgets, deliberately extractive/weak readers, or datasets
   constructed against single-fact shortcuts (MuSiQue). "Improves
   HotpotQA F1 with a strong reader" is, by itself, weak evidence about
   retrieval.

None of this says HotpotQA is a bad dataset or that strong readers are
bad. It says the *product* "HotpotQA answer metric × strong reader ×
generous budget" is a low-power instrument for the specific question
*did retrieval get better*, and that this product is a common default.

## Scope and limitations

One reader (`llama3.1:8b-instruct-q5_K_M`, greedy), one prompt, one
corpus, one dataset family for the answer-side measurement (HotpotQA;
the retrieval-side magnitude is corroborated structurally on 2Wiki and
MuSiQue). The threshold claim is regime-specific and not a universal
constant; a different reader or budget moves the threshold. The point
is methodological: the saturation must be *checked*, not assumed
absent, and the check is cheap.

## Provenance and independence

These measurements were produced as the diagnostic and follow-up of a
pre-registered negative result in an unrelated research program
(MANDORLA, an intersection-primitive thesis); the retrieval method
labeled "alternative" above was that program's intervention, which was
found not to help. **This note's claim does not depend on that thesis
in any direction** — it is a property of the *reader and benchmark*,
established by holding everything except retrieved chunks fixed. It is
reported separately precisely because it outlives the project it came
from.

Reproduction: `runascode/mandorla`, experiment
`exp1-vesica-rag` (answer-side: `results/RESULTS.md`,
`results/DIAGNOSTIC.md`, producing commit `5a3b34a`) and
`02-retrieval-isolation` (retrieval-side: `results/RESULTS.md`,
producing commit `fdf49f8`). Master seed 1337; all decoding and
bootstrap parameters pinned in the respective `PRECOMMIT.md`.
