# Experiment 02 — Lab Notes

Append-only chronological log. Oldest at top. Project-wide notes go in the project-root `LAB-NOTES.md`.

---

## 2026-05-13 — Experiment scoped

Triggered by the Experiment 01 slice diagnostic
(`01-vesica-rag/results/DIAGNOSTIC.md`):

- Coverage-hit subset (n=323) showed F1 lift −0.01 (CI −0.04, +0.02) →
  H1 ("coverage is the bottleneck") falsified.
- Vesica-RAG dropped gold-pair-in-context from 41.39% to 30.74% → context
  displacement was a real cost, but on the coverage-hit subset (where
  displacement is bounded) the LLM still showed no F1 lift. So
  Llama-3.1-8B saturates at top-25 dense on HotpotQA dev; the screening
  slice was effectively an LLM-saturation test, not a retrieval-primitive
  test.

PRECOMMIT.md locked the same day. No code committed yet.

Expected throughput for the retrieval-only run: ~0.5 q/s × ~20k questions
≈ 11 hours wall, plus dataset-pull + corpus-coverage overhead. Single
worker per `LAB-NOTES.md` Lesson 6 (16 GB resident FAISS index).

## 2026-05-13 — Dataset corpus-coverage report

`scripts/02_corpus_coverage.py` results:

| Dataset | n questions | all-gold-in-corpus | any-gold-in-corpus |
|---|---:|---:|---:|
| HotpotQA dev | 7,405 | **100.00%** | 100.00% |
| 2WikiMultiHop dev | 12,576 | **83.47%** | 99.06% |
| MuSiQue-Ans dev | 2,417 | **91.52%** | 99.96% |

All three datasets are above the 70% PRECOMMIT inclusion threshold, so
all three contribute to the decision rule. HotpotQA was expected at
100% (it's the source corpus). 2Wiki's 83.5% reflects gold paragraphs
from articles that aren't in the HotpotQA Wikipedia abstract dump (e.g.
2Wiki includes some non-abstract paragraphs that don't map to a single
chunk in our corpus). MuSiQue at 91.5% similarly drops a few questions
whose evidence is in articles not present.

The hop distributions matter for interpretation:

- HotpotQA: all 2-hop.
- 2Wiki: 9,825 2-hop + 2,751 4-hop.
- MuSiQue: 54 1-hop + 1,237 2-hop + 745 3-hop + 381 4-hop.

So MuSiQue and 2Wiki contribute the 3+ hop questions HotpotQA lacks —
the harder retrieval regime where the slice's saturation finding is
least likely to apply.

## 2026-05-13 — Retrieval-only run started (kicked off)

Single worker, primary mandorla clone (not the site's submodule clone —
indices live only in the primary). Logs to `/tmp/exp02_retrieval.log`,
output to `results/raw/<dataset>.jsonl`.

Startup confirmed:
- FAISS retriever loaded: 5,233,329 passages indexed.
- 64-D box store loaded: 5,233,329 boxes.
- Random projection + τ_v = 214.6442 loaded.
- Title index + query encoder loaded.
- HotpotQA iteration started, 0 already done / 7,405 to process.

### Throughput surprise

First measurement: **0.13 q/s** on HotpotQA. That's 30% faster than the
slice's 0.10 q/s with Ollama in the loop, but **far below** the
PRECOMMIT §Budget estimate of "~0.5 q/s when no LLM is involved." The
budget assumed FAISS scan was the bottleneck and Ollama was the
serializing wait; removing the LLM should have unlocked the FAISS
scan's faster steady state.

The actual bottleneck is the **box-containment routine**
(`BoxStore.containing_indices`), which does an O(N) linear scan over
all 5.23M box centers per Vesica (5 Vesicas × 1 scan each = 5 full
scans of 5.23M × 64 per question). This was already flagged in the
project-level `LAB-NOTES.md` Lesson #2:

> Brute-force array scans dominate when they're O(N) per query. ...
> ~4–5 s/question of numpy, half the Vesica-RAG run's per-question
> time. Anything that does a linear scan over millions of rows inside
> a per-item loop should use a spatial index ... it's the same
> computation, done in ms instead of seconds.

Removing the LLM didn't fix the box-containment scan; the scan now
dominates the per-question wall.

### Updated wall-clock estimate

22,398 questions × 1 / 0.13 q/s ÷ 3600 ≈ **47 hours**.

PRECOMMIT §Budget said "<8 hours." That number was wrong. Recording
here, not amending PRECOMMIT — the budget section is not a binding
decision (it doesn't affect what or how we measure, only when we'll
finish). An amendment isn't required per `CLAUDE.md` §7. Logging the
correction here is sufficient and keeps the audit chain intact.

### Decision on whether to optimize mid-run

The right architectural fix is a 64-D FAISS index over the box centers
that returns the k-nearest-box-center indices, then filters by the
exact box-containment predicate post-FAISS. This would make box
containment ~ms instead of seconds, and the eval would finish in 8-12
hours.

But: implementing it now would (a) require an amendment because the
returned set of contained chunks might differ (FAISS-k-nearest is a
superset that we then filter, which could produce slightly different
contained-chunk orderings if multiple chunks tie on centroid-distance
to the Vesica center), and (b) interrupt a run already in flight,
which is fine for a resumable script but wastes the ~hour already
sunk.

**Decision:** let the current run continue. The 47h wall is annoying
but not blocking — the result will land in roughly two days. Build the
spatial-index version *now*, behind a feature flag, so it's ready for
the next experiment (or for a re-run if we end up wanting one).
Document the build of that index in a separate amendment when it's
needed.

## 2026-05-15 — Run complete; decisive NO-GO

All three dev sets processed (22,398 questions; HotpotQA 7,405 +
2Wiki 12,576 + MuSiQue 2,417), single worker, ~33 h wall. PID 41691
exited cleanly after writing `results/raw/musique.jsonl`. Scored with
`scripts/04_score.py` the same day.

### Headline (Pair-Recall@25, the decision metric)

| Dataset | Baseline | Vesica-aug | Lift | 95% CI |
|---|---|---|---|---|
| HotpotQA | 41.39% | 30.74% | **−10.65 pp** | (−11.41, −9.91) |
| 2Wiki | 25.85% | 18.86% | **−6.99 pp** | (−7.52, −6.48) |
| MuSiQue | 6.79% | 3.56% | **−3.23 pp** | (−4.01, −2.48) |

NO-GO, and not a marginal one — every CI is well clear of zero on the
negative side. The intersection primitive doesn't merely fail to help
retrieval; it **actively degrades** gold-pair recall over the same
corpus. Every secondary metric (Pair-Recall@10, Any-Gold-Recall@25,
RR-of-first-pair) is negative on all three datasets. No slice helps.

### Why this is the decisive result, not just another NO-GO

The Exp 01 slice's NO-GO was confoundable: "the LLM saturated, so we
couldn't see the retrieval signal." The diagnostic then falsified H1
and pointed at H3 (Vesica chunks displace useful baseline context
under the 25-chunk cap). Exp 02 removed the LLM entirely and tested
the retrieval claim directly on three datasets. The result confirms
H3 at the pure-retrieval level with no confound left to appeal to:
the primitive's contained-chunk sets evict higher-value contriever
hits, and the net effect on gold-pair recall is strongly negative
everywhere.

Internal-consistency check passed: HotpotQA baseline Pair-Recall@25
(41.39%) equals the slice diagnostic's baseline gold-pair-in-context
(41.39%) to the digit. Two independently written pipelines agree, so
the number is a property of the method, not of either harness.

### What this closes and what it doesn't

Closes: the *in-query intersection-as-Vesica retrieval primitive in
this projection* (B2 density-extent boxes, contriever-derived 64-D
random projection, no store). Three datasets, two pipelines, headline
+ four secondary metrics, all consistent. This operationalization is
retired from the active queue per PRECOMMIT.

Does **not** close: B3 (learned box head) — a trained box geometry
could in principle surface pairs B2's density-extent construction
misses. But B3 now carries a heavier burden of proof: it must beat
not just "no signal" but "actively harmful," and the displacement
mechanism (greedy contained-chunk sets evicting good dense hits under
a finite context budget) would still apply to B3 unless its retrieval
assembly is redesigned. That redesign — not just a better box — is
the real open question, and it belongs to a fresh PRECOMMIT, not an
amendment here.

The spatial-index optimization discussed in the prior entry is moot:
there is no reason to re-run this operationalization faster. If a
future experiment needs the box-containment speedup it will be built
then, against that experiment's PRECOMMIT.

### Project-level consequence

Recorded in the project-root `LAB-NOTES.md`: with both the slice
(NO-GO, LLM-saturation-confounded) and the retrieval-isolation test
(NO-GO, decisive, unconfounded) negative, the retrieval form of
Thesis 2 is not pursued further at the screening level. Highest-EV
next move is Experiment 03 (Hex-Vote) — a different projection of the
thesis on entirely different infrastructure — or writing up the
LLM-saturation finding as a standalone contribution. Neither requires
this machine or this index.
