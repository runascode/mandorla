# Experiment 02 — Lab Notes

Append-only chronological log. Oldest at top. Project-wide notes go in the project-root `LAB-NOTES.md`.

---

## 2026-05-13 — Experiment scoped

Triggered by the Experiment 01 slice diagnostic
(`exp1-vesica-rag/results/DIAGNOSTIC.md`):

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
