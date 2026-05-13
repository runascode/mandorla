# Experiment 02 — Retrieval-Isolation Test of the Intersection Primitive: Pre-Commit Decisions

**Status:** Locked. Do not modify without an explicit dated amendment block at the bottom of this file.

**Date locked:** 2026-05-13
**Author:** Jacob Patterson
**Paper version this experiment targets:** MANDORLA v1.0 (2026), §3.1 Experiment 1 (the broader version), §3.3 (falsifiability)
**Repo numbering:** This is the **second** experiment globally (per `CLAUDE.md` §2). In paper terms it is a sharpened follow-up to Experiment 1, prompted by the diagnostic on the Vesica-RAG screening slice (`experiments/exp1-vesica-rag/results/DIAGNOSTIC.md`).

---

## Purpose of this document

Experiment 01 (the Vesica-RAG screening slice) returned NO-GO. The post-hoc diagnostic on its outputs showed that the screening-slice design **confounded two distinct questions**:

1. *Does the intersection primitive retrieve different (and better) evidence than dense nearest-neighbor over the same chunks?*
2. *Does a frontier-class extractive QA model convert that retrieved evidence into better answers on HotpotQA?*

The screening slice answered (2) cleanly with "no" and treated that as a "no" for (1). The diagnostic shows the screening was effectively an LLM-saturation test, not a retrieval-primitive test: on the 323 questions where the gold pair *was* surfaced as a candidate Vesica, F1 lift was −0.01 (CI −0.04, +0.02) — the LLM extracted no extra signal even from perfect Vesica evidence. Llama-3.1-8B at HotpotQA dev's top-25-dense context is already saturated relative to the primitive's contribution.

This experiment **isolates (1)**. No LLM in the loop. The primitive is evaluated on retrieval-side metrics only, on three multi-hop datasets, so that downstream model behavior cannot mask or amplify the result.

Why this is binding before the run:

- The Vesica-RAG slice's spec was right for what it tested but wrong for *what we learned we needed to test*. The right discipline post-diagnostic is to lock a new spec for the new question, not to amend the old one.
- Without a pre-commit, post-hoc choice of which datasets / metrics to report after seeing the numbers is exactly the kind of researcher-degrees-of-freedom that would make a positive result here unconvincing.

---

## What this experiment tests

**Question.** Holding everything else equal (corpus, encoder, projection seed, box construction), does the intersection-as-Vesica retrieval primitive surface gold supporting-paragraph **pairs** at a higher rate than nearest-neighbor over the same chunks, across multiple multi-hop QA datasets?

**Not what this experiment tests** (deferred):

- Whether the resulting answers are better. That was Experiment 01.
- Whether a trained box-embedding head outperforms the density-extent construction (B3). Separate experiment.
- Whether the store/promotion/decay machinery (§2.4) helps. Separate experiment.
- Whether Vesica-RAG composes with a different downstream generator. Separate experiment.
- Whether the primitive helps on single-hop questions. The claim is about multi-hop; single-hop is for ablation, not the headline.

**Decision rule.**

| Outcome | Pair-Recall@25 lift, **median across the 3 datasets** | Action |
|---|---|---|
| GO | ≥ +5 absolute percentage points on **all three** datasets | Proceed to Exp 03 (B3 learned box head) with formal OSF pre-registration. |
| WEAK GO | ≥ +3 pp on at least 2 of 3 datasets, AND no dataset is *worse* by more than 1 pp | Re-evaluate. May warrant a dataset-restricted follow-up; not a green light for B3. |
| NO-GO | otherwise | Publish negative result. Retire the in-query intersection primitive from the project's active experiment queue; move attention to Exp 03 (Hex-Vote) and Exp 04 (Curriculum) as the surviving falsifiable tests of the broader thesis. |

The bar is set on **pair-recall@25** because (a) HotpotQA's 25-chunk LLM context cap from Experiment 01 makes this the matched ceiling, (b) gold pair retrieval is the operational definition of "Vesica surfaced the right evidence," and (c) it is interpretable independent of the downstream model. Single-chunk recall is reported as a secondary metric but is not part of the decision rule.

Thresholds are deliberately tighter than Experiment 01's vesica-coverage bar because:

- We are now measuring directly, not via a downstream proxy. CIs will be tighter, so the bar can be tighter.
- The screening slice already showed +4.36 pp on HotpotQA. The bar must require *more than the slice already produced as a noisy proxy*, otherwise the experiment is just re-confirming a number we have.

---

## Frozen decisions

### A — Datasets

Three multi-hop QA datasets, all with multiple gold supporting passages per question:

1. **HotpotQA fullwiki, dev split.** 7,405 questions. Already indexed end-to-end from Experiment 01; no new infrastructure. Identifies whether the diagnostic's retrieval signal extends beyond the screening slice's specific setting.
2. **2WikiMultiHop, dev split.** Ho et al., COLING 2020 (arXiv:2011.01060). Bridge and comparison questions with 2–4 supporting paragraphs each. Independent dataset, different paragraph distribution.
3. **MuSiQue-Ans, dev split.** Trivedi et al., TACL 2022 (arXiv:2108.00573). 2–4 hop questions explicitly designed against retrieval shortcuts; harder than HotpotQA on retrieval-side metrics.

All three are public, English, HuggingFace `datasets`-loadable. None overlap with HotpotQA's training. Choosing three (not two) so the "median across datasets" decision rule has a meaningful sense.

**Corpus.** All three datasets are run against the *same* fullwiki corpus already built for Experiment 01 (~5.23M HotpotQA-Wikipedia abstract chunks). This is a deliberate choice with a caveat:

- *Caveat:* 2WikiMultiHop's gold passages come from a 2018 Wikipedia dump that may not be a strict subset of HotpotQA's. MuSiQue's gold passages come from Wikipedia paragraphs that may also drift from the slice corpus.
- *Why we accept the caveat:* the *primitive*'s ability to retrieve gold pairs from a fixed corpus is what we're measuring. If the gold paragraph isn't in the corpus at all, that's a corpus-coverage issue, not a primitive issue; we report it as part of the result. Recomputing pair-recall over a same-corpus-substitution is more honest than swapping corpora across datasets, which would introduce three sets of confounders.
- A **corpus-coverage rate** is reported per dataset: the fraction of dev questions whose gold supporting paragraphs all exist (by title match) in our corpus. Pair-recall is computed only over questions whose gold paragraphs are *in* the corpus; the corpus-coverage rate is reported alongside so the reader can judge representativeness.

If a dataset's corpus-coverage rate is below 70%, that dataset is reported but excluded from the decision rule (and the rule is computed over the remaining datasets, with the n-of-3 / n-of-2 rule adjusted explicitly).

### B — Retrieval methods compared

Both methods produce a ranked list of up to **25 candidate chunks** per query (matching the slice's context cap so this experiment's results compose cleanly with the slice's).

1. **Contriever top-25** (baseline). Same encoder (`facebook/contriever-msmarco`, mean-pool, max_len=128), same FAISS `IndexFlatIP` already built in Experiment 01. No change.
2. **Vesica-augmented top-25** (test). Same architecture as the slice's Vesica-RAG retrieval stage (contriever top-20 points + top-5 Vesicas by E[volume]·cos score, taking up to 10 chunks per Vesica's containment box), deduped and capped at 25 chunks. **No LLM call.** The output is just the deduped 25-chunk ranked list.

The Vesica candidates' rank inside the union is determined by: (a) parent point-rank first, (b) Vesica-rank second, (c) for chunks contained-in-Vesica but not in any top-20 point retrieval, rank by Vesica-rank then by centroid distance to Vesica center.

This is the *same retrieval stage* that ran inside the slice. No new retrieval code is required beyond logging.

### C — Metrics

**Primary (decision metric).** *Pair-Recall@25.* For each dev question with gold supporting paragraphs $G = \{g_1, ..., g_k\}$ (with $k \geq 2$): the question contributes 1.0 if $\{g_1, ..., g_k\} \subseteq \text{retrieved}_{25}$, else 0.0. The dataset-level value is the mean across applicable questions (i.e. those whose gold chunks are in the corpus). 95% bootstrap CIs with 10,000 resamples, seed=1337.

**Secondary (reported but not decision).**

- **Any-Gold-Recall@25.** Fraction of questions where at least one gold chunk is in the top-25.
- **Pair-Recall@10.** Tighter version of the primary metric to test whether the primitive is helping under tighter context budgets.
- **Reciprocal Rank of First Gold Pair Completion.** For each question, the smallest k such that the top-k contains the full gold pair; if no such k ≤ 25 exists, the contribution is 0. Reports how *early* in the ranked list the pair is achieved.
- **Vesica-coverage (slice-compatible).** As defined in Experiment 01: fraction of questions whose gold pair is the parent pair of at least one of the top-5 candidate Vesicas. Reported for slice compatibility; not part of the decision rule.

### D — Decision-rule unit

The decision uses **per-dataset point estimates** of Pair-Recall@25 lift (Vesica-augmented − contriever baseline), evaluated against the thresholds in the table above. CIs are reported alongside for transparency about precision but the rule fires on point estimates (consistent with Experiment 01's discipline).

### E — Sample size and statistics

Bootstrap 95% CIs over questions, 10,000 resamples, seed=1337. Within-dataset bootstrap; no cross-dataset pooling (because the datasets differ in question difficulty and corpus-coverage). Per-dataset n is whatever the dataset's dev split has minus questions with no gold-in-corpus.

### F — Reproducibility

- Same master seed: **1337**, propagated to (a) box random projection (already on disk from Experiment 01), (b) any future Vesica construction, (c) bootstrap resampling.
- Encoder, FAISS index, and 64-D box index are reused **bit-for-bit** from Experiment 01. No re-indexing.
- `pyproject.toml` for this experiment inherits Experiment 01's pinned deps; a fresh `uv.lock` is committed at this experiment's root.
- 2WikiMultiHop and MuSiQue dev splits are pulled via HuggingFace `datasets`, with the revision SHA pinned in a script-level constant.

### G — Architecture spec

```
dev question (HotpotQA / 2WikiMultiHop / MuSiQue)
        │
        ▼
[1] contriever encode → 768-D point vector
        │
        ├─→ [2a] contriever top-25 over fullwiki  ───→ ranked_25_baseline
        │
        └─→ [2b] contriever top-20 + Vesica search:
              for each pair (A, B) in top-20:
                V_AB = GumbelBox.intersect(A.box, B.box)
                if E[volume(V_AB)] > τ_v (frozen at 214.6442 from Exp 01):
                  score = E[volume(V_AB)] × cos(q_64, centroid(V_AB))
              keep top-5 Vesicas by score
              build candidate set:
                (top-5 contriever points) ∪ (up-to-10 chunks contained in each
                 top-5 Vesica), deduped, capped at 25 by the priority rule in §B
              → ranked_25_vesica
        │
        ▼
[3] METRICS (no LLM)
    Pair-Recall@25 over both ranked sets
    Any-Gold-Recall@25 over both
    Pair-Recall@10 over both
    Reciprocal Rank of First Gold-Pair Completion
    Vesica-coverage (slice-compatible)
```

### H — Baselines worth naming (but not run as part of this experiment)

Documented here so a future reader doesn't think we missed them:

- **BM25 (`pyserini`):** lexical baseline, classic for HotpotQA. Skipped because this experiment is testing whether *the intersection primitive over a particular dense space* adds value over *that same dense space without the primitive*. BM25 changes the dense space underneath and would conflate the test.
- **ColBERTv2:** stronger dense baseline; same argument as BM25 for omission here. Becomes part of the formal Experiment 03 if this one returns GO.
- **HippoRAG, HyperGraphRAG, RAPTOR:** these are *different retrieval architectures* that incorporate cross-passage structure differently than Vesica-RAG. Comparing against them is a fair test of "does the primitive earn its keep relative to other structural retrievers" — but it's a different experiment. Goes in the broader Experiment 1 once / if Exp 02 + Exp 03 return GO.

### I — What "ship" means for this experiment

On completion the following artifacts exist in `experiments/02-retrieval-isolation/`:

1. `PRECOMMIT.md` (this file), unchanged from its locked state except for dated amendments.
2. `RESULTS.md` — per-dataset Pair-Recall@25 with CIs (baseline, Vesica-augmented, lift), corpus-coverage rates, decision verdict, commit SHA.
3. `LAB-NOTES.md` — chronological log of throughput, dataset-load surprises, corpus-coverage findings.
4. `results/raw/<dataset>.jsonl` — per-question retrieval outputs (top-25 chunk ids + gold ids) for both conditions.
5. `src/` + `scripts/` with the eval pipeline (one new script: `01_evaluate_retrieval.py`).
6. A follow-up section on the existing blog post at `runascode.com/results/vesica-rag-slice`, *or* a new post at `/results/retrieval-isolation`. Decided by whether the verdict materially changes the slice's interpretation; binding either way to publish.

---

## What is deferred

- B3 (learned box-embedding head). Activated only if this experiment is GO.
- RAG-style downstream evaluation against MuSiQue + 2WikiMultiHop. Activated only if this experiment is GO and the proposed downstream LLM has been re-selected to be less retrieval-saturated than Llama-3.1-8B.
- Comparison against BM25, ColBERTv2, RAPTOR, HippoRAG, HyperGraphRAG. Belongs to the formal Experiment 1.
- Vesica store / promotion / decay. Belongs to formal Experiment 1.

---

## Budget

Target: **1 week of focused work** including dataset pulls, corpus-coverage analysis, eval script, and write-up. The eval itself is cheap (no generation): same FAISS index, same boxes, same projection, just compute Recall@k on three datasets.

The 16 GB resident FAISS index is the same single-machine constraint documented in `LAB-NOTES.md` Lesson 6; one worker on the dev machine is the right configuration. Estimated wall-clock for retrieval-only over the three datasets' combined dev sets (~20k questions): **<8 hours** at the slice's measured 5,233,329-passage scan throughput (~0.5 q/s when no LLM is involved, since FAISS scan is the bottleneck and Ollama generation is no longer alternating with it).

---

## Amendments

*None yet.*

---

## Sign-off

- **Author:** Jacob Patterson
- **Date locked:** 2026-05-13
- **Locked-before:** any code under `experiments/02-retrieval-isolation/scripts/` is run against dev data of any of the three datasets.
