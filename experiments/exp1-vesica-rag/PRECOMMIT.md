# Experiment 01 — Vesica-RAG Screening Slice: Pre-Commit Decisions

**Status:** Locked. Do not modify without an explicit dated amendment block at the bottom of this file.

**Date locked:** 2026-05-10
**Author:** Jacob Patterson
**Paper version this slice targets:** MANDORLA v1.0 (2026), §3.1 Experiment 1, §3.4 Build Path

---

## Purpose of this document

The MANDORLA paper §3.4 specifies a "smallest defensible slice" as a screening go/no-go before the formal Experiment 1 is run. This document freezes the design of that slice. It is the analog of an OSF pre-registration but lighter-weight, kept in the repo for transparency.

Why a pre-commit rather than an OSF pre-registration:

- The slice is a **screening run**, not the formal pre-registered prediction. Per §3.1, the formal prediction (≥10% relative F1/EM lift, ≤2% degradation on 1-hop) is registered against **MuSiQue + 2WikiMultiHop**, not HotpotQA. The slice's job is to decide whether to proceed to the formal experiment at all.
- Filing OSF for the slice risks locking in dataset and metric choices we may want to refine after seeing the screening result. OSF is the right discipline for the formal Experiment 1, not for the screening.
- Committing this file before any code that touches dev data runs achieves the equivalent epistemic guarantee at slice-scale.

The discipline this file commits to: **every decision below is frozen before any code runs against dev data. If a decision needs to change mid-run, an amendment block is added at the bottom of this file, dated, with reasoning. No silent redefinitions.**

---

## What this slice tests

**Question.** Does indexing intersections-as-Vesicas surface the correct evidence pair more often than nearest-neighbor over the same chunks, when retrieval is the binding constraint?

**Not what this slice tests** (these are deferred to the broader Experiment 1):

- Whether cross-query promotion/decay (§2.4 step 6-7) compounds the retrieval signal across queries. This is a separable claim about Hebbian-style learning at the region level, and conflating it with the retrieval-primitive claim in screening would make a positive result uninterpretable.
- Whether a trained box-embedding head outperforms a density-extent box construction. This is a separable claim about whether the box geometry needs to be learned vs. derived; same conflation risk.
- Whether Vesica-RAG holds up on MuSiQue or 2WikiMultiHop. Those are the targets of the formal pre-registered prediction; the slice is HotpotQA-only.
- 1-hop non-degradation. Belongs to the formal prediction, not the screening.

**Decision rule.** This slice produces one headline number and one diagnostic number. The exact go/no-go criterion is:

| Outcome | Headline (F1 lift over contriever on HotpotQA dev) | Diagnostic (vesica-coverage uplift on gold supporting-paragraph pairs) | Action |
|---|---|---|---|
| **GO** | ≥ +2 absolute F1 points | ≥ +5 absolute percentage points | Proceed to formal Experiment 1 (file OSF pre-registration with the same architecture). |
| **WEAK GO** | ≥ +1 F1 point | ≥ +3 pp | Run a second small slice on 2WikiMultiHop 1k subsample before committing to the full formal experiment. |
| **NO-GO** | < +1 F1 point AND < +3 pp coverage | (either condition) | Publish the negative result. Revise Thesis 2 or revise the operationalization. Do not proceed to formal Experiment 1 unchanged. |

These thresholds are not the paper's formal prediction (which is registered for MuSiQue + 2WikiMultiHop at ≥10% relative F1). They are slice-level go/no-go heuristics, set before seeing any results, and chosen to be more permissive than the formal prediction because the slice is on a single dataset with no replication.

---

## Frozen decisions

### Drift-level decisions (the four resolved with the analysis)

**D1 — Slice role.** Screening, not formal pre-registration. Formal OSF pre-reg filed before broader Experiment 1, not before this slice.

**D2 — Corpus.** Fullwiki. Index the HotpotQA Wikipedia abstract dump (~5.2M passages, chunked per §3.4 to roughly 1M chunks). Distractor setting is rejected: it neuters vesica-coverage (always ~100% in a 10-paragraph pool) and collapses the test to "did the LLM answer with mostly-correct paragraphs in context," which measures the LLM, not the retrieval claim.

**D3 — Vesica store.** Deferred to broader Experiment 1. Slice runs **in-query Vesicas only**: each question independently computes pairwise GumbelBox intersections over its top-k, scores them, retrieves accordingly, throws them away. No persistent store, no promotion, no decay, no order-dependence.

Rationale for deferring: the store tests a separable claim (Hebbian compounding across queries). Bundling it into the slice conflates two claims into one number and introduces order-dependence as a debugging tax. The store is implemented as an *optional code path behind a config flag, default off* — ready for broader Experiment 1, not included in slice numbers. The paper's §2.4 operationalization of "promotion_score" is also under-specified (what counts as "productively used"?), and inventing the operationalization in the slice would lock in a choice before any ablation has been run.

**D4 — 1-hop check.** Dropped from slice. Consequence of D1. The non-degradation criterion lives in the formal pre-registered prediction on MuSiQue + 2WikiMultiHop, both of which have natural 1-hop comparison subsets.

### Implementation decisions (A–F)

**A — Box dimensionality: 64-D.**

Project contriever 768-D vectors to 64-D as the box space. Matches Dasgupta et al. NeurIPS 2020's published dimensionality directly. Rejects 768-D (high-dimensional intersection sparsity is exactly the F4-style failure mode the paper warns about), 128-D (free parameter the slice doesn't need), and 32-D (information loss risk).

Projection method: **fixed random Gaussian projection** (seeded), not PCA. Reasons: PCA on 5.2M passages requires a fit step that itself takes hours and introduces a second source of variance; random projection has Johnson-Lindenstrauss guarantees at 64-D for cosine preservation and is reproducible from the seed alone. If the slice signals positive, PCA vs. random projection becomes an ablation in the broader experiment.

**B — Box construction: B2 (local-density extent).**

Each chunk's box is constructed as:
1. Center = the 64-D projection of the contriever vector.
2. Per-dimension half-width = a function of the chunk's k-NN distance in the 64-D projection space (k=10), specifically `half_width_i = α · mean_knn_distance` (isotropic across dimensions for the slice; per-dimension anisotropy deferred).
3. α is a single global scalar set so that the **median pairwise expected intersection volume over a random 10k-chunk sample is ~0.05** of the median single-box volume. This is a calibration step, not training; it picks α once before indexing begins. Documented in the run log.

Why B2, not B3 (trained box head): same separability argument as D3. B3 bundles two claims ("intersection helps" + "this specific training procedure produces useful boxes"). B2 tests the cleaner claim ("given contriever's geometry, does adding per-chunk extent + intersection scoring help over nearest-neighbor?"). If B2 shows a lift, B3 becomes a focused follow-up study with a sharp question. If B2 shows nothing, B3 might still work — but is run as a separate experiment, not a slice fallback.

**The "B3 with B2 fallback" framing is rejected.** Fallbacks-in-name-only become the actual plan when timelines slip. B2 is the correct first thing to try, not a retreat from B3.

**C — Vesica-to-text contract: C1 (contained chunks).**

A Vesica's textual content is the union of chunks whose 64-D box centers lie inside the Vesica box. Implementation: box-containment query against the 64-D index.

Cap: at most **10 chunks per Vesica**, ranked by centroid distance to the Vesica center, closer first. Rationale: LLM context windows are finite; a Vesica that pulls in 200 chunks is unhelpful and contaminates the retrieval signal with what is effectively a wide nearest-neighbor sweep.

The two parent chunks of a Vesica are always included in its content set (they are guaranteed to lie in the intersection by construction).

C2 (endpoint-only) and C3 (centroid-nearest) are rejected: they collapse the Vesica into either a pair-annotation or a single-chunk lookup, neither of which tests intersection-as-region. C1 is the operationalization that earns the paper's vocabulary.

**D — Promotion threshold and decay: N/A.**

No store in the slice. Deferred to broader Experiment 1 (D3 above). If/when the store is implemented, calibration of θ_promote and γ becomes its own pre-commit decision at that time.

**E — Vesica store implementation: N/A for the slice.**

When the store is implemented in the broader experiment: SQLite for metadata (id, parent_ids, birth_time, promotion_score, citations), `.npz` for box parameters (lower corner, upper corner, center). Documented here for the future.

**F — Reproducibility.**

- Single master seed: **1337**. Propagated to: Python `random`, NumPy, PyTorch, sentence-transformers, FAISS (where applicable), Ollama temperature (=0), and the random-projection matrix construction.
- HotpotQA dev question order: fixed by the Hugging Face `datasets` library's default load order. Documented in the run log.
- All chunk IDs are deterministic from the HotpotQA Wikipedia dump revision (pinned in `pyproject.toml`).
- The 64-D random projection matrix is materialized once, saved to disk, and committed (or its seed is committed and re-derived). The slice must be bit-for-bit reproducible from `pyproject.toml` + this file + the dataset checkpoint.

### Architecture spec for the slice

```
HotpotQA dev question
        │
        ▼
[1] INVOCATION
    contriever encode q → v_q ∈ ℝ^768
    project (fixed random Gaussian) → q_64 ∈ ℝ^64
    (q has no box extent; it is a point query)
        │
        ▼
[2] DESCENT
    contriever top-k=20 over ~5.2M chunks (FAISS, cosine)
        │
        ▼
[3] VESICA SEARCH (in-query only)
    for each pair (A, B) in the top-20 (C(20,2) = 190 pairs):
        V_AB = GumbelBox.intersect(A.box, B.box)
        if E[volume(V_AB)] > τ_v:
            score = E[volume(V_AB)] × cos(q_64, centroid(V_AB))
            keep V_AB as candidate
        │
        ▼
[4] CANDIDATE SELECTION
    keep top-m=5 scored candidate Vesicas
        │
        ▼
[5] RETRIEVAL UNION → LLM context
    chunks fed to LLM =
        (top-5 contriever points by cosine)
        ∪ (for each of top-m Vesicas: up to 10 chunks contained in V's box)
    deduplicated by chunk_id; capped at ~25 chunks total to fit context window
        │
        ▼
[6] LLM ANSWER
    llama3.1:8b-instruct-q5_K_M via Ollama, temperature=0, single deterministic pass
        │
        ▼
[7] METRICS (no promotion, no decay, no across-question state)
    headline:    F1, EM on HotpotQA dev
    diagnostic:  vesica-coverage =
                 fraction of HotpotQA dev questions where the gold
                 supporting-paragraph pair was contained in at least one
                 of the top-m candidate Vesicas before the LLM saw context
```

τ_v (minimum expected intersection volume to consider) is set to **the 50th percentile of expected intersection volumes observed in a 1k-question dry run on HotpotQA train**. Calibrated once before dev eval, then frozen. This calibration step is part of the screening, not training.

### Baselines

One baseline only for the slice:

- **Contriever** (Izacard et al., TMLR 2022, arXiv:2112.09118). Top-5 by cosine over the same fullwiki index, same LLM, same prompt, same decoding config. Identical retrieval pipeline up to and including the cap at ~25 chunks (achieved by top-k=25 for the baseline, vs. top-5 points + top-m Vesicas contributing up to 20 more chunks for Vesica-RAG).

Other baselines (BM25, ColBERTv2, HyDE, RAPTOR, HippoRAG, hypergraph methods) are part of the formal Experiment 1, not the screening. Adding them here would slow the slice without changing the go/no-go signal.

### Sample size and statistics

- Full HotpotQA dev set: 7,405 questions.
- Bootstrap 95% confidence intervals over questions for both headline (F1) and diagnostic (vesica-coverage), 10,000 resamples.
- Both metrics reported with CIs; go/no-go decision uses point estimates against the thresholds above; CIs are reported alongside for transparency about precision.

### What "ship" means for this slice

When the slice is complete, the following artifacts exist in `experiments/01-vesica-rag/`:

1. `PRECOMMIT.md` (this file), unchanged from its pre-run state except for any dated amendment blocks.
2. `RESULTS.md` — headline numbers, diagnostic numbers, CIs, plots, go/no-go decision, link to commit SHA of the code that produced the numbers, exact LLM string and prompt, all seeds, calibrated values of α and τ_v.
3. `src/` — code, structured per the repository layout template, with `pyproject.toml` pinned.
4. `results/raw/*.jsonl` — per-question outputs, queryable for post-hoc analysis.
5. A blog post on `runascode.com/results/vesica-rag-slice` summarizing the run. Published whether positive or negative.

The slice is not "in progress" if any of these is missing. The slice is not "done" until all are pushed.

---

## What is deferred to broader Experiment 1

Captured here so nothing gets lost:

- **Vesica store with promotion/decay** (§2.4 step 6–7). Tested as a *separate variant* in the formal experiment: Vesica-RAG (in-query) vs. Vesica-RAG+store vs. contriever, all on MuSiQue + 2WikiMultiHop.
- **Trained box-embedding head (B3)**, using Wikipedia hyperlink co-occurrence as the training signal via `iesl/box-embeddings`. Run as a focused study with the sharp question: "does the trained head recover signal that the density-extent construction (B2) misses?" Requires its own design doc before code starts.
- **Full baseline suite** (BM25, ColBERTv2, HyDE, RAPTOR, HippoRAG, HippoRAG 2, HyperGraphRAG, Hyper-RAG, Cog-RAG). The slice tests against contriever only.
- **MuSiQue and 2WikiMultiHop datasets**. The slice is HotpotQA-only.
- **1-hop non-degradation check**. Tested on the formal datasets' 1-hop subsets.
- **Per-dimension anisotropic box extents**. Slice uses isotropic; per-dimension is an ablation.
- **Calibrated thresholds at OSF-pre-registration time** with formal power analysis. Slice thresholds are first-pass guesses set in this document.

---

## Budget

Target: **two weeks of focused work**, per §3.4.

Honest accounting:

| Phase | Effort |
|---|---|
| Region/Vesica primitives + GumbelBox integration via `iesl/box-embeddings` | 2 days |
| Encode HotpotQA Wikipedia (~5.2M chunks) with contriever on MPS | 1.5–2 days |
| Build FAISS index (768-D for contriever) + 64-D index (boxes) | 0.5 day |
| α calibration on 10k random chunk sample | 0.5 day |
| τ_v calibration on 1k train questions | 0.5 day |
| Run baseline + Vesica-RAG on HotpotQA dev (7,405 questions × 2 conditions) | 1.5–2 days |
| Bootstrap CIs, plots, RESULTS.md, blog post | 1 day |
| **Total** | **9–10 working days** |

This fits a two-week calendar window with slack. If a phase blows up beyond its estimate by more than 1.5×, that is an amendment-block-triggering event: pause, document what went wrong, decide whether to scope-cut or extend, and update this file before resuming.

The original engineer's three-week estimate assumed B3 (trained box head, ~4–6 days) and the cross-query store (~2 days). Both are deferred, recovering the §3.4 timeline.

---

## Amendments

(None. The first entry, if any, goes here, dated, with a clear statement of what changed and why.)

---

## Sign-off

This slice is committed to as designed. The go/no-go decision rule above is binding: a NO-GO result will be published as such, in `RESULTS.md` and on `runascode.com`, without revision after seeing the numbers.

— Jacob Patterson, 2026-05-10
