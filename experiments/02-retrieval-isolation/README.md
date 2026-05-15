# Experiment 02 — Retrieval-Isolation Test of the Intersection Primitive

*Last updated: 2026-05-15*

**Status:** ✅ **complete — decisive NO-GO** (2026-05-15). Pair-Recall@25 lift −10.65 / −6.99 / −3.23 pp (HotpotQA / 2Wiki / MuSiQue); every CI clear of zero, every secondary metric consistent. See [`results/RESULTS.md`](./results/RESULTS.md). PRECOMMIT.md locked 2026-05-13.
**Paper section:** §3.1 (broader Experiment 1) + §3.3 (falsifiability).

## What this is

A sharpened follow-up to the Experiment 01 screening slice (`exp1-vesica-rag/`), prompted by the diagnostic at `exp1-vesica-rag/results/DIAGNOSTIC.md`. The slice's NO-GO confounded "the primitive doesn't retrieve better evidence" with "Llama-3.1-8B saturates at top-25 dense retrieval." This experiment isolates retrieval from the downstream model: it measures Pair-Recall@25 of contriever-baseline vs. Vesica-augmented retrieval across three multi-hop datasets, with **no LLM in the loop**.

Binding decisions are in [`PRECOMMIT.md`](./PRECOMMIT.md).

## Setup

| Component | Value |
|---|---|
| Datasets | HotpotQA fullwiki, 2WikiMultiHop, MuSiQue-Ans (dev splits) |
| Corpus | Same fullwiki corpus from `exp1-vesica-rag/` (~5.23M passages) |
| Encoder | `facebook/contriever-msmarco`, mean-pool, max_len=128 |
| Box index | d=64, random-projection seed=1337, α=22.0 (reused from Exp 01) |
| τ_v | 214.6442 (frozen from Exp 01) |
| Generator | none |

## Decision rule (summary; full rationale in PRECOMMIT)

| Outcome | Pair-Recall@25 lift |
|---|---|
| GO | ≥+5 pp on all three datasets |
| WEAK GO | ≥+3 pp on at least 2 of 3, none worse by >1 pp |
| NO-GO | otherwise |

## Reproduce

Not yet runnable. Scripts to be added under `scripts/`:

```
uv run python scripts/01_pull_datasets.py        # MuSiQue + 2WikiMultiHop dev
uv run python scripts/02_corpus_coverage.py      # report gold-in-corpus rates
uv run python scripts/03_evaluate_retrieval.py   # retrieval-only, both conditions, all 3 datasets
uv run python scripts/04_score.py                # Pair-Recall@25 + CIs + decision verdict
```

The FAISS index and box index are reused bit-for-bit from `exp1-vesica-rag/index/`; no rebuild.

## Artifacts on completion

- `RESULTS.md` with per-dataset Pair-Recall@25 (baseline / Vesica-augmented / lift), corpus-coverage rates, decision verdict, commit SHA.
- `results/raw/<dataset>.jsonl` (per-question top-25 chunk ids for both conditions).
- `LAB-NOTES.md` updated with throughput and surprises.
- Blog post update (decision-dependent, per PRECOMMIT §I).

## References

- Trivedi et al., *MuSiQue: Multi-hop Questions via Single-hop Question Composition*, TACL 2022 (arXiv:2108.00573).
- Ho et al., *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps*, COLING 2020 (arXiv:2011.01060).
- `../exp1-vesica-rag/results/DIAGNOSTIC.md` — motivating diagnostic.
