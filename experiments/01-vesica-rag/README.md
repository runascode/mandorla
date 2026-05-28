# Experiment 01 — Vesica-RAG Screening Slice

**Status:** In setup. Binding design document is [`PRECOMMIT.md`](./PRECOMMIT.md). This README is the operator-facing summary; if it disagrees with PRECOMMIT.md, PRECOMMIT.md wins.

## What this is

The MANDORLA paper (`mandorla.md`) §3.4 specifies a **"smallest defensible slice"** for Experiment 1 (`mandorla.md` §3.1, Thesis 2 — Connection over Hierarchy) — a single screening run intended to produce one headline number and one diagnostic number on HotpotQA dev, against a single dense baseline (contriever), to decide whether the broader formal Experiment 1 (MuSiQue + 2WikiMultiHop + full baseline suite) is justified.

This directory implements that slice.

**It is not the formal pre-registered prediction.** The formal prediction (≥10% relative F1/EM lift with ≤2% relative 1-hop degradation) is filed on OSF Registries before the broader Experiment 1 runs, not before this slice. The screening's go/no-go thresholds are defined in [`PRECOMMIT.md`](./PRECOMMIT.md).

## Setup

| Component | Choice |
|---|---|
| Corpus | HotpotQA Wikipedia abstract dump, fullwiki setting, chunked to ~1M chunks |
| Dense retriever | `facebook/contriever-msmarco` (768-D point embeddings) |
| Box space | 64-D, fixed Gaussian random projection of contriever output (seed = 1337) |
| Box construction | B2 — local-density extent, isotropic, α calibrated to median pairwise intersection ≈ 5% of median box volume on a 10k sample |
| Box intersection | GumbelBox per Dasgupta et al. NeurIPS 2020, via `iesl/box-embeddings==0.1.0` |
| Vesica scoring | E[volume] × cos(q_64, centroid(V)) |
| Candidate selection | top-k=20 contriever points → C(20,2)=190 pairwise intersections → keep those with E[volume] > τ_v → top-m=5 Vesicas |
| Vesica → text | C1 — chunks contained in the Vesica box, capped at 10 per Vesica, parents always included |
| LLM context | top-5 contriever points ∪ chunks from top-m Vesicas, deduped, capped at ~25 chunks |
| Generator | `llama3.1:8b-instruct-q5_K_M` via Ollama, `temperature=0`, `seed=1337` (see [`Modelfile`](./Modelfile)) |
| Baseline | Contriever top-25 by cosine (matched chunk budget), same LLM, same prompt, same decoding |
| Cross-query store / promotion / decay | **Deferred** to broader Experiment 1. Slice is in-query only. |

Full rationale for every choice is in [`PRECOMMIT.md`](./PRECOMMIT.md).

## Metrics

| Metric | Role |
|---|---|
| **F1** (token-level, HotpotQA normalization) | Headline. Slice number. |
| **EM** (same normalization) | Headline secondary. |
| **vesica-coverage** | Diagnostic. Fraction of dev questions where the gold supporting-paragraph pair was contained in at least one of the top-m candidate Vesicas before generation. This is the intrinsic-to-MANDORLA metric per §3.1 and is decisive for the slice's go/no-go regardless of LLM behavior. |

Bootstrap 95% CIs over questions, 10,000 resamples, reported for all three.

## Go / No-Go

| Outcome | F1 lift vs. contriever | Vesica-coverage uplift | Action |
|---|---|---|---|
| **GO** | ≥ +2 abs F1 | ≥ +5 pp | Proceed to formal Experiment 1, file OSF pre-reg. |
| **WEAK GO** | ≥ +1 abs F1 | ≥ +3 pp | Add a 2WikiMultiHop 1k subsample slice before formal. |
| **NO-GO** | < +1 F1 AND < +3 pp coverage | (either) | Publish the negative result. Revisit Thesis 2's operationalization. |

These thresholds are set before any code runs against dev data. See [`PRECOMMIT.md`](./PRECOMMIT.md) §"Decision rule" for the rationale.

## Reproduce

Single-machine, Apple Silicon, ~48 GB RAM recommended.

```bash
# from this directory
ollama pull llama3.1:8b-instruct-q5_K_M
uv sync

# data + index (dominated by the ~5.2M-passage contriever encode; ~10 h on M4 Pro MPS)
uv run python scripts/01_pull_hotpotqa.py        # HotpotQA dev (7,405) + train (90,447)
uv run python scripts/02_pull_wiki_corpus.py     # verify BeIR/hotpotqa corpus + supporting-fact audit
uv run python scripts/03_encode_corpus.py        # contriever-encode 5.2M passages → fp16 shards
uv run python scripts/04_build_faiss.py          # IndexFlatIP over the contriever shards
uv run python scripts/05_build_box_index.py      # 64-D random projection + per-chunk boxes + α calibration

# calibration
uv run python scripts/06_calibrate_tau_v.py      # τ_v = p50 of pairwise E[intersection log-vol] on 1k train questions

# slice eval
uv run python scripts/07_run_baseline.py         # contriever top-25 → Ollama → answer
uv run python scripts/08_run_vesica.py           # in-query Vesicas + retrieval union → Ollama → answer
uv run python scripts/09_score.py                # F1, EM, vesica-coverage, bootstrap CIs, go/no-go → RESULTS.md
```

`scripts/03` is resumable (`--resume` picks up at the last completed shard). `scripts/07` and `scripts/08` are resumable by question id. Run `uv run pytest tests/` at any point to verify the primitives.

Outputs land in `results/raw/*.jsonl` (per-question) and `results/RESULTS.md` (headline + diagnostic + CIs + go/no-go) and `results/scores.json` (machine-readable).

## Artifacts on completion

Slice is not "done" until all five exist and are pushed:

1. [`PRECOMMIT.md`](./PRECOMMIT.md) — unchanged from pre-run, except dated amendment blocks
2. `RESULTS.md` — headline + diagnostic + CIs + go/no-go + commit SHA + seeds + α + τ_v
3. `src/` and `scripts/` — pinned `pyproject.toml` / `uv.lock`
4. `results/raw/*.jsonl` — per-question records for post-hoc
5. Blog post at `runascode.com/results/vesica-rag-slice` — published whether positive or negative

## References

- Paper: `mandorla.md` §3.1 (Experiment 1), §3.4 (Build Path), §2.1 (Vesica primitive + GumbelBox), §2.4 (cognitive cycle)
- HotpotQA: Yang et al. 2018, [arXiv:1809.09600](https://arxiv.org/abs/1809.09600)
- contriever: Izacard et al., TMLR 2022, [arXiv:2112.09118](https://arxiv.org/abs/2112.09118)
- GumbelBox: Dasgupta et al., NeurIPS 2020, [arXiv:2010.04831](https://arxiv.org/abs/2010.04831); package: [iesl/box-embeddings](https://github.com/iesl/box-embeddings)
- Box-embeddings library demo: Chheda et al., EMNLP 2021 demo, [arXiv:2109.04997](https://arxiv.org/abs/2109.04997)
