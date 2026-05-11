# Experiment 1 — Vesica-RAG vs. Dense Baseline (Smallest Defensible Slice)

**Mandorla framework pre-registration.** Tests **Thesis 2 — Connection over Hierarchy** (`mandorla.md §1.4`). Scoped per `mandorla.md §3.4` ("smallest defensible slice"): **HotpotQA dev set, single dense baseline (contriever), Vesica-RAG, two weeks of focused work, one number on one benchmark.**

This README is the **pre-registration document.** It must be committed before any evaluation is run, and must also be filed on [OSF Registries](https://osf.io/registries) (or equivalent) before the full-dev-set eval. Commit hash of this file = the timestamp of pre-registration.

---

## Hypothesis

A retrieval system that explicitly indexes and retrieves *intersections* (promoted Vesicas, `mandorla.md §2.1`) outperforms vanilla nearest-neighbor RAG on tasks requiring multi-concept reasoning — queries that name two or more distinct concepts whose answer lives at their intersection.

A 2-hop question is, definitionally, a question asking for the intersection of two facts. HotpotQA bridge-type questions and the `hard` split are the natural test bed.

## Falsifiable Prediction (Pre-Registered)

Both conditions must hold simultaneously; failure of either undermines Thesis 2 and **will be reported**:

1. **Primary:** Vesica-RAG shows **≥10% relative improvement** in answer F1 *and* in EM over the contriever baseline on the **2+ hop subset** of HotpotQA dev (operationalized as `level ∈ {medium, hard}` and `type = bridge`).
2. **Non-degradation:** Vesica-RAG loses **no more than 2% relative** on the 1-hop / comparison subset (`type = comparison` or `level = easy` bridge questions used as proxy).

Auxiliary measurement (intrinsic to MANDORLA, independent of LLM strength):

3. **Vesica-coverage:** Fraction of multi-hop questions whose ground-truth supporting-paragraph pair was identified by the retriever as a single Vesica (intersection node) before the LLM saw the context. Reported but not used as a pass/fail criterion in this slice.

## Metrics

| Metric | Definition |
|---|---|
| **Answer F1** | Standard HotpotQA token-level F1 between predicted answer string and gold answer string, after normalization (lowercase, strip articles, strip punctuation). |
| **Answer EM** | Exact match after the same normalization. |
| **Vesica-coverage** | For Vesica-RAG only. Of all multi-hop questions, the fraction whose gold supporting-paragraph pair (`supporting_facts`) was identified by the retriever as a top-k Vesica (i.e., a high-scoring pairwise box intersection) before generation. |

All metrics are reported overall and stratified by `level` (`easy`/`medium`/`hard`) and `type` (`bridge`/`comparison`).

## Setup

### Models

| Role | Model | Pinning |
|---|---|---|
| Answer generation | `llama3.1:8b-instruct-q5_K_M` via Ollama | Digest `27fe1b0ab52c` (blob `sha256:1ae48274baafb576c66af17eca484ba3d44759316a0e9cbef252b6235af9ceef`); decoding params pinned in [`Modelfile`](./Modelfile) (`temperature=0`, `seed=1337`, `num_ctx=8192`) |
| Dense retrieval (baseline) | `facebook/contriever-msmarco` | HuggingFace |
| Box embeddings | `iesl/box-embeddings==0.1.0` (GumbelBox) | PyPI |

### Data

- **HotpotQA `distractor` validation split** (7,405 examples), loaded via `datasets.load_dataset('hotpot_qa', 'distractor', split='validation')`.
- For the slice, the retrievable corpus is the union of distractor + supporting paragraphs **per question** (the standard HotpotQA retrieval-eval setup). Full-Wikipedia retrieval is reserved for the broader Experiment 1.

### Conditions

| Condition | Retrieval | Generation |
|---|---|---|
| **Baseline** | contriever top-k over the per-question paragraph pool | Ollama (above), prompt = retrieved chunks + question |
| **Vesica-RAG** | contriever top-k points **+** box-embedding pairwise intersection scoring of the top-k, retrieve the union of points and the top-m promoted Vesicas (see `src/retrieve_vesica.py`) | Same prompt template, same model, same decoding |

The only independent variable is the retriever. The generative LLM is held fixed (same model, same decoding parameters, same prompt template, same in-context format).

### Hyperparameters (Frozen Before Eval)

- Retrieval `k = 5` (top-k chunks fed to the LLM)
- Box-intersection scoring: pairwise over the contriever top-20; keep top-`m=3` promoted Vesicas
- Prompt template: see `src/generate.py` (committed before any results)
- Random seed: `1337` everywhere (NumPy, PyTorch, Ollama)

## Conditions for "Reporting Failure Honestly"

If either condition (1) or condition (2) above fails, the result will be reported as a failure of Thesis 2 as specified, with no rescue analyses post-hoc. Subgroup analyses (by `level` / `type`) are pre-registered to be **reported regardless of primary outcome**, but no subgroup that wasn't pre-registered will be used to claim success. The pre-registered prediction can only be confirmed or falsified; it cannot be redefined after seeing the data.

## Repo Layout

```
experiments/exp1-vesica-rag/
├── README.md                # this file (pre-registration)
├── Modelfile                # pinned Ollama config
├── pyproject.toml           # uv-managed Python 3.12 env
├── src/
│   ├── data.py              # HotpotQA loading + per-question pool assembly
│   ├── encode.py            # contriever encode
│   ├── retrieve_dense.py    # baseline retrieval
│   ├── retrieve_vesica.py   # box-embedding intersection scoring
│   ├── generate.py          # Ollama answer generation
│   ├── eval.py              # F1, EM, vesica-coverage
│   └── promote.py           # Vesica store promotion/decay (mandorla.md §2.4)
├── scripts/
│   ├── 01_pull_data.py
│   ├── 02_build_index.py
│   ├── 03_run_baseline.py
│   ├── 04_run_vesica.py
│   └── 05_score.py
├── data/                    # gitignored: HotpotQA cache
├── index/                   # gitignored: built indices
└── results/                 # JSON per run + summary CSV (committed)
```

## Reproducing

```bash
# from this directory
ollama pull llama3.1:8b-instruct-q5_K_M
uv sync
uv run python scripts/01_pull_data.py
uv run python scripts/02_build_index.py
uv run python scripts/03_run_baseline.py
uv run python scripts/04_run_vesica.py
uv run python scripts/05_score.py
```

Hardware reference: Apple M4 Pro, 48 GB RAM, macOS. Different hardware may give different throughput but should give identical answers (decoding is deterministic at `temperature=0`).

## References

- Vesica primitive: `mandorla.md §2.1`
- Promotion/decay: `mandorla.md §2.4`
- Full Experiment 1 spec: `mandorla.md §3.1` (this slice is §3.4's "smallest defensible slice")
- HotpotQA: Yang et al. 2018, [arXiv:1809.09600](https://arxiv.org/abs/1809.09600)
- contriever: Izacard et al., TMLR 2022, [arXiv:2112.09118](https://arxiv.org/abs/2112.09118)
- GumbelBox: Dasgupta et al. 2020, [arXiv:2010.04831](https://arxiv.org/abs/2010.04831); package: [iesl/box-embeddings](https://github.com/iesl/box-embeddings)
