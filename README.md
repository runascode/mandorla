# MANDORLA

> A geometric foundation for machine cognition.
> The primitive is the *intersection*, not the point.

**[Read the paper](./mandorla.md)** · **[runascode.com/mandorla](https://runascode.com/mandorla)** · **arXiv: TBD**

*Last updated: 2026-05-15 — see [Status](#status) for live state.*

---

## What is this?

MANDORLA is a research framework that proposes the **vesica** — the almond-shaped overlap between two regions of meaning — as the fundamental primitive of distributed cognition. The current paper is a manifesto, specification, and pre-registered research blueprint, drawing on geometric deep learning, the Tolman–Eichenbaum Machine, the Thousand Brains Project, sparse distributed memory, box embeddings, and the Free Energy Principle.

Three falsifiable experiments are specified:

- **Vesica-RAG** — intersection-indexed retrieval vs. nearest-neighbor RAG, on multi-hop QA (HotpotQA, MuSiQue, 2WikiMultiHop).
- **Hex-Vote** — a 7-agent Seed configuration vs. flat debate, hub-and-spoke, and single-agent CoT, on multi-domain synthesis benchmarks.
- **Mandorla Curriculum** — a training objective that forces the model to construct representations *at the intersection* of existing ones, evaluated on COGS, SCAN, and ReCOGS.

Each experiment has pre-registered predictions (OSF Registries) and explicit failure modes. The geometry is on trial.

## Why care?

If MANDORLA's predictions hold, current AI architecture has been organizing itself around the wrong primitive — and a small set of geometric corrections, drawn from neuroscience and topological deep learning, would close real gaps in compositional generalization, multi-hop reasoning, and multi-agent coordination.

If they don't hold, the paper says so in §3.3, and the negative result gets reported.

## Status

| Component | Status |
|---|---|
| Position paper v1.0 (`mandorla.md`, `tex/mandorla.tex`) | ✅ Published — 2026-05-10 |
| arXiv preprint | 🟡 Pending endorsement |
| Site at runascode.com/mandorla | ✅ Live |
| **Experiment 01 — Vesica-RAG screening slice** ([`experiments/exp1-vesica-rag/`](./experiments/exp1-vesica-rag/)) | ✅ **Complete — NO-GO** (2026-05-13). See [`RESULTS.md`](./experiments/exp1-vesica-rag/results/RESULTS.md) and the post-hoc [`DIAGNOSTIC.md`](./experiments/exp1-vesica-rag/results/DIAGNOSTIC.md). |
| **Experiment 02 — Retrieval-isolation test** ([`experiments/02-retrieval-isolation/`](./experiments/02-retrieval-isolation/)) | ✅ **Complete — decisive NO-GO** (2026-05-15). Pair-Recall@25 lift −10.65 / −6.99 / −3.23 pp (HotpotQA / 2Wiki / MuSiQue). See [`RESULTS.md`](./experiments/02-retrieval-isolation/results/RESULTS.md). |
| Experiment 03 — Hex-Vote ([`experiments/03-hex-vote/`](./experiments/03-hex-vote/)) | 🔲 Design sketch — pre-PRECOMMIT. **Now the highest-priority surviving falsifiable test** (retrieval form of the thesis is closed). |
| Experiment 04 — Mandorla Curriculum ([`experiments/04-curriculum/`](./experiments/04-curriculum/)) | 🔲 Design sketch — pre-PRECOMMIT, blocking decisions in README. |
| Vesica-RAG (formal, full RAG) | ❌ Not pursued — retrieval form closed by Exp 02. Would require a redesigned retrieval-assembly (not just a better box) and a fresh PRECOMMIT. |

**Where the program stands as of 2026-05-15.** Two screening experiments on the retrieval form of Thesis 2, both negative:

- **Experiment 01** (HotpotQA dev, full RAG): NO-GO. F1 lift −1.64 pts. The post-hoc diagnostic showed the setup was confounded by LLM saturation (Llama-3.1-8B extracts near-everything from top-25 dense, so the retrieval signal — if any — couldn't show through).
- **Experiment 02** (HotpotQA + 2Wiki + MuSiQue dev, retrieval-only, no LLM): **decisive NO-GO**. Pair-Recall@25 lift −10.65 / −6.99 / −3.23 pp; every CI clear of zero, every secondary metric consistent across two independently written pipelines. The intersection primitive doesn't merely fail to help retrieval — it actively degrades gold-pair recall, by displacing higher-value dense hits under a finite context budget.

Together these close the **in-query intersection-as-Vesica retrieval primitive in this projection** (B2 density-extent boxes over a contriever-derived 64-D random projection, no store). The thesis is *not* falsified — only this operationalization of its retrieval form. The surviving falsifiable tests are different projections: **Experiment 03 (Hex-Vote)** tests the geometry in multi-agent topology (Thesis 1, entirely different infrastructure — LLM message-graph, no FAISS), and **Experiment 04 (Mandorla Curriculum)** tests it as a training objective (Thesis 3). The independently citable byproduct is the LLM-saturation finding from the Exp 01 diagnostic, which limits what HotpotQA-style RAG benchmarks can teach about *any* retrieval intervention.

## How research discipline is enforced here

Every experiment in [`experiments/`](./experiments/) carries three documents and follows one engineering standard. See **[`CLAUDE.md`](./CLAUDE.md)** for the full discipline. The short version:

- **`PRECOMMIT.md`** — binding design decisions, locked before code runs against eval data; changes only via dated amendment blocks.
- **`LAB-NOTES.md`** — chronological observations: throughputs, calibrations, surprises, debugging journeys. Non-binding; cited by amendments.
- **`README.md`** — operator-facing summary, defers to `PRECOMMIT.md` on every binding decision.

There is also a project-level [`LAB-NOTES.md`](./LAB-NOTES.md) for repo-wide events (e.g. infrastructure shifts).

## Stack (current slice)

Pinned per [`experiments/exp1-vesica-rag/pyproject.toml`](./experiments/exp1-vesica-rag/pyproject.toml):

- **Python 3.12** (newer versions lack ML wheels at time of writing)
- **`uv`** for environment + dependency management
- **PyTorch 2.11** with Apple MPS backend (Apple Silicon)
- **`facebook/contriever-msmarco`** via `transformers` for 768-D dense retrieval
- **`iesl/box-embeddings`** (`pip install box-embeddings`) for the GumbelBox primitive
- **`faiss-cpu`** for the contriever index (IndexFlatIP, exact) and the 64-D box-space kNN (HNSW)
- **`datasets`** (HuggingFace) for HotpotQA and the BeIR/hotpotqa Wikipedia corpus
- **Ollama** running `llama3.1:8b-instruct-q5_K_M` for answer generation (decoding pinned in [`experiments/exp1-vesica-rag/Modelfile`](./experiments/exp1-vesica-rag/Modelfile) and in code; `temperature=0`, `seed=1337`)

The deferred / future-work stack (agent topology for Experiment 2, training infrastructure for Experiment 3, NATS / Qdrant / Pydantic-as-schema-bus) is specified in `mandorla.md` §2.3 and §3.4 — not in scope for the current slice.

## Repository layout

```
mandorla/
├── README.md                    # this file
├── CLAUDE.md                    # engineering + research-discipline standards
├── LAB-NOTES.md                 # project-level chronological log (repo-wide events)
├── mandorla.md                  # the full paper (canonical text)
├── CITATION.cff
├── LICENSE-PAPER                # CC BY 4.0
├── LICENSE-CODE                 # MIT
├── LICENSE                      # mirror of LICENSE-CODE
├── tex/
│   ├── mandorla.tex             # arXiv LaTeX source
│   └── mandorla.pdf             # compiled paper
└── experiments/
    ├── exp1-vesica-rag/         # ✅ complete — NO-GO
    │   ├── PRECOMMIT.md         # binding design (locked 2026-05-10)
    │   ├── LAB-NOTES.md         # experiment-level chronological log
    │   ├── BENCHMARKS.md        # throughput / cost measurements
    │   ├── README.md            # operator-facing reproduce guide
    │   ├── pyproject.toml, uv.lock, .python-version, Modelfile
    │   ├── src/                 # Region/Vesica primitives, retrieval,
    │   │                        # eval, generation, calibration
    │   ├── tests/               # pytest unit tests
    │   ├── scripts/             # 01..10 numbered pipeline scripts
    │   ├── data/                # gitignored — HF cache, HotpotQA dumps
    │   ├── index/               # gitignored — FAISS, box indices
    │   └── results/             # raw per-question JSONL,
    │                            # RESULTS.md, DIAGNOSTIC.md
    ├── 02-retrieval-isolation/  # 🟡 planned — PRECOMMIT locked 2026-05-13
    ├── 03-hex-vote/             # 🔲 design sketch (pre-PRECOMMIT)
    └── 04-curriculum/           # 🔲 design sketch (pre-PRECOMMIT)
```

## Cite

```bibtex
@article{patterson2026mandorla,
  title  = {MANDORLA: A Geometric Foundation for Machine Cognition},
  author = {Patterson, Jacob},
  year   = {2026},
  eprint = {arXiv:TBD},
  primaryClass = {cs.AI},
  url    = {https://runascode.com/mandorla}
}
```

The `CITATION.cff` in this repo gives GitHub's "Cite this repository" button the same metadata.

## Contributing

This is, at present, a single-author research program. I welcome:

- **Critical reading.** If §2.5 (the vesica-as-Markov-blanket identity) is mathematically broken, please open an issue. Falsification of a load-bearing claim is the most valuable possible contribution at this stage.
- **Experimental collaboration.** If you have compute, retrieval infrastructure, or compositional-generalization expertise and want to co-run one of the three pre-registered experiments, open an issue tagged `experiment-01` / `02` / `03`, or email.
- **Adjacent work.** Pointers to prior art that overlap with §2.5 or §3.2's open questions are welcome via issue.

Pull requests against the paper text are not currently accepted — the document is a position; revisions are by author. PRs against experiment code are welcome.

## Contact

Jacob Patterson · runascode@protonmail.com · [runascode.com](https://runascode.com)

## License

- **Paper text** (`mandorla.md`, `mandorla.pdf`, `tex/`): [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Code** (`experiments/`, `src/`, scripts, configs): MIT.

## Acknowledgments

The paper stands on the shoulders of, among others, Lemanski (2019), Bronstein–Bruna–Cohen–Veličković (2021), Hawkins and the Thousand Brains team, the Kanerva lineage, the Whittington / Behrens / Moser line of grid-cell research, the Bodnar–Bronstein topological-deep-learning program, the box-embedding lineage from McCallum's group, and Friston's Free Energy Principle community. Specific citations are in the paper.

---

*The metaphor's job is to make the engineering memorable. The engineering's job is to make the metaphor true.*
