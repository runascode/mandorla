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
| **Experiment 01 — Vesica-RAG screening slice** ([`experiments/01-vesica-rag/`](./experiments/01-vesica-rag/)) | ✅ **Complete — NO-GO** (2026-05-13). See [`RESULTS.md`](./experiments/01-vesica-rag/results/RESULTS.md) and the post-hoc [`DIAGNOSTIC.md`](./experiments/01-vesica-rag/results/DIAGNOSTIC.md). |
| **Experiment 02 — Retrieval-isolation test** ([`experiments/02-retrieval-isolation/`](./experiments/02-retrieval-isolation/)) | ✅ **Complete — decisive NO-GO** (2026-05-15). Pair-Recall@25 lift −10.65 / −6.99 / −3.23 pp (HotpotQA / 2Wiki / MuSiQue). See [`RESULTS.md`](./experiments/02-retrieval-isolation/results/RESULTS.md). |
| Experiment 04 — Mandorla Curriculum ([`experiments/04-curriculum/`](./experiments/04-curriculum/)) | ⛔ **Both pilots complete — do not pre-register** (2026-05-15). Curriculum pilot: Q1 PASS, Q2 NO (perfectly-learned intersection construction → no held-out transfer). Operator pilot ([`pilot2-operator/`](./experiments/04-curriculum/pilot2-operator/)): **decisive negative**, intersection-on-critical-path/no-bypass/capacity-matched still loses to point composition (Δ_AND −0.013, all seeds, both tasks). |
| Experiment 03 — Hex-Vote ([`experiments/03-hex-vote/`](./experiments/03-hex-vote/)) | 🔲 Design sketch — pre-PRECOMMIT. The **only** projection not tested; runs on independent infra; prior now very low (see narrative). |
| Vesica-RAG (formal, full RAG) | ❌ Not pursued — retrieval form closed by Exp 02. |

**Where the program stands as of 2026-05-15.** Two screening experiments on the retrieval form of Thesis 2, both negative:

- **Experiment 01** (HotpotQA dev, full RAG): NO-GO. F1 lift −1.64 pts. The post-hoc diagnostic showed the setup was confounded by LLM saturation (Llama-3.1-8B extracts near-everything from top-25 dense, so the retrieval signal — if any — couldn't show through).
- **Experiment 02** (HotpotQA + 2Wiki + MuSiQue dev, retrieval-only, no LLM): **decisive NO-GO**. Pair-Recall@25 lift −10.65 / −6.99 / −3.23 pp; every CI clear of zero, every secondary metric consistent across two independently written pipelines. The intersection primitive doesn't merely fail to help retrieval — it actively degrades gold-pair recall, by displacing higher-value dense hits under a finite context budget.

Together these close the **in-query intersection-as-Vesica retrieval primitive in this projection** (B2 density-extent boxes over a contriever-derived 64-D random projection, no store).

**Then the curriculum projection (Thesis 3) was de-risked and also came back negative.** A cheap synthetic pilot ([`experiments/04-curriculum/pilot/`](./experiments/04-curriculum/pilot/)) tested the load-bearing assumption of the expensive Exp 04 — that training an explicit intersection-construction objective yields representations whose compositionality *generalizes* — in the cleanest possible isolation (exact ground truth, controlled 2-hop latent-composition task, demonstrated baseline headroom, capacity-matched generic-auxiliary control). Result: **Q1 PASS, Q2 NO.** The box-intersection losses train flawlessly (vesica/parent → 0.000, bit-accuracy 1.000, no collapse) — and that perfectly-learned construction produces *no* held-out compositional transfer (curriculum 0.570 ± 0.017 vs. baseline 0.588 ± 0.014 vs. generic-aux 0.580 ± 0.012; curriculum marginally worst, every seed). The mechanism is fully trainable and **inert** for the capability it was posited to induce. Per the pilot's pre-stated decision rule, Exp 04 is **not** pre-registered. This does not formally falsify Thesis 3 (that needs the benchmark run) but removes the warrant for it, at ~3 h of laptop compute instead of a multi-GPU quarter.

**Then the construction projection's residual loophole was closed too.** The curriculum pilot's negative could be waved away with "intersection was only an auxiliary loss the model routed around." A second pilot ([`experiments/04-curriculum/pilot2-operator/`](./experiments/04-curriculum/pilot2-operator/)) removed that: the intersection operator **on the critical path, no bypass, capacity-matched** against a point bottleneck, on an intersective task *and* a non-intersective control, verdict rule locked before any number. Result: **decisive negative** — Δ_AND = −0.013, INTERSECTION below a param-matched POINT bottleneck on both tasks and all three seeds, *even on the task whose ground truth is literally a set intersection*.

**Net standing as of 2026-05-15 — the cheap-to-screen falsification program is complete.** Every projection of the MANDORLA thesis that could be screened cheaply has returned negative: retrieval-as-cognition (Exp 01 + 02, decisively, two pipelines), and recursive-construction (Exp 04 curriculum pilot + operator pilot, the latter with the strongest controls of the series). Across the series a pattern recurred: each negative was followed by a plausible "it wasn't tested fairly" rescue (LLM saturation → removed; borrowed geometry → trained from scratch; auxiliary/bypass → put on the critical path, capacity-matched), and **every rescue's condition, once met, still produced a negative**. The thesis is *not* formally falsified — Thesis 1 (Hex-Vote) is untested and the mathematical arguments are untouched — but the one remaining cheap test now carries a very low prior, and the only surviving rescue ("the *architecture* must be intersection-native") is no longer an experiment but a multi-year program; the pattern of an ever-purer, ever-costlier precondition demanded after each disconfirmation is itself the signature this program has now formalized. The most portable, thesis-independent result is the **LLM-saturation finding** ([`findings/reader-saturation-hotpotqa.md`](./findings/reader-saturation-hotpotqa.md)) — true and citable regardless of MANDORLA. The disciplined path forward is repositioning the paper from manifesto to *a pre-registered program and its honest disconfirmation* — a rarer and more credible artifact than the manifesto would have been.

## How research discipline is enforced here

Every experiment in [`experiments/`](./experiments/) carries three documents and follows one engineering standard. See **[`CLAUDE.md`](./CLAUDE.md)** for the full discipline. The short version:

- **`PRECOMMIT.md`** — binding design decisions, locked before code runs against eval data; changes only via dated amendment blocks.
- **`LAB-NOTES.md`** — chronological observations: throughputs, calibrations, surprises, debugging journeys. Non-binding; cited by amendments.
- **`README.md`** — operator-facing summary, defers to `PRECOMMIT.md` on every binding decision.

There is also a project-level [`LAB-NOTES.md`](./LAB-NOTES.md) for repo-wide events (e.g. infrastructure shifts).

## Stack (current slice)

Pinned per [`experiments/01-vesica-rag/pyproject.toml`](./experiments/01-vesica-rag/pyproject.toml):

- **Python 3.12** (newer versions lack ML wheels at time of writing)
- **`uv`** for environment + dependency management
- **PyTorch 2.11** with Apple MPS backend (Apple Silicon)
- **`facebook/contriever-msmarco`** via `transformers` for 768-D dense retrieval
- **`iesl/box-embeddings`** (`pip install box-embeddings`) for the GumbelBox primitive
- **`faiss-cpu`** for the contriever index (IndexFlatIP, exact) and the 64-D box-space kNN (HNSW)
- **`datasets`** (HuggingFace) for HotpotQA and the BeIR/hotpotqa Wikipedia corpus
- **Ollama** running `llama3.1:8b-instruct-q5_K_M` for answer generation (decoding pinned in [`experiments/01-vesica-rag/Modelfile`](./experiments/01-vesica-rag/Modelfile) and in code; `temperature=0`, `seed=1337`)

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
    ├── 01-vesica-rag/         # ✅ complete — NO-GO
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
