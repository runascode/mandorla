# MANDORLA

> A geometric foundation for machine cognition.
> The primitive is the *intersection*, not the point.

**[Read the paper](./mandorla.md)** · **[runascode.com/mandorla](https://runascode.com/mandorla)** · **arXiv: TBD**

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
| Position paper (v1.0) | ✅ Published |
| Site at runascode.com/mandorla | 🟡 Coming online |
| arXiv preprint | 🟡 Pending endorsement |
| Experiment 1 (Vesica-RAG) | 🔲 Planned — `experiments/01-vesica-rag/` |
| Experiment 2 (Hex-Vote) | 🔲 Planned |
| Experiment 3 (Curriculum) | 🔲 Planned |
| Reference implementation | 🔲 Planned (Python 3.11+ / PyTorch / Pydantic v2) |

## Stack

The implementation is Python-first:

- **Python 3.11+**, PEP-604 unions, `from __future__ import annotations`
- **PyTorch** for any learned components
- **Pydantic v2** for cross-process schemas (CMP messages, persisted Region payloads)
- **`dataclasses`** for in-process structures
- **NATS JetStream** *or* **Celery + Redis Streams** for the agent message bus
- **Qdrant** for vector storage with payload indices for Vesica lineage
- **`iesl/box-embeddings`** (`pip install box-embeddings`) for GumbelBox

The spec uses dataclasses for readability; production code uses Pydantic.

## Repository layout

```
mandorla/
├── mandorla.md              # The full paper (canonical text)
├── tex/
│   └── mandorla.tex         # arXiv LaTeX source
├── experiments/             # Forthcoming
│   ├── 01-vesica-rag/
│   ├── 02-hex-vote/
│   └── 03-curriculum/
├── figures/                 # Diagrams (vesica, seed, fruit, K_13)
├── CITATION.cff
├── LICENSE-PAPER            # CC BY 4.0
├── LICENSE-CODE             # MIT (when code is added)
└── README.md
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
- **Experimental collaboration.** If you have compute, retrieval infrastructure, or compositional-generalization expertise and want to co-run one of the three pre-registered experiments, open an issue tagged `experiment-01` / `02` / `03`, or email me.
- **Adjacent work.** Pointers to prior art that overlap with §2.5 or §3.2's open questions are welcome via issue.

Pull requests against the paper text are not currently accepted — the document is a position; revisions are by author. PRs against experiment code are welcome once the directories are populated.

## Contact

Jacob Patterson · runascode@protonmail.com · [runascode.com](https://runascode.com)

## License

- **Paper text** (`mandorla.md`, `mandorla.pdf`, `tex/`): [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Code** (when added): MIT.

## Acknowledgments

The paper stands on the shoulders of, among others, Lemanski (2019), Bronstein–Bruna–Cohen–Veličković (2021), Hawkins and the Thousand Brains team, the Kanerva lineage, the Whittington / Behrens / Moser line of grid-cell research, the Bodnar–Bronstein topological-deep-learning program, the box-embedding lineage from McCallum's group, and Friston's Free Energy Principle community. Specific citations are in the paper.

---

*The metaphor's job is to make the engineering memorable. The engineering's job is to make the metaphor true.*
