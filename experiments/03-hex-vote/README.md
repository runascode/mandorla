# Experiment 03 — Hex-Vote: A 7-Node Cortical Council

*Last updated: 2026-05-13*

**Status:** **design sketch** (pre-PRECOMMIT). No PRECOMMIT.md is locked yet. This file is a stub: it captures the design space and the blocking decisions that must be resolved *before* a PRECOMMIT.md can be written.
**Paper section:** §3.1 — Experiment 2.

## What this is

A test of **Thesis 1 (Geometric Primacy)**: a multi-agent reasoning system in *Seed configuration* (1 coordinator + 6 specialists, each adjacent specialist-pair sharing a persistent Vesica subject) outperforms (a) single-agent CoT, (b) flat 7-agent debate, (c) hub-and-spoke 7-agent orchestrator on tasks requiring synthesis across multiple specialty domains, **at matched token budget**.

This experiment runs against *different* infrastructure than Experiments 01–02. There is no retrieval index here; there are 7 LLM instances and a message-graph between them. The intersection primitive shows up as the *persistent shared topic between adjacent specialists* (a Vesica subject = a shared message log that both adjacent specialists write to and read from).

## Falsifiable prediction (paper-level, to be made formal by PRECOMMIT)

Seed (Hex-Vote) outperforms hub-and-spoke and flat debate on synthesis-heavy subsets by ≥5% **relative** accuracy, with token cost within ±3% of the highest-cost baseline. If Seed loses to either alternative on the synthesis subset under matched compute, Thesis 1 is weakened (per paper §3.3, falsifier **F2**).

## Setup (from paper §3.1)

- 7 LLM instances, same base model, role-prompted.
- Coordinator publishes the query.
- Specialists work in their specialty.
- Each adjacent specialist-pair has a Vesica subject they both publish to and read from.
- Coordinator reads 6 Vesica subjects + 6 specialist outputs and synthesizes.
- Topology variants:
  - **Seed (Hex-Vote)**: 6 lateral Vesica subjects.
  - **Flat debate** (control): all-to-all $K_7$.
  - **Hub-and-spoke** (control): coordinator + 6 leaves, no lateral Vesicas.
  - **Single-agent CoT** (floor): same base model, no multi-agent.
- Benchmarks: MMLU-Pro (Wang et al. 2024), GPQA Diamond (Rein et al. 2024), and a custom synthesis benchmark held out by the project (each item explicitly requires combining knowledge from ≥3 of the 6 specialty domains).
- Token budget held constant across conditions.

## Blocking decisions (must resolve before PRECOMMIT.md can be locked)

1. **Base model selection.** Llama-3.1-8B (used in Exp 01) is plausibly too small for synthesis tasks; the slice's coverage-doesn't-convert-to-F1 finding suggests model strength matters. Candidate set: `llama3.1:70b`, `qwen2.5:32b`, `mistral-small:22b` (local via Ollama/MLX), or `claude-haiku-4-5` / `gpt-4.1-mini` via API. Need to decide local-vs-API before committing — affects budget, reproducibility, and whether the 7-instance topology is feasible. **Lean:** API + a frontier model, since matched-token-budget is the controlling axiom and local 70B isn't fast enough for the iteration cycle.
2. **Specialty domains.** §3.1 calls for "6 specialties" but doesn't enumerate them. Candidate: the six MMLU-Pro top-level categories (STEM, humanities, social sciences, law, business, health) — clean partition, defensible. Alternatives: GPQA's three (bio, chem, phys) plus three engineered domains. **Lean:** MMLU-Pro's six top-level categories.
3. **Operationalization of "Vesica subject."** This is the geometric primitive in the multi-agent setting. Concrete candidates:
   - (a) A *named shared message log*: both adjacent specialists read all messages on the log; they write new messages tagged to it. Cheap to implement. No geometric content; just a topic name.
   - (b) A *running summary* maintained at the intersection: both specialists co-edit a third document that is the synthesis of their joint work. The Vesica is then a *written artifact*, not just a message tag.
   - (c) An *embedding-space intersection* of the two specialists' rolling thought representations (closer to the paper's geometric framing). Requires structured rollouts; expensive.
   - **Lean:** (b), because it (i) preserves the geometric metaphor (the Vesica = a third object that depends on both parents), (ii) is implementable on existing chat APIs, (iii) makes "what the Vesica produced" inspectable, which matters for diagnostic analysis if NO-GO.
4. **Custom synthesis benchmark.** §3.1 says "a custom synthesis benchmark (held out)." This needs to *exist* before PRECOMMIT. Open question: build it in-house (50–100 hand-curated items with ≥3-domain composition) or repurpose an existing benchmark (e.g. SCIENCEBENCH cross-domain subset). **Lean:** build a small held-out set (50 items), with the build process and item provenance fully documented before the eval runs.
5. **Token-budget enforcement.** "Within ±3% of highest-cost baseline" is the spec. Need a concrete mechanism: per-turn token caps, total-cost caps, retry budgets. **Lean:** total-cost cap per query (counting all 7 agents' input+output tokens), with a 3% tolerance band measured post-hoc, and a re-run with adjusted caps if any condition is outside the band.
6. **Number of trials per benchmark item.** Multi-agent runs are stochastic; a single rollout per item is noisy. **Lean:** 3 rollouts per item, majority vote or first-correct, decided before the run.

## What this experiment does *not* test

- Whether more specialists is better (the count is fixed at 6 lateral + 1 coordinator by the hex constraint).
- Whether different topologies of the *same* node count behave differently (that's what the controls already test).
- Whether the primitive is useful in retrieval (that's Exp 01–02).
- Whether the primitive is useful at training time (that's Exp 04).

## Dependencies on Experiments 01–02

**None for the test itself.** Exp 03 stands on its own and tests a different projection of the thesis.

But: the *project-level prioritization* of running Exp 03 depends on Exp 02's verdict. If Exp 02 returns NO-GO, Exp 03 becomes the highest-priority surviving falsifiable test of the thesis and gets locked next. If Exp 02 returns GO/WEAK GO, the retrieval-form of the thesis is alive and Exp 03 may be sequenced after the retrieval-side follow-ups (B3, etc.) for resource reasons.

## Timeline (paper §3.1 estimate)

4–6 weeks once PRECOMMIT.md is locked and the synthesis benchmark exists.

## References

- Wu et al., *AutoGen*, arXiv:2308.08155.
- Wang et al., *MMLU-Pro*, 2024.
- Rein et al., *GPQA*, 2024.
- Paper §3.1 "EXPERIMENT 2" and §3.3 falsifier F2.
