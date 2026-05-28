# Experiment 06 — Substrate Pilot (pre-PRECOMMIT design)

**Status**: Pre-PRECOMMIT design sketch. Not yet runnable.

**Author**: Jacob Patterson · 2026-05-28

**Two-stage program**:
- **Stage A** — thesis-independent literature survey, deliverable in
  [`findings/substrate-rethinks-survey.md`](../../findings/substrate-rethinks-survey.md).
- **Stage B** — PRECOMMIT-locked toy pilot under this directory.

A must complete before B's PRECOMMIT can be locked. See
[Sequencing](#sequencing).

**Related project documents**:
- [`README.md`](../../README.md) §"Net standing" — the disconfirmation
  history this experiment responds to.
- [`findings/reader-saturation-hotpotqa.md`](../../findings/reader-saturation-hotpotqa.md)
  — existing thesis-independent finding; Stage A targets the same shape
  (citable, survives whatever MANDORLA decides about its future).
- [`CLAUDE.md`](../../CLAUDE.md) — discipline this experiment follows.

---

## 1. Purpose

Test whether the surviving rescue of MANDORLA — *"the architecture
itself must be intersection-native"* — has signal at the cheapest
possible scale, before any commitment to a multi-year program.

The cheap form: a full-factorial ablation that asks whether
region-nativeness must hold *across all three primitives* (input,
representation, objective) **jointly** to produce the predicted lift,
or whether the lift (if any) is carried by a single axis. If the
latter, "all three together" is decorative and the rescue collapses to
a single-axis claim that the prior literature has typically already
tested.

## 2. What this slice tests / does not test

### Tests

Whether at small scale (1–10M params), on a synthetic compositional
region-prediction task, a region-native (input × representation ×
objective) model outperforms every "drop-one-axis" variant at matched
capacity.

### Does **not** test

- **Scale-up** of any region-native architecture. A positive result
  warrants Stage C (a small-LM build at 10–100M params on
  natural-language data); it does **not** itself ship a model.
- **Natural-language transfer.** Pilot is on synthetic data by design
  — small, controllable, deterministic ground truth. Natural-language
  questions are deferred to C, contingent on B-warrant.
- **Absolute competitiveness** of any specific prior region/HD/
  predictive-coding architecture. This is an isolation test, not a
  benchmark.

## 3. Methodological context

The disconfirmation history is in
[`README.md` §"Net standing"](../../README.md). Across four screening
experiments (01, 02, 04-pilot, 04-pilot2-operator), every rescue of the
intersection thesis came back negative once tested under conditions
that closed its plausible escape route:

- borrowed geometry → *trained from scratch*
- auxiliary / bypass route → *on critical path*
- capacity unmatched → *capacity-matched*

The pattern — escalating purity required after each disconfirmation —
is the signature of a thesis sliding toward unfalsifiability. This
pilot is the **last cheap-to-screen form** of that escalation. It
tests the architectural rescue at the smallest scale that can decide
the question, with the same capacity-matching and no-bypass discipline
that decided the operator pilot.

## 4. Stage A — Literature survey (thesis-independent)

### 4.1. Scope

Five lineages have made serious attempts at non-point, non-token, or
non-autoregressive substrates. Each is read for: what was attempted,
at what scale, what failed and why, and which of the three primitive
axes (input / representation / objective) it actually moved.

1. **Sparse Distributed Memory & Hyperdimensional Computing** —
   Kanerva; Plate (HRRs); Schlegel et al. review.
2. **Box / region embeddings** — Vilnis, Li, Patel, Boratko (McCallum
   group). Already cited in MANDORLA.
3. **Topological / geometric deep learning** — Bodnar, Bronstein.
   Already cited.
4. **Predictive coding & active inference for language** — Friston;
   Whittington–Behrens (TEM); Bogacz formalisms.
5. **Non-autoregressive sequence models & latent diffusion LMs** —
   LeCun (JEPA); Bakhtin (EBLM); Lou (SEDD, score-entropy discrete
   diffusion); Gulrajani et al.

### 4.2. Deliverable

A single file: [`findings/substrate-rethinks-survey.md`](../../findings/substrate-rethinks-survey.md).

**Thesis-independent.** Stands on its own, citable, survives whatever
MANDORLA decides about its future. Same shape as
[`findings/reader-saturation-hotpotqa.md`](../../findings/reader-saturation-hotpotqa.md).

Required content:

- One section per lineage: attempt, scale, failure mode (if any), why.
- A `lineage × {input, representation, objective}` matrix — most
  attempts move 1 axis; almost none move all 3. Empty cells motivate
  the pilot.
- The load-bearing claim of "all three together," sharpened from
  paragraph form to a single PRECOMMIT-ready sentence with operationalized
  metric M, Δ_warrant numerics, and toy-task family.
- A short "what's left to test" section identifying whether prior
  work already closes the question (in which case B does not run).

### 4.3. Effort

1–2 person-weeks. ~50–100 papers shortlisted, ~15–25 read deeply.

### 4.4. Output that B needs

- Operationalized load-bearing claim (M, Δ_warrant numerics).
- Specific toy task within the region-prediction family.
- Any prior negative result that already closes the question — in
  which case the gate at §7 closes 06 without running B.

## 5. Stage B — Toy pilot

### 5.1. Provisional load-bearing claim

> There exists a compositional task on which a region-native
> (input × representation × objective) model outperforms every
> "drop-one-axis" variant at matched capacity, by ≥ Δ_warrant on
> metric M with 95% CI clear of zero, across all seeds.

To be sharpened by A and locked in `PRECOMMIT.md` before any number
reaches a screen.

### 5.2. Full 2³ factorial design

Eight conditions, capacity-matched (identical parameter count and
identical training compute):

| Cond | Input | Rep | Obj | Role |
|---|---|---|---|---|
| **A** | reg | reg | reg | The "all three together" claim |
| B1 | pt  | reg | reg | Drop input-axis |
| B2 | reg | pt  | reg | Drop rep-axis |
| B3 | reg | reg | pt  | Drop obj-axis |
| C1 | reg | pt  | pt  | Keep only input-axis (diagnostic) |
| C2 | pt  | reg | pt  | Keep only rep-axis (diagnostic) |
| C3 | pt  | pt  | reg | Keep only obj-axis (diagnostic) |
| D  | pt  | pt  | pt  | Standard transformer baseline floor |

- `reg` = region-native primitive at that axis (boxes via
  `iesl/box-embeddings`, or equivalent specified by A).
- `pt`  = point-native primitive at that axis (standard embeddings /
  point hidden states / next-point-or-token objective).

### 5.3. Capacity matching

**Constrained and audited**: parameter count and FLOPs per training
step, held identical across all 8 conditions.

**Reported but not constrained**: total training compute per condition
(steps × per-step FLOPs). Each condition trains to its own loss
plateau on a held-out probe (see §6), so per-condition total compute
may differ. If a region condition needs more total compute to
converge, that asymmetry is a result and is reported as such, not
papered over. The PRECOMMIT statement must make this distinction
explicit to forestall the "region conditions trained longer = unfair"
reading.

The Exp 04-pilot2-operator lesson lifted here: without parameter +
per-step FLOP matching we are measuring parameter count, not
primitives. Box-embedding parameters per region (centre + extent) are
counted toward total param count; the point baselines receive the
equivalent additional parameter budget elsewhere in the network
(e.g., wider hidden dim, additional layer).

### 5.4. Decision rule (locked in PRECOMMIT before any number)

- **A > max(B1, B2, B3)** by ≥ Δ_warrant on M, 95% CIs clear of zero,
  all seeds → **warrant for C** (small-LM build, 10–100M params on
  natural-language data — separately scoped, separately
  pre-registered).
- **A ≤ any of {B1, B2, B3}** → **"all three together" decisively
  negative**; do not pre-register C; close the architectural rescue.
- **Ambiguous** (overlapping CIs, mixed across seeds) → run additional
  seeds; if still ambiguous after 6 seeds total, close as inconclusive
  (not warrant).

C1–C3 are **diagnostic**: when A fails, they identify *which* single
axis (if any) carries weight. They do not enter the decision rule.
If any C variant beats A, that is a **separate finding** — single-axis
sufficiency on a region-prediction task, contra what some prior
literature implies — and warrants its own write-up under `findings/`,
not a rescue of the "all three together" claim and not a warrant for
Stage C.

Δ_warrant provisional: **0.05** on a 0–1 metric. Sharpened in
PRECOMMIT after A — informed by what effect sizes prior literature
reports as meaningful on similar tasks.

### 5.5. Toy task family

Specific task selected in A, locked in PRECOMMIT.

**Family**: synthetic compositional region-prediction.

- **Inputs**: encoded "facts," each natively a region in some semantic
  space (centre + extent in ℝ^d).
- **Output**: a target region — intersection, union, or hierarchical
  containment — of the input facts.
- **Metric M (provisional)**: containment-IoU on output region.
  Alternatives reported alongside: symmetric-difference volume,
  vesica overlap.
- **Design constraint**: the task must be one where the *region*
  representation of inputs and outputs is **not optional information**
  — point representations of either strictly lose what the task
  requires. This makes "all three together" a falsifiable claim
  rather than a self-confirming one.
- **Headroom check before lock**: a saturation diagnostic
  ([`findings/reader-saturation-hotpotqa.md`](../../findings/reader-saturation-hotpotqa.md)
  pattern) — verify the metric is responsive to known interventions
  before training the 8 conditions. PRECOMMIT must specify (i) the
  intervention — a known-magnitude degradation of inputs or ground
  truth — and (ii) the threshold of metric movement that counts as
  passing. If M does not move by at least the locked threshold, the
  task or metric is saturated for these conditions and the spec
  returns to Stage A for redesign **before** any of the 8 main runs.

### 5.6. Compute scope

- 1–10M params per model, scaled to fit on a single Apple Silicon
  or single consumer GPU.
- 8 conditions × 3 seeds = **24 training runs**.
- Days of laptop / single-GPU time, not weeks. Falls within the
  project's cheap-to-screen budget (Exp 04 pilot reference: ~3h
  laptop).

### 5.7. Pre-registration discipline

Per [`CLAUDE.md` §1](../../CLAUDE.md):

- [`PRECOMMIT.md`](PRECOMMIT.md) locked after A produces the
  sharpened load-bearing claim; **before any data hits a model**.
- [`LAB-NOTES.md`](LAB-NOTES.md) appended chronologically from this
  design doc forward.
- [`results/RESULTS.md`](results/RESULTS.md) on completion: all
  pairwise Δs (A vs B1, A vs B2, A vs B3, plus C1/C2/C3 diagnostics)
  with **10K-resample bootstrap CIs**; loss curves for all eight
  conditions; capacity-match audit (param count, FLOPs per step);
  commit SHA; master seed **1337**; calibrated values for any free
  parameters.

## 6. Architecture spec (provisional)

To be sharpened in PRECOMMIT after A. Provisional choices:

- **Region primitive**: GumbelBox via `iesl/box-embeddings` (already
  in [exp01 stack](../../experiments/01-vesica-rag/pyproject.toml)).
  A may recommend an alternative (e.g., HD-binary vectors) if the
  literature suggests one is more credible for this test.
- **Model body**: small transformer (≤ 4 layers, ≤ 256 dim) for
  point conditions; matched-capacity region-native variant for
  region conditions. No attention is required if the synthetic task
  is short-sequence; a 2–4 layer MLP with optional sequence mixing
  may be sufficient.
- **Training**: AdamW, cosine schedule, fp32 to start (avoid bf16
  numerical interactions with box volumes until verified). Early-stop
  on a held-out probe rather than fixed step count, so each condition
  reaches its own plateau. Total training compute per condition is
  **reported alongside results** (not constrained — see §5.3 for what
  *is* constrained: parameter count and per-step FLOPs).
- **Stack**: Python 3.12, `uv`, PyTorch 2.11+, `box-embeddings`,
  pytest. Mirrors the exp1 / exp2 / exp4 stack.

## 7. Sequencing and gates

1. **Now → +1–2 wks**: Stage A. Reading and synthesis. Output:
   `findings/substrate-rethinks-survey.md` v1.
2. **+1 day after A**: **Decision gate G1.** If A surfaces a prior
   negative result that closes the question, B does not run; a brief
   negative-by-prior-art note is written under `findings/` and 06 is
   closed. Otherwise, proceed.
3. **+2–3 days**: Lock [`PRECOMMIT.md`](PRECOMMIT.md) with
   operationalized claim, task, Δ_warrant, metric.
4. **+1 wk**: Implement primitives + training loop. Capacity-match
   audit before any seeded run.
5. **+days**: Run all 8 conditions × 3 seeds (or up to 6 seeds if
   §5.4 triggers). Write `results/RESULTS.md`.
6. **Decision per §5.4.**

## 8. Budget (honest accounting)

| Stage | Effort | Compute | Output |
|---|---|---|---|
| A | 1–2 person-weeks | 0 | `findings/substrate-rethinks-survey.md` |
| Gate G1 | 1 day | 0 | Pass/close note |
| PRECOMMIT lock | 2–3 days | 0 | `PRECOMMIT.md` |
| Impl + audit | 1 week | minimal | code, capacity audit |
| Pilot run | days | ~24 small-model runs | `results/RESULTS.md` |
| **Total to decision** | **~3–4 weeks** | **days of GPU** | **go/no-go on the architectural rescue** |

If B comes back positive, **Stage C** (the small-LM build) is a
separately scoped follow-on: months of design, weeks of GPU, its own
PRECOMMIT and pre-registration. **The current design does not commit
to C.**

## 9. What "ship" means

- [`findings/substrate-rethinks-survey.md`](../../findings/substrate-rethinks-survey.md)
  v1 published.
- [`experiments/06-substrate-pilot/PRECOMMIT.md`](PRECOMMIT.md)
  locked (or, if G1 closes 06, the negative-by-prior-art note is
  published).
- [`experiments/06-substrate-pilot/results/RESULTS.md`](results/RESULTS.md)
  with the decision and CIs.
- Root [`README.md`](../../README.md) status table updated to reflect
  the outcome.

## 10. What is deferred

- **Stage C** — small-LM build at 10–100M params on natural-language
  data. Out of scope; only entered on a B-warrant.
- **Scale-up arguments** of the form *"this would work if only we had
  a frontier-scale model."* By construction, the pilot tests for an
  effect at small scale. If the effect requires frontier scale to
  appear, it is not falsifiable by the cheap-to-screen program — and
  the rescue pattern documented in §3 says we have already paid that
  bill.
- **Architectural variants beyond box-style regions** (simplicial
  complexes, HD-binary vectors as the rep primitive, JEPA-style
  energy objectives). A may recommend one in place of the
  provisional `iesl/box-embeddings` default; otherwise deferred.

## 11. Risks and pre-mortem

- **Region representation parameter accounting.** Boxes have 2× the
  parameter cost per unit (centre + extent). Capacity-matched
  comparison must count those parameters honestly. *Mitigation:*
  param-count audit in `RESULTS.md`, with point baselines awarded
  matching parameter budget elsewhere.
- **Loss landscape asymmetry making fixed training-compute budgets
  inadequate for region conditions.** *Mitigation:* each condition
  trained to its own loss plateau (early-stopping on a held-out
  probe), with training compute audited per-condition. If region
  conditions need more compute to converge, that asymmetry is itself
  a result and is reported, not hidden.
- **Toy-task choice biases toward region-native by construction.**
  *Mitigation:* A's deliverable includes a paragraph defending the
  task choice against this objection; the task must be one where
  point representations *can in principle* succeed, only with a
  measured loss of information.
- **"Bitter lesson" applies.** Possibility that even a positive
  small-scale result fails to scale. Known limitation of the
  cheap-to-screen program; B's positive is *necessary but not
  sufficient* warrant for C. PRECOMMIT for C, if it happens, includes
  its own scaling-curve prediction.
- **The rescue pattern recurses.** If B comes back ambiguous and the
  rescue is *"the task wasn't designed cleanly enough"*, that is the
  signature documented in §3 reappearing one level deeper. Treat as
  such. Do not redesign and re-run; close as inconclusive.

## 12. Amendments

*(none — to be dated and numbered if the design changes between this
doc and PRECOMMIT lock)*

## 13. Sign-off

- **Author**: Jacob Patterson — 2026-05-28
- **Status**: Pre-PRECOMMIT design sketch. Not yet runnable. Stage A
  pending.
