# Experiment 04 — Curriculum Pilot (pre-PRECOMMIT de-risking)

**This is NOT the experiment. This is NOT a PRECOMMIT.** Nothing here is
binding. This document and the code under `pilot/` exist solely to
answer, cheaply and locally, whether the full Experiment 04 is worth
locking a `PRECOMMIT.md` for. Writing a PRECOMMIT for the pilot itself
would be a category error — a pilot's job is to *inform* the decision to
pre-register, not to be pre-registered.

If the pilot passes, its findings feed the blocking-decision list in
[`../README.md`](../README.md) and a real `experiments/04-curriculum/PRECOMMIT.md`
gets written and locked *before* any code runs against the real
benchmarks (COGS/SCAN/ReCOGS). If the pilot fails, Experiment 04 does
not get a PRECOMMIT and the curriculum projection is recorded as
not-worth-the-compute, with the pilot as the evidence.

**Date:** 2026-05-15. **Author:** Jacob Patterson. Compute: local MPS
(Apple M4 Pro). No cloud. Pre-PRECOMMIT.

---

## Why this pilot exists

Experiments 01 and 02 closed the *retrieval* form of the MANDORLA
thesis: the intersection primitive, imposed at inference time on a
*borrowed* geometry (contriever), is net-harmful. The curriculum
(Thesis 3) is structurally different: it puts the intersection in the
*training objective*, so the representation space is *shaped to make
intersections meaningful*. That is the only surviving projection where
the thesis is tested as a representation-learning claim rather than an
inference-time trick.

But the full Experiment 04 (100–300M params from scratch on
C4/RedPajama, COGS/SCAN/ReCOGS, 8–12 weeks, multi-GPU) is the most
expensive and slowest-to-falsify thing in the program, and it rests on
two unproven assumptions. Going straight to it would repeat the Exp 01
mistake: a large commitment before the core assumption is de-risked.
This pilot de-risks both assumptions at toy scale (~weeks → ~hours,
multi-GPU → one laptop).

## The two questions this pilot answers

**Q1 — Stability.** Can the Vesica-prediction and parent-reconstruction
losses, with gradients flowing through a differentiable box-intersection,
train *at all* inside a from-scratch transformer without collapsing
(intersections → ∅, or → constant, or reconstruction stuck at chance)?
The slice already hit box-embedding underflow at small β with a *static*
index; doing this *inside* a training loop is materially harder
optimization. If it won't train cleanly at toy scale it certainly won't
at 300M, and the answer is known for ~$0.

**Q2 — Signal (the load-bearing one).** Does the curriculum produce
*compositional transfer* that standard CLM does not, on a synthetic
probe where compositional ground truth is exact and controllable? This
directly tests the assumption the paper asserts but does not derive: that
training an explicit intersection operation yields representations whose
compositionality *generalizes to combinations never seen in training*.

Q2 is deliberately tested on a synthetic world, not COGS/SCAN. The
pilot's job is to isolate the mechanism with exact ground truth, not to
claim a benchmark result. A synthetic win does not prove Exp 04 will
work; a synthetic *failure* is strong evidence Exp 04 is not worth the
compute. The asymmetry is the point — this is a screening test, like the
slice was.

## Design revision R1 — 2026-05-15 (recall+AND world abandoned)

> **Non-binding** (this is a pre-PRECOMMIT pilot) but the reason is
> recorded per the project's discipline ethos.
>
> The original world emitted `E_i E_j SHARE P_k` and held out *pairs*
> while exposing every entity individually. Empirically, plain CLM hit
> held-out F1@G ≈ 0.994 by step 500 and 1.0 by step 1500 — with and
> without the direct-attribute (`HAS`) paragraphs. The cause is
> structural, not a hyperparameter: holding out *pairs* while each
> entity is individually observed makes the held-out task equal to
> "recall two attribute vectors, elementwise-AND them." Attention does
> associative recall + AND natively in one step, so the baseline
> ceilings and **Q2 becomes untestable** (no headroom to detect any
> curriculum effect). Partner-sparsification was rejected because it
> makes some entity attributes unobservable → probe ill-posed.
>
> R1 replaces the task with a **2-hop latent relational composition**
> where the held-out generalization cannot be reduced to recall+AND,
> and adds a third condition (generic-auxiliary control) so a positive
> result can be attributed to the *intersection* specifically rather
> than to "more training signal / more parameters." The sections below
> describe the R1 design; the R0 description is superseded.

## The synthetic world (R1)

A 2-hop relational world. The held-out task requires *constructing and
using an intermediate* the model was never directly asked for.

- **Entities** `E_0 … E_{N-1}`, each with a fixed seeded attribute
  vector `a_i ∈ {0,1}^K`.
- **Partner relation** `π`: a fixed seeded map with `π(i) ≠ i` (hop 1).
- **Property tokens** `P_0 … P_{K-1}`. Relation tokens `ATTR`,
  `PARTNER`, `QUERY`.
- **Paragraph types:**
  - `BOS E_i ATTR P_k EOS` for every `k` with `a_i[k]=1` — teaches each
    entity's attributes directly (attribute *observation* is easy by
    design; the difficulty is the composition, which is the lever R1
    chose after R0 showed attribute-sparsity breaks well-posedness).
  - `BOS E_i PARTNER E_{π(i)} EOS` for **all** `i` — teaches hop 1.
  - `BOS E_i E_j QUERY P_k EOS` for every `k ∈ (a_{π(i)} AND a_j)` —
    teaches the 2-hop composition, **only for non-held-out `i`**.
- **Ground truth** `Q(i, j) := a_{π(i)} AND a_j`. Answering requires
  hop 1 (`i → π(i)`) then hop 2 (intersect `a_{π(i)}` with `a_j`).

## The compositional-transfer probe (R1)

- A set of **entities `H`** is held out: for `i ∈ H` **no `QUERY`
  paragraph is ever emitted** (in any loss). Their `ATTR` and `PARTNER`
  facts *are* present, and `π(i)`'s `ATTR` facts are present. So every
  *part* needed to answer `Q(i,j)` was taught as a separate single-hop
  fact; the (i,j) query itself never was, for any j.
- **comp-OOD (the signal).** For `i ∈ H`, predict `Q(i,j)` for sampled
  `j`. The model cannot have memorized i's answer (i was never
  queried); it must chain two separately-taught facts latently — the
  exact OOD-composition the latent-multi-hop-reasoning literature shows
  plain transformers struggle with.
- **seen-entity control.** For `i ∉ H` with a *specific* `(i,j)` not in
  training, predict `Q(i,j)`. i *was* queried (with other j), so a
  shortcut-memorized `a_{π(i)}` can solve this. A curriculum gap that
  appears here too is *not* novel-composition transfer.
- **Metric.** F1@G (rank properties, cut at the true shared-set size).
  Signal = curriculum − baseline on comp-OOD, net of the seen-entity
  control gap **and** net of the generic-auxiliary control.

## Conditions (R1 — three, not two)

Same tiny transformer, same corpus, same seed, same token budget. Only
the auxiliary loss differs. The third condition is the load-bearing
addition R1 makes: the field already knows *generic* auxiliary
objectives shift compositional generalization, so a curriculum win is
only attributable to the intersection if it beats a capacity-matched
*non-intersection* auxiliary objective.

- **Baseline:** CLM (next-token) only.
- **Generic-aux control:** CLM **+** a capacity-matched auxiliary head
  off the same hidden state, predicting a *non-compositional* target
  (the queried entity's own attributes `a_i`) — adds parameters and
  structured signal but no intersection geometry.
- **Vesica curriculum:** CLM **+** two intersection losses, both read
  off the model's **`QUERY`-position hidden state** (the representation
  the model itself constructed — it is *not* handed `π(i)`; the only
  extra supervision is the target, the same information CLM gets from
  the `QUERY` paragraph's continuation):
  1. *Vesica prediction* — box(query hidden state) ∩ box(`E_j`) → predict
     `Q(i,j)`.
  2. *Parent reconstruction* — intersection → reconstruct `a_{π(i)}`
     and `a_j` (forbids the collapse that discards parent info).

  Trained **only on non-held-out entities'** queries. Held-out entities
  appear in *no* auxiliary loss — no probe contamination.

## What "pilot passed" means (decided now, before any run)

Both must hold:

- **Q1 pass:** over a full toy training run, the auxiliary losses
  decrease and stabilize; box-intersection log-volume stays finite and
  non-degenerate (not collapsed to ∅ or to a constant); parent
  reconstruction exceeds chance by a clear margin. Recorded from the
  training-stability JSONL, not eyeballed.
- **Q2 pass:** curriculum probe metric on held-out pairs exceeds the CLM
  baseline's by a margin that is (a) large relative to seed-to-seed
  noise across ≥3 seeds and (b) not explained by the i.i.d. (seen-pair)
  control also differing. The exact numeric bar is intentionally *not*
  fixed here — the pilot is exploratory and pre-PRECOMMIT; fixing a bar
  would imply a rigor this stage doesn't have. The full Exp 04 PRECOMMIT
  is where a numeric, pre-registered bar gets locked. The pilot reports
  effect size + across-seed spread and makes a *judgement* call,
  explicitly labeled as such.

If Q1 fails: Exp 04 is not viable as specified; the box-loss
optimization needs rethinking before any pre-registration.
If Q1 passes but Q2 fails: the transfer assumption is unsupported even
in the cleanest possible setting; Exp 04 on noisy real text is very
unlikely to work; do not pre-register.
If both pass: write `experiments/04-curriculum/PRECOMMIT.md`, lock it,
then proceed to the real benchmarks.

## Explicit non-goals

- Not COGS/SCAN/ReCOGS. Those are for the pre-registered Exp 04 only.
- Not 100–300M params. Toy scale (~few M) is sufficient to answer Q1/Q2.
- Not a publishable result. A synthetic-probe win is *internal evidence
  to justify pre-registration*, nothing more, and must be described that
  way everywhere.
- Not cloud. Local MPS; the pilot's whole value is cheap fast iteration.
- Not tuned for SOTA. The baseline is honest standard CLM on the same
  budget; the question is *differential*, not absolute.

## Layout

```
pilot/
├── PILOT.md            # this file
├── pyproject.toml
├── .python-version     # 3.12
├── LAB-NOTES.md        # chronological; pilot observations
├── src/
│   ├── synthetic.py    # seeded world + corpus + held-out probe split
│   ├── box.py          # differentiable soft box + intersection + log-vol
│   ├── model.py        # tiny transformer + CLM head + box head
│   ├── losses.py       # vesica-prediction + parent-reconstruction
│   └── probe.py        # held-out-pair compositional metric
├── scripts/
│   ├── 01_gen_corpus.py
│   ├── 02_train.py     # both conditions, stability JSONL
│   └── 03_evaluate_probe.py
└── tests/              # box analytic limits, corpus invariants, probe
```

## Sign-off

Pre-PRECOMMIT pilot. Nothing locked. Author: Jacob Patterson,
2026-05-15.
