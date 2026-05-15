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

## The synthetic world

Designed so the "Vesica of two entities" has an *exact, checkable*
ground truth: set intersection.

- **Entities.** `N` entity tokens `E_0 … E_{N-1}`. Each entity `i` has a
  fixed latent attribute vector `a_i ∈ {0,1}^K` — membership in `K`
  binary properties. The `a_i` are sampled once, seeded, and fixed.
- **Property tokens.** `P_0 … P_{K-1}`.
- **Paragraphs.** A paragraph is a short token sequence asserting a
  shared property: `E_i E_j SHARE P_k`, emitted iff `a_i[k] = a_j[k] =
  1`. Filler/among other paragraph types are added so the corpus isn't
  trivially one template (the model must actually represent entities,
  not pattern-match a fixed slot).
- **Ground-truth Vesica.** `V(E_i, E_j) := a_i AND a_j` (bitwise) — the
  exact set of properties the pair shares. This is a *literal* set
  intersection, the cleanest possible operationalization of "the
  intersection of two regions."

## The compositional-transfer probe

- A subset of entity pairs `H` is **held out**: those pairs *never*
  co-occur in any training paragraph (neither as `E_i E_j` nor `E_j
  E_i`). The model sees every entity individually (in pairs with other
  entities) but never the held-out pair *together*.
- **Probe task.** For each held-out pair `(E_i, E_j) ∈ H`, predict
  `V(E_i, E_j) = a_i AND a_j` — the shared-property set the model was
  never shown for this pair.
- **Metric.** Mean per-bit F1 (or exact-set accuracy) over `H`.
- **Why this is the right probe.** A model that learned the *operation*
  (represent each entity's attributes, intersect them) generalizes to
  unseen pairs. A model that memorized co-occurrence cannot — the pair
  was never in training. The gap between the curriculum model and the
  CLM baseline *on `H`* is the pilot signal. (On *seen* pairs both can
  win by memorization; that's the i.i.d. control, expected ≈ equal.)

This mirrors the structure of COGS/SCAN (train on parts, test on novel
combinations) at a scale where ground truth is exact and a result is
unambiguous.

## Conditions

Same tiny transformer, same corpus, same seed, same token budget. Only
the loss differs.

- **Baseline:** standard CLM (next-token) only.
- **Curriculum:** CLM **plus** two auxiliary losses:
  1. *Vesica prediction* — entity embeddings → per-entity box →
     differentiable box-intersection → a head predicting the shared
     property set `a_i AND a_j`. Trained only on *seen* pairs.
  2. *Parent reconstruction* — intersection representation → reconstruct
     `a_i` and `a_j` (denoising/invertibility; forbids the degenerate
     collapse where the intersection discards parent information).

The probe is evaluated on **held-out pairs** for *both* conditions. The
curriculum never sees held-out pairs in *any* loss, including the
auxiliary ones — otherwise the probe would be contaminated.

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
