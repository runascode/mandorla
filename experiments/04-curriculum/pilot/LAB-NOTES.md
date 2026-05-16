# Curriculum Pilot — Lab Notes

Append-only. Oldest at top. This is a pre-PRECOMMIT pilot; nothing here
is binding (see `PILOT.md`).

---

## 2026-05-15 — Pilot built and smoke-verified

Scaffolded the synthetic-world pilot to de-risk Experiment 04 before
any PRECOMMIT lock. Local MPS, no cloud (per the pilot's cheap-fast-
iteration purpose).

### Synthetic world (scripts/01)

64 entities × 16 binary properties, prop_density 0.5, 15% of unordered
pairs held out. Materialized: 7,916 paragraphs, 1,714 training pairs,
302 held-out pairs, mean 8.3 attrs/entity, mean shared-set size ≈ 4.3
for both train and held-out pairs, 99.3% of held-out pairs have a
non-empty shared set (probe is well-posed). Ground-truth Vesica =
exact bitwise-AND; held-out isolation is unit-tested (no pair leakage
in either order).

### Box math

Differentiable soft box (center + softplus half-extent), softplus side
length so disjoint boxes still get gradient *in the training regime*.
Documented limitation, tested explicitly: at extreme separation
(≈40 units, β=0.1) the softplus tail saturates and the gradient
underflows to exactly 0 — no fixed-β smoothing avoids this. Mitigation
is keeping box centers bounded (small init + weight decay on the box
head); recorded as a tracked property so it can't surprise the Q1
stability read.

### Smoke run (50 steps, both conditions, seed 1337)

- 642,453-param model, MPS, executes cleanly.
- Curriculum: all three losses decrease monotonically over 50 steps
  (clm 4.60→2.27, vesica 0.70→0.57, parent 0.70→0.67). Intersection
  log-volume stays finite and non-degenerate (≈4.5, not → −∞, not
  constant). No collapse signature in the smoke window.
- Baseline: CLM decreases 2.59→2.24 as expected.
- Both probe to ≈chance at step 50 (held-out F1@G ≈ 0.36–0.37) — that
  is the expected null at 50 steps; the smoke test only proves the
  loop runs, not the signal.
- 20/20 unit tests green (box analytic limits + world invariants +
  held-out isolation). No autograd warnings after detaching loss
  scalars in the metrics dict.

Setup objective met: the losses train, the box gradient flows in the
operating regime, the probe is well-posed and contamination-free, the
harness is deterministic and multi-seed-ready. The full pilot run
(Q1 stability over a real training horizon + Q2 across-seed signal)
is the next step.

---

## 2026-05-15 — R0 world falsified by its own baseline; pivot to R1

Launched the full run (3 seeds × 2 conditions × 4000 steps). Killed it
at the first probe checkpoint.

**Finding.** The R0 baseline (plain CLM, no curriculum) reached
held-out F1@G ≈ 0.994 at step 500 and 1.0 by step 1500 — *with* the
`HAS` paragraphs and again *without* them (SHARE-only). The synthetic
task as designed is solved near-perfectly by a plain transformer, so
there is zero headroom for a curriculum effect and **Q2 is untestable
in the R0 design**.

**Root cause (structural, not a hyperparameter).** Holding out entity
*pairs* while every entity is individually observed reduces the
held-out task to "recall two attribute vectors, elementwise-AND them."
Attention does associative recall + elementwise AND natively in one
step. So R0 was never testing compositional *construction*; it was
testing recall+AND, which transformers already do. Partner-sparsifying
to harden it was rejected: it makes some entities' properties
unobservable from training co-occurrence → probe ill-posed (an
unlearnable component would muddy Q2).

This is the pilot working exactly as intended: ~30 min of compute
caught a design flaw that would have made a 3-month experiment
unattributable. It is itself a recorded result — the *obvious*
synthetic isolator for this curriculum is dominated by trivial recall,
which raises the bar for cleanly attributing any future COGS/SCAN
effect to the intersection mechanism rather than to capacity.

**Pivot — R1 (see PILOT.md "Design revision R1").** Task replaced with
2-hop latent relational composition (`i → π(i)`, then intersect
`a_{π(i)}` with `a_j`); held out at the *entity* level (held-out
entities are never queried, so the held-out answer cannot be
shortcut-memorized and must be composed latently from two separately-
taught single-hop facts). Third condition added: a capacity-matched
generic-auxiliary control, because the field already knows generic
auxiliary objectives move compositional generalization — a curriculum
win is only attributable to the *intersection* if it also beats the
generic-aux control, not just plain CLM. Box module (box.py) is
task-agnostic and carried over unchanged; synthetic/model/losses/probe
rebuilt for R1.

---

## 2026-05-15 — R1 full run complete: Q1 PASS, Q2 NO (decisive)

9 runs (3 conditions × 3 seeds × 6000 steps), MPS, ~1.5 h.

### Q1 — stability: clean PASS

Curriculum auxiliary losses on supervised (trained-entity) queries,
final step, all three seeds:

| seed | vesica loss | parent loss | ves bit-acc | par bit-acc | inter log-vol | inter-emb batch-std |
|---|---|---|---|---|---|---|
| 1337 | 0.000 | 0.000 | 1.000 | 1.000 | 80.3 ± 18.4 | 7.86 |
| 1338 | 0.000 | 0.000 | 1.000 | 1.000 | 76.2 ± 21.8 | 7.89 |
| 1339 | 0.000 | 0.000 | 1.000 | 1.000 | 80.6 ± 16.6 | 7.45 |

The box-intersection losses train flawlessly inside a from-scratch
transformer. No collapse: intersection log-volume is finite and
well-dispersed (not → −∞, not constant), intersection embeddings vary
across the batch. The Exp 01 slice's static-index box-underflow worry
does **not** generalize to the in-training setting. Q1 is an
unambiguous PASS — the mechanism is fully trainable.

### Q2 — signal: NO

comp-OOD F1@G (held-out-entity 2-hop latent composition), mean ± std
over 3 seeds:

| condition | comp-OOD F1@G | seen-ctrl F1@G |
|---|---|---|
| baseline | 0.5876 ± 0.0143 | 1.0000 |
| generic_aux | 0.5796 ± 0.0123 | 1.0000 |
| curriculum | **0.5702 ± 0.0166** | 1.0000 |

- curriculum − baseline    = **−0.0174**
- curriculum − generic_aux = **−0.0094**  (−0.65 pooled SD)
- seen-control gap is identically 0 (all conditions ceiling at 1.0)

The curriculum does not close the −0.40 latent-2-hop gap. It is
indistinguishable from — if anything marginally *below* — both plain
CLM and the capacity-matched generic-auxiliary control. Q2 is NO on
every seed.

### The shape is the finding

This is **not** "the box losses won't train" (Q1 falsified that — they
train to zero, bit-accuracy 1.000). It is: the model **perfectly
masters the explicit intersection construction on supervised data and
that mastery transfers nothing** to held-out latent composition. The
thesis's load-bearing assumption — that *training an explicit
intersection operation yields representations whose compositionality
generalizes to unseen combinations* — fails in the cleanest possible
setting, where it had every structural advantage: exact ground truth,
controlled difficulty, demonstrated baseline headroom (0.59, not
ceiling), auxiliary losses converging to exactly zero, and a fair
LM-head readout shared across conditions. A perfectly-learned
intersection operator is **inert** for the capability it was posited
to induce.

### Decision (per PILOT.md "what pilot passed means")

"If Q1 passes but Q2 fails: the transfer assumption is unsupported
even in the cleanest possible setting; Exp 04 on noisy real text is
very unlikely to work; do not pre-register."

→ **Do not write `experiments/04-curriculum/PRECOMMIT.md`. Do not run
the full Exp 04 (100–300M from-scratch, COGS/SCAN/ReCOGS, 8–12 weeks,
multi-GPU).** The pilot did its job: ~3 h of laptop compute removed the
warrant for a quarter of multi-GPU work by falsifying, in isolation,
the assumption that work depended on.

### Scope discipline (what this does and does not claim)

This pilot is pre-PRECOMMIT and synthetic. It does **not** formally
falsify Thesis 3 — that would require the pre-registered COGS/SCAN
experiment. What it does: removes the justification for that
experiment by showing its central assumption fails where it had every
advantage. Formal-falsification language is not used; "the warrant for
Exp 04 is gone" is the correct and sufficient claim.
