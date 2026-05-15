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
