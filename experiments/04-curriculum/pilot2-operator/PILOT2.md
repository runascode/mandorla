# Pilot 2 — Composition-operator test (pre-PRECOMMIT, decision rule locked before any run)

**This is NOT the experiment and NOT a PRECOMMIT.** Pre-PRECOMMIT
de-risking, like `../pilot/`. Nothing here binds anything. But the
verdict rule below is **fixed before a single number exists** — that
fixity is the entire epistemic value of this pilot, given it is the
last surviving "but it was never tested fairly" escape for the
MANDORLA thesis. Author: Jacob Patterson. Date: 2026-05-15. Local MPS.

## Why this exists (the residual after three negatives)

Exp 01/02 (retrieval) and the `../pilot/` curriculum run all returned
negative. The curriculum pilot's residual loophole: intersection was an
*auxiliary* loss inside an attention model with a co-present next-token
objective, so the model could — and did — learn the intersection head
perfectly and route the actual answer around it (CLM offered a bypass).
The unfalsified claim left standing: *intersection-as-the-computation,
on the critical path with no bypass, might confer a compositional
advantage even though intersection-as-a-side-loss did not.* This pilot
tests exactly that, and nothing larger. It does not test a fully
intersection-native architecture (that is a multi-year program, not a
pilot, and committing to it now would repeat the mistake the discipline
exists to prevent).

## The confound this design must defeat

The `../pilot/` world's ground truth is literally a set intersection
(`a_{π(i)} AND a_j`). If we only tested an intersection-shaped
bottleneck against a point bottleneck on that task, a win would be
near-tautological: *the tool matches the data generator*. That tells us
nothing about whether intersection is a useful **general** primitive —
it is the R0 recall+AND trap in new clothes. The only way a positive is
interpretable is to **also** run a task whose ground truth is **not** an
intersection and see whether the intersection bottleneck still wins.

## Design (2 × 2, identical everything except the two factors)

Shared, held constant across all cells: the synthetic 2-hop relational
world and its 2-hop structure (`i → π(i)` taught only as a separate
`PARTNER` fact); the `TinyTransformer` encoder; data; seeds protocol;
optimizer/budget; the **identical answer head** (same architecture,
same input width); the held-out-entity split and the comp-OOD probe.

**No-bypass construction (fixes the `../pilot/` flaw):** CLM is trained
**only on `ATTR` and `PARTNER` paragraphs** — enough for the encoder to
learn each entity's attributes and the partner map, but it does *not*
teach the composition. The 2-hop answer is produced **exclusively**
through the bottleneck → answer head, supervised only on
trained-entity queries. There is no alternative path to the answer, so
the bottleneck operator is genuinely on the critical path. The probe
reads the **answer head** (through the bottleneck), never an LM head.

**Factor 1 — composition bottleneck** (the isolated variable). Both
take the encoder's QUERY-position hidden (must encode `a_{π(i)}`) and
the `E_j` hidden, and produce a vector of width `2·box_dim`, then the
*identical* answer head maps it to K logits.
- **POINT:** `MLP([h_query, h_j]) → 2·box_dim`. Capacity-matched to the
  INTERSECTION path's box-head; exact param counts logged and asserted
  within tolerance.
- **INTERSECTION:** `h → box` (center, log-half) for each operand; the
  `2·box_dim` vector is `intersection_embedding(box_q, box_j)`
  (midpoint ++ soft side). The intersection operator *is* the
  bottleneck; no MLP can bypass it.

**Factor 2 — task ground truth.**
- **AND (intersective):** `Q(i,j) = a_{π(i)} AND a_j`.
- **NONINT (non-intersective control):** `Q'(i,j) = 𝟙[ M·[a_{π(i)};a_j]
  > θ ]`, a fixed seeded random linear-threshold relation over the
  joint 2K-bit operand. Not a per-operand intersection, not XOR/OR
  (which would be a region-hostile strawman) — a generic learnable
  2-hop relation. Per-bit θ calibrated on the entity population so the
  positive rate per bit ≈ that of AND (≈0.25), keeping F1@G label
  balance comparable across tasks. A test asserts NONINT disagrees with
  AND on >20% of (i,j,bit) so it is demonstrably a different relation.

Cells: {POINT, INTERSECTION} × {AND, NONINT} × seeds {1337,1338,1339}
= 12 runs. comp-OOD F1@G on held-out entities; seen-entity control for
shortcut detection.

## Gating check (before trusting any verdict)

If **POINT on AND** ceilings on comp-OOD (≈1.0, like the R0 baseline),
there is no headroom and the contrast is void → stop, report the task
as too easy, do not read the verdict. A short POINT/AND run must show
clear sub-ceiling comp-OOD first. (Same discipline that caught R0.)

## Verdict rule — LOCKED before any run

Let Δ_T = mean over seeds of (INTERSECTION comp-OOD − POINT comp-OOD)
on task T, judged against across-seed pooled SD and net of the same
contrast on the seen-entity control.

1. **Decisive NEGATIVE — every escape closed.** Δ_AND ≤ 0 (≤ noise).
   The intersection bottleneck fails to beat a param-matched point
   bottleneck *even on the task whose ground truth is literally an
   intersection, on the critical path, with no bypass.* Nothing further
   to test cheaply; the residual loophole is closed; the paper's
   limitations section states this with evidence.
2. **CIRCULAR positive — uninformative, escape still closed.** Δ_AND
   clearly > 0 **but** Δ_NONINT ≤ 0 (≤ noise or negative). The AND win
   is the tool matching the generator; intersection is not a general
   compositional primitive. Reported as such; does **not** license
   further investment.
3. **GENERAL positive — the first thesis-supporting signal.** Δ_AND
   clearly > 0 **and** Δ_NONINT also clearly > 0, both large vs.
   across-seed SD and not mirrored in the seen-entity control. The
   intersection bottleneck confers a compositional advantage even when
   the target is not an intersection. This is a *de-risk that licenses
   exactly one* pre-registered follow-up (a real PRECOMMIT, non-toy,
   non-intersective targets, ideally non-synthetic) — it is **not**
   itself a validation of MANDORLA and must be described that way
   everywhere.

"Clearly > 0" = point estimate positive and ≥ ~1 across-seed pooled SD,
with the seen-control contrast not explaining it. The pilot is
exploratory; this is a judgement framed by a pre-fixed rule, not a
p-value. The rule's job is to remove post-hoc freedom, not to imply
rigor this stage doesn't have.

## Priors (recorded before running, for honesty)

Three projections negative, two mechanistically. Prior on outcome 3
(general positive): low — ~15–25%. Most of this pilot's expected value
is in outcome 1 (decisive negative closes the last cheap escape). A
positive's value is entirely contingent on it being outcome 3, not
outcome 2.

## Explicit non-goals

Not a fully intersection-native architecture. Not real data. Not COGS/
SCAN. Not a publishable validation. Not >few-M params. The encoder is
transformer-based for *both* arms by design — architecture is held
constant; only the composition operator varies.

## Outcome — 2026-05-15: VERDICT 1, DECISIVE NEGATIVE

Full 2×2×3 run (12 cells, 5000 steps each, MPS). Gate passed before the
run (POINT/AND plateaus at comp-OOD ≈0.59, not ceiling). Capacity
audited: intersection bottleneck = 49,600 params vs point = 51,424
(int/point = 0.965 — matched; not capacity-confounded in either
direction).

comp-OOD F1@G (held-out entities), mean ± std over seeds {1337,1338,1339}:

| task | POINT | INTERSECTION | Δ (int − pt) | pooled SD | net of seen-ctrl |
|---|---|---|---|---|---|
| AND (intersective) | 0.5866 ± 0.0155 | 0.5733 ± 0.0081 | **−0.0133** | 0.0124 | −0.0050 |
| NONINT (control) | 0.5397 ± 0.0108 | 0.5251 ± 0.0107 | −0.0145 | 0.0108 | +0.0016 |

**Δ_AND = −0.013 ≤ 0.** Per the rule locked above (before any number
existed): **Outcome 1 — DECISIVE NEGATIVE.** The intersection
bottleneck does not beat a parameter-matched point bottleneck *even on
the task whose ground truth is literally a set intersection, with the
intersection operator on the critical path and no associative bypass*
(the construction that was specifically built to defeat the
curriculum pilot's "it was only an auxiliary loss" loophole). The
effect is negative on **both** tasks and **all three** seeds; INTERSECTION
is, if anything, marginally worse everywhere.

This was the last cheap, decisive test available. It closes the final
"but intersection was never the actual computation" escape, with
capacity controlled and the bypass removed. There is no remaining
cheap experiment that could rescue the retrieval/construction forms of
the thesis.

**Scope, stated precisely (unchanged discipline):** this does *not*
formally falsify the MANDORLA thesis. It does not test a fully
intersection-native architecture (held constant by design — that is a
multi-year program, not a pilot, and the pattern of needing an
ever-purer, ever-costlier setting after each negative is itself the
signature this result formalizes). Thesis 1 (Hex-Vote) remains
untested. The paper's mathematical arguments are untouched by any
empirical result. What is established: **every projection of the thesis
that was cheap enough to screen has now returned negative**, this one
with the strongest controls of the series.
