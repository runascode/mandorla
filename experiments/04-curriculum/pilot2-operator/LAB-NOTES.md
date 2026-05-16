# Pilot 2 — Lab Notes

Append-only. Pre-PRECOMMIT; binds nothing. Verdict rule was locked in
PILOT2.md before any number existed.

---

## 2026-05-15 — Built, gated, run; VERDICT 1 (decisive negative)

Built to close the residual loophole the curriculum pilot left:
intersection there was an *auxiliary* loss the model could route
around (CLM bypass). Pilot 2 puts the intersection operator **on the
critical path with no bypass** — CLM trains the encoder on ATTR/PARTNER
only; the 2-hop answer is producible *exclusively* through the
bottleneck; the probe reads the answer head, never an LM head. 2×2:
{POINT, INTERSECTION} bottleneck × {AND intersective, NONINT
non-intersective control} × 3 seeds.

Reused `box.py` unchanged (task-agnostic, tested). 20 tests green
including: held-out-entity isolation both tasks, no QUERY in CLM
corpus, NONINT differs from AND on >20% of bits, NONINT label balance
≈ AND, capacity int ≤ point.

Capacity tightened *before* any real run (pre-numbers design fix, not
goalpost-moving): first build had int/point = 0.24 (a negative would
be capacity-starved-confounded). Widened the post-intersection mix to
a 2-layer MLP so int = 49,600 vs point = 51,424 (0.965) — matched both
directions. Gate passed: POINT/AND plateaus comp-OOD ≈0.59, not
ceiling → contrast valid.

### Result

comp-OOD F1@G, mean ± std over 3 seeds:

| task | POINT | INTERSECTION | Δ(int−pt) | pooled SD |
|---|---|---|---|---|
| AND | 0.5866 ± 0.0155 | 0.5733 ± 0.0081 | −0.0133 | 0.0124 |
| NONINT | 0.5397 ± 0.0108 | 0.5251 ± 0.0107 | −0.0145 | 0.0108 |

Δ_AND ≤ 0 → locked Outcome 1, **DECISIVE NEGATIVE**. Negative on both
tasks, all three seeds. Intersection-on-the-critical-path, capacity
matched, no bypass — and it still does not beat point composition even
on the task literally defined as an intersection.

### Lesson #10 — when the purity ratchet bottoms out, that *is* the answer

Each negative in this program was followed by a plausible "it wasn't
tested fairly" rescue: LLM saturation → removed (Exp 02). Borrowed
geometry → trained from scratch (curriculum pilot). Auxiliary/bypass →
put on the critical path, no bypass, capacity-matched (this pilot).
Every time the rescue's condition was met, the result stayed negative.
The one remaining rescue ("the *architecture* must be intersection-
native") is no longer a pilot — it is a multi-year program, and the
*pattern itself* (an ever-purer, ever-costlier precondition demanded
after each disconfirmation, each one failing once met) is exactly how a
false thesis behaves. The disciplined terminus is to name that pattern
and stop, not to pay the next escalation. This pilot is that terminus
for the cheap-to-screen program.

### Decision

No PRECOMMIT for Exp 04 (already decided by the curriculum pilot; this
reinforces it and additionally closes the operator-on-critical-path
loophole). The cheap falsification program is complete. Remaining:
Hex-Vote (Thesis 1, independent infra, untested) and the thesis-
independent assets (the saturation note; the methodology). Project
LAB-NOTES carries the final branching.
