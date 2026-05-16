"""Aggregate the 2×2×seeds and apply the LOCKED PILOT2.md verdict rule.

Δ_T = mean_seeds( INTERSECTION comp-OOD − POINT comp-OOD ) on task T,
judged vs across-seed pooled SD and net of the same contrast on the
seen-entity control. Rule (verbatim from PILOT2.md):

  1 NEGATIVE  : Δ_AND ≤ 0 (≤ noise)                  → escape closed
  2 CIRCULAR  : Δ_AND > 0  but Δ_NONINT ≤ 0          → uninformative
  3 GENERAL   : Δ_AND > 0  and Δ_NONINT > 0          → first real signal
"clearly > 0" = positive and ≥ ~1 across-seed pooled SD, seen-control
not explaining it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"


def collect(arm: str, task: str):
    ood, seen, seeds = [], [], []
    for p in sorted(RES.glob(f"final_{arm}_{task}_seed*.json")):
        seeds.append(int(re.search(r"seed(\d+)", p.name).group(1)))
        d = json.loads(p.read_text())
        ood.append(d["comp_ood"]["f1_at_g"])
        seen.append(d["seen_entity_control"]["f1_at_g"])
    return np.array(ood), np.array(seen), seeds


def main() -> int:
    cells = {(a, t): collect(a, t)
             for a in ("point", "intersection") for t in ("and", "nonint")}
    if any(v[0].size == 0 for v in cells.values()):
        miss = [f"{a}/{t}" for (a, t), v in cells.items() if v[0].size == 0]
        print(f"missing cells: {miss}; run scripts/02_train.py for all 12.")
        return 1

    def delta(task: str):
        io, isn, _ = cells[("intersection", task)]
        po, psn, _ = cells[("point", task)]
        d_ood = io.mean() - po.mean()
        d_seen = isn.mean() - psn.mean()
        pooled_sd = float(np.sqrt((io.var() + po.var()) / 2.0))
        return d_ood, d_seen, pooled_sd

    dA, dA_seen, sdA = delta("and")
    dN, dN_seen, sdN = delta("nonint")

    print("=== Pilot 2 — composition-operator test ===\n")
    for t in ("and", "nonint"):
        io, isn, _ = cells[("intersection", t)]
        po, psn, _ = cells[("point", t)]
        print(f"  task={t}")
        print(f"    POINT        comp-OOD {po.mean():.4f}±{po.std():.4f} "
              f"seen {psn.mean():.4f}")
        print(f"    INTERSECTION comp-OOD {io.mean():.4f}±{io.std():.4f} "
              f"seen {isn.mean():.4f}")
    print()
    print(f"  Δ_AND    (int−pt) comp-OOD = {dA:+.4f}  "
          f"(seen Δ {dA_seen:+.4f}, net {dA-dA_seen:+.4f}, pooled SD {sdA:.4f})")
    print(f"  Δ_NONINT (int−pt) comp-OOD = {dN:+.4f}  "
          f"(seen Δ {dN_seen:+.4f}, net {dN-dN_seen:+.4f}, pooled SD {sdN:.4f})")

    def clearly_pos(d, d_seen, sd):
        return (d > 0) and (sd > 0) and ((d - d_seen) >= sd)

    A_pos = clearly_pos(dA, dA_seen, sdA)
    N_pos = clearly_pos(dN, dN_seen, sdN)
    if not A_pos and dA <= 0:
        verdict = "1 — DECISIVE NEGATIVE (intersection bottleneck does not beat a param-matched point bottleneck even on the intersective task; the last cheap escape is closed)"
    elif A_pos and not N_pos:
        verdict = "2 — CIRCULAR POSITIVE (wins only where the task IS an intersection; the tool matched the generator; not a general primitive; does not license further investment)"
    elif A_pos and N_pos:
        verdict = "3 — GENERAL POSITIVE (intersection bottleneck wins even on a non-intersective target; first thesis-supporting signal; licenses ONE pre-registered follow-up — not a validation)"
    else:
        verdict = "AMBIGUOUS (Δ_AND positive but below the locked 'clearly > 0' bar; treat as NOT outcome 3; report exactly as measured)"

    print(f"\n  capacity: int/point bottleneck params = "
          f"{json.loads(next(RES.glob('final_intersection_and_seed*.json')).read_text())['capacity']['int_over_point']}")
    print(f"\n=== VERDICT (locked rule): {verdict} ===")
    (RES / "verdict.json").write_text(json.dumps({
        "delta_and": dA, "delta_and_seen": dA_seen, "pooled_sd_and": sdA,
        "delta_nonint": dN, "delta_nonint_seen": dN_seen, "pooled_sd_nonint": sdN,
        "verdict": verdict,
    }, indent=2))
    print(f"\nWrote {(RES / 'verdict.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
