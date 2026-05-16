"""Aggregate the R1 pilot signal across seeds and conditions.

Reads results/final_{condition}_seed{S}.json for the three conditions
(baseline, generic_aux, curriculum) over all seeds present and lays out:

  - comp-OOD F1@G  mean ± std across seeds, per condition
  - seen-entity-control F1@G mean ± std, per condition
  - the two gaps that matter for Q2:
      curriculum − baseline      (raw effect)
      curriculum − generic_aux   (intersection-specific effect)
    each on comp-OOD, net of the same gap on the seen-entity control.

Per PILOT.md the pilot is exploratory/pre-PRECOMMIT, so this prints the
numbers for a judgement, it does not hard-pass/fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
RES = EXP_ROOT / "results"
CONDS = ["baseline", "generic_aux", "curriculum"]


def _collect(cond: str) -> dict:
    ood, seen, seeds = [], [], []
    for p in sorted(RES.glob(f"final_{cond}_seed*.json")):
        m = re.search(r"seed(\d+)", p.name)
        seeds.append(int(m.group(1)) if m else -1)
        d = json.loads(p.read_text())
        ood.append(d["comp_ood"]["f1_at_g"])
        seen.append(d["seen_entity_control"]["f1_at_g"])
    return {"seeds": seeds, "ood": np.array(ood), "seen": np.array(seen)}


def _f(a: np.ndarray) -> str:
    return "—" if a.size == 0 else f"{a.mean():.4f} ± {a.std():.4f} (n={a.size})"


def main() -> int:
    data = {c: _collect(c) for c in CONDS}
    if any(data[c]["ood"].size == 0 for c in CONDS):
        missing = [c for c in CONDS if data[c]["ood"].size == 0]
        print(f"Missing final_*.json for: {missing}. Run scripts/02_train.py "
              f"for all three conditions / seeds first.")
        return 1

    print("=== R1 pilot signal (comp-OOD = held-out-entity 2-hop) ===\n")
    for c in CONDS:
        print(f"  {c:12s} comp-OOD F1@G : {_f(data[c]['ood'])}")
    print()
    for c in CONDS:
        print(f"  {c:12s} seen-ctrl F1@G: {_f(data[c]['seen'])}")
    print()

    b, g, cu = data["baseline"], data["generic_aux"], data["curriculum"]
    raw = cu["ood"].mean() - b["ood"].mean()
    raw_seen = cu["seen"].mean() - b["seen"].mean()
    spec = cu["ood"].mean() - g["ood"].mean()
    spec_seen = cu["seen"].mean() - g["seen"].mean()
    pooled = float(np.sqrt((b["ood"].var() + g["ood"].var() + cu["ood"].var()) / 3.0))

    print("Gaps on comp-OOD (net of the same gap on the seen-entity control):\n")
    print(f"  curriculum − baseline    : {raw:+.4f}   "
          f"(seen-ctrl gap {raw_seen:+.4f}  → net {raw - raw_seen:+.4f})")
    print(f"  curriculum − generic_aux : {spec:+.4f}   "
          f"(seen-ctrl gap {spec_seen:+.4f}  → net {spec - spec_seen:+.4f})")
    print(f"  across-seed pooled SD    : {pooled:.4f}")
    if pooled > 0:
        print(f"  intersection-specific net effect / pooled SD : "
              f"{(spec - spec_seen) / pooled:+.2f}")

    print("\nJudgement (PILOT.md, exploratory — not a hard bar):")
    print("  • Q2 leans PASS only if curriculum − generic_aux on comp-OOD is")
    print("    clearly positive, large vs across-seed SD, and not mirrored by")
    print("    the seen-entity control. Beating *baseline* alone is NOT enough")
    print("    (generic auxiliary signal is known to help — the intersection")
    print("    has to beat the capacity-matched generic control).")
    print("  • Q1 (stability) read from results/train_curriculum_*.jsonl.")

    (RES / "pilot_summary.json").write_text(json.dumps({
        "comp_ood_mean": {c: float(data[c]["ood"].mean()) for c in CONDS},
        "seen_ctrl_mean": {c: float(data[c]["seen"].mean()) for c in CONDS},
        "curr_minus_baseline_net": float(raw - raw_seen),
        "curr_minus_generic_net": float(spec - spec_seen),
        "pooled_sd": pooled,
    }, indent=2))
    print(f"\nWrote {(RES / 'pilot_summary.json').relative_to(EXP_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
