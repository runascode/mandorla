"""Aggregate the pilot signal across seeds.

Reads results/final_{condition}_seed{S}.json for both conditions over
all seeds present and reports, per condition:

  - mean ± std held-out F1@G across seeds
  - mean ± std seen-pair-control F1@G across seeds

and the curriculum−baseline gap on held-out pairs with across-seed
spread. Q2 of the pilot is a *judgement* on whether that gap is large
relative to seed noise AND not explained by a matching seen-control
gap; this script lays the numbers out for that judgement (it does not
itself pass/fail — PILOT.md is explicit that the pilot is exploratory
and pre-PRECOMMIT, so no numeric bar is hard-coded).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
RES = EXP_ROOT / "results"


def _collect(condition: str) -> dict:
    held, seen = [], []
    seeds = []
    for p in sorted(RES.glob(f"final_{condition}_seed*.json")):
        m = re.search(r"seed(\d+)", p.name)
        seeds.append(int(m.group(1)) if m else -1)
        d = json.loads(p.read_text())
        held.append(d["heldout"]["f1_at_g"])
        seen.append(d["seen_control"]["f1_at_g"])
    return {"seeds": seeds, "heldout": np.array(held), "seen": np.array(seen)}


def _fmt(a: np.ndarray) -> str:
    if a.size == 0:
        return "—"
    return f"{a.mean():.4f} ± {a.std():.4f} (n={a.size})"


def main() -> int:
    base = _collect("baseline")
    curr = _collect("curriculum")
    if base["heldout"].size == 0 or curr["heldout"].size == 0:
        print("Need at least one final_*.json for each condition. "
              "Run scripts/02_train.py for both conditions / seeds first.")
        return 1

    print("=== Pilot signal (held-out compositional transfer) ===\n")
    print(f"  baseline   held-out F1@G : {_fmt(base['heldout'])}")
    print(f"  curriculum held-out F1@G : {_fmt(curr['heldout'])}")
    print(f"  baseline   seen-ctrl F1@G: {_fmt(base['seen'])}")
    print(f"  curriculum seen-ctrl F1@G: {_fmt(curr['seen'])}\n")

    held_gap = curr["heldout"].mean() - base["heldout"].mean()
    seen_gap = curr["seen"].mean() - base["seen"].mean()
    pooled_sd = float(np.sqrt(
        (base["heldout"].var() + curr["heldout"].var()) / 2.0
    )) or float("nan")
    print(f"  held-out gap (curr − base) : {held_gap:+.4f}")
    print(f"  seen-ctrl gap (curr − base): {seen_gap:+.4f}  "
          f"(control: should be ≪ held-out gap for real transfer)")
    print(f"  across-seed pooled SD      : {pooled_sd:.4f}")
    if pooled_sd and not np.isnan(pooled_sd) and pooled_sd > 0:
        print(f"  held-out gap / pooled SD   : {held_gap / pooled_sd:+.2f}")

    print("\nJudgement (per PILOT.md, exploratory — not a hard bar):")
    print("  • Q2 leans PASS if held-out gap is clearly positive, large")
    print("    vs across-seed SD, and substantially exceeds the seen-ctrl gap.")
    print("  • Q1 (stability) is read from results/train_*.jsonl separately.")
    summary = {
        "baseline_heldout_mean": float(base["heldout"].mean()),
        "curriculum_heldout_mean": float(curr["heldout"].mean()),
        "held_gap": float(held_gap),
        "seen_gap": float(seen_gap),
        "pooled_sd": pooled_sd,
        "seeds_baseline": base["seeds"],
        "seeds_curriculum": curr["seeds"],
    }
    (RES / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {(RES / 'pilot_summary.json').relative_to(EXP_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
