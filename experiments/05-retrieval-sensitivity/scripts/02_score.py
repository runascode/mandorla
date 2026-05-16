"""Score the sweep: per-condition F1/EM + pair-in-context, with
bootstrap CIs. Emits RESULTS.md — the paper's central table/curve."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results"
RAW = OUT / "sweep.jsonl"
CONDS = ("oracle", "dense", "gold_removed", "random")
SEED = 1337


def boot(xs: np.ndarray, n=10000, seed=SEED):
    if len(xs) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = np.array([xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(n)])
    return float(xs.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    rows = [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]
    lines = ["# Experiment 05 — Retrieval-sensitivity sweep: Results\n",
             "Reader llama3.1:8b-instruct-q5_K_M, T=0, seed=1337; budget=25; "
             "HotpotQA dev sample (seed 1337). Bootstrap 10k.\n",
             "| Condition | n | F1 | EM | pair-in-context |",
             "|---|--:|---|---|---|"]
    table = {}
    for c in CONDS:
        sub = [r for r in rows if r["condition"] == c]
        if not sub:
            continue
        f1 = np.array([r["f1"] for r in sub])
        em = np.array([r["em"] for r in sub])
        pic = np.array([r["pair_in_context"] for r in sub], float)
        fm, fl, fh = boot(f1)
        em_m, _, _ = boot(em)
        lines.append(f"| {c} | {len(sub)} | {fm*100:.2f} (CI {fl*100:.2f}–{fh*100:.2f}) "
                     f"| {em_m*100:.2f} | {pic.mean()*100:.1f}% |")
        table[c] = fm
    if "oracle" in table and "gold_removed" in table:
        drop = (table["oracle"] - table["gold_removed"]) * 100
        lines.append(
            f"\n**Headline.** F1(oracle) − F1(gold_removed) = **{drop:.2f} points**, "
            f"while pair-in-context goes ~100% → ~0%. "
            + ("Saturation claim holds: the answer metric is near-flat across the "
               "full retrieval-quality range."
               if abs(drop) < 10 else
               "Saturation claim does NOT generalize; standalone paper withdrawn.")
        )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
