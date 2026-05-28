"""Score the baseline and Vesica-RAG runs; write RESULTS.md with the
go/no-go decision.

Reads results/raw/baseline.jsonl and results/raw/vesica.jsonl, matches by
question id, computes per-question F1/EM for both conditions and
vesica-coverage for Vesica-RAG, bootstraps 95% CIs (overall and stratified
by HotpotQA `level` and `type`), applies the PRECOMMIT.md decision rule,
and writes results/RESULTS.md plus results/scores.json (machine-readable).

Decision rule (PRECOMMIT.md §"Decision rule", with Amendment 1 and 2 not
affecting it):

  GO       : F1 lift ≥ +2 absolute  AND  vesica-coverage ≥ 0.05
  WEAK GO  : F1 lift ≥ +1 absolute  AND  vesica-coverage ≥ 0.03   (and not GO)
  NO-GO    : otherwise

The contriever baseline has zero vesica-coverage by construction (it never
forms Vesicas), so the diagnostic "uplift" equals the raw Vesica-RAG
coverage. If the result falls in a region the table doesn't cleanly cover
(e.g. strong F1 lift but weak coverage, or vice versa), RESULTS.md flags it
for human adjudication rather than silently bucketing it.

Requires scripts/07 and scripts/08 to have run.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import REPO_ROOT
from src.eval import (
    BootstrapResult,
    bootstrap_mean,
    bootstrap_paired_difference,
    exact_match,
    f1_score,
    vesica_covered,
)

BASELINE_PATH = REPO_ROOT / "results" / "raw" / "baseline.jsonl"
VESICA_PATH = REPO_ROOT / "results" / "raw" / "vesica.jsonl"
RESULTS_MD = REPO_ROOT / "results" / "RESULTS.md"
SCORES_JSON = REPO_ROOT / "results" / "scores.json"

# Decision-rule thresholds (PRECOMMIT.md §"Decision rule").
GO_F1_LIFT = 2.0          # absolute F1 points (×100 scale)
GO_COVERAGE = 0.05
WEAK_F1_LIFT = 1.0
WEAK_COVERAGE = 0.03

N_BOOTSTRAP = 10_000
SEED = 1337


def _load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the corresponding run script first.")
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def _fmt_ci(b: BootstrapResult, scale: float = 1.0, pct: bool = False) -> str:
    p = b.point_estimate * scale
    lo = b.ci_low * scale
    hi = b.ci_high * scale
    suffix = "%" if pct else ""
    return f"{p:.2f}{suffix} (95% CI {lo:.2f}–{hi:.2f}{suffix})"


def _decide(f1_lift_abs: float, coverage: float) -> tuple[str, str]:
    """Return (decision, note). decision ∈ {GO, WEAK GO, NO-GO}.

    The PRECOMMIT.md table requires *both* conditions for GO and *both* for
    WEAK GO; everything else is NO-GO. If the result is "split" (one metric
    in a higher tier, the other in a lower tier), we still bucket it per the
    table but attach a note so a human sees the split.
    """
    f1_go = f1_lift_abs >= GO_F1_LIFT
    cov_go = coverage >= GO_COVERAGE
    f1_weak = f1_lift_abs >= WEAK_F1_LIFT
    cov_weak = coverage >= WEAK_COVERAGE

    note = ""
    if f1_go and cov_go:
        decision = "GO"
    elif f1_weak and cov_weak:
        decision = "WEAK GO"
    else:
        decision = "NO-GO"

    # Flag splits for human attention.
    if (f1_go and not cov_weak) or (cov_go and not f1_weak):
        note = (
            "SPLIT RESULT — one metric clears the GO bar while the other is "
            "below the WEAK GO bar. The table buckets this as the lower tier; "
            "a human should adjudicate whether the strong metric warrants "
            "deviating from the literal rule. See PRECOMMIT.md decision rule."
        )
    elif decision == "NO-GO" and (f1_weak or cov_weak):
        note = (
            "One metric reached the WEAK GO bar but the other did not; per "
            "the table, both are required, so this is NO-GO. Recorded for "
            "transparency."
        )
    return decision, note


def _strata(records: list[dict]) -> dict[str, list[int]]:
    """Map stratum name → list of indices into the records list."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by["all"].append(i)
        by[f"level={r['level']}"].append(i)
        by[f"type={r['type']}"].append(i)
    return by


def main() -> int:
    print("Loading run outputs...", flush=True)
    base = _load_jsonl(BASELINE_PATH)
    ves = _load_jsonl(VESICA_PATH)

    common_ids = sorted(set(base) & set(ves))
    if not common_ids:
        print("ERROR: no question ids in common between the two runs.")
        return 1
    n_base_only = len(set(base) - set(ves))
    n_ves_only = len(set(ves) - set(base))
    if n_base_only or n_ves_only:
        print(f"WARNING: {n_base_only} ids only in baseline, {n_ves_only} only "
              f"in vesica. Scoring the {len(common_ids)} common ids.")

    # Per-question metrics, aligned by id.
    records: list[dict] = []
    base_f1, ves_f1, base_em, ves_em, ves_cov = [], [], [], [], []
    for qid in common_ids:
        b, v = base[qid], ves[qid]
        bf1 = f1_score(b["prediction"], b["gold_answer"])
        vf1 = f1_score(v["prediction"], v["gold_answer"])
        bem = exact_match(b["prediction"], b["gold_answer"])
        vem = exact_match(v["prediction"], v["gold_answer"])
        gold_chunks = set(v.get("gold_chunk_ids", []))
        cand_parents = [tuple(p) for p in v.get("candidate_vesica_parents", [])]
        cov = 1.0 if vesica_covered(gold_chunks, cand_parents) else 0.0
        base_f1.append(bf1); ves_f1.append(vf1)
        base_em.append(bem); ves_em.append(vem); ves_cov.append(cov)
        records.append({"id": qid, "level": v["level"], "type": v["type"]})

    base_f1 = np.array(base_f1); ves_f1 = np.array(ves_f1)
    base_em = np.array(base_em); ves_em = np.array(ves_em)
    ves_cov = np.array(ves_cov)
    n = len(common_ids)

    # Overall + stratified bootstrap.
    strata = _strata(records)
    summary: dict[str, dict] = {}
    for name, idxs in strata.items():
        idxs_arr = np.array(idxs, dtype=int)
        summary[name] = {
            "n": len(idxs),
            "baseline_f1": bootstrap_mean(base_f1[idxs_arr], N_BOOTSTRAP, SEED),
            "vesica_f1": bootstrap_mean(ves_f1[idxs_arr], N_BOOTSTRAP, SEED + 1),
            "f1_lift": bootstrap_paired_difference(base_f1[idxs_arr], ves_f1[idxs_arr], N_BOOTSTRAP, SEED + 2),
            "baseline_em": bootstrap_mean(base_em[idxs_arr], N_BOOTSTRAP, SEED + 3),
            "vesica_em": bootstrap_mean(ves_em[idxs_arr], N_BOOTSTRAP, SEED + 4),
            "em_lift": bootstrap_paired_difference(base_em[idxs_arr], ves_em[idxs_arr], N_BOOTSTRAP, SEED + 5),
            "vesica_coverage": bootstrap_mean(ves_cov[idxs_arr], N_BOOTSTRAP, SEED + 6),
        }

    overall = summary["all"]
    # F1 lift in "absolute F1 points" — metrics here are 0–1, PRECOMMIT
    # speaks in "F1 points" (0–100), so multiply by 100 for the decision.
    f1_lift_abs = overall["f1_lift"].point_estimate * 100.0
    coverage = overall["vesica_coverage"].point_estimate
    decision, note = _decide(f1_lift_abs, coverage)

    # --- write RESULTS.md ---
    lines: list[str] = []
    lines.append("# Experiment 1 — Vesica-RAG Screening Slice: Results\n")
    lines.append(f"**Date:** {date.today().isoformat()}  ")
    lines.append("**Design:** see [`PRECOMMIT.md`](./../PRECOMMIT.md) (and its amendments).  ")
    lines.append(f"**Questions scored:** {n} (HotpotQA `distractor` validation).  ")
    lines.append(f"**Bootstrap:** {N_BOOTSTRAP} resamples, seed={SEED}.\n")
    lines.append("> ⚠️ Fill in the **commit SHA**, **encoding/index meta**, and **α / τ_v** "
                 "values below before treating this file as the final record. They are "
                 "available from `git rev-parse HEAD`, `index/contriever_meta.json`, "
                 "`index/box.meta.json`, `index/tau_v.json`.\n")

    lines.append("## Decision\n")
    lines.append(f"### **{decision}**\n")
    lines.append(f"- F1 lift over contriever: **{_fmt_ci(overall['f1_lift'], scale=100.0)} F1 points**")
    lines.append(f"  - (GO bar: ≥ +{GO_F1_LIFT}; WEAK GO bar: ≥ +{WEAK_F1_LIFT})")
    lines.append(f"- Vesica-coverage: **{_fmt_ci(overall['vesica_coverage'], scale=100.0, pct=True)}**")
    lines.append(f"  - (GO bar: ≥ {GO_COVERAGE*100:.0f}%; WEAK GO bar: ≥ {WEAK_COVERAGE*100:.0f}%)")
    lines.append(f"  - the contriever baseline forms no Vesicas, so this coverage *is* the uplift")
    if note:
        lines.append(f"\n> **Note:** {note}\n")
    lines.append("\nPer PRECOMMIT.md, this decision is binding: a NO-GO result is published "
                 "as such, on `runascode.com/results/vesica-rag-slice`, without revision "
                 "after seeing the numbers.\n")

    lines.append("## Overall metrics\n")
    lines.append("| Metric | Baseline (contriever top-25) | Vesica-RAG | Lift |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Answer F1 | {_fmt_ci(overall['baseline_f1'], 100.0)} | {_fmt_ci(overall['vesica_f1'], 100.0)} | {_fmt_ci(overall['f1_lift'], 100.0)} |")
    lines.append(f"| Answer EM | {_fmt_ci(overall['baseline_em'], 100.0)} | {_fmt_ci(overall['vesica_em'], 100.0)} | {_fmt_ci(overall['em_lift'], 100.0)} |")
    lines.append(f"| Vesica-coverage | 0.00% (no Vesicas) | {_fmt_ci(overall['vesica_coverage'], 100.0, pct=True)} | — |")
    lines.append("")

    lines.append("## Stratified (by HotpotQA `level` and `type`)\n")
    lines.append("| Stratum | n | Baseline F1 | Vesica-RAG F1 | F1 lift | Vesica-coverage |")
    lines.append("|---|---:|---|---|---|---|")
    order = ["level=easy", "level=medium", "level=hard", "type=bridge", "type=comparison"]
    for name in order:
        if name not in summary:
            continue
        s = summary[name]
        lines.append(
            f"| {name} | {s['n']} | {_fmt_ci(s['baseline_f1'], 100.0)} | "
            f"{_fmt_ci(s['vesica_f1'], 100.0)} | {_fmt_ci(s['f1_lift'], 100.0)} | "
            f"{_fmt_ci(s['vesica_coverage'], 100.0, pct=True)} |"
        )
    lines.append("")

    lines.append("## Provenance (fill in)\n")
    lines.append("- Commit SHA: `<git rev-parse HEAD>`")
    lines.append("- Generator: `llama3.1:8b-instruct-q5_K_M` via Ollama; temperature=0, seed=1337, num_ctx=8192")
    lines.append("- Encoder: `facebook/contriever-msmarco`, mean-pool, max_len=128 (see `index/contriever_meta.json`)")
    lines.append("- Box index: d=64, random-projection seed=1337, α=`<index/box.meta.json>`")
    lines.append("- τ_v (log-volume): `<index/tau_v.json>`")
    lines.append("- Master seed: 1337")
    lines.append("")
    lines.append("---\n*Generated by `scripts/09_score.py`. Re-run to regenerate.*")

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS_MD}")

    # machine-readable
    def _br_dict(b: BootstrapResult) -> dict:
        return {"point": b.point_estimate, "ci_low": b.ci_low, "ci_high": b.ci_high, "n_resamples": b.n_resamples}

    scores = {
        "date": date.today().isoformat(),
        "n_questions": n,
        "decision": decision,
        "decision_note": note,
        "f1_lift_abs_points": f1_lift_abs,
        "vesica_coverage": coverage,
        "strata": {
            name: {
                "n": s["n"],
                "baseline_f1": _br_dict(s["baseline_f1"]),
                "vesica_f1": _br_dict(s["vesica_f1"]),
                "f1_lift": _br_dict(s["f1_lift"]),
                "baseline_em": _br_dict(s["baseline_em"]),
                "vesica_em": _br_dict(s["vesica_em"]),
                "em_lift": _br_dict(s["em_lift"]),
                "vesica_coverage": _br_dict(s["vesica_coverage"]),
            }
            for name, s in summary.items()
        },
    }
    SCORES_JSON.write_text(json.dumps(scores, indent=2))
    print(f"Wrote {SCORES_JSON}")

    print(f"\n=== DECISION: {decision} ===")
    print(f"F1 lift: {f1_lift_abs:+.2f} points | vesica-coverage: {coverage*100:.2f}%")
    if note:
        print(f"Note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
