"""Post-hoc diagnostic on the NO-GO slice.

Decides between three hypotheses for why the +4.36% vesica-coverage signal
didn't convert to an F1 lift:

  H1 (coverage-is-the-bottleneck): On the subset of questions where the gold
     pair WAS surfaced as a candidate Vesica, Vesica-RAG outperforms the
     baseline. The limit is just that coverage is too low (4.36%).

  H2 (vesicas-don't-help-the-LLM): Even on coverage-hit questions, F1 is
     flat. The LLM doesn't differentially use Vesica-augmented context.

  H3 (vesicas-displace-good-context): The context cap (~25 chunks) means
     adding Vesica chunks removes useful contriever hits, hurting bridge
     questions where Vesica-coverage is low. Test: is Vesica-RAG specifically
     worse than baseline on questions where the gold pair was NOT in a
     Vesica? Is gold-chunk recall higher for the baseline than for
     Vesica-RAG?

Runs only on the existing results/raw/{baseline,vesica}.jsonl from the
slice run. No new generation. No new retrieval. Output: a Markdown report
appended to RESULTS.md as a §Diagnostic section, plus a JSON dump.

This is exploratory analysis labeled as such; it does not change the
binding NO-GO decision recorded in PRECOMMIT.md.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import (  # noqa: E402
    BootstrapResult,
    bootstrap_mean,
    bootstrap_paired_difference,
    exact_match,
    f1_score,
    vesica_covered,
)

EXP_ROOT = Path(__file__).resolve().parents[1]
RAW = EXP_ROOT / "results" / "raw"
OUT_REPORT = EXP_ROOT / "results" / "DIAGNOSTIC.md"
OUT_JSON = EXP_ROOT / "results" / "diagnostic.json"
N_BOOTSTRAP = 10000
SEED = 1337


def _load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def _ci(br: BootstrapResult, scale: float = 1.0, pct: bool = False) -> str:
    if pct:
        return f"{br.point_estimate * scale:.2f}% (CI {br.ci_low * scale:.2f}–{br.ci_high * scale:.2f}%)"
    return f"{br.point_estimate * scale:.2f} (CI {br.ci_low * scale:.2f}, {br.ci_high * scale:.2f})"


def main() -> int:
    base = _load_jsonl(RAW / "baseline.jsonl")
    ves = _load_jsonl(RAW / "vesica.jsonl")
    common = sorted(set(base) & set(ves))
    if not common:
        print("ERROR: no overlapping ids", file=sys.stderr)
        return 1
    print(f"Loaded {len(common)} aligned questions.")

    bf1 = np.array([f1_score(base[q]["prediction"], base[q]["gold_answer"]) for q in common])
    vf1 = np.array([f1_score(ves[q]["prediction"], ves[q]["gold_answer"]) for q in common])
    bem = np.array([exact_match(base[q]["prediction"], base[q]["gold_answer"]) for q in common])
    vem = np.array([exact_match(ves[q]["prediction"], ves[q]["gold_answer"]) for q in common])

    gold_chunks = [set(ves[q].get("gold_chunk_ids", [])) for q in common]
    cand_parents = [[tuple(p) for p in ves[q].get("candidate_vesica_parents", [])] for q in common]
    covered = np.array([vesica_covered(g, c) for g, c in zip(gold_chunks, cand_parents)])
    types = np.array([ves[q]["type"] for q in common])

    base_ctx = [set(base[q].get("retrieved_chunk_ids", [])) for q in common]
    ves_ctx = [set(ves[q].get("retrieved_chunk_ids", [])) for q in common]
    overlap_sizes = np.array([len(b & v) for b, v in zip(base_ctx, ves_ctx)])
    base_only = np.array([len(b - v) for b, v in zip(base_ctx, ves_ctx)])
    base_gold_recall = np.array(
        [int(len(g & b) > 0) for g, b in zip(gold_chunks, base_ctx)], dtype=float
    )
    ves_gold_recall = np.array(
        [int(len(g & v) > 0) for g, v in zip(gold_chunks, ves_ctx)], dtype=float
    )
    base_gold_recall_pair = np.array(
        [int(len(g & b) == len(g) and len(g) > 0) for g, b in zip(gold_chunks, base_ctx)],
        dtype=float,
    )
    ves_gold_recall_pair = np.array(
        [int(len(g & v) == len(g) and len(g) > 0) for g, v in zip(gold_chunks, ves_ctx)],
        dtype=float,
    )

    def slice_stats(name: str, mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if n < 5:
            return {"name": name, "n": n, "skipped": True}
        idx = np.where(mask)[0]
        return {
            "name": name,
            "n": n,
            "baseline_f1": bootstrap_mean(bf1[idx], N_BOOTSTRAP, SEED).__dict__,
            "vesica_f1": bootstrap_mean(vf1[idx], N_BOOTSTRAP, SEED + 1).__dict__,
            "f1_lift": bootstrap_paired_difference(
                bf1[idx], vf1[idx], N_BOOTSTRAP, SEED + 2
            ).__dict__,
            "baseline_em": bootstrap_mean(bem[idx], N_BOOTSTRAP, SEED + 3).__dict__,
            "vesica_em": bootstrap_mean(vem[idx], N_BOOTSTRAP, SEED + 4).__dict__,
        }

    slices = {
        "all": slice_stats("all", np.ones_like(covered, dtype=bool)),
        "coverage_hit": slice_stats("vesica_covered=True", covered),
        "coverage_miss": slice_stats("vesica_covered=False", ~covered),
        "bridge_coverage_hit": slice_stats("bridge & covered", covered & (types == "bridge")),
        "bridge_coverage_miss": slice_stats("bridge & not covered", (~covered) & (types == "bridge")),
        "comparison_coverage_hit": slice_stats(
            "comparison & covered", covered & (types == "comparison")
        ),
        "comparison_coverage_miss": slice_stats(
            "comparison & not covered", (~covered) & (types == "comparison")
        ),
    }

    overlap_summary = {
        "mean_context_overlap": float(overlap_sizes.mean()),
        "median_context_overlap": float(np.median(overlap_sizes)),
        "p10_context_overlap": float(np.percentile(overlap_sizes, 10)),
        "p90_context_overlap": float(np.percentile(overlap_sizes, 90)),
        "mean_baseline_only_chunks": float(base_only.mean()),
        "baseline_size_mean": float(np.mean([len(b) for b in base_ctx])),
        "vesica_size_mean": float(np.mean([len(v) for v in ves_ctx])),
    }
    gold_recall_summary = {
        "baseline_any_gold_in_context": float(base_gold_recall.mean()),
        "vesica_any_gold_in_context": float(ves_gold_recall.mean()),
        "baseline_pair_in_context": float(base_gold_recall_pair.mean()),
        "vesica_pair_in_context": float(ves_gold_recall_pair.mean()),
    }

    type_counts = dict(Counter(types.tolist()))

    payload = {
        "n": len(common),
        "slices": slices,
        "context_overlap": overlap_summary,
        "gold_recall": gold_recall_summary,
        "type_counts": type_counts,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=lambda o: o.__dict__))
    print(f"Wrote {OUT_JSON.relative_to(EXP_ROOT)}")

    def br_from_dict(d: dict) -> BootstrapResult:
        return BootstrapResult(
            point_estimate=d["point_estimate"],
            ci_low=d["ci_low"],
            ci_high=d["ci_high"],
            n_resamples=d.get("n_resamples", N_BOOTSTRAP),
        )

    lines: list[str] = []
    lines.append("# Slice Diagnostic (post-hoc, exploratory)\n")
    lines.append("**This is exploratory analysis on the existing slice JSONLs. It does not change the binding NO-GO decision in [`PRECOMMIT.md`](../PRECOMMIT.md). It exists to inform the next pre-commit.**\n")
    lines.append("## H1 — is coverage the bottleneck?\n")
    lines.append("If the gold pair *is* surfaced as a candidate Vesica, does Vesica-RAG outperform the contriever baseline?\n")
    lines.append("| Slice | n | Baseline F1 | Vesica-RAG F1 | F1 lift |")
    lines.append("|---|---:|---|---|---|")
    for key in ["coverage_hit", "coverage_miss", "all"]:
        s = slices[key]
        if s.get("skipped"):
            continue
        lines.append(
            "| {name} | {n} | {bf} | {vf} | **{lf}** |".format(
                name=s["name"],
                n=s["n"],
                bf=_ci(br_from_dict(s["baseline_f1"])),
                vf=_ci(br_from_dict(s["vesica_f1"])),
                lf=_ci(br_from_dict(s["f1_lift"])),
            )
        )
    lines.append("\n## By question type × coverage\n")
    lines.append("| Slice | n | Baseline F1 | Vesica-RAG F1 | F1 lift |")
    lines.append("|---|---:|---|---|---|")
    for key in [
        "bridge_coverage_hit",
        "bridge_coverage_miss",
        "comparison_coverage_hit",
        "comparison_coverage_miss",
    ]:
        s = slices[key]
        if s.get("skipped"):
            lines.append(f"| {s['name']} | {s['n']} | — | — | (n<5, skipped) |")
            continue
        lines.append(
            "| {name} | {n} | {bf} | {vf} | **{lf}** |".format(
                name=s["name"],
                n=s["n"],
                bf=_ci(br_from_dict(s["baseline_f1"])),
                vf=_ci(br_from_dict(s["vesica_f1"])),
                lf=_ci(br_from_dict(s["f1_lift"])),
            )
        )
    lines.append("\n## H3 — does Vesica context displace useful baseline context?\n")
    lines.append(f"- Mean baseline context size: **{overlap_summary['baseline_size_mean']:.1f}** chunks")
    lines.append(f"- Mean Vesica-RAG context size: **{overlap_summary['vesica_size_mean']:.1f}** chunks")
    lines.append(f"- Mean overlap between the two contexts: **{overlap_summary['mean_context_overlap']:.1f}** chunks (median {overlap_summary['median_context_overlap']:.0f}, p10–p90 {overlap_summary['p10_context_overlap']:.0f}–{overlap_summary['p90_context_overlap']:.0f})")
    lines.append(f"- Mean **baseline-only** chunks (dropped by Vesica-RAG): **{overlap_summary['mean_baseline_only_chunks']:.1f}**")
    lines.append("")
    lines.append("Gold-chunk recall in the final LLM context:\n")
    lines.append("| Recall metric | Baseline | Vesica-RAG |")
    lines.append("|---|---|---|")
    lines.append(f"| Any gold chunk present | {gold_recall_summary['baseline_any_gold_in_context']*100:.2f}% | {gold_recall_summary['vesica_any_gold_in_context']*100:.2f}% |")
    lines.append(f"| Full gold pair present | {gold_recall_summary['baseline_pair_in_context']*100:.2f}% | {gold_recall_summary['vesica_pair_in_context']*100:.2f}% |")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("(filled in by the reader of the numbers above — kept out of code by design)\n")
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_REPORT.relative_to(EXP_ROOT)}")

    print()
    print("=== Headline numbers ===")
    for key in ["coverage_hit", "coverage_miss"]:
        s = slices[key]
        if s.get("skipped"):
            continue
        lift = br_from_dict(s["f1_lift"])
        bf = br_from_dict(s["baseline_f1"])
        vf = br_from_dict(s["vesica_f1"])
        print(f"  {s['name']} (n={s['n']}): baseline {bf.point_estimate:.2f} | vesica {vf.point_estimate:.2f} | lift {lift.point_estimate:+.2f} (CI {lift.ci_low:+.2f}, {lift.ci_high:+.2f})")
    print(f"  Gold-pair-in-context: baseline {gold_recall_summary['baseline_pair_in_context']*100:.2f}% | vesica {gold_recall_summary['vesica_pair_in_context']*100:.2f}%")
    print(f"  Any-gold-chunk-in-context: baseline {gold_recall_summary['baseline_any_gold_in_context']*100:.2f}% | vesica {gold_recall_summary['vesica_any_gold_in_context']*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
