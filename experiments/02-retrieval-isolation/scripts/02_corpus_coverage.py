"""Corpus-coverage analysis for the three datasets.

For each dev question, ask: are *all* of its gold supporting titles
present in the slice corpus (by normalized-title match)? Report the
per-dataset fraction.

The PRECOMMIT decision rule excludes any dataset with coverage <70%
from the GO/WEAK-GO computation (and adjusts the n-of-3 rule).

The slice corpus's title index lives at
`../01-vesica-rag/index/title_to_chunk_id.json`. We load it via the
slice's TitleIndex helper to avoid re-implementing the title→chunk
lookup.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_ROOT))
os.environ.setdefault("HF_HOME", str(EXP_ROOT / "data" / "hf_home"))

from src import _exp01_bridge as bridge  # noqa: E402
from src.datasets import DATASET_LOADERS, normalize_title  # noqa: E402

TitleIndex = bridge.exp01_data.TitleIndex

OUT_PATH = EXP_ROOT / "results" / "corpus_coverage.json"
DATASETS = ["hotpotqa", "2wiki", "musique"]


def main() -> int:
    print("Loading slice corpus title index...", flush=True)
    titles = TitleIndex()
    title_set: set[str] = {normalize_title(t) for t in titles.title_to_id.keys()}
    print(f"  {len(title_set):,} unique titles in corpus")

    summary: dict[str, dict] = {}
    for name in DATASETS:
        print(f"\n=== {name} ===", flush=True)
        qs = list(DATASET_LOADERS[name]())
        n_total = len(qs)
        n_no_gold = 0
        n_all_in_corpus = 0
        n_any_in_corpus = 0
        miss_counter: Counter = Counter()
        hop_dist: Counter = Counter()
        for q in qs:
            if not q.gold_titles:
                n_no_gold += 1
                continue
            hop_dist[q.n_hops] += 1
            in_corpus = [t in title_set for t in q.gold_titles]
            if all(in_corpus):
                n_all_in_corpus += 1
            if any(in_corpus):
                n_any_in_corpus += 1
            for t, ok in zip(q.gold_titles, in_corpus):
                if not ok:
                    miss_counter[t] += 1
        n_scored = n_total - n_no_gold
        rate_all = n_all_in_corpus / n_scored if n_scored else 0.0
        rate_any = n_any_in_corpus / n_scored if n_scored else 0.0
        summary[name] = {
            "n_total": n_total,
            "n_with_gold": n_scored,
            "n_no_gold_titles": n_no_gold,
            "n_all_gold_in_corpus": n_all_in_corpus,
            "n_any_gold_in_corpus": n_any_in_corpus,
            "rate_all_in_corpus": rate_all,
            "rate_any_in_corpus": rate_any,
            "hop_distribution": dict(hop_dist),
            "top_missing_titles": miss_counter.most_common(20),
        }
        print(f"  total questions: {n_total}")
        print(f"  with gold titles: {n_scored} (no-gold: {n_no_gold})")
        print(f"  all-gold-in-corpus: {n_all_in_corpus} ({rate_all * 100:.2f}%)")
        print(f"  any-gold-in-corpus: {n_any_in_corpus} ({rate_any * 100:.2f}%)")
        print(f"  hops: {dict(hop_dist)}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_PATH.relative_to(EXP_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
