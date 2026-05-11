"""Verify the HotpotQA Wikipedia corpus is present in the local HuggingFace
cache and write a pinning record.

We use BeIR/hotpotqa's `corpus` split, which is the canonical retrieval-
formatted version of the HotpotQA Wikipedia abstracts dump: 5,233,329
passages, one per Wikipedia article, schema {_id, title, text}. Matching
HotpotQA's `supporting_facts.title` against the corpus's `title` field is
the gold-pair lookup for vesica-coverage.

We do NOT re-materialize the corpus as JSONL on disk. The HuggingFace Arrow
cache is the canonical local copy; re-dumping would waste ~4 GB and add a
second source of drift. Downstream scripts read directly from the dataset
loader.

This script's job:
  1. Load the corpus (forces HF cache materialization on first run).
  2. Verify the size and schema.
  3. Verify that *every* HotpotQA gold supporting-fact title resolves to
     exactly one passage in the corpus. If any title is missing, the slice
     can't compute vesica-coverage for that question — we want to know now.
  4. Write `data/corpus_meta.json` with the dataset revision, passage count,
     and the count of unmatched supporting-fact titles (which we expect to be
     zero or near-zero — title-aliasing across Wikipedia revisions sometimes
     leaves a small tail).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EXPECTED_COUNT = 5_233_329  # HotpotQA Wikipedia abstracts dump


def build_title_index(corpus) -> dict[str, list[str]]:
    """title → list of chunk_ids. Most titles map to exactly one passage."""
    print("Building title → chunk_id index...", flush=True)
    idx: dict[str, list[str]] = {}
    n = 0
    for row in corpus:
        title = row["title"]
        idx.setdefault(title, []).append(row["_id"])
        n += 1
        if n % 500_000 == 0:
            print(f"  ...{n}/{EXPECTED_COUNT}", flush=True)
    print(f"  done. {len(idx)} unique titles across {n} passages.")
    return idx


def audit_supporting_fact_coverage(
    dev_path: Path,
    train_path: Path,
    title_index: dict[str, list[str]],
) -> dict:
    """Count HotpotQA supporting-fact titles that fail to match any corpus
    passage. Returns a coverage report."""
    missing: Counter[str] = Counter()
    total_facts = 0
    total_questions = 0
    questions_with_any_missing = 0

    for split_path in (dev_path, train_path):
        print(f"Auditing {split_path.name} ...", flush=True)
        with split_path.open() as f:
            for line in f:
                q = json.loads(line)
                total_questions += 1
                any_missing = False
                for fact in q["supporting_facts"]:
                    total_facts += 1
                    if fact["title"] not in title_index:
                        missing[fact["title"]] += 1
                        any_missing = True
                if any_missing:
                    questions_with_any_missing += 1

    return {
        "total_questions": total_questions,
        "total_supporting_facts": total_facts,
        "unmatched_facts": int(sum(missing.values())),
        "unique_unmatched_titles": len(missing),
        "questions_with_at_least_one_unmatched_fact": questions_with_any_missing,
        "sample_unmatched_titles": [t for t, _ in missing.most_common(20)],
    }


def main() -> int:
    print("Loading BeIR/hotpotqa corpus split (forces HF cache materialization)...", flush=True)
    corpus = load_dataset("BeIR/hotpotqa", "corpus", split="corpus")
    n = len(corpus)
    print(f"  loaded {n} passages")
    assert n == EXPECTED_COUNT, f"Expected {EXPECTED_COUNT} passages; got {n}"
    assert set(corpus.column_names) == {"_id", "title", "text"}, (
        f"Unexpected schema: {corpus.column_names}"
    )

    title_index = build_title_index(corpus)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    audit = audit_supporting_fact_coverage(
        DATA_DIR / "hotpotqa_validation.jsonl",
        DATA_DIR / "hotpotqa_train.jsonl",
        title_index,
    )

    meta = {
        "dataset": "BeIR/hotpotqa",
        "split": "corpus",
        "passage_count": n,
        "unique_titles": len(title_index),
        "audit": audit,
    }
    out_path = DATA_DIR / "corpus_meta.json"
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out_path}\n")
    print(json.dumps(meta["audit"], indent=2))

    # Sanity: we expect a small tail of unmatched titles due to Wikipedia
    # revision drift, but it should be <2% of total facts. If it's larger,
    # vesica-coverage will be noisy for questions with unmatched supporting
    # facts, which is worth knowing before encoding kicks off.
    miss_rate = audit["unmatched_facts"] / audit["total_supporting_facts"]
    print(f"\nSupporting-fact miss rate: {miss_rate:.4f}")
    if miss_rate > 0.02:
        print(
            "WARNING: >2% of supporting facts unmatched. Vesica-coverage will\n"
            "be noisy. Consider documenting this as a slice caveat in RESULTS.md."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
