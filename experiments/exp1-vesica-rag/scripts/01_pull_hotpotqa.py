"""Pull HotpotQA from HuggingFace and persist a normalized JSONL.

We pull both the validation split (7,405 questions — the slice's eval set) and
the train split (90,447 — used downstream for τ_v calibration on a 1k
sub-sample).

We use the `distractor` configuration. Important: we are not running the
slice's evaluation under the distractor setting (which would defeat the
retrieval test — see PRECOMMIT.md D2). We pull the distractor configuration
because it carries the `supporting_facts` annotation we need for the
vesica-coverage diagnostic. The actual retrieval happens against the
separately-pulled fullwiki corpus (script 02).

Schema written per question:
  {
    "id":               str  (HotpotQA id, e.g. "5a8b57f25542995d1e6f1371"),
    "question":         str,
    "answer":           str,
    "type":             "bridge" | "comparison",
    "level":            "easy" | "medium" | "hard",
    "supporting_facts": [ {"title": str, "sent_id": int}, ... ],
    "context_titles":   [ str, ... ]    # the 10 paragraph titles from
                                        # the distractor pool; useful for
                                        # debugging and as a 1-hop sanity
                                        # check at vesica-coverage time
  }

The full distractor context paragraphs are NOT written — at fullwiki time we
match by title against the wiki corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def normalize_example(ex: dict) -> dict:
    """Convert a HuggingFace HotpotQA example to our flat JSONL schema."""
    sf_titles = ex["supporting_facts"]["title"]
    sf_sent_ids = ex["supporting_facts"]["sent_id"]
    supporting_facts = [
        {"title": t, "sent_id": int(s)} for t, s in zip(sf_titles, sf_sent_ids)
    ]
    context_titles = list(ex["context"]["title"])
    return {
        "id": ex["id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex["type"],
        "level": ex["level"],
        "supporting_facts": supporting_facts,
        "context_titles": context_titles,
    }


def dump_split(split: str, out_path: Path) -> int:
    print(f"Loading hotpot_qa/distractor split={split} ...", flush=True)
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for ex in ds:
            f.write(json.dumps(normalize_example(ex)) + "\n")
            n += 1
    print(f"  wrote {n} examples → {out_path}")
    return n


def main() -> int:
    # File names use the HuggingFace split names ("validation", "train") so
    # they line up with src.data.load_hotpotqa(split) → "hotpotqa_{split}.jsonl".
    # (The paper / README call this the "dev" set; that's a colloquialism for
    # the same split.)
    n_val = dump_split("validation", DATA_DIR / "hotpotqa_validation.jsonl")
    n_train = dump_split("train", DATA_DIR / "hotpotqa_train.jsonl")
    print(f"\nDone. val={n_val} train={n_train}")
    # Sanity asserts from the published HotpotQA paper.
    assert n_val == 7405, f"Expected 7405 validation examples; got {n_val}"
    assert n_train == 90447, f"Expected 90447 train examples; got {n_train}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
