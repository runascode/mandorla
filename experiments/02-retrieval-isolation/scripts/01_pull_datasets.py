"""Pull dev splits of the three multi-hop QA datasets into the local HF
datasets cache.

After this runs, the rest of the pipeline can iterate over the three
loaders in `src/datasets.py` without re-downloading. The cache lives in
`./data/` per the standard HF env var.

Idempotent: re-running just hits the cache.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXP_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(EXP_ROOT / "data" / "hf_home"))
(EXP_ROOT / "data").mkdir(exist_ok=True)

from src.datasets import (  # noqa: E402
    load_2wiki_dev,
    load_hotpotqa_dev,
    load_musique_dev,
)


def main() -> int:
    print("Pulling HotpotQA dev (distractor split's validation = 7,405 q)...", flush=True)
    n_hp = sum(1 for _ in load_hotpotqa_dev())
    print(f"  HotpotQA dev: {n_hp} questions")

    print("Pulling 2WikiMultiHopQA dev...", flush=True)
    n_2w = sum(1 for _ in load_2wiki_dev())
    print(f"  2Wiki dev: {n_2w} questions")

    print("Pulling MuSiQue-Ans dev...", flush=True)
    n_mq = sum(1 for _ in load_musique_dev())
    print(f"  MuSiQue dev: {n_mq} questions")

    print()
    print(f"All three datasets cached in {os.environ['HF_HOME']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
