"""Materialize the synthetic world and print/record its statistics.

The corpus itself is regenerated deterministically in-memory by the
train script; this script exists to make the world's shape inspectable
and to record it as a provenance artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import World, WorldConfig  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cfg = WorldConfig()
    w = World(cfg)
    corpus = w.corpus()
    shared_sizes = [int(w.shared(i, j).sum()) for (i, j) in w.train_pairs]
    held_sizes = [int(w.shared(i, j).sum()) for (i, j) in w.heldout_pairs]
    stats = {
        "config": cfg.__dict__,
        "vocab_size": w.vocab_size,
        "n_paragraphs": len(corpus),
        "n_train_pairs": len(w.train_pairs),
        "n_heldout_pairs": len(w.heldout_pairs),
        "mean_attrs_per_entity": float(w.A.sum(axis=1).mean()),
        "mean_shared_size_train": float(np.mean(shared_sizes)),
        "mean_shared_size_heldout": float(np.mean(held_sizes)),
        "frac_heldout_pairs_with_nonempty_shared": float(
            np.mean([s > 0 for s in held_sizes])
        ),
    }
    out = EXP_ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {(out / 'corpus_stats.json').relative_to(EXP_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
