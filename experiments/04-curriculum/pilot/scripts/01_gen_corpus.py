"""Materialize the R1 2-hop world and record its statistics."""

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
    ood = w.comp_ood_probe()
    seen = w.seen_entity_control()
    ans_sizes = [int(w.answer(i, j).sum()) for (i, j, _) in w.supervised_queries()]
    ood_sizes = [int(a.sum()) for (_, _, a) in ood]
    stats = {
        "config": cfg.__dict__,
        "vocab_size": w.vocab_size,
        "n_paragraphs": len(corpus),
        "n_trained_entities": len(w.trained_entities),
        "n_heldout_entities": len(w.heldout_entities),
        "n_supervised_queries": len(w.supervised_queries()),
        "mean_attrs_per_entity": float(w.A.sum(axis=1).mean()),
        "mean_answer_size_train": float(np.mean(ans_sizes)),
        "mean_answer_size_comp_ood": float(np.mean(ood_sizes)),
        "frac_comp_ood_nonempty": float(np.mean([s > 0 for s in ood_sizes])),
        "n_comp_ood_examples": len(ood),
        "n_seen_control_examples": len(seen),
        "partner_is_derangement": bool(all(w.pi[i] != i for i in range(cfg.n_entities))),
    }
    out = EXP_ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
