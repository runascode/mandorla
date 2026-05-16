"""Pilot 2 probe — reads the ANSWER HEAD (through the bottleneck),
never an LM head. comp-OOD = held-out entities; seen-entity control.

F1@G: rank the K property logits, cut at G = |true answer|, score
overlap. Applied identically to both arms and both tasks, so the
INTERSECTION−POINT gap is the robust signal.
"""

from __future__ import annotations

import numpy as np
import torch

from .synthetic import BOS, QUERY, World


@torch.no_grad()
def _logits(model, world: World, i: int, j: int, device) -> np.ndarray:
    prompt = torch.tensor(
        [[BOS, world.entity_tok(i), world.entity_tok(j), QUERY]], device=device
    )
    return model.answer(prompt)[0].cpu().numpy()


def evaluate(model, world: World, examples, device) -> dict:
    model.eval()
    f1s, exacts = [], []
    for (i, j, ans) in examples:
        g = int(ans.sum())
        if g == 0:
            continue
        lg = _logits(model, world, i, j, device)
        topg = set(np.argsort(-lg)[:g].tolist())
        truth = set(np.where(ans == 1)[0].tolist())
        f1s.append(len(topg & truth) / g)
        exacts.append(1.0 if topg == truth else 0.0)
    return {
        "n": len(f1s),
        "f1_at_g": float(np.mean(f1s)) if f1s else float("nan"),
        "exact_set_acc": float(np.mean(exacts)) if exacts else float("nan"),
    }


def probe_report(model, world: World, task: str, device) -> dict:
    ood = evaluate(model, world, world.comp_ood_probe(task), device)
    seen = evaluate(model, world, world.seen_entity_control(task), device)
    return {"comp_ood": ood, "seen_entity_control": seen,
            "ood_minus_seen": ood["f1_at_g"] - seen["f1_at_g"]}
