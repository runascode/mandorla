"""Compositional-transfer probe (R1).

Readout is identical for all three conditions and uses only the LM
head: feed `BOS E_i E_j QUERY`, read the next-token distribution at the
QUERY position restricted+renormalized over property tokens. In
training, a queried pair emits one `... QUERY P_k EOS` paragraph per
property k in Q(i,j), so the post-QUERY distribution should spread over
exactly the answer set.

Primary metric F1@G: rank properties by probability, cut at G = |true
answer|. G-cutoff removes threshold choice and is applied identically
to every condition, so the *gaps* are the robust signal.

- comp_ood: held-out entities (never queried) → the signal.
- seen_entity_control: trained entities, (i,j) not in training → guards
  against "the gap is just new-j", not novel composition.
"""

from __future__ import annotations

import numpy as np
import torch

from .synthetic import BOS, QUERY, World


@torch.no_grad()
def _property_probs(model, world: World, i: int, j: int, device: torch.device) -> np.ndarray:
    prompt = torch.tensor(
        [[BOS, world.entity_tok(i), world.entity_tok(j), QUERY]], device=device
    )
    logits = model(prompt)[0, -1]
    p0 = world.prop_tok(0)
    return torch.softmax(logits[p0 : p0 + world.cfg.n_props], dim=-1).cpu().numpy()


def evaluate(model, world: World, examples, device: torch.device) -> dict:
    model.eval()
    f1s, exacts = [], []
    for (i, j, ans) in examples:
        g = int(ans.sum())
        if g == 0:
            continue
        probs = _property_probs(model, world, i, j, device)
        topg = set(np.argsort(-probs)[:g].tolist())
        truth = set(np.where(ans == 1)[0].tolist())
        f1s.append(len(topg & truth) / g)
        exacts.append(1.0 if topg == truth else 0.0)
    return {
        "n": len(f1s),
        "f1_at_g": float(np.mean(f1s)) if f1s else float("nan"),
        "exact_set_acc": float(np.mean(exacts)) if exacts else float("nan"),
    }


def probe_report(model, world: World, device: torch.device) -> dict:
    ood = evaluate(model, world, world.comp_ood_probe(), device)
    seen = evaluate(model, world, world.seen_entity_control(), device)
    return {
        "comp_ood": ood,
        "seen_entity_control": seen,
        "ood_minus_seen": ood["f1_at_g"] - seen["f1_at_g"],
    }
