"""Compositional-transfer probe.

Readout is identical for both conditions and uses only the LM head, so
the curriculum's extra heads never advantage it: feed `BOS E_i E_j
SHARE`, read the next-token distribution restricted to property tokens.
In training, for any pair that co-occurs there is one
`... SHARE P_k EOS` paragraph per shared property k, so the LM's
post-`SHARE` distribution should spread over exactly the shared set.

Primary metric: **F1@G** — rank properties by model probability, take
the top-G where G = |true shared set|, score overlap. G-cutoff removes
threshold choice and is applied identically to both conditions, so the
curriculum−baseline *gap* is the robust pilot signal. Exact-set match
is reported as a secondary, stricter view.

The signal of interest is the gap on **held-out** pairs (never seen
together in training). The seen-pair control should show little/no gap
(both can win there by memorization); a held-out gap that merely
mirrors a seen-pair gap is not compositional transfer.
"""

from __future__ import annotations

import numpy as np
import torch

from .synthetic import BOS, SHARE, World


@torch.no_grad()
def _property_probs(
    model, world: World, ei: int, ej: int, device: torch.device
) -> np.ndarray:
    """Model P(next token) restricted+renormalized over property tokens,
    after the prompt `BOS E_i E_j SHARE`."""
    prompt = torch.tensor(
        [[BOS, world.entity_tok(ei), world.entity_tok(ej), SHARE]],
        device=device,
    )
    logits = model(prompt)[0, -1]                       # (V,)
    p0 = world.prop_tok(0)
    prop_logits = logits[p0 : p0 + world.cfg.n_props]
    return torch.softmax(prop_logits, dim=-1).cpu().numpy()


def evaluate(
    model,
    world: World,
    examples: list[tuple[int, int, np.ndarray]],
    device: torch.device,
) -> dict:
    """Mean F1@G and exact-set accuracy over the given (i,j,shared)
    examples."""
    model.eval()
    f1s: list[float] = []
    exacts: list[float] = []
    for (i, j, shared) in examples:
        probs = _property_probs(model, world, i, j, device)
        g = int(shared.sum())
        if g == 0:
            continue
        topg = set(np.argsort(-probs)[:g].tolist())
        truth = set(np.where(shared == 1)[0].tolist())
        hit = len(topg & truth)
        # |pred| == |truth| == g  ⇒  precision == recall == F1
        f1s.append(hit / g)
        exacts.append(1.0 if topg == truth else 0.0)
    return {
        "n": len(f1s),
        "f1_at_g": float(np.mean(f1s)) if f1s else float("nan"),
        "exact_set_acc": float(np.mean(exacts)) if exacts else float("nan"),
    }


def probe_report(model, world: World, device: torch.device) -> dict:
    """Held-out (the signal) + seen-pair control (the i.i.d. baseline)."""
    held = evaluate(model, world, world.probe_examples(), device)
    seen = evaluate(model, world, world.seen_pair_control(), device)
    return {
        "heldout": held,
        "seen_control": seen,
        "transfer_gap_vs_self": held["f1_at_g"] - seen["f1_at_g"],
    }
