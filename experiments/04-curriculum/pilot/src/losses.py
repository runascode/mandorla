"""Losses for the pilot.

- CLM: standard next-token cross-entropy (baseline trains only this;
  curriculum trains this too — the LM objective is never removed).
- Vesica prediction: box-intersection of E_i, E_j → predict the shared
  property set a_i AND a_j. The paper phrases this contrastively; with
  an *exact* synthetic ground truth a per-property BCE against the true
  shared set is the clean, stable equivalent and is what the pilot
  uses. (This simplification is a pilot-only choice, recorded in
  PILOT.md's spirit: the full Exp 04 PRECOMMIT picks the real
  contrastive form.)
- Parent reconstruction: from the same intersection embedding,
  reconstruct a_i and a_j. Forbids the degenerate collapse where the
  intersection throws away parent information.

`curriculum_loss` also returns a metrics dict (component losses + box
stats + collapse detectors) consumed by the training-stability JSONL —
Q1 of the pilot is literally "do these train without collapsing", so
the instrumentation is a first-class output, not a print.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .box import intersection_embedding, intersection_log_volume

PAD = 0


def clm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Next-token CE. logits (B,T,V) predict tokens[:,1:]. PAD ignored."""
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    tgt = tokens[:, 1:].reshape(-1)
    return F.cross_entropy(pred, tgt, ignore_index=PAD)


def _box_params(model, entity_ids: torch.Tensor):
    return model.entity_box(entity_ids)


def vesica_and_parent_losses(
    model,
    heads,
    ei: torch.Tensor,             # (B,) entity token ids
    ej: torch.Tensor,             # (B,)
    shared_multihot: torch.Tensor,   # (B, K) float {0,1}
    attr_i: torch.Tensor,            # (B, K) float {0,1}
    attr_j: torch.Tensor,            # (B, K)
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Returns (vesica_loss, parent_loss, box_stats)."""
    c_i, lh_i = _box_params(model, ei)
    c_j, lh_j = _box_params(model, ej)

    inter = intersection_embedding(c_i, lh_i, c_j, lh_j)
    v_logits, p_logits = heads(inter)

    vesica = F.binary_cross_entropy_with_logits(v_logits, shared_multihot)

    K = attr_i.size(-1)
    parent_tgt = torch.cat([attr_i, attr_j], dim=-1)
    parent = F.binary_cross_entropy_with_logits(p_logits, parent_tgt)

    with torch.no_grad():
        ilv = intersection_log_volume(c_i, lh_i, c_j, lh_j)
        # collapse detectors
        box_stats = {
            "inter_logvol_mean": float(ilv.mean()),
            "inter_logvol_std": float(ilv.std()),
            "center_norm_mean": float(c_i.norm(dim=-1).mean()),
            # fraction of pairs whose intersection embedding is ~constant
            # across the batch (a collapse signature)
            "inter_emb_batch_std": float(inter.std(dim=0).mean()),
            "vesica_bit_acc": float(
                ((v_logits > 0).float() == shared_multihot).float().mean()
            ),
            "parent_bit_acc": float(
                ((p_logits > 0).float() == parent_tgt).float().mean()
            ),
        }
    return vesica, parent, box_stats


def curriculum_loss(
    model,
    heads,
    tokens: torch.Tensor,
    pair_batch: dict,
    lambda_v: float = 1.0,
    lambda_p: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Total = CLM + λ_v·vesica + λ_p·parent. Returns (loss, metrics)."""
    lm = clm_loss(model(tokens), tokens)
    v, p, box_stats = vesica_and_parent_losses(
        model, heads,
        pair_batch["ei"], pair_batch["ej"],
        pair_batch["shared"], pair_batch["attr_i"], pair_batch["attr_j"],
    )
    total = lm + lambda_v * v + lambda_p * p
    metrics = {
        "loss_total": float(total.detach()),
        "loss_clm": float(lm.detach()),
        "loss_vesica": float(v.detach()),
        "loss_parent": float(p.detach()),
        **box_stats,
    }
    return total, metrics


def baseline_loss(model, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """CLM only — the honest baseline. Same model class; the box/aux
    heads simply receive no gradient."""
    lm = clm_loss(model(tokens), tokens)
    return lm, {"loss_total": float(lm.detach()), "loss_clm": float(lm.detach())}
