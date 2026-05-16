"""Losses for the R1 pilot — three conditions.

All three keep the CLM objective (never removed). They differ only in
the auxiliary signal applied to the model's **QUERY-position hidden
state** (the intermediate the model itself constructed for the 2-hop
query — it is never handed π(i)):

  baseline      : CLM only
  generic_aux   : CLM + BCE( generic_head(h_query) , a_i )       [control]
  curriculum    : CLM + vesica + parent, both from
                  box(h_query) ∩ box(h_Ej)

Auxiliary losses train ONLY on supervised (trained-entity) queries;
held-out entities never enter any auxiliary loss, so the comp-OOD probe
is uncontaminated.

`curriculum_loss` returns a metrics dict (component losses + box stats +
collapse detectors) for the Q1 stability JSONL.

The query prompt is the fixed 4-token layout `BOS E_i E_j QUERY`, so
the E_j hidden state is at position 1-indexed 2 and the QUERY hidden
state at position 3 (0-indexed).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .box import intersection_embedding, intersection_log_volume

PAD = 0
EJ_POS = 2       # 0-indexed position of E_j in `BOS E_i E_j QUERY`
QUERY_POS = 3    # 0-indexed position of QUERY


def clm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    tgt = tokens[:, 1:].reshape(-1)
    return F.cross_entropy(pred, tgt, ignore_index=PAD)


def baseline_loss(model, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
    lm = clm_loss(model(tokens), tokens)
    return lm, {"loss_total": float(lm.detach()), "loss_clm": float(lm.detach())}


def generic_aux_loss(
    model, generic_head, tokens: torch.Tensor, q: dict, lam: float = 1.0
) -> tuple[torch.Tensor, dict]:
    logits, h = model(q["prompt"], return_hidden=True)
    lm = clm_loss(model(tokens), tokens)
    h_q = h[:, QUERY_POS, :]
    aux_logits = generic_head(h_q)
    aux = F.binary_cross_entropy_with_logits(aux_logits, q["attr_i"])
    total = lm + lam * aux
    with torch.no_grad():
        acc = float(((aux_logits > 0).float() == q["attr_i"]).float().mean())
    return total, {
        "loss_total": float(total.detach()),
        "loss_clm": float(lm.detach()),
        "loss_generic_aux": float(aux.detach()),
        "generic_aux_bit_acc": acc,
    }


def curriculum_loss(
    model, heads, tokens: torch.Tensor, q: dict,
    lambda_v: float = 1.0, lambda_p: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    lm = clm_loss(model(tokens), tokens)

    _, h = model(q["prompt"], return_hidden=True)
    h_q = h[:, QUERY_POS, :]      # constructed intermediate (must encode π(i))
    h_j = h[:, EJ_POS, :]         # E_j representation
    c_q, lh_q = model.box_from_hidden(h_q)
    c_j, lh_j = model.box_from_hidden(h_j)

    inter = intersection_embedding(c_q, lh_q, c_j, lh_j)
    v_logits, p_logits = heads(inter)

    vesica = F.binary_cross_entropy_with_logits(v_logits, q["answer"])
    parent_tgt = torch.cat([q["attr_pi"], q["attr_j"]], dim=-1)
    parent = F.binary_cross_entropy_with_logits(p_logits, parent_tgt)

    total = lm + lambda_v * vesica + lambda_p * parent
    with torch.no_grad():
        ilv = intersection_log_volume(c_q, lh_q, c_j, lh_j)
        metrics = {
            "loss_total": float(total.detach()),
            "loss_clm": float(lm.detach()),
            "loss_vesica": float(vesica.detach()),
            "loss_parent": float(parent.detach()),
            "inter_logvol_mean": float(ilv.mean()),
            "inter_logvol_std": float(ilv.std()),
            "inter_emb_batch_std": float(inter.std(dim=0).mean()),
            "qbox_center_norm": float(c_q.norm(dim=-1).mean()),
            "vesica_bit_acc": float(((v_logits > 0).float() == q["answer"]).float().mean()),
            "parent_bit_acc": float(((p_logits > 0).float() == parent_tgt).float().mean()),
        }
    return total, metrics
