"""Pilot 2 losses.

total = CLM(ATTR/PARTNER corpus)  +  λ · BCE( answer_head(bottleneck(...)),
                                              Q(i,j) )

CLM is identical across both arms and trains only the shared encoder on
ATTR/PARTNER paragraphs (no QUERY → no bypass of the composition). The
answer is produced exclusively through the bottleneck, so the bottleneck
operator is genuinely on the critical path. The only thing that differs
between arms is which bottleneck module is used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

PAD = 0


def clm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    tgt = tokens[:, 1:].reshape(-1)
    return F.cross_entropy(pred, tgt, ignore_index=PAD)


def step_loss(
    model, corpus_tokens: torch.Tensor, q: dict, lam: float = 1.0
) -> tuple[torch.Tensor, dict]:
    lm = clm_loss(model.encoder.lm_logits(corpus_tokens), corpus_tokens)
    ans_logits = model.answer(q["prompt"])
    ans = F.binary_cross_entropy_with_logits(ans_logits, q["target"])
    total = lm + lam * ans
    with torch.no_grad():
        bit_acc = float(((ans_logits > 0).float() == q["target"]).float().mean())
    return total, {
        "loss_total": float(total.detach()),
        "loss_clm": float(lm.detach()),
        "loss_answer": float(ans.detach()),
        "answer_bit_acc": bit_acc,
    }
