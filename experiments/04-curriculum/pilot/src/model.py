"""Tiny causal transformer + box head for the curriculum pilot (R1).

Key R1 change: the box geometry is read off the model's **constructed
hidden state at the `QUERY` position**, not a static entity embedding.
For the 2-hop task the model is *not* handed `π(i)`; to answer it must
resolve the partner internally, so the QUERY-position hidden state is
exactly "the intermediate the model built." Forcing *that* to support a
meaningful intersection is the faithful test of Thesis 3.

The probe readout uses the LM head for *all three* conditions (feed
`BOS E_i E_j QUERY`, read property-token logits), so no condition's
extra heads ever advantage it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 8
    box_dim: int = 32
    dropout: float = 0.0
    seed: int = 1337


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model, dropout=cfg.dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        # hidden state → (center, log_half) box params
        self.box_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 2 * cfg.box_dim),
        )

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        h = self.tok(tokens) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        return self.encoder(h, mask=mask, is_causal=True)        # (B,T,d)

    def forward(self, tokens: torch.Tensor, return_hidden: bool = False):
        h = self.encode(tokens)
        logits = self.lm_head(h)
        return (logits, h) if return_hidden else logits

    def box_from_hidden(self, hidden_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, d_model) → (center (B, box_dim), log_half (B, box_dim))."""
        cl = self.box_head(hidden_vec)
        return cl[..., : self.cfg.box_dim], cl[..., self.cfg.box_dim :]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class VesicaHeads(nn.Module):
    """Consumes the intersection embedding (midpoint ++ side, 2*box_dim):

      - vesica head → K logits  (predict Q(i,j) = a_{π(i)} AND a_j)
      - parent head → 2K logits (reconstruct a_{π(i)} and a_j)
    """

    def __init__(self, box_dim: int, n_props: int, hidden: int = 128) -> None:
        super().__init__()
        d = 2 * box_dim
        self.vesica = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_props)
        )
        self.parent = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 2 * n_props)
        )

    def forward(self, inter_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vesica(inter_emb), self.parent(inter_emb)


class GenericAuxHead(nn.Module):
    """Capacity-comparable NON-intersection control. Off the same QUERY
    hidden state, predicts the queried entity's *own* attributes a_i —
    structured extra signal + parameters, but no box geometry. Sized to
    sit in the same ballpark as box_head+VesicaHeads combined; the train
    script logs exact param counts so the match is auditable."""

    def __init__(self, d_model: int, n_props: int, hidden: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_props),
        )

    def forward(self, hidden_vec: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_vec)
