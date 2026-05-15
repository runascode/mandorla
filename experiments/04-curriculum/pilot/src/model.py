"""Tiny causal transformer + box head for the curriculum pilot.

One shared token-embedding table feeds both the language-model head
(next-token, the only thing the baseline trains) and the box head
(entity token → box center + log-half). The curriculum's auxiliary
losses push gradient into the *same* embedding table the LM uses — that
is the whole point: it tests whether shaping the shared representation
so intersections are meaningful changes what the model generalizes.

The probe readout uses the LM head for *both* conditions (feed
`BOS E_i E_j SHARE`, read property-token logits), so the comparison is
symmetric and never advantages the curriculum's extra heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 3
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
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

        # Box head: entity token embedding → (center, log_half) in box_dim.
        self.box_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2 * cfg.box_dim),
        )

    # ── language model ───────────────────────────────────────────────
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, T) → next-token logits (B, T, V), causal mask."""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        h = self.tok(tokens) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.lm_head(h)

    # ── box geometry off the shared embedding table ──────────────────
    def entity_box(self, entity_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """entity_token_ids (B,) → (center (B, box_dim), log_half (B, box_dim)).

        Reads the *same* embedding table the LM trains, so the auxiliary
        losses and the LM objective share representational substrate."""
        emb = self.tok(entity_token_ids)
        cl = self.box_head(emb)
        center, log_half = cl[..., : self.cfg.box_dim], cl[..., self.cfg.box_dim :]
        return center, log_half

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class VesicaHeads(nn.Module):
    """Two heads consuming the intersection embedding (midpoint ++ side,
    width 2*box_dim):

      - vesica head → K logits  (predict the shared property set)
      - parent head → 2K logits (reconstruct a_i and a_j)
    """

    def __init__(self, box_dim: int, n_props: int, hidden: int = 128) -> None:
        super().__init__()
        in_dim = 2 * box_dim
        self.vesica = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, n_props)
        )
        self.parent = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * n_props)
        )

    def forward(self, inter_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vesica(inter_emb), self.parent(inter_emb)
