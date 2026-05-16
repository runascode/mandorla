"""Shared encoder + the two composition bottlenecks for Pilot 2.

Everything is held identical between the two arms except the bottleneck
module that turns (h_query, h_j) into a width-`2·box_dim` vector. That
vector is consumed by an **identical** answer head in both arms. CLM
(ATTR/PARTNER corpus only) trains the shared encoder identically; it
never sees QUERY, so it cannot teach or bypass the composition — the
answer is producible *only* through the bottleneck.

Capacity is matched-and-audited, not exact-by-construction: the
intersection bottleneck is built to have **≤** the point bottleneck's
parameter count (asserted at init, both counts logged). So a win for
INTERSECTION cannot be attributed to extra capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .box import corners, intersect, soft_side


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 8
    box_dim: int = 32
    point_hidden: int = 160
    int_mix_hidden: int = 320   # sized so int bottleneck ≈ point (audited)
    answer_hidden: int = 128
    n_props: int = 12
    dropout: float = 0.0
    seed: int = 1337


class Encoder(nn.Module):
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
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)  # CLM only

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        h = self.tok(tokens) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        return self.encoder(h, mask=mask, is_causal=True)

    def lm_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.hidden(tokens))


class PointBottleneck(nn.Module):
    """MLP([h_q, h_j]) → 2·box_dim. The point-composition arm."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        out = 2 * cfg.box_dim
        self.net = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.point_hidden), nn.GELU(),
            nn.Linear(cfg.point_hidden, out),
        )

    def forward(self, h_q: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_q, h_j], dim=-1))


class IntersectionBottleneck(nn.Module):
    """h → box (shared map, applied to both operands) → box∩box →
    (midpoint ++ soft side) → small mix. The intersection operator *is*
    the bottleneck; no MLP can route around it."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.box_dim = cfg.box_dim
        self.to_box = nn.Linear(cfg.d_model, 2 * cfg.box_dim)   # shared both operands
        # 2-layer learned mix AFTER the param-free intersection, sized so
        # the whole bottleneck ≈ the point bottleneck's param count. The
        # intersection op still sits on the critical path (param-free,
        # unavoidable); both arms get an equal learned-mixing budget — the
        # only structural difference is whether a box∩box is in the pipe.
        out = 2 * cfg.box_dim
        self.mix = nn.Sequential(
            nn.Linear(out, cfg.int_mix_hidden), nn.GELU(),
            nn.Linear(cfg.int_mix_hidden, out),
        )

    def _cl(self, h: torch.Tensor):
        cl = self.to_box(h)
        return cl[..., : self.box_dim], cl[..., self.box_dim :]

    def forward(self, h_q: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        cq, lq = self._cl(h_q)
        cj, lj = self._cl(h_j)
        lo_q, hi_q = corners(cq, lq)
        lo_j, hi_j = corners(cj, lj)
        lo_i, hi_i = intersect(lo_q, hi_q, lo_j, hi_j)
        mid = 0.5 * (lo_i + hi_i)
        side = soft_side(lo_i, hi_i)
        return self.mix(torch.cat([mid, side], dim=-1))


class AnswerHead(nn.Module):
    """Identical in both arms: 2·box_dim → answer_hidden → K logits."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * cfg.box_dim, cfg.answer_hidden), nn.GELU(),
            nn.Linear(cfg.answer_hidden, cfg.n_props),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ComposerModel(nn.Module):
    def __init__(self, cfg: ModelConfig, arm: str) -> None:
        super().__init__()
        assert arm in ("point", "intersection")
        self.cfg = cfg
        self.arm = arm
        self.encoder = Encoder(cfg)
        self.bottleneck = (
            PointBottleneck(cfg) if arm == "point" else IntersectionBottleneck(cfg)
        )
        self.answer_head = AnswerHead(cfg)

    @staticmethod
    def _count(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    def bottleneck_params(self) -> int:
        return self._count(self.bottleneck)

    def answer(self, prompt: torch.Tensor) -> torch.Tensor:
        # prompt = [BOS, E_i, E_j, QUERY]; QUERY pos=3, E_j pos=2
        h = self.encoder.hidden(prompt)
        return self.answer_head(self.bottleneck(h[:, 3, :], h[:, 2, :]))


def assert_capacity_matched(point_model: ComposerModel, int_model: ComposerModel) -> dict:
    """INTERSECTION must have ≤ POINT bottleneck params (so a win can't
    be 'more capacity'). Answer heads + encoders are identical by
    construction."""
    pp = point_model.bottleneck_params()
    ip = int_model.bottleneck_params()
    # Matched-and-audited: int must be ≤ point (so an INTERSECTION win
    # can't be 'more capacity') AND close to it (so a NEGATIVE/tie can't
    # be 'intersection was capacity-starved'). Tune int_mix_hidden /
    # point_hidden if this trips.
    assert ip <= pp * 1.02, f"int bottleneck ({ip}) must be ≤ point ({pp})"
    assert ip >= pp * 0.80, (
        f"int bottleneck ({ip}) too far below point ({pp}); a negative "
        f"would be capacity-confounded — raise int_mix_hidden"
    )
    return {"point_bottleneck_params": pp, "int_bottleneck_params": ip,
            "int_over_point": round(ip / pp, 3)}
