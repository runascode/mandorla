"""Seeded synthetic relational world for the curriculum pilot.

The world is designed so the "Vesica of two entities" has an exact,
checkable ground truth: bitwise-AND of their latent property vectors.

Vocabulary (fixed id layout so tests can assert it):

    0 PAD   1 BOS   2 EOS   3 HAS   4 SHARE
    5 .. 5+N-1            entity tokens   E_0 .. E_{N-1}
    5+N .. 5+N+K-1        property tokens P_0 .. P_{K-1}

Two paragraph templates:

  - `BOS E_i HAS P_k EOS`         for every entity i and property k with
                                  a_i[k] = 1   (single-entity attribute
                                  exposure — lets a compositional model
                                  learn each entity's attributes from
                                  contexts that never pair it)
  - `BOS E_i E_j SHARE P_k EOS`   for every *training* pair (i,j) and
                                  property k with a_i[k]=a_j[k]=1

Held-out pairs never appear in any SHARE paragraph (in either order).
Their entities still appear in HAS paragraphs and in SHARE paragraphs
with *other* entities, so the only way to predict a held-out pair's
shared set is to have learned the per-entity attributes and the
*operation* of intersecting them — which is exactly the thesis claim
under test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAD, BOS, EOS, HAS, SHARE = 0, 1, 2, 3, 4
N_SPECIAL = 5


@dataclass(frozen=True)
class WorldConfig:
    n_entities: int = 64
    n_props: int = 16
    prop_density: float = 0.5      # P(a_i[k] = 1)
    heldout_frac: float = 0.15     # fraction of all unordered pairs held out
    seed: int = 1337


class World:
    """A deterministic synthetic relational world."""

    def __init__(self, cfg: WorldConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        # Latent attributes, fixed once. Reject any all-zero entity so
        # every entity has at least one property (keeps HAS paragraphs
        # non-empty and the probe well-posed).
        A = (rng.random((cfg.n_entities, cfg.n_props)) < cfg.prop_density).astype(np.int8)
        for i in range(cfg.n_entities):
            if A[i].sum() == 0:
                A[i, rng.integers(0, cfg.n_props)] = 1
        self.A = A

        # Token id layout.
        self.entity_tok = lambda i: N_SPECIAL + i
        self.prop_tok = lambda k: N_SPECIAL + cfg.n_entities + k
        self.vocab_size = N_SPECIAL + cfg.n_entities + cfg.n_props

        # Held-out unordered pair split.
        all_pairs = [
            (i, j)
            for i in range(cfg.n_entities)
            for j in range(i + 1, cfg.n_entities)
        ]
        rng.shuffle(all_pairs)
        n_held = int(round(len(all_pairs) * cfg.heldout_frac))
        self._heldout = set(all_pairs[:n_held])
        self._train_pairs = all_pairs[n_held:]

    # ── ground truth ────────────────────────────────────────────────
    def shared(self, i: int, j: int) -> np.ndarray:
        """Ground-truth Vesica: bitwise-AND of the two attribute rows."""
        return (self.A[i] & self.A[j]).astype(np.int8)

    def is_heldout(self, i: int, j: int) -> bool:
        a, b = (i, j) if i < j else (j, i)
        return (a, b) in self._heldout

    @property
    def train_pairs(self) -> list[tuple[int, int]]:
        return list(self._train_pairs)

    @property
    def heldout_pairs(self) -> list[tuple[int, int]]:
        return sorted(self._heldout)

    # ── corpus ──────────────────────────────────────────────────────
    def _has_paragraphs(self) -> list[list[int]]:
        out: list[list[int]] = []
        for i in range(self.cfg.n_entities):
            for k in range(self.cfg.n_props):
                if self.A[i, k]:
                    out.append([BOS, self.entity_tok(i), HAS, self.prop_tok(k), EOS])
        return out

    def _share_paragraphs(self) -> list[list[int]]:
        out: list[list[int]] = []
        for (i, j) in self._train_pairs:
            sh = self.shared(i, j)
            for k in range(self.cfg.n_props):
                if sh[k]:
                    out.append(
                        [BOS, self.entity_tok(i), self.entity_tok(j),
                         SHARE, self.prop_tok(k), EOS]
                    )
        return out

    def corpus(self, shuffle_seed: int = 1337) -> list[list[int]]:
        """All paragraphs (HAS + training-pair SHARE), shuffled
        deterministically. Held-out pairs are absent by construction."""
        paras = self._has_paragraphs() + self._share_paragraphs()
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(paras)
        return paras

    # ── supervised pair examples (for the auxiliary losses) ─────────
    def supervised_pairs(self) -> list[tuple[int, int, np.ndarray]]:
        """(i, j, shared_multihot) over TRAINING pairs only. The
        curriculum's Vesica/parent losses train on these; held-out
        pairs are never supervised, in any loss."""
        return [(i, j, self.shared(i, j)) for (i, j) in self._train_pairs]

    # ── probe ───────────────────────────────────────────────────────
    def probe_examples(self) -> list[tuple[int, int, np.ndarray]]:
        """(i, j, shared_multihot) over HELD-OUT pairs. The pilot signal
        is the gap between curriculum and baseline on this set."""
        return [(i, j, self.shared(i, j)) for (i, j) in self.heldout_pairs]

    def seen_pair_control(self, n: int = 256, seed: int = 99) -> list[tuple[int, int, np.ndarray]]:
        """A sample of TRAINING pairs, used as the i.i.d. control: both
        conditions can win here by memorization, so a curriculum lift on
        held-out pairs is only meaningful if it exceeds any lift here."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self._train_pairs), size=min(n, len(self._train_pairs)), replace=False)
        return [
            (self._train_pairs[t][0], self._train_pairs[t][1],
             self.shared(*self._train_pairs[t]))
            for t in idx
        ]
