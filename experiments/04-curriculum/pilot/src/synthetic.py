"""Seeded synthetic 2-hop relational world for the curriculum pilot (R1).

R0 (held-out *pairs*, `E_i E_j SHARE P_k`) was abandoned: plain CLM
solved it near-perfectly because, with every entity individually
observed, the held-out task reduces to recall + elementwise-AND, which
attention does natively. See PILOT.md "Design revision R1" and
LAB-NOTES.md (2026-05-15).

R1 makes the held-out generalization a **2-hop latent composition** that
cannot be reduced to recall+AND:

    Q(i, j) := a_{π(i)} AND a_j

Answering requires hop 1 (`i → π(i)`, taught only as a separate
single-hop `PARTNER` fact) then hop 2 (intersect `a_{π(i)}` with
`a_j`). Entities in the held-out set `H` are **never queried** in
training, so their answer cannot be shortcut-memorized; the model must
chain two separately-taught single-hop facts latently.

Vocabulary (fixed id layout; asserted by tests):

    0 PAD  1 BOS  2 EOS  3 ATTR  4 PARTNER  5 QUERY
    6 .. 6+N-1            entity tokens   E_0 .. E_{N-1}
    6+N .. 6+N+K-1        property tokens P_0 .. P_{K-1}

Paragraph types:

  - `BOS E_i ATTR P_k EOS`            for every k with a_i[k]=1
  - `BOS E_i PARTNER E_{π(i)} EOS`    for ALL i                  (hop 1)
  - `BOS E_i E_j QUERY P_k EOS`       for every k in Q(i,j),
                                      ONLY for non-held-out i    (hop 2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAD, BOS, EOS, ATTR, PARTNER, QUERY = 0, 1, 2, 3, 4, 5
N_SPECIAL = 6


@dataclass(frozen=True)
class WorldConfig:
    n_entities: int = 96
    n_props: int = 12
    prop_density: float = 0.5          # P(a_i[k] = 1)
    heldout_entity_frac: float = 0.20  # entities never queried in training
    queries_per_entity: int = 24       # training (i, j) queries per trained i
    seed: int = 1337


class World:
    def __init__(self, cfg: WorldConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        A = (rng.random((cfg.n_entities, cfg.n_props)) < cfg.prop_density).astype(np.int8)
        for i in range(cfg.n_entities):
            if A[i].sum() == 0:
                A[i, rng.integers(0, cfg.n_props)] = 1
        self.A = A

        # Partner map π: a derangement (π(i) ≠ i), fixed and seeded.
        perm = rng.permutation(cfg.n_entities)
        for i in range(cfg.n_entities):
            if perm[i] == i:
                swap = (i + 1) % cfg.n_entities
                perm[i], perm[swap] = perm[swap], perm[i]
        self.pi = perm

        # Token id layout.
        self.entity_tok = lambda i: N_SPECIAL + i
        self.prop_tok = lambda k: N_SPECIAL + cfg.n_entities + k
        self.vocab_size = N_SPECIAL + cfg.n_entities + cfg.n_props

        # Held-out *entities* (never appear as the queried entity i in any
        # QUERY paragraph or in any auxiliary loss).
        n_held = int(round(cfg.n_entities * cfg.heldout_entity_frac))
        order = rng.permutation(cfg.n_entities)
        self._heldout = set(int(x) for x in order[:n_held])
        self._trained = [int(x) for x in order[n_held:]]

        # Training queries: for each trained i, a fixed seeded sample of
        # j partners (j may be any entity ≠ i, held-out or not — held-out
        # entities are excluded only as the *queried* entity, not as j).
        self._train_queries: list[tuple[int, int]] = []
        for i in self._trained:
            js = rng.choice(
                [x for x in range(cfg.n_entities) if x != i],
                size=min(cfg.queries_per_entity, cfg.n_entities - 1),
                replace=False,
            )
            for j in js:
                self._train_queries.append((i, int(j)))
        self._train_query_set = set(self._train_queries)

    # ── ground truth ────────────────────────────────────────────────
    def answer(self, i: int, j: int) -> np.ndarray:
        """Q(i,j) = a_{π(i)} AND a_j  (the 2-hop composition target)."""
        return (self.A[self.pi[i]] & self.A[j]).astype(np.int8)

    @property
    def trained_entities(self) -> list[int]:
        return list(self._trained)

    @property
    def heldout_entities(self) -> list[int]:
        return sorted(self._heldout)

    def is_heldout(self, i: int) -> bool:
        return i in self._heldout

    # ── corpus ──────────────────────────────────────────────────────
    def _attr_paragraphs(self) -> list[list[int]]:
        out = []
        for i in range(self.cfg.n_entities):
            for k in range(self.cfg.n_props):
                if self.A[i, k]:
                    out.append([BOS, self.entity_tok(i), ATTR, self.prop_tok(k), EOS])
        return out

    def _partner_paragraphs(self) -> list[list[int]]:
        # Hop 1 taught for ALL entities, including held-out ones.
        return [
            [BOS, self.entity_tok(i), PARTNER, self.entity_tok(int(self.pi[i])), EOS]
            for i in range(self.cfg.n_entities)
        ]

    def _query_paragraphs(self) -> list[list[int]]:
        out = []
        for (i, j) in self._train_queries:           # trained i only
            ans = self.answer(i, j)
            for k in range(self.cfg.n_props):
                if ans[k]:
                    out.append(
                        [BOS, self.entity_tok(i), self.entity_tok(j),
                         QUERY, self.prop_tok(k), EOS]
                    )
        return out

    def corpus(self, shuffle_seed: int = 1337) -> list[list[int]]:
        paras = (
            self._attr_paragraphs()
            + self._partner_paragraphs()
            + self._query_paragraphs()
        )
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(paras)
        return paras

    # ── supervised examples for the auxiliary losses ────────────────
    def supervised_queries(self) -> list[tuple[int, int, np.ndarray]]:
        """(i, j, answer) over TRAINING queries only (trained i). The
        curriculum / generic-aux losses train on these; held-out
        entities never appear, in any loss."""
        return [(i, j, self.answer(i, j)) for (i, j) in self._train_queries]

    # ── probes ──────────────────────────────────────────────────────
    def comp_ood_probe(self, n: int = 600, seed: int = 7) -> list[tuple[int, int, np.ndarray]]:
        """The signal: held-out entities (never queried) × sampled j.
        Requires latent 2-hop composition of two separately-taught
        single-hop facts."""
        rng = np.random.default_rng(seed)
        held = self.heldout_entities
        out = []
        for _ in range(n):
            i = int(rng.choice(held))
            j = int(rng.choice([x for x in range(self.cfg.n_entities) if x != i]))
            out.append((i, j, self.answer(i, j)))
        return out

    def seen_entity_control(self, n: int = 600, seed: int = 9) -> list[tuple[int, int, np.ndarray]]:
        """Control: trained entities, but (i,j) combos NOT in training.
        i was queried (with other j) so a shortcut-memorized a_{π(i)}
        can win here. A curriculum gap that also shows up here is not
        novel-composition transfer."""
        rng = np.random.default_rng(seed)
        out = []
        guard = 0
        while len(out) < n and guard < n * 50:
            guard += 1
            i = int(rng.choice(self._trained))
            j = int(rng.choice([x for x in range(self.cfg.n_entities) if x != i]))
            if (i, j) in self._train_query_set:
                continue
            out.append((i, j, self.answer(i, j)))
        return out
