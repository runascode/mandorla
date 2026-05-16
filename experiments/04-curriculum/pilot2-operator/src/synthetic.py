"""Synthetic 2-hop world for Pilot 2 (composition-operator test).

Reuses the `../pilot/` R1 2-hop structure (entities with attribute
bits; a fixed partner map π; `i → π(i)` taught only as a `PARTNER`
fact; held-out *entities* never queried). Two differences from the
curriculum pilot, both required by PILOT2.md:

1. **Two task families.**
   - AND     : Q(i,j)  = a_{π(i)} AND a_j      (intersective)
   - NONINT  : Q'(i,j) = 1[ M·[a_{π(i)};a_j] > θ ]  (a fixed seeded
               random linear-threshold relation over the joint 2K-bit
               operand; per-bit θ calibrated so the positive rate ≈
               that of AND, keeping F1@G label balance comparable).
     NONINT is *not* a per-operand intersection and *not* XOR/OR; it
     is a generic learnable 2-hop relation. A test asserts it disagrees
     with AND on >20% of (i,j,bit).

2. **No-bypass corpus.** CLM trains on `ATTR` and `PARTNER` paragraphs
   only (encoder learns attributes + partner map). The composition into
   the answer is learned **exclusively** through the bottleneck +
   answer head from supervised queries — there is no `QUERY` paragraph
   in the CLM corpus, so the LM objective cannot teach (or bypass) the
   composition.

Vocab (fixed; asserted by tests):
    0 PAD  1 BOS  2 EOS  3 ATTR  4 PARTNER  5 QUERY
    6.. entities ; then properties
`QUERY` is reserved for the (non-CLM) probe/answer-head prompt only.
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
    prop_density: float = 0.5
    heldout_entity_frac: float = 0.20
    queries_per_entity: int = 24
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

        perm = rng.permutation(cfg.n_entities)
        for i in range(cfg.n_entities):
            if perm[i] == i:
                s = (i + 1) % cfg.n_entities
                perm[i], perm[s] = perm[s], perm[i]
        self.pi = perm

        self.entity_tok = lambda i: N_SPECIAL + i
        self.prop_tok = lambda k: N_SPECIAL + cfg.n_entities + k
        self.vocab_size = N_SPECIAL + cfg.n_entities + cfg.n_props

        n_held = int(round(cfg.n_entities * cfg.heldout_entity_frac))
        order = rng.permutation(cfg.n_entities)
        self._heldout = set(int(x) for x in order[:n_held])
        self._trained = [int(x) for x in order[n_held:]]

        self._train_queries: list[tuple[int, int]] = []
        for i in self._trained:
            js = rng.choice(
                [x for x in range(cfg.n_entities) if x != i],
                size=min(cfg.queries_per_entity, cfg.n_entities - 1),
                replace=False,
            )
            for j in js:
                self._train_queries.append((i, int(j)))

        # NONINT relation: fixed random linear-threshold over [x ; y],
        # x = a_{π(i)}, y = a_j, joint dim 2K. Per-bit θ calibrated on
        # the realized population of (trained ∪ held) i × all j so the
        # per-bit positive rate ≈ AND's.
        K = cfg.n_props
        self._M = rng.standard_normal((K, 2 * K)).astype(np.float32)
        all_i = list(range(cfg.n_entities))
        sample = [(i, int(j)) for i in all_i
                  for j in rng.choice([x for x in all_i if x != i], size=12, replace=False)]
        feats = np.stack([
            np.concatenate([self.A[self.pi[i]], self.A[j]]).astype(np.float32)
            for (i, j) in sample
        ])                                                    # (S, 2K)
        pre = feats @ self._M.T                                # (S, K)
        and_rate = float((self.A[:, None, :] & self.A[None, :, :]).mean())
        # per-bit threshold = (1 - and_rate) quantile of that bit's pre-activation
        self._theta = np.quantile(pre, 1.0 - and_rate, axis=0).astype(np.float32)

    # ── ground truth ────────────────────────────────────────────────
    def answer_and(self, i: int, j: int) -> np.ndarray:
        return (self.A[self.pi[i]] & self.A[j]).astype(np.int8)

    def answer_nonint(self, i: int, j: int) -> np.ndarray:
        x = np.concatenate([self.A[self.pi[i]], self.A[j]]).astype(np.float32)
        return ((self._M @ x) > self._theta).astype(np.int8)

    def answer(self, task: str, i: int, j: int) -> np.ndarray:
        return self.answer_and(i, j) if task == "and" else self.answer_nonint(i, j)

    @property
    def trained_entities(self) -> list[int]:
        return list(self._trained)

    @property
    def heldout_entities(self) -> list[int]:
        return sorted(self._heldout)

    def is_heldout(self, i: int) -> bool:
        return i in self._heldout

    # ── corpus (CLM): ATTR + PARTNER only — no QUERY, no bypass ──────
    def corpus(self, shuffle_seed: int = 1337) -> list[list[int]]:
        paras: list[list[int]] = []
        for i in range(self.cfg.n_entities):
            for k in range(self.cfg.n_props):
                if self.A[i, k]:
                    paras.append([BOS, self.entity_tok(i), ATTR, self.prop_tok(k), EOS])
        for i in range(self.cfg.n_entities):
            paras.append([BOS, self.entity_tok(i), PARTNER,
                          self.entity_tok(int(self.pi[i])), EOS])
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(paras)
        return paras

    # ── supervised query examples for the answer head ───────────────
    def supervised_queries(self, task: str) -> list[tuple[int, int, np.ndarray]]:
        return [(i, j, self.answer(task, i, j)) for (i, j) in self._train_queries]

    # ── probes ──────────────────────────────────────────────────────
    def comp_ood_probe(self, task: str, n: int = 600, seed: int = 7):
        rng = np.random.default_rng(seed)
        held = self.heldout_entities
        out = []
        for _ in range(n):
            i = int(rng.choice(held))
            j = int(rng.choice([x for x in range(self.cfg.n_entities) if x != i]))
            out.append((i, j, self.answer(task, i, j)))
        return out

    def seen_entity_control(self, task: str, n: int = 600, seed: int = 9):
        rng = np.random.default_rng(seed)
        trainset = set(self._train_queries)
        out, guard = [], 0
        while len(out) < n and guard < n * 50:
            guard += 1
            i = int(rng.choice(self._trained))
            j = int(rng.choice([x for x in range(self.cfg.n_entities) if x != i]))
            if (i, j) in trainset:
                continue
            out.append((i, j, self.answer(task, i, j)))
        return out
