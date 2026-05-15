"""Invariants for the synthetic world and the probe split.

These guard the properties the pilot's validity depends on: exact
ground truth, true held-out isolation (no leakage), deterministic
seeding, correct vocab layout.
"""

from __future__ import annotations

import numpy as np

from src.synthetic import (
    BOS,
    EOS,
    HAS,
    N_SPECIAL,
    SHARE,
    World,
    WorldConfig,
)


def _world() -> World:
    return World(WorldConfig(n_entities=40, n_props=12, seed=7))


def test_shared_is_exact_bitwise_and():
    w = _world()
    for (i, j) in w.train_pairs[:50]:
        assert np.array_equal(w.shared(i, j), w.A[i] & w.A[j])


def test_heldout_pairs_never_appear_in_corpus():
    """The validity of the whole pilot rests on this: a held-out pair
    must never co-occur in any paragraph, in either order."""
    w = _world()
    held = set(w.heldout_pairs)
    ent0 = N_SPECIAL
    for para in w.corpus():
        # entity tokens in this paragraph
        ents = [t - ent0 for t in para if N_SPECIAL <= t < N_SPECIAL + w.cfg.n_entities]
        if len(ents) == 2:
            a, b = sorted(ents)
            assert (a, b) not in held, f"held-out pair ({a},{b}) leaked into corpus"


def test_heldout_and_train_pairs_are_disjoint_and_cover_all():
    w = _world()
    held = set(w.heldout_pairs)
    train = set(tuple(p) for p in w.train_pairs)
    assert held.isdisjoint(train)
    n = w.cfg.n_entities
    assert len(held) + len(train) == n * (n - 1) // 2


def test_determinism_same_seed_same_world():
    a = World(WorldConfig(n_entities=20, n_props=8, seed=42))
    b = World(WorldConfig(n_entities=20, n_props=8, seed=42))
    assert np.array_equal(a.A, b.A)
    assert a.heldout_pairs == b.heldout_pairs
    assert a.corpus() == b.corpus()


def test_different_seed_different_world():
    a = World(WorldConfig(n_entities=20, n_props=8, seed=1))
    b = World(WorldConfig(n_entities=20, n_props=8, seed=2))
    assert not np.array_equal(a.A, b.A)


def test_no_all_zero_entities():
    w = World(WorldConfig(n_entities=128, n_props=6, prop_density=0.1, seed=3))
    assert (w.A.sum(axis=1) > 0).all()


def test_vocab_layout_and_token_helpers():
    w = _world()
    assert (BOS, EOS, HAS, SHARE) == (1, 2, 3, 4)
    assert w.entity_tok(0) == N_SPECIAL
    assert w.prop_tok(0) == N_SPECIAL + w.cfg.n_entities
    assert w.vocab_size == N_SPECIAL + w.cfg.n_entities + w.cfg.n_props


def test_share_paragraphs_only_assert_true_shared_properties():
    w = _world()
    p0 = w.prop_tok(0)
    for para in w.corpus():
        if SHARE in para:
            # [BOS, E_i, E_j, SHARE, P_k, EOS]
            ei = para[1] - N_SPECIAL
            ej = para[2] - N_SPECIAL
            pk = para[4] - p0
            assert w.A[ei, pk] == 1 and w.A[ej, pk] == 1


def test_supervised_pairs_are_training_only():
    w = _world()
    held = set(w.heldout_pairs)
    for (i, j, _) in w.supervised_pairs():
        a, b = sorted((i, j))
        assert (a, b) not in held
