"""Invariants for the R1 2-hop relational world.

The pilot's validity rests on these: held-out *entities* are never
queried (no comp-OOD contamination), the partner map is a derangement,
ground truth is the exact 2-hop composition, seeding is deterministic.
"""

from __future__ import annotations

import numpy as np

from src.synthetic import (
    ATTR,
    BOS,
    EOS,
    N_SPECIAL,
    PARTNER,
    QUERY,
    World,
    WorldConfig,
)


def _w() -> World:
    return World(WorldConfig(n_entities=40, n_props=8, queries_per_entity=10, seed=7))


def test_answer_is_exact_two_hop_composition():
    w = _w()
    for (i, j, ans) in w.supervised_queries()[:50]:
        assert np.array_equal(ans, w.A[w.pi[i]] & w.A[j])


def test_partner_is_derangement():
    w = _w()
    assert all(w.pi[i] != i for i in range(w.cfg.n_entities))
    assert sorted(w.pi.tolist()) == list(range(w.cfg.n_entities))  # permutation


def test_heldout_entities_never_queried_in_corpus():
    """Core validity invariant: a held-out entity must never be the
    queried entity i in any QUERY paragraph."""
    w = _w()
    held = set(w.heldout_entities)
    e0 = N_SPECIAL
    for para in w.corpus():
        if QUERY in para:
            # [BOS, E_i, E_j, QUERY, P_k, EOS]
            i = para[1] - e0
            assert i not in held, f"held-out entity {i} was queried in training"


def test_heldout_entities_still_have_attr_and_partner_facts():
    """The parts must be learnable: held-out entities need ATTR and
    PARTNER facts present (only the QUERY is withheld)."""
    w = _w()
    held = set(w.heldout_entities)
    e0 = N_SPECIAL
    attr_ents, partner_ents = set(), set()
    for para in w.corpus():
        if ATTR in para:
            attr_ents.add(para[1] - e0)
        if PARTNER in para:
            partner_ents.add(para[1] - e0)
    for h in held:
        assert h in attr_ents and h in partner_ents


def test_heldout_and_trained_partition_all_entities():
    w = _w()
    held = set(w.heldout_entities)
    tr = set(w.trained_entities)
    assert held.isdisjoint(tr)
    assert held | tr == set(range(w.cfg.n_entities))


def test_supervised_queries_are_trained_entities_only():
    w = _w()
    held = set(w.heldout_entities)
    for (i, _, _) in w.supervised_queries():
        assert i not in held


def test_comp_ood_probe_is_heldout_entities_only():
    w = _w()
    held = set(w.heldout_entities)
    for (i, _, _) in w.comp_ood_probe(n=200):
        assert i in held


def test_seen_control_excludes_training_queries():
    w = _w()
    trainq = set((i, j) for (i, j) in w._train_queries)
    held = set(w.heldout_entities)
    for (i, j, _) in w.seen_entity_control(n=200):
        assert i not in held
        assert (i, j) not in trainq


def test_determinism_same_seed():
    a = World(WorldConfig(n_entities=24, n_props=6, seed=42))
    b = World(WorldConfig(n_entities=24, n_props=6, seed=42))
    assert np.array_equal(a.A, b.A)
    assert np.array_equal(a.pi, b.pi)
    assert a.heldout_entities == b.heldout_entities
    assert a.corpus() == b.corpus()


def test_query_paragraphs_assert_only_true_answer_props():
    w = _w()
    p0 = w.prop_tok(0)
    for para in w.corpus():
        if QUERY in para:
            i = para[1] - N_SPECIAL
            j = para[2] - N_SPECIAL
            pk = para[4] - p0
            assert (w.A[w.pi[i]] & w.A[j])[pk] == 1


def test_vocab_layout():
    w = _w()
    assert (BOS, EOS, ATTR, PARTNER, QUERY) == (1, 2, 3, 4, 5)
    assert w.entity_tok(0) == N_SPECIAL
    assert w.prop_tok(0) == N_SPECIAL + w.cfg.n_entities
