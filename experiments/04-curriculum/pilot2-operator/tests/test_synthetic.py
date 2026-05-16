"""Pilot 2 world invariants. The verdict is only interpretable if:
held-out entities are never queried/supervised; the CLM corpus has no
QUERY (no bypass); NONINT is genuinely a different relation than AND;
NONINT's label balance is comparable to AND's (fair F1@G)."""

from __future__ import annotations

import numpy as np

from src.synthetic import N_SPECIAL, QUERY, World, WorldConfig


def _w():
    return World(WorldConfig(n_entities=48, n_props=10, queries_per_entity=12, seed=7))


def test_clm_corpus_has_no_query_token():
    """No-bypass: CLM never sees QUERY, so it cannot teach the
    composition the bottleneck is supposed to be the sole path to."""
    w = _w()
    for para in w.corpus():
        assert QUERY not in para


def test_heldout_entities_never_supervised_either_task():
    w = _w()
    held = set(w.heldout_entities)
    for task in ("and", "nonint"):
        for (i, _, _) in w.supervised_queries(task):
            assert i not in held


def test_comp_ood_is_heldout_only_both_tasks():
    w = _w()
    held = set(w.heldout_entities)
    for task in ("and", "nonint"):
        for (i, _, _) in w.comp_ood_probe(task, n=150):
            assert i in held


def test_and_is_exact_two_hop_intersection():
    w = _w()
    for (i, j, a) in w.supervised_queries("and")[:40]:
        assert np.array_equal(a, w.A[w.pi[i]] & w.A[j])


def test_nonint_is_genuinely_not_and():
    """The whole point of the control: NONINT must differ from AND on a
    large fraction of (i,j,bit), else a 'win' on it is still circular."""
    w = _w()
    diff = tot = 0
    for i in range(w.cfg.n_entities):
        for j in range(w.cfg.n_entities):
            if i == j:
                continue
            a = w.answer_and(i, j)
            n = w.answer_nonint(i, j)
            diff += int((a != n).sum())
            tot += a.size
    assert diff / tot > 0.20, f"NONINT differs from AND on only {diff/tot:.1%} of bits"


def test_nonint_label_balance_comparable_to_and():
    w = _w()
    ar = np.mean([w.answer_and(i, j).mean()
                  for i in range(w.cfg.n_entities)
                  for j in range(w.cfg.n_entities) if i != j])
    nr = np.mean([w.answer_nonint(i, j).mean()
                  for i in range(w.cfg.n_entities)
                  for j in range(w.cfg.n_entities) if i != j])
    assert abs(ar - nr) < 0.08, f"AND rate {ar:.3f} vs NONINT rate {nr:.3f}"


def test_partner_is_derangement_and_permutation():
    w = _w()
    assert all(w.pi[i] != i for i in range(w.cfg.n_entities))
    assert sorted(w.pi.tolist()) == list(range(w.cfg.n_entities))


def test_determinism():
    a = World(WorldConfig(n_entities=20, n_props=6, seed=5))
    b = World(WorldConfig(n_entities=20, n_props=6, seed=5))
    assert np.array_equal(a.A, b.A) and np.array_equal(a.pi, b.pi)
    assert a.corpus() == b.corpus()
    assert np.array_equal(a.answer_nonint(1, 2), b.answer_nonint(1, 2))


def test_capacity_matched_intersection_le_point():
    from src.model import ComposerModel, ModelConfig, assert_capacity_matched
    cfg = ModelConfig(vocab_size=_w().vocab_size, n_props=10)
    cap = assert_capacity_matched(ComposerModel(cfg, "point"),
                                  ComposerModel(cfg, "intersection"))
    assert cap["int_bottleneck_params"] <= cap["point_bottleneck_params"]
