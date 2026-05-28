"""Unit tests for RandomProjection.

Verifies the determinism, shape, batch/single equivalence, and save/load
round-trip — all of which are load-bearing for reproducibility per
PRECOMMIT.md decision F.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.projection import DEFAULT_SEED, RandomProjection


def test_same_seed_yields_identical_matrix() -> None:
    p1 = RandomProjection(d_in=768, d_out=64, seed=DEFAULT_SEED)
    p2 = RandomProjection(d_in=768, d_out=64, seed=DEFAULT_SEED)
    np.testing.assert_array_equal(p1.matrix, p2.matrix)


def test_different_seeds_yield_different_matrices() -> None:
    p1 = RandomProjection(d_in=768, d_out=64, seed=1337)
    p2 = RandomProjection(d_in=768, d_out=64, seed=42)
    assert not np.array_equal(p1.matrix, p2.matrix)


def test_projection_matrix_shape() -> None:
    assert RandomProjection(d_in=768, d_out=64).matrix.shape == (64, 768)


def test_project_single_vector_shape_and_dtype() -> None:
    p = RandomProjection(d_in=768, d_out=64)
    v = np.random.RandomState(0).standard_normal(768).astype(np.float32)
    out = p.project(v)
    assert out.shape == (64,)
    assert out.dtype == np.float32


def test_project_batch_shape_and_dtype() -> None:
    p = RandomProjection(d_in=768, d_out=64)
    vs = np.random.RandomState(0).standard_normal((10, 768)).astype(np.float32)
    out = p.project(vs)
    assert out.shape == (10, 64)
    assert out.dtype == np.float32


def test_batched_matches_per_vector() -> None:
    p = RandomProjection(d_in=768, d_out=64)
    vs = np.random.RandomState(0).standard_normal((5, 768)).astype(np.float32)
    batched = p.project(vs)
    one_by_one = np.stack([p.project(v) for v in vs])
    np.testing.assert_allclose(batched, one_by_one, atol=1e-5)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    p = RandomProjection(d_in=768, d_out=64, seed=DEFAULT_SEED)
    save_path = tmp_path / "proj.npz"
    p.save(save_path)
    p2 = RandomProjection.load(save_path)
    np.testing.assert_array_equal(p.matrix, p2.matrix)
    assert p.d_in == p2.d_in
    assert p.d_out == p2.d_out
    assert p.seed == p2.seed


def test_load_detects_tampered_matrix(tmp_path: Path) -> None:
    """If the on-disk matrix doesn't match the seed-derivation, load must raise."""
    p = RandomProjection(d_in=8, d_out=4, seed=DEFAULT_SEED)
    save_path = tmp_path / "proj.npz"
    p.save(save_path)
    data = dict(np.load(save_path))
    # Corrupt the matrix
    data["matrix"] = (data["matrix"] + 1.0).astype(np.float32)
    np.savez(save_path, **data)
    try:
        RandomProjection.load(save_path)
    except RuntimeError as e:
        assert "reproducibility" in str(e).lower() or "match" in str(e).lower()
    else:
        raise AssertionError("Expected RuntimeError on tampered matrix")


def test_project_wrong_dim_raises() -> None:
    p = RandomProjection(d_in=768, d_out=64)
    bad = np.zeros(100, dtype=np.float32)
    try:
        p.project(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError on wrong input dim")


def test_norm_approximately_preserved() -> None:
    """Under the 1/sqrt(d_out) scaling, expected squared norm of projection
    matches input squared norm on average."""
    p = RandomProjection(d_in=768, d_out=64)
    rng = np.random.default_rng(0)
    vs = rng.standard_normal((512, 768)).astype(np.float32)
    in_norms = np.linalg.norm(vs, axis=1) ** 2
    out_norms = np.linalg.norm(p.project(vs), axis=1) ** 2
    # Mean ratio should be near 1 with Johnson-Lindenstrauss scaling.
    assert 0.7 < float(np.mean(out_norms) / np.mean(in_norms)) < 1.4
