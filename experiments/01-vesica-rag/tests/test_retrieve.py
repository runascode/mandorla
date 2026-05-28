"""Unit tests for BoxStore, BaselineRetriever, and VesicaRetriever.

We construct a tiny synthetic corpus (N=20 chunks in 8-D, projected to 4-D
boxes) where we can pre-compute the expected retrieval behavior. This
exercises the full retrieve pipeline without the 5.2M-passage real corpus.
"""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from src.box import intersect_boxes
from src.projection import RandomProjection
from src.regions import BoxExtent
from src.retrieve import (
    BaselineRetriever,
    BoxStore,
    DenseRetriever,
    FaissDenseRetriever,
    VesicaRetriever,
)


# ─── BoxStore ───────────────────────────────────────────────────────────────


def make_box_store(n: int = 6, d: int = 4, half: float = 0.5) -> BoxStore:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((n, d)).astype(np.float32)
    half_widths = np.full((n, d), half, dtype=np.float32)
    return BoxStore(
        centers=centers,
        half_widths=half_widths,
        chunk_ids=[f"chunk_{i}" for i in range(n)],
    )


def test_box_store_shape_consistency_required() -> None:
    with pytest.raises(ValueError):
        BoxStore(
            centers=np.zeros((3, 4), dtype=np.float32),
            half_widths=np.zeros((3, 5), dtype=np.float32),
            chunk_ids=["a", "b", "c"],
        )


def test_box_store_id_count_required() -> None:
    with pytest.raises(ValueError):
        BoxStore(
            centers=np.zeros((3, 4), dtype=np.float32),
            half_widths=np.zeros((3, 4), dtype=np.float32),
            chunk_ids=["a", "b"],
        )


def test_box_for_index_round_trips() -> None:
    store = make_box_store()
    box = store.box_for_index(2)
    np.testing.assert_allclose(box.center, store.centers[2], atol=1e-6)
    np.testing.assert_allclose(box.half_width, store.half_widths[2], atol=1e-6)


def test_containing_indices_includes_centers_inside() -> None:
    # Place 3 chunks at known centers; box around the middle includes one.
    centers = np.array([[0, 0], [1, 1], [5, 5]], dtype=np.float32)
    half_widths = np.full_like(centers, 0.1)
    store = BoxStore(centers=centers, half_widths=half_widths, chunk_ids=["a", "b", "c"])
    # Vesica box: [-0.5, -0.5] to [1.5, 1.5] — should contain a and b but not c.
    v = BoxExtent(
        min_corner=np.array([-0.5, -0.5], dtype=np.float32),
        max_corner=np.array([1.5, 1.5], dtype=np.float32),
    )
    idxs = store.containing_indices(v)
    assert set(idxs.tolist()) == {0, 1}


def test_containing_indices_respects_cap() -> None:
    n = 20
    centers = np.zeros((n, 2), dtype=np.float32)   # all at origin
    half_widths = np.full_like(centers, 0.1)
    store = BoxStore(centers=centers, half_widths=half_widths,
                     chunk_ids=[f"c{i}" for i in range(n)])
    v = BoxExtent(
        min_corner=np.array([-1.0, -1.0], dtype=np.float32),
        max_corner=np.array([1.0, 1.0], dtype=np.float32),
    )
    idxs = store.containing_indices(v, cap=5)
    assert idxs.size == 5


# ─── BaselineRetriever ──────────────────────────────────────────────────────


def test_baseline_top_k_returns_k() -> None:
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((10, 8)).astype(np.float32)
    br = BaselineRetriever(embeddings=embs, chunk_ids=[f"c{i}" for i in range(10)])
    out = br.top_k(np.ones(8, dtype=np.float32), k=3)
    assert len(out) == 3


def test_baseline_top_k_returns_closest_first() -> None:
    """Most cosine-similar should come first."""
    embs = np.array([
        [1.0, 0.0, 0.0],     # aligned with query
        [0.0, 1.0, 0.0],     # orthogonal
        [-1.0, 0.0, 0.0],    # anti-aligned
    ], dtype=np.float32)
    br = BaselineRetriever(embeddings=embs, chunk_ids=["a", "b", "c"])
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out = br.top_k(q, k=3)
    assert out[0].chunk_id == "a"
    assert out[2].chunk_id == "c"


def test_baseline_top_k_caps_at_n() -> None:
    embs = np.zeros((3, 4), dtype=np.float32)
    embs[0, 0] = 1
    br = BaselineRetriever(embeddings=embs, chunk_ids=["a", "b", "c"])
    out = br.top_k(np.ones(4, dtype=np.float32), k=10)
    assert len(out) == 3


def test_baseline_top_k_dim_check() -> None:
    embs = np.zeros((3, 4), dtype=np.float32)
    br = BaselineRetriever(embeddings=embs, chunk_ids=["a", "b", "c"])
    with pytest.raises(ValueError):
        br.top_k(np.zeros(5, dtype=np.float32), k=1)


# ─── FaissDenseRetriever ────────────────────────────────────────────────────


def _build_faiss(embs: np.ndarray) -> faiss.Index:
    """IndexFlatIP over L2-normalized rows — cosine via inner product."""
    e = embs.astype(np.float32)
    norms = np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-12)
    e_hat = e / norms
    idx = faiss.IndexFlatIP(e.shape[1])
    idx.add(e_hat)
    return idx


def test_faiss_retriever_satisfies_protocol() -> None:
    embs = np.eye(4, dtype=np.float32)
    fr = FaissDenseRetriever(index=_build_faiss(embs), chunk_ids=["a", "b", "c", "d"])
    assert isinstance(fr, DenseRetriever)


def test_faiss_retriever_count_check() -> None:
    embs = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError):
        FaissDenseRetriever(index=_build_faiss(embs), chunk_ids=["a", "b"])


def test_faiss_retriever_returns_closest_first() -> None:
    embs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.7, 0.0],   # between a and b, closer to query than b
    ], dtype=np.float32)
    fr = FaissDenseRetriever(index=_build_faiss(embs), chunk_ids=["a", "b", "c"])
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out = fr.top_k(q, k=3)
    assert out[0].chunk_id == "a"
    assert out[1].chunk_id == "c"
    assert out[2].chunk_id == "b"


def test_faiss_and_baseline_agree_on_ranking() -> None:
    """Same data → same top-k ordering from FAISS and the numpy reference."""
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((50, 16)).astype(np.float32)
    ids = [f"c{i}" for i in range(50)]
    br = BaselineRetriever(embeddings=embs, chunk_ids=ids)
    fr = FaissDenseRetriever(index=_build_faiss(embs), chunk_ids=ids)
    q = rng.standard_normal(16).astype(np.float32)
    b_ids = [s.chunk_id for s in br.top_k(q, k=10)]
    f_ids = [s.chunk_id for s in fr.top_k(q, k=10)]
    assert b_ids == f_ids


def test_faiss_retriever_caps_at_n() -> None:
    embs = np.eye(3, dtype=np.float32)
    fr = FaissDenseRetriever(index=_build_faiss(embs), chunk_ids=["a", "b", "c"])
    out = fr.top_k(np.ones(3, dtype=np.float32), k=10)
    assert len(out) == 3


def test_vesica_retriever_works_with_faiss_dense() -> None:
    """VesicaRetriever should accept a FaissDenseRetriever for descent."""
    rng = np.random.default_rng(99)
    n, d_768, d_box = 25, 32, 8
    embs = rng.standard_normal((n, d_768)).astype(np.float32)
    ids = [f"c{i}" for i in range(n)]
    fr = FaissDenseRetriever(index=_build_faiss(embs), chunk_ids=ids)
    projection = RandomProjection(d_in=d_768, d_out=d_box, seed=7)
    centers = projection.project(embs)
    half_widths = np.full_like(centers, 0.3)
    store = BoxStore(centers=centers, half_widths=half_widths, chunk_ids=ids)
    vr = VesicaRetriever(baseline=fr, boxes=store, projection=projection)
    result = vr.retrieve(rng.standard_normal(d_768).astype(np.float32),
                         k_points_used=5, k_descent=10, m_vesicas=3)
    assert len(result.points) == 5
    assert len(result.vesicas) <= 3
    for p in result.points:
        assert p.chunk_id in result.retrieved_chunk_ids


# ─── VesicaRetriever end-to-end ─────────────────────────────────────────────


def test_vesica_retrieve_produces_expected_structure() -> None:
    """Full retrieval cycle on a tiny synthetic corpus.

    We don't assert specific scores — just structural invariants:
      - points has k_points_used entries
      - vesicas has at most m_vesicas entries
      - retrieved_chunk_ids includes all `points`
      - retrieved_chunk_ids includes each Vesica's parents
      - retrieved_chunk_ids has no duplicates and is capped
    """
    rng = np.random.default_rng(1337)
    n = 30
    d_768 = 64    # use small dim for test speed
    d_box = 8
    embs = rng.standard_normal((n, d_768)).astype(np.float32)
    chunk_ids = [f"c{i}" for i in range(n)]
    br = BaselineRetriever(embeddings=embs, chunk_ids=chunk_ids)

    # Construct boxes by projecting embeddings and adding small isotropic extent
    projection = RandomProjection(d_in=d_768, d_out=d_box, seed=1337)
    centers = projection.project(embs)
    half_widths = np.full_like(centers, 0.3)
    store = BoxStore(centers=centers, half_widths=half_widths, chunk_ids=chunk_ids)

    vr = VesicaRetriever(baseline=br, boxes=store, projection=projection)

    query = rng.standard_normal(d_768).astype(np.float32)
    result = vr.retrieve(
        query,
        k_points_used=5,
        k_descent=10,
        m_vesicas=3,
        max_chunks_per_vesica=4,
        max_total_chunks=15,
    )

    assert len(result.points) == 5
    assert len(result.vesicas) <= 3
    # All point chunk_ids in retrieved union
    for p in result.points:
        assert p.chunk_id in result.retrieved_chunk_ids
    # All Vesica parents in retrieved union
    for v in result.vesicas:
        assert v.parent_a_id in result.retrieved_chunk_ids
        assert v.parent_b_id in result.retrieved_chunk_ids
    # Dedup
    assert len(result.retrieved_chunk_ids) == len(set(result.retrieved_chunk_ids))
    # Cap respected
    assert len(result.retrieved_chunk_ids) <= 15


def test_vesica_retrieve_sorted_by_score() -> None:
    rng = np.random.default_rng(7)
    n = 20
    d_768 = 32
    d_box = 8
    embs = rng.standard_normal((n, d_768)).astype(np.float32)
    chunk_ids = [f"c{i}" for i in range(n)]
    br = BaselineRetriever(embeddings=embs, chunk_ids=chunk_ids)
    projection = RandomProjection(d_in=d_768, d_out=d_box, seed=42)
    centers = projection.project(embs)
    half_widths = np.full_like(centers, 0.3)
    store = BoxStore(centers=centers, half_widths=half_widths, chunk_ids=chunk_ids)
    vr = VesicaRetriever(baseline=br, boxes=store, projection=projection)
    result = vr.retrieve(rng.standard_normal(d_768).astype(np.float32), m_vesicas=5)
    scores = [v.log_score for v in result.vesicas]
    assert scores == sorted(scores, reverse=True)
