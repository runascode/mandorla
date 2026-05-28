"""Unit tests for Region/Vesica/BoxExtent primitives.

Verifies the data-model invariants from mandorla.md §2.2: primordial-vs-vesica
distinction, lineage depth incrementing, citation tracking, and
isolation of mutable per-region metadata across instances.
"""

from __future__ import annotations

import numpy as np

from src.box import intersect_boxes
from src.regions import (
    BoxExtent,
    Lineage,
    Region,
    Vesica,
    make_chunk_region,
    make_vesica,
    new_id,
)


def cube(center: float, half: float, dim: int = 4) -> BoxExtent:
    return BoxExtent(
        min_corner=np.full(dim, center - half, dtype=np.float32),
        max_corner=np.full(dim, center + half, dtype=np.float32),
    )


def test_new_id_unique_at_scale() -> None:
    assert len({new_id() for _ in range(1000)}) == 1000


def test_box_extent_dim_64() -> None:
    assert cube(0.0, 1.0, dim=64).dim == 64


def test_box_extent_center_and_half_width() -> None:
    box = BoxExtent(
        min_corner=np.array([0.0, 2.0], dtype=np.float32),
        max_corner=np.array([2.0, 6.0], dtype=np.float32),
    )
    np.testing.assert_allclose(box.center, [1.0, 4.0])
    np.testing.assert_allclose(box.half_width, [1.0, 2.0])


def test_chunk_region_is_primordial() -> None:
    box = cube(0.0, 1.0)
    r = make_chunk_region("chunk-0", box.center, box)
    assert r.is_primordial()
    assert r.lineage.depth == 0
    assert r.id == "chunk-0"


def test_vesica_records_parents_and_citations() -> None:
    box_a = cube(0.0, 1.0)
    box_b = cube(0.5, 1.0)
    a = make_chunk_region("a", box_a.center, box_a)
    b = make_chunk_region("b", box_b.center, box_b)
    inter = intersect_boxes(box_a, box_b)
    assert inter is not None
    v = make_vesica(a, b, inter)
    assert not v.is_primordial()
    assert v.parent_ids() == ("a", "b")
    assert v.lineage.depth == 1
    assert v.citations == {"a", "b"}


def test_vesica_depth_increments_max_of_parents() -> None:
    box = cube(0.0, 1.0)
    shallow = make_chunk_region("s", box.center, box)
    deep = Region(
        id="d",
        center=box.center,
        extent=box,
        birth_time=0.0,
        lineage=Lineage(parents=("x", "y"), depth=5),
    )
    inter = intersect_boxes(box, box)
    assert inter is not None
    v = make_vesica(shallow, deep, inter)
    assert v.lineage.depth == 6


def test_region_meta_default_factory_isolated() -> None:
    """Two regions must not share mutable metadata via class-level defaults."""
    box = cube(0.0, 1.0)
    r1 = make_chunk_region("a", box.center, box)
    r2 = make_chunk_region("b", box.center, box)
    r1.meta.promotion_score = 1.0
    r1.meta.overlap_history.append(0.5)
    r1.citations.add("x")
    assert r2.meta.promotion_score == 0.0
    assert r2.meta.overlap_history == []
    assert r2.citations == set()


def test_vesica_inherits_region_interface() -> None:
    """Vesica is a Region; should pass isinstance checks and have the same fields."""
    box = cube(0.0, 1.0)
    a = make_chunk_region("a", box.center, box)
    b = make_chunk_region("b", box.center, box)
    inter = intersect_boxes(box, box)
    assert inter is not None
    v = make_vesica(a, b, inter)
    assert isinstance(v, Region)
    assert isinstance(v, Vesica)
