"""Region, Vesica, and BoxExtent primitives per mandorla.md §2.2.

Slice scope (per PRECOMMIT.md): BoxExtent only; ball and SDM extents are not
implemented in this slice. Region / Vesica match the §2.2 reference interface
including lineage, citations, and RegionMeta — these are recorded but not yet
acted on by a store (the cross-query Vesica store is deferred to the broader
Experiment 1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from ulid import ULID

Vec: TypeAlias = NDArray[np.float32]


def new_id() -> str:
    """Generate a fresh ULID string. Sortable, opaque, unique."""
    return str(ULID())


class BoxExtent(BaseModel):
    """Axis-aligned hyperrectangle. Used both as the deterministically-
    constructed box (B2: center ± local-density half-width) and as the
    GumbelBox under the closed-form softplus expectation when scoring
    intersections.

    Immutable. `dim`, `center`, and `half_width` are derived.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    kind: str = "box"
    min_corner: Vec
    max_corner: Vec

    @property
    def dim(self) -> int:
        return int(self.min_corner.shape[0])

    @property
    def center(self) -> Vec:
        return ((self.min_corner + self.max_corner) * 0.5).astype(np.float32)

    @property
    def half_width(self) -> Vec:
        return ((self.max_corner - self.min_corner) * 0.5).astype(np.float32)


@dataclass
class Lineage:
    """Region birth record. parents=None for primordial Regions (raw chunks);
    parents=(a_id, b_id) for Vesicas born from intersections."""

    parents: Optional[tuple[str, str]] = None
    depth: int = 0  # 0 for primordial; max(parent depths) + 1 for Vesicas


@dataclass
class RegionMeta:
    """Mutable per-region metadata used by the cognitive cycle (§2.4).

    For the slice (no store, no promotion), these are recorded but not acted on.
    Kept here so the data model is consistent with the broader experiment.
    """

    overlap_history: list[float] = field(default_factory=list)
    co_retrieval_count: int = 0
    promotion_score: float = 0.0


@dataclass
class Region:
    """A node in the Mandorla memory graph. center + extent define its
    geometric body; lineage and citations record its position in the
    construction history."""

    id: str
    center: Vec
    extent: BoxExtent
    birth_time: float
    lineage: Lineage = field(default_factory=Lineage)
    citations: set[str] = field(default_factory=set)
    meta: RegionMeta = field(default_factory=RegionMeta)

    def is_primordial(self) -> bool:
        return self.lineage.parents is None


@dataclass
class Vesica(Region):
    """A Region whose lineage records exactly two non-null parents.

    Constructed via `make_vesica` after `box.intersect_boxes` has produced a
    non-degenerate intersection.
    """

    def parent_ids(self) -> tuple[str, str]:
        assert self.lineage.parents is not None, "Vesica must have two parents"
        return self.lineage.parents


def make_chunk_region(
    chunk_id: str,
    center: Vec,
    extent: BoxExtent,
    birth_time: Optional[float] = None,
) -> Region:
    """Construct a primordial Region for a Wikipedia chunk."""
    return Region(
        id=chunk_id,
        center=center.astype(np.float32, copy=False),
        extent=extent,
        birth_time=birth_time if birth_time is not None else time.monotonic(),
        lineage=Lineage(parents=None, depth=0),
    )


def make_vesica(
    parent_a: Region,
    parent_b: Region,
    extent: BoxExtent,
    birth_time: Optional[float] = None,
) -> Vesica:
    """Construct a Vesica from two parent Regions and a pre-computed
    intersection extent. Caller is responsible for computing the intersection
    box (see `box.intersect_boxes`); this factory wires up lineage and
    citations."""
    return Vesica(
        id=new_id(),
        center=extent.center,
        extent=extent,
        birth_time=birth_time if birth_time is not None else time.monotonic(),
        lineage=Lineage(
            parents=(parent_a.id, parent_b.id),
            depth=max(parent_a.lineage.depth, parent_b.lineage.depth) + 1,
        ),
        citations={parent_a.id, parent_b.id},
    )
