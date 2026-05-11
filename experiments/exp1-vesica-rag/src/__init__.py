"""Mandorla Experiment 1 slice — Region/Vesica primitives, GumbelBox math,
and the contriever→box projection.

See ../mandorla.md §2.2 for the reference data-model interface and
../PRECOMMIT.md for the frozen slice-specific decisions (64-D box space, B2
density-extent construction, in-query Vesicas only, etc.).
"""

from .box import (
    DEFAULT_BETA,
    box_log_volume,
    box_volume,
    expected_intersection_log_volume,
    expected_intersection_sides,
    expected_intersection_volume,
    intersect_boxes,
)
from .projection import DEFAULT_SEED, RandomProjection
from .regions import (
    BoxExtent,
    Lineage,
    Region,
    RegionMeta,
    Vec,
    Vesica,
    make_chunk_region,
    make_vesica,
    new_id,
)

__all__ = [
    "BoxExtent",
    "DEFAULT_BETA",
    "DEFAULT_SEED",
    "Lineage",
    "RandomProjection",
    "Region",
    "RegionMeta",
    "Vec",
    "Vesica",
    "box_log_volume",
    "box_volume",
    "expected_intersection_log_volume",
    "expected_intersection_sides",
    "expected_intersection_volume",
    "intersect_boxes",
    "make_chunk_region",
    "make_vesica",
    "new_id",
]
