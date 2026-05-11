"""Retrieval logic — baseline contriever top-k and Vesica-RAG.

This module implements the per-question cognitive cycle from PRECOMMIT.md
§"Architecture spec for the slice":

  invocation → descent → vesica-search → candidate-selection → retrieval-union

It is written to operate against in-memory arrays. The index-building scripts
(04, 05) produce the arrays; this module consumes them. That separation keeps
retrieval logic unit-testable without requiring 16 GB of real embeddings.

All scoring is in log-volume space to avoid underflow in 64-D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .box import (
    DEFAULT_BETA,
    expected_intersection_log_volume,
    intersect_boxes,
)
from .projection import RandomProjection
from .regions import BoxExtent, Vec

if TYPE_CHECKING:  # pragma: no cover - typing only
    import faiss


# ─── Box store (boxes per chunk_id, in 64-D) ────────────────────────────────


@dataclass
class BoxStore:
    """In-memory store of per-chunk BoxExtents in 64-D.

    Layout: parallel arrays — centers (N, d_box), half_widths (N, d_box),
    chunk_ids (length N). Constructed by scripts/05_build_box_index.py.
    """

    centers: NDArray[np.float32]        # (N, d_box)
    half_widths: NDArray[np.float32]    # (N, d_box)
    chunk_ids: list[str]

    def __post_init__(self) -> None:
        if self.centers.shape != self.half_widths.shape:
            raise ValueError(
                f"shape mismatch: centers {self.centers.shape} vs "
                f"half_widths {self.half_widths.shape}"
            )
        if self.centers.shape[0] != len(self.chunk_ids):
            raise ValueError(
                f"row count mismatch: {self.centers.shape[0]} centers vs "
                f"{len(self.chunk_ids)} chunk_ids"
            )

    @property
    def n(self) -> int:
        return self.centers.shape[0]

    @property
    def d_box(self) -> int:
        return self.centers.shape[1]

    def box_for_index(self, idx: int) -> BoxExtent:
        c = self.centers[idx]
        h = self.half_widths[idx]
        return BoxExtent(
            min_corner=(c - h).astype(np.float32),
            max_corner=(c + h).astype(np.float32),
        )

    def containing_indices(
        self,
        vesica: BoxExtent,
        cap: Optional[int] = None,
    ) -> NDArray[np.intp]:
        """Indices of chunks whose centers lie inside the Vesica box.

        If `cap` is provided, return at most `cap` indices, ranked by
        distance from chunk center to Vesica center (closer first). The
        unranked variant is useful for the vesica-coverage diagnostic.
        """
        mn = vesica.min_corner
        mx = vesica.max_corner
        inside = ((self.centers >= mn) & (self.centers <= mx)).all(axis=1)
        idxs = np.nonzero(inside)[0]
        if cap is None or idxs.size <= cap:
            return idxs
        v_center = vesica.center
        dists = np.linalg.norm(self.centers[idxs] - v_center, axis=1)
        order = np.argsort(dists)[:cap]
        return idxs[order]


# ─── Baseline retriever (contriever top-k over 768-D embeddings) ────────────


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float           # cosine similarity


@runtime_checkable
class DenseRetriever(Protocol):
    """Anything that returns top-k cosine-similar chunks for a 768-D query.

    Both `BaselineRetriever` (numpy reference, used in tests) and
    `FaissDenseRetriever` (production, used by scripts 07/08) satisfy this.
    `VesicaRetriever` is written against this Protocol so it doesn't care
    which is plugged in.
    """

    def top_k(self, query: Vec, k: int) -> list[ScoredChunk]:
        ...


@dataclass
class BaselineRetriever:
    """Top-k cosine retrieval over a contriever-encoded corpus — numpy
    reference implementation.

    Operates in float32, uses np.argpartition for top-k without a full
    sort. The testing surface for retrieval logic; the production path at
    5.2M passages uses `FaissDenseRetriever` instead (avoids holding a
    16 GB embedding matrix when the FAISS index already has it).

    Assumes embeddings are NOT pre-normalized; normalizes both sides for
    cosine.
    """

    embeddings: NDArray[np.float32]    # (N, 768)
    chunk_ids: list[str]

    def __post_init__(self) -> None:
        if self.embeddings.shape[0] != len(self.chunk_ids):
            raise ValueError(
                f"embeddings rows {self.embeddings.shape[0]} != "
                f"len(chunk_ids) {len(self.chunk_ids)}"
            )

    @property
    def n(self) -> int:
        return self.embeddings.shape[0]

    @property
    def d(self) -> int:
        return self.embeddings.shape[1]

    def top_k(self, query: Vec, k: int) -> list[ScoredChunk]:
        if query.shape[0] != self.d:
            raise ValueError(f"query dim {query.shape[0]} != index dim {self.d}")
        q = query.astype(np.float32, copy=False)
        q_norm = q / max(float(np.linalg.norm(q)), 1e-12)
        e_norms = np.linalg.norm(self.embeddings, axis=1).clip(min=1e-12)
        e_hat = self.embeddings / e_norms[:, None]
        scores = e_hat @ q_norm
        k = min(k, self.n)
        idx_part = np.argpartition(-scores, kth=k - 1)[:k]
        idx_sorted = idx_part[np.argsort(-scores[idx_part])]
        return [
            ScoredChunk(chunk_id=self.chunk_ids[int(i)], score=float(scores[int(i)]))
            for i in idx_sorted
        ]


@dataclass
class FaissDenseRetriever:
    """Top-k cosine retrieval backed by a FAISS IndexFlatIP over
    L2-normalized vectors (built by scripts/04). Production path for the
    5.2M-passage corpus — exact cosine, no ANN approximation, ~16 GB
    resident.

    The query is normalized before search so inner-product == cosine.
    """

    index: "faiss.Index"
    chunk_ids: list[str]

    def __post_init__(self) -> None:
        if self.index.ntotal != len(self.chunk_ids):
            raise ValueError(
                f"index.ntotal {self.index.ntotal} != len(chunk_ids) "
                f"{len(self.chunk_ids)}"
            )

    @property
    def n(self) -> int:
        return self.index.ntotal

    def top_k(self, query: Vec, k: int) -> list[ScoredChunk]:
        q = query.astype(np.float32, copy=False)
        q_norm = (q / max(float(np.linalg.norm(q)), 1e-12)).reshape(1, -1)
        k = min(k, self.n)
        scores, idxs = self.index.search(q_norm, k)
        scores, idxs = scores[0], idxs[0]
        return [
            ScoredChunk(chunk_id=self.chunk_ids[int(i)], score=float(s))
            for s, i in zip(scores, idxs)
            if i >= 0
        ]


# ─── Candidate Vesica (a scored intersection of two retrieved chunks) ───────


@dataclass(frozen=True)
class CandidateVesica:
    """A scored candidate Vesica produced during retrieval.

    parent_a_id, parent_b_id — the two contriever-retrieved chunks that birthed
    this Vesica. score — E[volume] × cos(query_64, vesica_center). box — the
    BoxExtent of the intersection in 64-D. contained_chunk_ids — chunks whose
    boxes contain a center inside this Vesica's box (the C1 contract from
    PRECOMMIT.md).
    """

    parent_a_id: str
    parent_b_id: str
    box: BoxExtent
    log_volume: float
    log_score: float
    contained_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class VesicaRetrievalResult:
    """Output of one VesicaRetriever.retrieve() call.

    `points` — the top-`k_points` contriever chunks (the "Descent" output).
    `vesicas` — the top-`m_vesicas` candidate Vesicas, sorted by score desc.
    `retrieved_chunk_ids` — the union, deduplicated and capped, that will be
                             fed to the LLM.
    """

    points: tuple[ScoredChunk, ...]
    vesicas: tuple[CandidateVesica, ...]
    retrieved_chunk_ids: tuple[str, ...]


# ─── Vesica retriever ───────────────────────────────────────────────────────


@dataclass
class VesicaRetriever:
    """The slice's Vesica-RAG retriever.

    Composes:
      - a `DenseRetriever` for the Descent step (contriever top-k points) —
        `BaselineRetriever` in tests, `FaissDenseRetriever` in production
      - `BoxStore` for box geometry per chunk_id
      - `RandomProjection` for query 768-D → 64-D
    """

    baseline: DenseRetriever
    boxes: BoxStore
    projection: RandomProjection
    beta: float = DEFAULT_BETA
    min_log_volume: Optional[float] = None   # τ_v in log space

    # Index from chunk_id → row in BoxStore.centers, built lazily once.
    _chunk_id_to_box_idx: dict[str, int] = field(default_factory=dict, repr=False)

    def _box_idx(self, chunk_id: str) -> int:
        if not self._chunk_id_to_box_idx:
            self._chunk_id_to_box_idx = {cid: i for i, cid in enumerate(self.boxes.chunk_ids)}
        return self._chunk_id_to_box_idx[chunk_id]

    def retrieve(
        self,
        query_768: Vec,
        k_points_used: int = 5,
        k_descent: int = 20,
        m_vesicas: int = 5,
        max_chunks_per_vesica: int = 10,
        max_total_chunks: int = 25,
    ) -> VesicaRetrievalResult:
        """Run the full slice cognitive cycle for one query.

        Args:
          query_768: contriever-encoded query (768-D float32)
          k_points_used: how many of the descent top-k to include in the
                         final retrieval union (the "top-5 contriever points"
                         per PRECOMMIT.md)
          k_descent: how many top-k descent points to consider for pairwise
                     Vesica candidates (the C(k_descent,2) pair count)
          m_vesicas: how many top-scoring candidate Vesicas to keep
          max_chunks_per_vesica: cap on chunks per Vesica's contained set
                                 (PRECOMMIT.md C1)
          max_total_chunks: cap on the final union fed to the LLM
        """
        # --- DESCENT ---
        descent = self.baseline.top_k(query_768, k_descent)

        # --- VESICA SEARCH (pairwise intersection scoring over descent) ---
        query_64 = self.projection.project(query_768.astype(np.float32))
        candidates: list[CandidateVesica] = []
        for a, b in combinations(descent, 2):
            ai = self._box_idx(a.chunk_id)
            bi = self._box_idx(b.chunk_id)
            box_a = self.boxes.box_for_index(ai)
            box_b = self.boxes.box_for_index(bi)
            log_vol = expected_intersection_log_volume(box_a, box_b, beta=self.beta)
            if self.min_log_volume is not None and log_vol < self.min_log_volume:
                continue
            # Intersection box (closed-form softplus)
            v_box = intersect_boxes(box_a, box_b, beta=self.beta)
            if v_box is None:
                continue
            # score = E[volume] × cos(query_64, v_center)
            v_center = v_box.center
            q_norm = float(np.linalg.norm(query_64))
            c_norm = float(np.linalg.norm(v_center))
            if q_norm > 0 and c_norm > 0:
                cos = float(np.dot(query_64, v_center) / (q_norm * c_norm))
            else:
                cos = 0.0
            # cosine ∈ [-1, 1]; pull into (0, 2] before log so negative cosines
            # don't make log_score undefined. score order is preserved.
            log_score = log_vol + float(np.log(max(cos + 1.0, 1e-12)))
            candidates.append(
                CandidateVesica(
                    parent_a_id=a.chunk_id,
                    parent_b_id=b.chunk_id,
                    box=v_box,
                    log_volume=log_vol,
                    log_score=log_score,
                    contained_chunk_ids=(),  # populated below
                )
            )

        # --- CANDIDATE SELECTION ---
        candidates.sort(key=lambda c: c.log_score, reverse=True)
        candidates = candidates[:m_vesicas]

        # --- RETRIEVAL UNION ---
        # For each candidate Vesica, populate contained_chunk_ids (parents
        # always included; cap at max_chunks_per_vesica total).
        final_candidates: list[CandidateVesica] = []
        union: list[str] = [s.chunk_id for s in descent[:k_points_used]]
        seen: set[str] = set(union)
        for cand in candidates:
            contained_idx = self.boxes.containing_indices(
                cand.box, cap=max_chunks_per_vesica
            )
            contained_ids = [self.boxes.chunk_ids[int(i)] for i in contained_idx]
            # Parents always included
            for pid in (cand.parent_a_id, cand.parent_b_id):
                if pid not in contained_ids:
                    contained_ids.insert(0, pid)
            # Trim to cap (parents have priority via the insert)
            contained_ids = contained_ids[:max_chunks_per_vesica]
            final_candidates.append(
                CandidateVesica(
                    parent_a_id=cand.parent_a_id,
                    parent_b_id=cand.parent_b_id,
                    box=cand.box,
                    log_volume=cand.log_volume,
                    log_score=cand.log_score,
                    contained_chunk_ids=tuple(contained_ids),
                )
            )
            for cid in contained_ids:
                if cid not in seen and len(union) < max_total_chunks:
                    union.append(cid)
                    seen.add(cid)

        return VesicaRetrievalResult(
            points=tuple(descent[:k_points_used]),
            vesicas=tuple(final_candidates),
            retrieved_chunk_ids=tuple(union[:max_total_chunks]),
        )
