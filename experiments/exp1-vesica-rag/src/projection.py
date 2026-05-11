"""Fixed Gaussian random projection from contriever's 768-D output to the
64-D box space.

Per PRECOMMIT.md decision A (64-D box space) and decision F (seeded
reproducibility). The projection matrix is materialized once from seed=1337
and is bit-for-bit reproducible. It is also serialized to disk so subsequent
loads avoid even the cost of re-deriving it (and the load path verifies the
on-disk matrix matches the seed-derived matrix as a paranoia check).

Why random projection rather than PCA: PRECOMMIT.md A — PCA on 5.2M passages
requires a fit step that itself takes hours and introduces a second source of
variance. Random Gaussian projection has Johnson-Lindenstrauss guarantees at
64 target dimensions and is reproducible from the seed alone. PCA vs.
random-projection becomes an ablation in the broader Experiment 1.

Matrix entries are drawn from N(0, 1/sqrt(d_out)) so the expected squared norm
of a projected vector matches the input squared norm in expectation (cosine
similarity is approximately preserved at the JL bound).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_SEED: int = 1337


class RandomProjection:
    """Deterministic Gaussian random projection v_in → v_out.

    Construct with `RandomProjection(d_in, d_out, seed)` to derive the matrix
    fresh. Use `.save(path)` to serialize and `.load(path)` to restore. The
    load path re-derives the matrix from the stored seed and compares — if
    they don't match, the file is corrupt or the implementation changed.
    """

    def __init__(self, d_in: int = 768, d_out: int = 64, seed: int = DEFAULT_SEED):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_out)
        self.matrix: NDArray[np.float32] = (
            rng.standard_normal((d_out, d_in), dtype=np.float32) * scale
        ).astype(np.float32)
        self.d_in = d_in
        self.d_out = d_out
        self.seed = seed

    def project(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Project a single vector (shape (d_in,)) or a batch (shape
        (B, d_in)). Returns the same leading-dim structure projected to d_out.
        """
        if vectors.ndim == 1:
            if vectors.shape[0] != self.d_in:
                raise ValueError(f"Expected dim {self.d_in}; got {vectors.shape[0]}")
            return (self.matrix @ vectors.astype(np.float32, copy=False)).astype(np.float32)
        if vectors.ndim == 2:
            if vectors.shape[1] != self.d_in:
                raise ValueError(f"Expected dim {self.d_in}; got {vectors.shape[1]}")
            return (vectors.astype(np.float32, copy=False) @ self.matrix.T).astype(np.float32)
        raise ValueError(f"Expected 1-D or 2-D array; got {vectors.ndim}-D")

    def save(self, path: str | Path) -> None:
        """Serialize the projection matrix and its derivation parameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            matrix=self.matrix,
            d_in=np.array(self.d_in),
            d_out=np.array(self.d_out),
            seed=np.array(self.seed),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RandomProjection":
        """Load a serialized projection. Re-derives the matrix from the
        stored seed and verifies it matches; raises if not."""
        data = np.load(Path(path))
        obj = cls.__new__(cls)
        obj.matrix = data["matrix"].astype(np.float32)
        obj.d_in = int(data["d_in"])
        obj.d_out = int(data["d_out"])
        obj.seed = int(data["seed"])

        expected = cls(d_in=obj.d_in, d_out=obj.d_out, seed=obj.seed)
        if not np.array_equal(obj.matrix, expected.matrix):
            raise RuntimeError(
                "Loaded projection matrix does not match the deterministic "
                "derivation from its declared seed. Reproducibility is broken; "
                "regenerate the projection or fix the implementation."
            )
        return obj
