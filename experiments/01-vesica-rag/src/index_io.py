"""Loaders for the built indices and the contriever question encoder.

scripts/06–09 use this to turn the on-disk artifacts into ready-to-use
objects:

  - the FAISS contriever index + parallel chunk_ids (scripts/04 output)
  - the 64-D box index (centers, half_widths, chunk_ids) (scripts/05 output)
  - the seeded random projection matrix (scripts/05 output)
  - the calibrated τ_v threshold, if scripts/06 has run
  - a `QueryEncoder` that encodes HotpotQA questions with the *same* model
    and *same* pooling/truncation the corpus was encoded with — this match
    is load-bearing for retrieval to make sense.

The QueryEncoder uses `max_len=128` to match the corpus encode
(`scripts/03_encode_corpus.py`, see `BENCHMARKS.md`). Same model, same
mean-pooling over non-pad tokens (Izacard 2021 §3.2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer

from .data import INDEX_DIR
from .projection import RandomProjection
from .retrieve import BoxStore, FaissDenseRetriever

CONTRIEVER_MODEL = "facebook/contriever-msmarco"
QUERY_MAX_LEN = 128   # match scripts/03_encode_corpus.py

# Artifact paths (relative to INDEX_DIR)
FAISS_PATH = INDEX_DIR / "contriever.faiss"
FAISS_IDS_PATH = INDEX_DIR / "chunk_ids.npy"
BOX_CENTERS_PATH = INDEX_DIR / "box_centers.npy"
BOX_HALF_PATH = INDEX_DIR / "box_half_widths.npy"
BOX_IDS_PATH = INDEX_DIR / "box_chunk_ids.npy"
PROJECTION_PATH = INDEX_DIR / "projection.npz"
TAU_V_PATH = INDEX_DIR / "tau_v.json"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class QueryEncoder:
    """Encodes HotpotQA questions with contriever, matching the corpus encode.

    `encode(text)` → (768,) float32. `encode_batch(texts)` → (B, 768).
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_MODEL)
        self.model = AutoModel.from_pretrained(CONTRIEVER_MODEL).to(self.device).eval()

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=QUERY_MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb.detach().to(torch.float32).cpu().numpy()

    def encode(self, text: str) -> NDArray[np.float32]:
        return self._encode([text])[0]

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> NDArray[np.float32]:
        out: list[NDArray[np.float32]] = []
        for start in range(0, len(texts), batch_size):
            out.append(self._encode(texts[start:start + batch_size]))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype=np.float32)


def load_faiss_retriever() -> FaissDenseRetriever:
    """Load the FAISS IndexFlatIP + chunk_ids built by scripts/04."""
    import faiss  # local import: faiss pulls in a heavy native lib

    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"{FAISS_PATH} not found. Run scripts/04_build_faiss.py.")
    index = faiss.read_index(str(FAISS_PATH))
    chunk_ids = list(np.load(FAISS_IDS_PATH, allow_pickle=True))
    return FaissDenseRetriever(index=index, chunk_ids=chunk_ids)


def load_box_store() -> BoxStore:
    """Load the 64-D box index built by scripts/05."""
    for p in (BOX_CENTERS_PATH, BOX_HALF_PATH, BOX_IDS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts/05_build_box_index.py.")
    centers = np.load(BOX_CENTERS_PATH).astype(np.float32)
    half_widths = np.load(BOX_HALF_PATH).astype(np.float32)
    chunk_ids = list(np.load(BOX_IDS_PATH, allow_pickle=True))
    return BoxStore(centers=centers, half_widths=half_widths, chunk_ids=chunk_ids)


def load_projection() -> RandomProjection:
    """Load the seeded random projection built by scripts/05."""
    if not PROJECTION_PATH.exists():
        raise FileNotFoundError(f"{PROJECTION_PATH} not found. Run scripts/05_build_box_index.py.")
    return RandomProjection.load(PROJECTION_PATH)


def load_tau_v() -> Optional[float]:
    """Load the calibrated τ_v (expected-intersection log-volume threshold)
    from scripts/06. Returns None if calibration hasn't been run yet."""
    if not TAU_V_PATH.exists():
        return None
    return float(json.loads(TAU_V_PATH.read_text())["tau_v_log_volume"])


def save_tau_v(tau_v_log_volume: float, meta: dict) -> Path:
    """Persist the calibrated τ_v + its calibration metadata."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"tau_v_log_volume": float(tau_v_log_volume), **meta}
    TAU_V_PATH.write_text(json.dumps(payload, indent=2))
    return TAU_V_PATH
