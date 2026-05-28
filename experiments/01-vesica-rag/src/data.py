"""Data loaders shared across scripts: HotpotQA questions, the BeIR corpus,
the contriever-shard manifest, and the title→chunk_id index.

This is the canonical place to add per-question pool assembly, per-chunk
text lookup, and supporting-fact matching. Keeping these in one place lets
the eval and the vesica-coverage diagnostic share exact lookup semantics.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INDEX_DIR = REPO_ROOT / "index"
CONTRIEVER_SHARD_DIR = INDEX_DIR / "contriever_shards"
CONTRIEVER_META_PATH = INDEX_DIR / "contriever_meta.json"


@dataclass(frozen=True)
class HotpotQuestion:
    """A flat HotpotQA record from scripts/01."""

    id: str
    question: str
    answer: str
    type: str               # "bridge" | "comparison"
    level: str              # "easy" | "medium" | "hard"
    supporting_titles: tuple[str, ...]   # the 2 (or rarely more) gold titles


def load_hotpotqa(split: str) -> Iterator[HotpotQuestion]:
    """Stream HotpotQuestion records from data/hotpotqa_{split}.jsonl."""
    path = DATA_DIR / f"hotpotqa_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/01_pull_hotpotqa.py first."
        )
    with path.open() as f:
        for line in f:
            ex = json.loads(line)
            titles = tuple(sorted({sf["title"] for sf in ex["supporting_facts"]}))
            yield HotpotQuestion(
                id=ex["id"],
                question=ex["question"],
                answer=ex["answer"],
                type=ex["type"],
                level=ex["level"],
                supporting_titles=titles,
            )


def load_corpus():
    """Load the BeIR/hotpotqa corpus from the HF cache. Returns a Dataset."""
    return load_dataset("BeIR/hotpotqa", "corpus", split="corpus")


class TitleIndex:
    """Bidirectional title ↔ chunk_id mapping over the BeIR/hotpotqa corpus.

    The corpus is one-passage-per-Wikipedia-article, so the mapping is
    title → exactly one chunk_id under our assumed schema. Verified by
    scripts/02 (audit recorded zero unmatched supporting-fact titles).
    """

    def __init__(self) -> None:
        self.corpus = load_corpus()
        self._title_to_id: Optional[dict[str, str]] = None
        self._id_to_idx: Optional[dict[str, int]] = None

    @cached_property
    def title_to_id(self) -> dict[str, str]:
        if self._title_to_id is None:
            self._title_to_id = {
                t: cid for cid, t in zip(self.corpus["_id"], self.corpus["title"])
            }
        return self._title_to_id

    @cached_property
    def id_to_idx(self) -> dict[str, int]:
        if self._id_to_idx is None:
            self._id_to_idx = {cid: i for i, cid in enumerate(self.corpus["_id"])}
        return self._id_to_idx

    def chunk_id_for_title(self, title: str) -> Optional[str]:
        return self.title_to_id.get(title)

    def passage(self, chunk_id: str) -> dict:
        """Return the {_id, title, text} record for a chunk_id."""
        idx = self.id_to_idx[chunk_id]
        return {
            "_id": self.corpus[idx]["_id"],
            "title": self.corpus[idx]["title"],
            "text": self.corpus[idx]["text"],
        }


@dataclass(frozen=True)
class ContrieverMeta:
    model: str
    embed_dim: int
    max_len: int
    shard_size: int
    passages_encoded: int
    device: str
    passage_format: str
    pool: str


def load_contriever_meta() -> ContrieverMeta:
    if not CONTRIEVER_META_PATH.exists():
        raise FileNotFoundError(
            f"{CONTRIEVER_META_PATH} not found. Run scripts/03_encode_corpus.py first."
        )
    d = json.loads(CONTRIEVER_META_PATH.read_text())
    return ContrieverMeta(**d)


def iter_shards() -> Iterator[tuple[int, Path]]:
    """Yield (shard_idx, path) pairs for all completed contriever shards,
    in order."""
    if not CONTRIEVER_SHARD_DIR.exists():
        return
    for path in sorted(CONTRIEVER_SHARD_DIR.glob("shard_*.npz")):
        idx = int(path.stem.split("_")[1])
        yield idx, path


def load_shard(path: Path) -> tuple[NDArray[np.float16], list[str]]:
    """Load (embeddings, chunk_ids) from a shard file."""
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], list(data["chunk_ids"])


def load_all_embeddings() -> tuple[NDArray[np.float32], list[str]]:
    """Load every encoded passage into memory as a single (N, 768) float32
    matrix and a parallel chunk_id list. Costs ~16 GB at 5.2M passages
    in float32 (8 GB at fp16 on disk, doubled on conversion). For the slice
    machine (48 GB RAM) this is fine; if it ever needs to scale, switch to
    memory-mapped access."""
    parts_e: list[NDArray] = []
    parts_ids: list[str] = []
    for _, path in iter_shards():
        e, ids = load_shard(path)
        parts_e.append(e.astype(np.float32))
        parts_ids.extend(ids)
    if not parts_e:
        raise RuntimeError("No contriever shards found.")
    return np.concatenate(parts_e, axis=0), parts_ids
