"""Loaders for the three multi-hop QA dev sets used by Experiment 02.

We normalize three different on-disk schemas to a common
`MultiHopQuestion` so the eval and corpus-coverage code don't branch on
dataset. The common fields are exactly what the retrieval-isolation
metrics need (gold_titles for title-match scoring against the slice
corpus, plus a canonical id for resumable JSONL output).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class MultiHopQuestion:
    id: str
    dataset: str  # "hotpotqa" | "2wiki" | "musique"
    question: str
    answer: str
    gold_titles: tuple[str, ...]  # canonical supporting-article titles
    n_hops: int


def _norm_title(s: str) -> str:
    """Title normalization for cross-dataset matching against the corpus.

    The HotpotQA Wikipedia abstract dump's titles are e.g. 'Albert
    Einstein', 'Tucker Carlson' — Title Case, English. 2Wiki and MuSiQue
    sometimes carry parenthetical disambiguations or underscores; we
    strip those for matching.
    """
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_hotpotqa_dev() -> Iterator[MultiHopQuestion]:
    """HotpotQA dev: same 7,405 questions used in Exp 01. We load via the
    distractor split's `validation` (same question set as fullwiki) and
    keep only the question text + gold titles."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    for r in ds:
        # `supporting_facts` is {'title': [...], 'sent_id': [...]}
        titles = tuple(dict.fromkeys(_norm_title(t) for t in r["supporting_facts"]["title"]))
        yield MultiHopQuestion(
            id=r["id"],
            dataset="hotpotqa",
            question=r["question"],
            answer=r["answer"],
            gold_titles=titles,
            n_hops=len(titles),
        )


_DATA_DIR = None  # populated lazily relative to this module


def _data_dir() -> "Path":
    """Local cache for the raw JSON/JSONL files we pull outside the HF
    datasets machinery. HF's `load_dataset` no longer supports
    loading-script datasets; both 2Wiki and MuSiQue ship their dev splits
    as plain JSON/JSONL on the Hub, which we fetch directly. This avoids
    a moving target as the `datasets` library evolves."""
    from pathlib import Path

    global _DATA_DIR
    if _DATA_DIR is None:
        _DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def _fetch(url: str, local_name: str) -> "Path":
    """Idempotent download. Returns the local path."""
    import urllib.request

    dst = _data_dir() / local_name
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"  downloading {url} -> {dst}", flush=True)
    urllib.request.urlretrieve(url, dst)
    return dst


# Raw-file URLs (pinned implicitly by the file's content-hash on the Hub;
# updates would change the on-disk cache).
_2WIKI_DEV_URL = "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json"
_MUSIQUE_DEV_URL = (
    "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/"
    "musique_ans_v1.0_dev.jsonl"
)


def load_2wiki_dev() -> Iterator[MultiHopQuestion]:
    """2WikiMultiHopQA dev. Each item has `supporting_facts` listing the
    titles of supporting paragraphs.

    Loaded from the raw `dev.json` published on `voidful/2WikiMultihopQA`
    (an array of question dicts). We avoid `datasets.load_dataset` here
    because the original dataset script (`xanhho/2WikiMultihopQA`) is no
    longer supported by recent `datasets` versions.
    """
    import json as _json

    path = _fetch(_2WIKI_DEV_URL, "2wiki_dev.json")
    with path.open() as f:
        items = _json.load(f)
    for r in items:
        sf = r.get("supporting_facts") or []
        # supporting_facts is a list of [title, sent_id] tuples
        titles = tuple(dict.fromkeys(_norm_title(t) for t, _ in sf))
        yield MultiHopQuestion(
            id=str(r.get("_id") or r.get("id")),
            dataset="2wiki",
            question=r["question"],
            answer=r.get("answer", ""),
            gold_titles=titles,
            n_hops=len(titles),
        )


def load_musique_dev() -> Iterator[MultiHopQuestion]:
    """MuSiQue-Ans dev. Each item has `paragraphs` (list of {idx, title,
    paragraph_text, is_supporting}). Gold titles are those with
    `is_supporting=True`.

    Loaded from the raw `musique_ans_v1.0_dev.jsonl` on the
    `dgslibisey/MuSiQue` HF dataset (LFS-tracked JSONL). Same
    rationale as 2Wiki: avoids the loading-script deprecation.
    """
    import json as _json

    path = _fetch(_MUSIQUE_DEV_URL, "musique_ans_dev.jsonl")
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _json.loads(line)
            paragraphs = r.get("paragraphs") or []
            sup_titles = [
                _norm_title(p["title"]) for p in paragraphs if p.get("is_supporting")
            ]
            titles = tuple(dict.fromkeys(sup_titles))
            yield MultiHopQuestion(
                id=str(r.get("id")),
                dataset="musique",
                question=r["question"],
                answer=str(r.get("answer", "")),
                gold_titles=titles,
                n_hops=len(titles),
            )


DATASET_LOADERS: dict[str, callable] = {
    "hotpotqa": load_hotpotqa_dev,
    "2wiki": load_2wiki_dev,
    "musique": load_musique_dev,
}


def load_dataset_by_name(name: str) -> list[MultiHopQuestion]:
    loader = DATASET_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name!r} (choose from {sorted(DATASET_LOADERS)})")
    return list(loader())


def normalize_title(s: str) -> str:
    """Public alias for cross-module title matching."""
    return _norm_title(s)
