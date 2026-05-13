"""Retrieval-only evaluation across the three datasets.

For each dev question on each dataset, produces TWO ranked lists of up
to 25 chunk_ids (with their titles): one from contriever-baseline, one
from Vesica-augmented. NO LLM is called.

Reuses Exp 01's FAISS index, 64-D box store, random projection, τ_v,
and QueryEncoder bit-for-bit — no re-indexing. The sys.path hack at the
top of this script imports from `../exp1-vesica-rag/src/`.

Resumable: per-question JSONL append; re-running skips ids already in
the output file. One JSONL per dataset.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_ROOT))
os.environ.setdefault("HF_HOME", str(EXP_ROOT / "data" / "hf_home"))

# NB: torch must load before faiss (see project LAB-NOTES Lesson #4). The
# bridge loads exp01.index_io which pulls in torch transitively, then we
# touch faiss only later when load_faiss_retriever() is called.
from src import _exp01_bridge as bridge  # noqa: E402
from src.datasets import (  # noqa: E402
    DATASET_LOADERS,
    MultiHopQuestion,
    normalize_title,
)

TitleIndex = bridge.exp01_data.TitleIndex
QueryEncoder = bridge.exp01_index_io.QueryEncoder
load_box_store = bridge.exp01_index_io.load_box_store
load_faiss_retriever = bridge.exp01_index_io.load_faiss_retriever
load_projection = bridge.exp01_index_io.load_projection
load_tau_v = bridge.exp01_index_io.load_tau_v
VesicaRetriever = bridge.exp01_retrieve.VesicaRetriever

# Same architectural constants as the slice — frozen in Exp 01 PRECOMMIT
# §"Architecture spec for the slice" and inherited by Exp 02 PRECOMMIT §G.
K_BASELINE = 25
K_POINTS_USED = 5
K_DESCENT = 20
M_VESICAS = 5
MAX_CHUNKS_PER_VESICA = 10
MAX_TOTAL_CHUNKS = 25

RAW_DIR = EXP_ROOT / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LOG_EVERY = 50


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        lines = f.readlines()
    for line in lines[:-1]:  # all but the last (may be torn on crash)
        try:
            done.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            pass
    if lines:
        try:
            done.add(json.loads(lines[-1])["id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return done


def _titles_for(chunk_ids, titles: TitleIndex) -> list[str]:
    out: list[str] = []
    for cid in chunk_ids:
        try:
            out.append(normalize_title(titles.passage(cid)["title"]))
        except KeyError:
            out.append("")
    return out


def _evaluate_one_dataset(
    name: str,
    questions: list[MultiHopQuestion],
    *,
    faiss_retriever,
    vesica_retriever: VesicaRetriever,
    encoder: QueryEncoder,
    titles: TitleIndex,
    out_path: Path,
) -> None:
    done = _load_done_ids(out_path)
    todo = [q for q in questions if q.id not in done]
    print(
        f"  [{name}] {len(questions)} total; {len(done)} already done; {len(todo)} to process",
        flush=True,
    )
    t0 = time.monotonic()
    with out_path.open("a") as f:
        for i, q in enumerate(todo, start=1):
            q_vec = encoder.encode(q.question)
            base_top = faiss_retriever.top_k(q_vec, K_BASELINE)
            base_ids = [s.chunk_id for s in base_top]
            base_titles = _titles_for(base_ids, titles)

            ves = vesica_retriever.retrieve(
                q_vec,
                k_points_used=K_POINTS_USED,
                k_descent=K_DESCENT,
                m_vesicas=M_VESICAS,
                max_chunks_per_vesica=MAX_CHUNKS_PER_VESICA,
                max_total_chunks=MAX_TOTAL_CHUNKS,
            )
            ves_ids = list(ves.retrieved_chunk_ids)
            ves_titles = _titles_for(ves_ids, titles)
            cand_parent_titles = [
                [
                    normalize_title(titles.passage(v.parent_a_id)["title"]) if v.parent_a_id else "",
                    normalize_title(titles.passage(v.parent_b_id)["title"]) if v.parent_b_id else "",
                ]
                for v in ves.vesicas
            ]
            record = {
                "id": q.id,
                "dataset": name,
                "question": q.question,
                "answer": q.answer,
                "n_hops": q.n_hops,
                "gold_titles": list(q.gold_titles),
                "baseline_chunk_ids": base_ids,
                "baseline_titles": base_titles,
                "vesica_chunk_ids": ves_ids,
                "vesica_titles": ves_titles,
                "candidate_vesica_parent_titles": cand_parent_titles,
            }
            f.write(json.dumps(record) + "\n")
            if i % LOG_EVERY == 0 or i == len(todo):
                dt = time.monotonic() - t0
                qps = i / dt if dt > 0 else 0.0
                eta_min = (len(todo) - i) / qps / 60.0 if qps > 0 else float("nan")
                print(
                    f"    [{name}] {i}/{len(todo)} ({qps:.2f} q/s, eta {eta_min:.1f} min)",
                    flush=True,
                )
                f.flush()


def main() -> int:
    print("Loading FAISS retriever (~16 GB resident)...", flush=True)
    faiss_retriever = load_faiss_retriever()
    print(f"  {faiss_retriever.n} passages indexed")

    print("Loading 64-D box store...", flush=True)
    boxes = load_box_store()
    print(f"  {boxes.n} boxes, d={boxes.d_box}")

    print("Loading random projection + τ_v...", flush=True)
    projection = load_projection()
    tau_v = load_tau_v()
    if tau_v is None:
        raise RuntimeError("τ_v not on disk — scripts/06 in exp01 must have run first")
    print(f"  τ_v (log-volume) = {tau_v:.4f}")

    print("Loading corpus title index (5.2M titles)...", flush=True)
    titles = TitleIndex()

    print("Loading query encoder (contriever)...", flush=True)
    encoder = QueryEncoder()

    vesica_retriever = VesicaRetriever(
        baseline=faiss_retriever,
        boxes=boxes,
        projection=projection,
        min_log_volume=tau_v,
    )

    for name in ["hotpotqa", "2wiki", "musique"]:
        print(f"\n=== {name} ===", flush=True)
        qs = list(DATASET_LOADERS[name]())
        out_path = RAW_DIR / f"{name}.jsonl"
        _evaluate_one_dataset(
            name,
            qs,
            faiss_retriever=faiss_retriever,
            vesica_retriever=vesica_retriever,
            encoder=encoder,
            titles=titles,
            out_path=out_path,
        )
        print(f"  -> {out_path.relative_to(EXP_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
