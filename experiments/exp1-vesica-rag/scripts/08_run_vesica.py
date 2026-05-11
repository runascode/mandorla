"""Run the Vesica-RAG condition.

Per-question cognitive cycle (PRECOMMIT.md §"Architecture spec for the
slice"):

  encode → contriever top-20 descent → C(20,2)=190 pairwise GumbelBox
  intersections → drop those below τ_v → score the rest by
  log E[volume] + log(cos(q_64, V.center) + 1) → keep top-5 candidate
  Vesicas → populate each Vesica's contained-chunk set (cap 10, parents
  always included) → union with the top-5 contriever points, dedup, cap
  25 chunks total → feed to llama3.1:8b-instruct-q5_K_M → record.

In addition to the answer, the record carries the candidate-Vesica parent
pairs and the gold supporting-paragraph chunk ids, so the scoring script
can compute the vesica-coverage diagnostic without re-running retrieval.

Resumable. Output: results/raw/vesica.jsonl

Requires scripts/04 (FAISS index), scripts/05 (box index + projection),
and scripts/06 (τ_v calibration) to have run against the full corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import REPO_ROOT, TitleIndex, load_hotpotqa
from src.generate import OllamaGenerator
from src.index_io import (
    QueryEncoder,
    load_box_store,
    load_faiss_retriever,
    load_projection,
    load_tau_v,
)
from src.retrieve import VesicaRetriever
from src.runner import run_eval

# Slice retrieval params — frozen in PRECOMMIT.md §"Architecture spec".
K_POINTS_USED = 5
K_DESCENT = 20
M_VESICAS = 5
MAX_CHUNKS_PER_VESICA = 10
MAX_TOTAL_CHUNKS = 25

CONDITION = "vesica_rag"
OUT_PATH = REPO_ROOT / "results" / "raw" / "vesica.jsonl"


def main() -> int:
    print("Loading FAISS retriever (loads ~16 GB into RAM)...", flush=True)
    faiss_retriever = load_faiss_retriever()
    print(f"  {faiss_retriever.n} passages indexed")

    print("Loading 64-D box store...", flush=True)
    boxes = load_box_store()
    print(f"  {boxes.n} boxes, d={boxes.d_box}")

    print("Loading random projection...", flush=True)
    projection = load_projection()

    tau_v = load_tau_v()
    if tau_v is None:
        print("ERROR: τ_v not calibrated. Run scripts/06_calibrate_tau_v.py first.")
        return 1
    print(f"  τ_v (log-volume) = {tau_v:.6g}")

    print("Loading corpus title index...", flush=True)
    titles = TitleIndex()
    _ = titles.id_to_idx
    _ = titles.title_to_id

    print("Loading query encoder (contriever)...", flush=True)
    encoder = QueryEncoder()

    print("Loading HotpotQA dev...", flush=True)
    questions = list(load_hotpotqa("validation"))
    print(f"  {len(questions)} dev questions")

    generator = OllamaGenerator()
    vr = VesicaRetriever(
        baseline=faiss_retriever,
        boxes=boxes,
        projection=projection,
        min_log_volume=tau_v,
    )

    def gold_chunk_ids(q) -> list[str]:
        return [titles.title_to_id[t] for t in q.supporting_titles if t in titles.title_to_id]

    def retrieve(q, q_vec):
        res = vr.retrieve(
            q_vec,
            k_points_used=K_POINTS_USED,
            k_descent=K_DESCENT,
            m_vesicas=M_VESICAS,
            max_chunks_per_vesica=MAX_CHUNKS_PER_VESICA,
            max_total_chunks=MAX_TOTAL_CHUNKS,
        )
        return (
            list(res.retrieved_chunk_ids),
            {
                "point_chunk_ids": [p.chunk_id for p in res.points],
                "candidate_vesica_parents": [
                    [v.parent_a_id, v.parent_b_id] for v in res.vesicas
                ],
                "vesica_log_scores": [round(v.log_score, 6) for v in res.vesicas],
                "vesica_contained_chunk_ids": [
                    list(v.contained_chunk_ids) for v in res.vesicas
                ],
                "gold_chunk_ids": gold_chunk_ids(q),
            },
        )

    def chunk_text(chunk_id: str) -> tuple[str, str]:
        p = titles.passage(chunk_id)
        return p["title"], p["text"]

    print(f"\nRunning condition {CONDITION!r} → {OUT_PATH}", flush=True)
    run_eval(
        condition=CONDITION,
        questions=questions,
        encode_fn=encoder.encode,
        retrieve_fn=retrieve,
        chunk_text_fn=chunk_text,
        generator=generator,
        out_path=OUT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
