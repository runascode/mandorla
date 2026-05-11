"""Run the baseline condition: contriever top-25 → Ollama → answer.

The baseline retrieves the top-25 passages by contriever cosine over the
full 5.2M-passage index, feeds them to llama3.1:8b-instruct-q5_K_M, and
records the answer. The chunk budget (25) matches the Vesica-RAG total cap
so the comparison is at matched context size (PRECOMMIT.md §"Baselines").

Resumable: re-running picks up where it left off (already-recorded ids are
skipped). Output: results/raw/baseline.jsonl

Requires scripts/04 (FAISS index over the full corpus) to have run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import REPO_ROOT, TitleIndex, load_hotpotqa
from src.generate import OllamaGenerator
from src.index_io import QueryEncoder, load_faiss_retriever
from src.runner import run_eval

K_RETRIEVE = 25
CONDITION = "baseline_contriever_top25"
OUT_PATH = REPO_ROOT / "results" / "raw" / "baseline.jsonl"
# Ollama generation dominates wall-clock; run several requests in parallel.
# Set OLLAMA_NUM_PARALLEL on the server to match (or rely on its default).
N_WORKERS = int(os.environ.get("MANDORLA_N_WORKERS", "4"))


def main() -> int:
    print("Loading FAISS retriever (loads ~16 GB into RAM)...", flush=True)
    retriever = load_faiss_retriever()
    print(f"  {retriever.n} passages indexed")

    print("Loading corpus title index (title↔chunk_id, chunk_id→passage)...", flush=True)
    titles = TitleIndex()
    # Force the id→idx map now so the first per-question lookup isn't slow.
    _ = titles.id_to_idx

    print("Loading query encoder (contriever)...", flush=True)
    encoder = QueryEncoder()

    print("Loading HotpotQA dev...", flush=True)
    questions = list(load_hotpotqa("validation"))
    print(f"  {len(questions)} dev questions")

    generator = OllamaGenerator()

    def retrieve(_q, q_vec):
        hits = retriever.top_k(q_vec, K_RETRIEVE)
        return (
            [h.chunk_id for h in hits],
            {"retrieval_scores": [round(h.score, 6) for h in hits]},
        )

    def chunk_text(chunk_id: str) -> tuple[str, str]:
        p = titles.passage(chunk_id)
        return p["title"], p["text"]

    print(f"\nRunning condition {CONDITION!r} → {OUT_PATH} (n_workers={N_WORKERS})", flush=True)
    run_eval(
        condition=CONDITION,
        questions=questions,
        encode_fn=encoder.encode,
        retrieve_fn=retrieve,
        chunk_text_fn=chunk_text,
        generator=generator,
        out_path=OUT_PATH,
        n_workers=N_WORKERS,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
