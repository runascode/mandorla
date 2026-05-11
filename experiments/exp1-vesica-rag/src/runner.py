"""Shared eval-loop machinery for the baseline and Vesica-RAG runs.

Each run is the same shape: for every HotpotQA dev question — encode the
question, retrieve a chunk set, fetch chunk texts, ask the LLM, append a
JSONL record. The loop is resumable: on restart, question ids already
present in the output file are skipped, so a crash at question 5,000
costs only the questions after it.

scripts/07 and scripts/08 supply the condition-specific bits:
  - `retrieve_fn(question, q_vec) -> (chunk_ids, extra_record_fields)`
    — what chunks to feed the LLM, plus any extra fields (retrieval
      scores, candidate-Vesica parents, gold chunk ids for the
      vesica-coverage diagnostic, …) to record alongside.
  - `chunk_text_fn(chunk_id) -> (title, text)` — chunk_id → passage text.

This module owns the loop, the resumability, and the record schema common
to both conditions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .data import HotpotQuestion
from .generate import OllamaGenerator, RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

    RetrieveFn = Callable[[HotpotQuestion, "NDArray[np.float32]"], tuple[list[str], dict]]
    ChunkTextFn = Callable[[str], tuple[str, str]]
    EncodeFn = Callable[[str], "NDArray[np.float32]"]


def _load_done_ids(path: Path) -> set[str]:
    """Question ids already present in an output JSONL (for resume)."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                # tolerate a torn last line from a previous crash
                continue
    return done


def run_eval(
    *,
    condition: str,
    questions: list[HotpotQuestion],
    encode_fn: "EncodeFn",
    retrieve_fn: "RetrieveFn",
    chunk_text_fn: "ChunkTextFn",
    generator: OllamaGenerator,
    out_path: Path,
    log_every: int = 50,
) -> None:
    """Run one condition over all (not-yet-done) questions, appending JSONL.

    Record schema (per line):
      {
        "id":                  HotpotQA question id,
        "condition":           the condition label,
        "question":            question text,
        "gold_answer":         HotpotQA gold answer,
        "prediction":          LLM answer (stripped; "" on generation error),
        "type":                "bridge" | "comparison",
        "level":               "easy" | "medium" | "hard",
        "retrieved_chunk_ids": chunks fed to the LLM, in order,
        "error":               None, or the generation error string,
        ...                    extra fields from retrieve_fn,
      }
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_ids(out_path)
    if done:
        print(f"  resuming: {len(done)} questions already recorded in {out_path.name}")
    todo = [q for q in questions if q.id not in done]
    print(f"  {len(todo)} questions to process for condition={condition!r}", flush=True)

    t0 = time.time()
    with out_path.open("a") as f:
        for i, q in enumerate(todo):
            q_vec = encode_fn(q.question)
            chunk_ids, extra = retrieve_fn(q, q_vec)
            chunks: list[RetrievedChunk] = []
            for cid in chunk_ids:
                title, text = chunk_text_fn(cid)
                chunks.append(RetrievedChunk(chunk_id=cid, title=title, text=text))
            gen = generator.answer_question(q.question, chunks)
            record = {
                "id": q.id,
                "condition": condition,
                "question": q.question,
                "gold_answer": q.answer,
                "prediction": gen.answer,
                "type": q.type,
                "level": q.level,
                "retrieved_chunk_ids": list(chunk_ids),
                "error": gen.error,
                **extra,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if (i + 1) % log_every == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                eta_min = (len(todo) - i - 1) / max(rate, 1e-6) / 60.0
                print(
                    f"  ...{i + 1}/{len(todo)} ({rate:.2f} q/s, eta {eta_min:.1f} min)",
                    flush=True,
                )
    elapsed_min = (time.time() - t0) / 60.0
    print(f"  done. processed {len(todo)} questions in {elapsed_min:.1f} min "
          f"→ {out_path}", flush=True)
