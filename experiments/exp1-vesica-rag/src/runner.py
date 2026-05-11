"""Shared eval-loop machinery for the baseline and Vesica-RAG runs.

Each run is the same shape: for every HotpotQA dev question — encode the
question, retrieve a chunk set, fetch chunk texts, ask the LLM, append a
JSONL record. The loop is resumable: on restart, question ids already
present in the output file are skipped, so a crash at question 5,000
costs only the questions after it.

The loop can run with a thread pool (`n_workers > 1`). The LLM call (Ollama)
dominates wall-clock; running several in parallel against an Ollama server
configured with `OLLAMA_NUM_PARALLEL > 1` gives a near-linear speedup. The
contriever encode is serialized with a lock (PyTorch MPS is not guaranteed
thread-safe); it's fast (~50 ms) so this doesn't bottleneck. FAISS search
and the numpy box operations are read-only on shared arrays and safe under
concurrent reads. Each output line is written under a lock. Output order
is not guaranteed under `n_workers > 1`, which is fine — scripts/09 matches
records by question id.

scripts/07 and scripts/08 supply the condition-specific bits:
  - `retrieve_fn(question, q_vec) -> (chunk_ids, extra_record_fields)`
  - `chunk_text_fn(chunk_id) -> (title, text)`

This module owns the loop, the resumability, the concurrency, and the
record schema common to both conditions.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    n_workers: int = 1,
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

    With `n_workers > 1`, questions are processed by a thread pool; output
    line order is not preserved (scripts/09 matches by id).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_ids(out_path)
    if done:
        print(f"  resuming: {len(done)} questions already recorded in {out_path.name}")
    todo = [q for q in questions if q.id not in done]
    print(f"  {len(todo)} questions to process for condition={condition!r} "
          f"(n_workers={n_workers})", flush=True)
    if not todo:
        return

    t0 = time.time()
    encode_lock = threading.Lock()
    write_lock = threading.Lock()
    state = {"done": 0}

    def build_record(q: HotpotQuestion) -> dict:
        # MPS encode is serialized; it's fast and the parallel win is the LLM call.
        with encode_lock:
            q_vec = encode_fn(q.question)
        chunk_ids, extra = retrieve_fn(q, q_vec)
        chunks: list[RetrievedChunk] = []
        for cid in chunk_ids:
            title, text = chunk_text_fn(cid)
            chunks.append(RetrievedChunk(chunk_id=cid, title=title, text=text))
        gen = generator.answer_question(q.question, chunks)
        return {
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

    def emit(f, record: dict) -> None:
        with write_lock:
            f.write(json.dumps(record) + "\n")
            f.flush()
            state["done"] += 1
            d = state["done"]
            if d % log_every == 0:
                rate = d / max(time.time() - t0, 1e-6)
                eta_min = (len(todo) - d) / max(rate, 1e-6) / 60.0
                print(f"  ...{d}/{len(todo)} ({rate:.2f} q/s, eta {eta_min:.1f} min)",
                      flush=True)

    with out_path.open("a") as f:
        if n_workers <= 1:
            for q in todo:
                emit(f, build_record(q))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(build_record, q) for q in todo]
                # Write each record as soon as it completes (unordered) — a
                # crash loses only the in-flight ~n_workers, not a buffer.
                for fut in as_completed(futures):
                    emit(f, fut.result())

    elapsed_min = (time.time() - t0) / 60.0
    print(f"  done. processed {len(todo)} questions in {elapsed_min:.1f} min "
          f"→ {out_path}", flush=True)
