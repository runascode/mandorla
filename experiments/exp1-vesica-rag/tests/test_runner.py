"""Unit tests for the shared eval loop (src/runner.py).

We fake the generator (no Ollama call) and a trivial retriever/encoder.
Tests cover: record schema, extra-field passthrough from retrieve_fn,
resumability (already-recorded ids skipped), and tolerance of a torn last
line in the output file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import HotpotQuestion
from src.generate import GenerationResult, RetrievedChunk
from src.runner import _load_done_ids, run_eval


class FakeGenerator:
    """Stand-in for OllamaGenerator — returns a deterministic canned answer
    and records the chunks it was handed. `label` is echoed into the answer
    so multi-generator tests can see which generator served which question."""

    def __init__(self, label: str = "fake") -> None:
        self.label = label
        self.host_label = label
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def answer_question(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        self.calls.append((question, list(chunks)))
        return GenerationResult(
            answer=f"answer-for::{question}",
            prompt="(fake prompt)",
            model=f"fake-model::{self.label}",
            options={},
            error=None,
        )


def make_questions(n: int) -> list[HotpotQuestion]:
    return [
        HotpotQuestion(
            id=f"q{i}",
            question=f"question {i}?",
            answer=f"gold{i}",
            type="bridge" if i % 2 == 0 else "comparison",
            level=["easy", "medium", "hard"][i % 3],
            supporting_titles=(f"Title A{i}", f"Title B{i}"),
        )
        for i in range(n)
    ]


def fake_encode(text: str) -> np.ndarray:
    # Deterministic 4-D vector from the string hash
    h = abs(hash(text)) % 1000
    return np.array([h, h + 1, h + 2, h + 3], dtype=np.float32)


def fake_retrieve(question: HotpotQuestion, q_vec: np.ndarray):
    # Return two chunk ids derived from the question, plus an extra field.
    return [f"{question.id}_chunkA", f"{question.id}_chunkB"], {
        "marker": f"ret::{question.id}"
    }


def fake_chunk_text(chunk_id: str) -> tuple[str, str]:
    return (f"Title({chunk_id})", f"Body of {chunk_id}.")


def test_run_eval_writes_expected_schema(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    gen = FakeGenerator()
    questions = make_questions(3)
    run_eval(
        condition="testcond",
        questions=questions,
        encode_fn=fake_encode,
        retrieve_fn=fake_retrieve,
        chunk_text_fn=fake_chunk_text,
        generator=gen,
        out_path=out,
        log_every=100,
    )
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    rec0 = json.loads(lines[0])
    assert rec0["id"] == "q0"
    assert rec0["condition"] == "testcond"
    assert rec0["question"] == "question 0?"
    assert rec0["gold_answer"] == "gold0"
    assert rec0["prediction"] == "answer-for::question 0?"
    assert rec0["type"] == "bridge"
    assert rec0["level"] == "easy"
    assert rec0["retrieved_chunk_ids"] == ["q0_chunkA", "q0_chunkB"]
    assert rec0["error"] is None
    # extra field from retrieve_fn
    assert rec0["marker"] == "ret::q0"


def test_run_eval_passes_chunks_to_generator(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    gen = FakeGenerator()
    run_eval(
        condition="c",
        questions=make_questions(1),
        encode_fn=fake_encode,
        retrieve_fn=fake_retrieve,
        chunk_text_fn=fake_chunk_text,
        generator=gen,
        out_path=out,
    )
    assert len(gen.calls) == 1
    _, chunks = gen.calls[0]
    assert [c.chunk_id for c in chunks] == ["q0_chunkA", "q0_chunkB"]
    assert chunks[0].title == "Title(q0_chunkA)"
    assert chunks[0].text == "Body of q0_chunkA."


def test_run_eval_resumes_skipping_done_ids(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    # Pre-populate with q0 and q1 already "done"
    with out.open("w") as f:
        f.write(json.dumps({"id": "q0", "prediction": "old"}) + "\n")
        f.write(json.dumps({"id": "q1", "prediction": "old"}) + "\n")
    gen = FakeGenerator()
    run_eval(
        condition="c",
        questions=make_questions(4),
        encode_fn=fake_encode,
        retrieve_fn=fake_retrieve,
        chunk_text_fn=fake_chunk_text,
        generator=gen,
        out_path=out,
    )
    # Only q2 and q3 should have been (re)generated
    assert len(gen.calls) == 2
    lines = out.read_text().strip().splitlines()
    ids = [json.loads(l)["id"] for l in lines]
    assert ids == ["q0", "q1", "q2", "q3"]
    # q0/q1 retain their old predictions (not overwritten)
    assert json.loads(lines[0])["prediction"] == "old"
    assert json.loads(lines[2])["prediction"] == "answer-for::question 2?"


def test_load_done_ids_tolerates_torn_last_line(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    with out.open("w") as f:
        f.write(json.dumps({"id": "q0"}) + "\n")
        f.write(json.dumps({"id": "q1"}) + "\n")
        f.write('{"id": "q2", "predict')  # torn
    done = _load_done_ids(out)
    assert done == {"q0", "q1"}


def test_load_done_ids_empty_file(tmp_path: Path) -> None:
    out = tmp_path / "missing.jsonl"
    assert _load_done_ids(out) == set()
    out.write_text("")
    assert _load_done_ids(out) == set()


def test_run_eval_parallel_processes_all_questions(tmp_path: Path) -> None:
    """With n_workers > 1, every question is processed exactly once; output
    order is not guaranteed but the id set is complete and unique."""
    out = tmp_path / "run.jsonl"
    gen = FakeGenerator()
    questions = make_questions(20)
    run_eval(
        condition="parcond",
        questions=questions,
        encode_fn=fake_encode,
        retrieve_fn=fake_retrieve,
        chunk_text_fn=fake_chunk_text,
        generator=gen,
        out_path=out,
        n_workers=4,
        log_every=100,
    )
    lines = out.read_text().strip().splitlines()
    ids = [json.loads(l)["id"] for l in lines]
    assert len(ids) == 20
    assert set(ids) == {f"q{i}" for i in range(20)}
    # Each question's record carries the right extra field
    by_id = {json.loads(l)["id"]: json.loads(l) for l in lines}
    assert by_id["q5"]["marker"] == "ret::q5"
    assert by_id["q5"]["prediction"] == "answer-for::question 5?"


def test_run_eval_parallel_resumes(tmp_path: Path) -> None:
    """Parallel path also honors the resume set."""
    out = tmp_path / "run.jsonl"
    with out.open("w") as f:
        for i in range(5):
            f.write(json.dumps({"id": f"q{i}", "prediction": "old"}) + "\n")
    gen = FakeGenerator()
    run_eval(
        condition="c",
        questions=make_questions(12),
        encode_fn=fake_encode,
        retrieve_fn=fake_retrieve,
        chunk_text_fn=fake_chunk_text,
        generator=gen,
        out_path=out,
        n_workers=4,
    )
    assert len(gen.calls) == 7   # q5..q11
    ids = [json.loads(l)["id"] for l in out.read_text().strip().splitlines()]
    assert set(ids) == {f"q{i}" for i in range(12)}


def test_run_eval_multi_generator_all_questions_covered(tmp_path: Path) -> None:
    """generators=[g1, g2]: every question processed exactly once, split
    across the two generators (each gets at least one)."""
    out = tmp_path / "run.jsonl"
    g1, g2 = FakeGenerator("g1"), FakeGenerator("g2")
    run_eval(
        condition="multigen",
        questions=make_questions(20),
        encode_fn=fake_encode,
        retrieve_fn=fake_retrieve,
        chunk_text_fn=fake_chunk_text,
        generators=[g1, g2],
        out_path=out,
        log_every=100,
    )
    ids = [json.loads(l)["id"] for l in out.read_text().strip().splitlines()]
    assert len(ids) == 20
    assert set(ids) == {f"q{i}" for i in range(20)}
    # both generators saw work, total = 20
    assert len(g1.calls) + len(g2.calls) == 20
    assert len(g1.calls) > 0 and len(g2.calls) > 0


def test_run_eval_multi_generator_resumes(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    with out.open("w") as f:
        for i in range(8):
            f.write(json.dumps({"id": f"q{i}", "prediction": "old"}) + "\n")
    g1, g2 = FakeGenerator("g1"), FakeGenerator("g2")
    run_eval(
        condition="c",
        questions=make_questions(14),
        encode_fn=fake_encode,
        retrieve_fn=fake_retrieve,
        chunk_text_fn=fake_chunk_text,
        generators=[g1, g2],
        out_path=out,
    )
    assert len(g1.calls) + len(g2.calls) == 6  # q8..q13
    ids = [json.loads(l)["id"] for l in out.read_text().strip().splitlines()]
    assert set(ids) == {f"q{i}" for i in range(14)}


def test_run_eval_requires_a_generator(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(ValueError):
        run_eval(
            condition="c",
            questions=make_questions(1),
            encode_fn=fake_encode,
            retrieve_fn=fake_retrieve,
            chunk_text_fn=fake_chunk_text,
            out_path=tmp_path / "x.jsonl",
        )
