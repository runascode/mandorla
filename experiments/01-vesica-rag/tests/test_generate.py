"""Unit tests for prompt assembly and generator option plumbing.

We don't mock or call Ollama here — the daemon is exercised end-to-end by
scripts/07–08 against the real model. Tests here are confined to deterministic
prompt-string construction and option-dict invariants.
"""

from __future__ import annotations

from src.generate import (
    DEFAULT_MODEL,
    OllamaGenerator,
    RetrievedChunk,
    SYSTEM_PROMPT,
    build_prompt,
)


def make_chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=f"c{i}", title=f"Title {i}", text=f"Text body {i}.")
        for i in range(n)
    ]


def test_build_prompt_contains_question_and_chunks() -> None:
    p = build_prompt("What is the capital of France?", make_chunks(3))
    assert "What is the capital of France?" in p
    for i in range(3):
        assert f"[{i + 1}]" in p
        assert f"Title {i}" in p
        assert f"Text body {i}." in p


def test_build_prompt_chunk_order_preserved() -> None:
    chunks = make_chunks(5)
    p = build_prompt("?", chunks)
    idx_1 = p.find("[1]")
    idx_2 = p.find("[2]")
    idx_3 = p.find("[3]")
    assert idx_1 < idx_2 < idx_3


def test_build_prompt_empty_chunks() -> None:
    p = build_prompt("any question?", [])
    assert "any question?" in p
    assert "Passages:" in p


def test_build_prompt_ends_with_answer_line() -> None:
    p = build_prompt("?", make_chunks(2))
    assert p.rstrip().endswith("Answer:")


def test_default_model_string() -> None:
    g = OllamaGenerator()
    assert g.model == DEFAULT_MODEL


def test_options_pinned() -> None:
    g = OllamaGenerator()
    opts = g.options_dict()
    assert opts["temperature"] == 0.0
    assert opts["seed"] == 1337
    assert opts["top_p"] == 1.0
    assert opts["num_ctx"] == 8192
    assert opts["num_predict"] == 128


def test_options_dict_is_copy() -> None:
    """Mutating the returned options dict shouldn't affect the generator."""
    g = OllamaGenerator()
    opts = g.options_dict()
    opts["temperature"] = 999
    assert g.options_dict()["temperature"] == 0.0


def test_system_prompt_constant_used() -> None:
    """The SYSTEM_PROMPT constant must be the one the generator sends — keep
    the contract explicit so we don't drift between Modelfile and code."""
    assert "extractive" in SYSTEM_PROMPT
    assert "shortest possible answer span" in SYSTEM_PROMPT


def test_host_label_default() -> None:
    assert OllamaGenerator().host_label == "default"


def test_host_label_set() -> None:
    g = OllamaGenerator(host="http://192.168.1.179:11434")
    assert g.host_label == "http://192.168.1.179:11434"
    assert g.host == "http://192.168.1.179:11434"


def test_construction_with_host_does_no_network_io() -> None:
    """Constructing a generator pointed at an unreachable host must not raise
    — the client is lazy; only generate() touches the network. Uses
    TEST-NET-1 (192.0.2.0/24, RFC 5737), guaranteed unroutable, valid port."""
    g = OllamaGenerator(host="http://192.0.2.1:11434")
    assert g.host_label == "http://192.0.2.1:11434"
    assert g.options_dict()["temperature"] == 0.0
