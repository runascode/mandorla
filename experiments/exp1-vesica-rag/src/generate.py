"""Ollama answer-generation wrapper.

The slice's generative LLM step. Identical prompt template and decoding
parameters across baseline and Vesica-RAG — the only variable between the
two conditions is the list of retrieved chunks fed into the prompt
(PRECOMMIT.md §"Architecture spec for the slice").

We pass parameters explicitly per-call rather than relying on the Modelfile
alone, so the code is the single source of truth for the decoding config.
The Modelfile is kept in sync for `ollama run` interactive use and as
documentation.

Defaults pinned (PRECOMMIT.md decisions A model, F seed):
  - model = "llama3.1:8b-instruct-q5_K_M"
  - temperature = 0
  - seed = 1337
  - top_p = 1
  - num_ctx = 8192
  - num_predict = 128

Retry policy: simple exponential backoff for transient connection/server
errors; a hard failure is logged and the answer is recorded as "" (empty)
so eval scoring still runs on the rest of the dev set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import ollama

DEFAULT_MODEL = "llama3.1:8b-instruct-q5_K_M"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 1337
DEFAULT_TOP_P = 1.0
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 128

SYSTEM_PROMPT = (
    "You answer extractive multi-hop questions from supplied context. "
    "Output the shortest possible answer span (an entity, date, yes/no, or "
    "short phrase) and nothing else — no preamble, no explanation, no quotes."
)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk to be placed in the LLM context."""
    chunk_id: str
    title: str
    text: str


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the user-side prompt from chunks + question.

    Format is deterministic and identical for baseline and Vesica-RAG. We
    enumerate chunks so the LLM can reference them, and we tag each with its
    Wikipedia title for disambiguation."""
    parts: list[str] = ["Passages:"]
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] ({c.title}) {c.text}")
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return "\n".join(parts)


@dataclass(frozen=True)
class GenerationResult:
    """One LLM generation."""
    answer: str
    prompt: str
    model: str
    options: dict
    error: Optional[str] = None


@dataclass
class OllamaGenerator:
    """Callable around an Ollama daemon (local or over the network).

    Constructed once per host per run; reused across questions; thread-safe
    (each `generate` call is independent — no chat history — and the
    underlying `ollama.Client` is a thin HTTP wrapper). Retries on transient
    errors with exponential backoff.

    `host`: None → the ollama library default (OLLAMA_HOST env var, else
    `http://localhost:11434`); otherwise a `host:port` or `http://host:port`
    string. Used to fan eval generation across multiple machines.
    """

    model: str = DEFAULT_MODEL
    host: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    seed: int = DEFAULT_SEED
    top_p: float = DEFAULT_TOP_P
    num_ctx: int = DEFAULT_NUM_CTX
    num_predict: int = DEFAULT_NUM_PREDICT
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    _options: dict = field(default_factory=dict, init=False, repr=False)
    _client: "ollama.Client" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._options = {
            "temperature": self.temperature,
            "seed": self.seed,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        # Client construction does no network I/O; it just configures the
        # base URL and an httpx session.
        self._client = ollama.Client(host=self.host) if self.host else ollama.Client()

    def options_dict(self) -> dict:
        return dict(self._options)

    @property
    def host_label(self) -> str:
        return self.host or "default"

    def generate(self, prompt: str) -> GenerationResult:
        last_err: Optional[Exception] = None
        backoff = self.initial_backoff_seconds
        for _ in range(self.max_retries):
            try:
                resp = self._client.generate(
                    model=self.model,
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    options=self._options,
                    stream=False,
                )
                return GenerationResult(
                    answer=resp["response"].strip(),
                    prompt=prompt,
                    model=self.model,
                    options=self.options_dict(),
                )
            except Exception as e:
                last_err = e
                time.sleep(backoff)
                backoff *= 2
        return GenerationResult(
            answer="",
            prompt=prompt,
            model=self.model,
            options=self.options_dict(),
            error=f"{type(last_err).__name__}: {last_err} (host={self.host_label})",
        )

    def answer_question(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        prompt = build_prompt(question, chunks)
        return self.generate(prompt)
