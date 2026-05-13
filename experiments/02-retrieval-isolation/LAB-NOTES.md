# Experiment 02 — Lab Notes

Append-only chronological log. Oldest at top. Project-wide notes go in the project-root `LAB-NOTES.md`.

---

## 2026-05-13 — Experiment scoped

Triggered by the Experiment 01 slice diagnostic
(`exp1-vesica-rag/results/DIAGNOSTIC.md`):

- Coverage-hit subset (n=323) showed F1 lift −0.01 (CI −0.04, +0.02) →
  H1 ("coverage is the bottleneck") falsified.
- Vesica-RAG dropped gold-pair-in-context from 41.39% to 30.74% → context
  displacement was a real cost, but on the coverage-hit subset (where
  displacement is bounded) the LLM still showed no F1 lift. So
  Llama-3.1-8B saturates at top-25 dense on HotpotQA dev; the screening
  slice was effectively an LLM-saturation test, not a retrieval-primitive
  test.

PRECOMMIT.md locked the same day. No code committed yet.

Expected throughput for the retrieval-only run: ~0.5 q/s × ~20k questions
≈ 11 hours wall, plus dataset-pull + corpus-coverage overhead. Single
worker per `LAB-NOTES.md` Lesson 6 (16 GB resident FAISS index).
