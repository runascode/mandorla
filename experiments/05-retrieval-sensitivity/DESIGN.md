# Experiment 05 — Retrieval-sensitivity sweep (locked design, measurement)

**Pre-PRECOMMIT measurement, not a thesis test.** Binds nothing, but
the claim and metric are fixed here before any number, because this is
the load-bearing figure for the standalone reader-saturation paper
(`findings/reader-saturation-hotpotqa.md`).

## Claim under test (fixed before running)

Holding reader, prompt, decoding and context budget constant, HotpotQA
answer-F1 is **near-flat across the full retrieval-quality range** —
from perfect (gold pair injected) to adversarial (gold pair forcibly
removed) to random — even though a retrieval-side metric (gold pair
in context) spans ≈100% → ≈0% across those same conditions. If F1
moves only a few points while the retrieval metric moves ~100, the
benchmark answer-metric is a low-power instrument for retrieval
quality, generalizing the Exp 01 diagnostic from one idiosyncratic
method to the whole quality axis.

## Conditions (same reader, prompt, budget=25 chunks)

- **oracle** — gold supporting chunks (by title) + dense top-k fill to 25.
- **dense** — contriever top-25 (the realistic operating point).
- **gold_removed** — contriever top-k with every gold-supporting-title
  chunk excluded, take 25 (adversarial but plausible distractors).
- **random** — 25 uniformly random corpus chunks.

## Fixed

- Reader: `llama3.1:8b-instruct-q5_K_M`, T=0, seed=1337 (Exp 01 config).
- Encoder/index: contriever + the Exp 01 FAISS index, bit-for-bit.
- Sample: 500 HotpotQA dev questions, seeded (1337). 500 is ample for
  a per-condition F1 mean ± bootstrap CI; the effect, if present, is
  an order of magnitude larger than the CI.
- Metrics: F1, EM (answer side); pair-in-context rate (retrieval side,
  the sanity that conditions truly differ in retrieval quality).

## Read

If F1(oracle) − F1(gold_removed) is small (single-digit points) while
pair-in-context goes ~100% → ~0%, the claim holds and the curve is the
paper's central figure. If F1 tracks retrieval quality strongly, the
saturation finding does *not* generalize beyond the Exp 01 method and
the standalone paper is withdrawn. Either way, published.

## Non-goals

Not BM25/ColBERT (a fuller quality axis is a follow-up; oracle →
dense → gold_removed → random already brackets the range with no new
deps). Not multi-reader/multi-budget (that is the Findings-tier
extension named in the venue analysis; this is the core figure).
