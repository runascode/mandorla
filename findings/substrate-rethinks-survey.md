# Substrate rethinks: a survey of non-point, non-token, non-autoregressive attempts

**Jacob Patterson** · runascode@protonmail.com · *draft — opened 2026-05-28, target v1 2026-06-11*

**Status**: STUB. This file is being populated as Stage A of
[`experiments/06-substrate-pilot/`](../experiments/06-substrate-pilot/PILOT-DESIGN.md).
It is thesis-independent — the survey content stands on its own
regardless of whatever MANDORLA decides about its own future.

---

## Intended summary

(To be written after the reading is done. Provisional shape: across the
five lineages surveyed below, [N] of the [K] attempts moved exactly one
of the three primitive axes — input / representation / objective. [X]
moved two; [Y] moved all three. The empty cells in the
`lineage × {input, representation, objective}` matrix motivate the
single test that would discriminate "all three together" from any
single-axis explanation: [...].)

## Why this is worth publishing on its own

If the dominant LLM stack — BPE tokens, point embeddings, autoregressive
prediction — is the only configuration that scales, the question "what
have non-point/non-token/non-AR substrates actually demonstrated?" is
worth a synthesis regardless of whether anyone *should* try to replace
the stack. Most published positions on this question are either
boosters (their own lineage works) or sceptics (the bitter lesson
forecloses it). A flat synthesis — at-what-scale, against-what-baseline,
with-what-load-bearing-claim — does not currently exist, as far as the
author has found.

## Scope

Five lineages, each surveyed for: what was attempted, at what scale,
what failed and why, and which of the three primitive axes —
**input** / **representation** / **objective** — it actually moved.

1. **Sparse Distributed Memory & Hyperdimensional Computing** (Kanerva;
   Plate's HRRs; Schlegel et al. review)
2. **Box / region embeddings** (Vilnis, Li, Patel, Boratko — McCallum
   group)
3. **Topological / geometric deep learning** (Bodnar, Bronstein)
4. **Predictive coding & active inference for language** (Friston;
   Whittington–Behrens TEM; Bogacz)
5. **Non-autoregressive sequence models & latent diffusion LMs** (LeCun
   JEPA; Bakhtin EBLM; Lou SEDD; Gulrajani et al.)

## Plan

- ~50–100 papers shortlisted from the five lineages.
- ~15–25 read deeply.
- One §-section per lineage, with the matrix below filled in.
- A "what's left to test" section identifying whether prior work
  already closes the question of "all three axes together"; if so, the
  associated experiment ([`experiments/06-substrate-pilot/`](../experiments/06-substrate-pilot/PILOT-DESIGN.md))
  does not run.

## Matrix (empty — to be filled by v1)

| Lineage | Input axis moved? | Representation axis moved? | Objective axis moved? | Demonstrated scale |
|---|---|---|---|---|
| SDM / HDC | | | | |
| Box / region embeddings | | | | |
| Topological / geometric DL | | | | |
| Predictive coding / FEP for language | | | | |
| Non-AR seq models / latent diffusion LMs | | | | |

## Provenance and independence

This survey is being assembled as the design-phase input to
[`experiments/06-substrate-pilot/`](../experiments/06-substrate-pilot/PILOT-DESIGN.md)
of the MANDORLA program — but the matrix and its conclusions do not
depend on MANDORLA's thesis in any direction. It is being filed under
[`findings/`](.) alongside
[`reader-saturation-hotpotqa.md`](reader-saturation-hotpotqa.md) for
the same reason that note was: a synthesis that outlives the project
it came from.
