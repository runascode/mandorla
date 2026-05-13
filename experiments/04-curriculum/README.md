# Experiment 04 — Mandorla Curriculum: Recursive Construction

*Last updated: 2026-05-13*

**Status:** **design sketch** (pre-PRECOMMIT). No PRECOMMIT.md is locked yet. This file captures the design space and the blocking decisions that must be resolved *before* a PRECOMMIT.md can be written.
**Paper section:** §3.1 — Experiment 3.

## What this is

A test of **Thesis 3 (Recursive Construction)**: a learning curriculum that forces the model to *construct new representations at the intersection of pre-existing representations* produces better compositional generalization than standard MLM/CLM training, while not degrading i.i.d. performance.

This experiment is *training* infrastructure-heavy. It is the longest of the three planned experiments (paper estimates 8–12 weeks) and requires from-scratch training of 100–300M-parameter transformers.

## Falsifiable prediction (paper-level, to be made formal by PRECOMMIT)

Mandorla-curriculum model shows ≥15% **relative** improvement in accuracy on the systematic-generalization splits (COGS Gen, SCAN length/primitive, ReCOGS structural) while losing no more than 2% **relative** on i.i.d. test accuracy. Both conditions must hold. Per paper §3.3, falsifier **F3** fires if no compositional advantage is observed.

## Setup (from paper §3.1)

- Two small transformers (100M–300M params), trained from scratch on the same corpus (clean subset of C4 or RedPajama).
- **Baseline:** standard CLM + span-corruption (T5-style auxiliary).
- **Mandorla-curriculum:** same model and corpus, plus two additional self-supervised losses:
  1. **Vesica prediction.** Given two co-occurring named entities $E_1, E_2$ in a paragraph, predict a representation of $V(E_1, E_2)$. Operationalized as a contrastive objective against a held-out paragraph containing both $E_1$ and $E_2$; entity-pair embeddings are trained so their box-intersection contains the third-entity-pair embedding from the held-out paragraph.
  2. **Parent reconstruction.** Given a Vesica representation, predict back its parents (a denoising objective).
- Benchmarks: COGS (Kim & Linzen, EMNLP 2020), SCAN (Lake & Baroni, ICML 2018), ReCOGS / ReCOGS_pos (Wu, Manning & Potts, TACL 2023).
- I.i.d. eval to confirm no degradation.

## Blocking decisions (must resolve before PRECOMMIT.md can be locked)

1. **Compute budget.** From-scratch training of even a 100M model on a meaningful corpus subset is multi-GPU-day-class work. The dev machine (M4 Pro, 48 GB unified) is not sufficient. Options:
   - (a) Rent cloud GPUs (A100 / H100 hours) — straightforward but burns budget; need to scope hours before committing.
   - (b) Apply for academic compute (e.g. CINECA / NCSA / Lambda academic credits).
   - (c) Defer until a partnership grants access.
   - **Open.** Blocking. Choose before locking PRECOMMIT.
2. **Corpus selection.** §3.1 names "clean subset of C4 or RedPajama." Need exact corpus revision, subset construction, and deduplication recipe documented. Both conditions train on identical corpora; the corpus revision goes in the PRECOMMIT.
3. **Operationalization of "Vesica prediction" loss.** §3.1 sketches the objective but several choices are open:
   - How are entity pairs $(E_1, E_2)$ identified at training time? Named-entity tagger + co-occurrence rule? Heuristic (capitalized tokens within N words)? Pre-extracted Wikipedia entity-link pairs?
   - What is the *target* representation for $V(E_1, E_2)$? A learned box-intersection? A pooled representation of the held-out paragraph?
   - Contrastive against what negative pool? Random paragraphs, hard negatives, in-batch?
   - **Open.** This is the single biggest design decision; a PRECOMMIT cannot lock without it.
4. **Box parameterization in the model.** Are box parameters (center, log-width) projected from token embeddings (cheap, ties parameters), or trained as a separate head (more capacity)? Paper is silent; default per `iesl/box-embeddings` is a separate head.
5. **Hyperparameter selection without test-set peeking.** COGS/SCAN/ReCOGS have known leaderboards; we must pre-commit hyperparameter selection on a held-out validation set (not the gen split) so we are not implicitly fishing.
6. **Both-conditions-trained-equivalently.** Total tokens, optimizer, learning-rate schedule, sequence length must be identical between baseline and Mandorla-curriculum conditions. Only the loss functions differ. Documented in PRECOMMIT.

## What this experiment does *not* test

- Whether the curriculum scales to large models (this is a 100–300M test).
- Whether the curriculum helps i.i.d. tasks (i.i.d. is the *non-degradation* constraint, not the prediction).
- Whether the primitive is useful in retrieval (Exp 01–02) or in multi-agent reasoning (Exp 03).

## Dependencies on other experiments

**None for the test itself.** Like Exp 03, this stands on its own.

For *prioritization*: this is the most expensive and slowest of the three. The decision to actually run it is the project's biggest single resource commitment and should follow either (a) at least one of Exp 02 or Exp 03 returning GO, providing convergent evidence that the primitive is doing something somewhere, or (b) a deliberate decision to test the training-time form of the thesis independently. Either path requires the compute decision (item 1 above) to be resolved.

## Timeline (paper §3.1 estimate)

8–12 weeks once PRECOMMIT.md is locked, compute is acquired, and the Vesica-prediction loss operationalization is frozen.

## References

- Kim & Linzen, *COGS*, EMNLP 2020, arXiv:2010.05465.
- Lake & Baroni, *SCAN*, ICML 2018.
- Wu, Manning & Potts, *ReCOGS*, TACL 2023, arXiv:2303.13716.
- Csordás et al. 2021, arXiv:2108.12284 (harder baseline at COGS).
- Paper §3.1 "EXPERIMENT 3" and §3.3 falsifier F3.
