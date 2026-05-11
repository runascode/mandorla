# Mandorla — Engineering Standards

This document codifies the engineering and research discipline that every
experiment under `experiments/` must follow. It exists so a fresh session
(human or AI) can pick up the work without losing the conventions, and so
the research record stays defensible.

**Doctrine in one line:** every decision and every change gets documented
with reasons. The reasons are what makes the result reproducible and the
negative result publishable.

The discipline is non-negotiable. If a shortcut feels tempting, that's the
moment to add a comment, an amendment, or a `LAB-NOTES.md` entry instead.

---

## 1. Document types (three-document discipline)

Each experiment carries three documents that play distinct roles. They
must be kept distinct — collapsing them into one file destroys the audit
trail.

### `PRECOMMIT.md` — binding pre-commit decisions

- **One per experiment.** Lives in the experiment directory root.
- **Locked before any code runs against evaluation data.** The dated lock
  is recorded at the top of the file.
- **Frozen content.** Anything that would change a number we report must
  be in this file, written before that number was produced.
- **Changes only via dated amendment blocks** at the bottom of the file.
  An amendment must:
  - have a unique number (`Amendment 1`, `Amendment 2`, …) and a date
  - quote the original passage being amended
  - quote (or write) the new passage
  - state **why** the change is being made (geometric, empirical, deadline,
    new finding — be specific)
  - state **what triggered it** (smoke-test result, paper-side correction,
    upstream library change — cite a `LAB-NOTES.md` entry or commit SHA
    where applicable)
  - state **scope** — what is and isn't changing
- **Required sections** (template):
  - Purpose
  - What this slice tests / does **not** test
  - Decision rule (go/no-go, formal prediction)
  - Frozen decisions (numbered, e.g. D1, D2 or A, B, C)
  - Architecture spec
  - Baselines
  - Sample size and statistics
  - What "ship" means (artifact checklist)
  - What is deferred to follow-on work
  - Budget (honest accounting of expected effort)
  - Amendments
  - Sign-off (author, date)

### `LAB-NOTES.md` — chronological observations

- **One per experiment**, in the experiment directory. Append-only, oldest
  at top.
- **One at the project root** for repo-wide events (identity / attribution
  changes, infrastructure shifts, large dependency upgrades, cross-
  experiment refactors). Use sparingly — experiment-specific findings
  always go in the per-experiment file.
- **Non-binding.** Nothing in this file changes the spec on its own — but
  `PRECOMMIT.md` amendments cite entries here when they are *triggered* by
  what is observed.
- **What to record:**
  - throughput / benchmark numbers
  - sanity-check results (e.g. "contriever cosines were 0.83 / 0.22 /
    0.22 — semantically sensible")
  - calibration values and grids
  - surprises (the encoder was 2× faster than estimated; α calibration
    plateaued; vesica-coverage was higher than predicted)
  - debugging journeys (what was tried, what failed, what we learned)
  - empirical observations that *might* trigger an amendment but haven't
    yet
- **Date every entry** with the ISO date (`YYYY-MM-DD`). Within a date,
  use sub-sections.
- **Quote numbers exactly.** When a result will be re-computed at a
  larger scale, mark the entry as "smoke test on N=200k" or similar so
  it's clear it's preliminary.

### `README.md` — operator-facing summary

- **One per experiment.** Lives in the experiment directory root.
- **Audience:** someone (or a future agent) who wants to run this
  experiment — not someone who wants to argue with its design.
- **Defers to PRECOMMIT.md** as the source of truth on every binding
  decision. If README disagrees with PRECOMMIT, PRECOMMIT wins; fix the
  README.
- **Required sections:**
  - What this is, in one paragraph, with a link to the paper section it
    implements
  - Setup table (corpus, models, key configs)
  - Metrics + go-no-go criteria (a summary; full rationale is in PRECOMMIT)
  - Reproduce (the exact `uv run python scripts/NN_*.py` sequence)
  - Artifacts on completion (what files must exist for the experiment to
    be "done")
  - References

### Other documents (optional, when they earn their place)

- **`BENCHMARKS.md`** — throughput / cost measurements that justify a
  configuration choice. Add when more than ~3 numbers need to be
  presented as a comparison; otherwise put the numbers in LAB-NOTES.
- **`RESULTS.md`** — written at the **end** of the experiment. Headline
  numbers, confidence intervals, plots, go-no-go decision, link to the
  commit SHA that produced the numbers, all seeds, all calibrated values,
  and any caveats. This is the artifact a reviewer would read.

---

## 2. Folder structure

Every experiment under `experiments/` follows this layout. The numbering
prefix (`01-`, `02-`, …) sorts experiments chronologically and resists
renaming when scope shifts.

```
experiments/{NN}-{slug}/
├── PRECOMMIT.md             # binding decisions (§1)
├── LAB-NOTES.md             # chronological log (§1)
├── README.md                # operator-facing (§1)
├── BENCHMARKS.md            # optional (§1)
├── RESULTS.md               # written at end (§1)
├── pyproject.toml           # uv-managed deps; pinned versions
├── uv.lock                  # committed
├── .python-version          # pinned to 3.12 for ML compatibility
├── Modelfile                # if Ollama is the generator; pin model + decoding
├── src/
│   ├── __init__.py
│   ├── <module>.py          # one concern per module; pure logic where possible
│   └── …
├── tests/
│   ├── __init__.py
│   ├── test_<module>.py     # one test file per src module; same name
│   └── …
├── scripts/
│   ├── 01_<verb>.py         # numbered, runnable, idempotent / resumable
│   ├── 02_<verb>.py
│   └── …
├── data/                    # gitignored — input data, HF cache, etc.
├── index/                   # gitignored — built indices, embeddings
└── results/                 # raw per-question JSONL (gitignored or LFS,
                             # decide per experiment); RESULTS.md committed.
```

**The slug** is short, hyphen-separated, descriptive of the *test*, not
the *method*. `01-vesica-rag` is the slug because the *experiment* is
Vesica-RAG; `01-multihop-retrieval` would be a slug for an experiment
that compares multiple methods.

**The numbering** is global across the project. Experiment 02 follows
Experiment 01 even if they live under different paper sections.

**`data/`, `index/`, and (typically) `results/raw/`** are gitignored.
Anything that can be regenerated from inputs + committed code should be
gitignored. Anything that summarizes the run (RESULTS.md, plots,
metadata JSONs) is committed.

---

## 3. Testing standards

### What we test

- **All pure-logic modules in `src/`** must have a corresponding
  `tests/test_<module>.py`. New code without tests does not get merged.
- **Index-building and eval scripts** are integration-tested end-to-end
  on a small subset (e.g. one shard of the corpus) before committing the
  pipeline. The result of that smoke test is logged in `LAB-NOTES.md`.
- **Numerical correctness** of math primitives (intersection, projection,
  bootstrap CI) is exercised against analytic cases at known limits
  (β→0, identical inputs, disjoint inputs).

### How we test

- **Pytest, no other runner.**
- **No live network in tests.** Datasets are not loaded in unit tests;
  ditto Ollama. Construct synthetic data inline.
- **Determinism.** Every test that uses randomness seeds it explicitly
  with `np.random.default_rng(seed)` (or `np.random.RandomState(seed)`
  where APIs require it). Never call `np.random.seed` globally — it
  pollutes other tests.
- **Independence.** Each test must pass in isolation. No shared mutable
  state, no test ordering dependencies. Use `tmp_path` for filesystem
  side-effects.
- **Behavioral names.** `test_disjoint_boxes_underflow_at_small_beta` —
  not `test_box_function_3`. The test name should read like a sentence
  about the contract.
- **Numerical tolerance is explicit.** `pytest.approx(x, rel=1e-3)` or
  `np.testing.assert_allclose(a, b, atol=…)`. Bare `==` on floats is a
  test bug.
- **Edge cases are part of the contract.** Empty inputs, dimension
  mismatches, degenerate boxes — every public function must declare its
  behavior at the boundary, and a test must witness it.

### How we run tests

- `uv run pytest tests/` from the experiment directory.
- The full suite must be green at every commit that touches `src/`,
  `scripts/`, `pyproject.toml`, or `uv.lock`. Document a test failure
  in `LAB-NOTES.md` and fix it before continuing.
- Pytest config lives in `pyproject.toml` (or `pytest.ini` if the
  experiment grows enough complexity that it warrants its own).

---

## 4. Code conventions

- **Python 3.12.** Pinned in `.python-version` and `pyproject.toml`'s
  `requires-python`. Newer Pythons may lack ML wheels; older Pythons
  miss type-system features.
- **`uv` for everything.** `uv init`, `uv add`, `uv run`. The `uv.lock`
  is committed.
- **Type annotations on every public function signature.** This is
  research code; future readers (us, six months from now) need the help.
- **Module docstrings explain WHY.** What the module does is visible in
  the names; why it does it that way is what we need to record. Cite
  the paper section, PRECOMMIT amendment, or LAB-NOTES entry that the
  decision came from.
- **Inline comments only where the WHY is non-obvious.** No comments
  that restate the line below them.
- **All randomness seeded.** Every `np.random.default_rng()` gets an
  explicit seed argument. Every PyTorch model run sets
  `torch.manual_seed(seed)`. Every Ollama call sets `seed` in options.
  The master seed for the slice is **1337**; derived seeds are
  `master_seed + N` and recorded in code.
- **All LLM decoding params explicit in code.** The Modelfile is
  documentation; the code is the source of truth. Pin `temperature`,
  `seed`, `top_p`, `num_ctx`, `num_predict` per call.
- **Scripts add the project root to `sys.path`** at the top:
  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  ```
  so `from src.module import …` works under `uv run python scripts/…`.
- **Scripts are idempotent or resumable.** Re-running script `NN` should
  either produce identical outputs (idempotent) or pick up from where it
  left off (resumable, e.g. encoder shard checkpointing).
- **Output files include enough metadata to reproduce.** Alongside any
  numerical artifact, write a `*.meta.json` recording the seed, the
  source dataset revision, the build parameters, and the script that
  produced it.

---

## 5. Reproducibility checklist (per experiment)

Before claiming a number, verify:

- [ ] Master seed (`1337` unless documented otherwise) propagated to
      every randomness source touched by the run.
- [ ] All Python deps pinned in `pyproject.toml` *and* `uv.lock`
      committed.
- [ ] All non-Python deps versions documented (Ollama version, model
      digest, OS, hardware in BENCHMARKS or RESULTS).
- [ ] All indices regenerable from raw inputs + committed code (no
      hand-edited artifacts).
- [ ] The exact commit SHA that produced the result is logged in
      `RESULTS.md`.
- [ ] The full `scripts/01_*.py … scripts/NN_*.py` sequence runs end-to-end
      from a clean clone, in order, with the published commands.

---

## 6. Commit discipline

- **Authored as the user.** No `Co-Authored-By` / "Generated by" trailers
  ever (project-wide rule; see global `~/.claude/CLAUDE.md`).
- **Commit titles are short, present tense, ≤72 chars.**
- **Commit bodies are long-form when the change involves a decision.**
  This is research; the commit log is a research log. Use the body to:
  - state what changed
  - state why
  - cite the PRECOMMIT amendment, LAB-NOTES entry, or paper section it
    came from
  - list affected files at a one-line-per-thing granularity if the diff
    is large
- **Related changes share a commit only if they share a single reason.**
  Adding a calibration module + the script that uses it + the tests for
  it + the LAB-NOTES entry that motivates it = one commit, one story.
  Adding eval metrics + adding box intersection math = two commits.
- **`git push` after every commit.** The remote is the durable record
  for collaborators and reviewers.

---

## 7. When PRECOMMIT.md must be amended

In short: any time a change would alter a *binding decision* (anything
that, if it had been different, would have produced a different number).
In particular:

- the target dataset, baseline, or model
- a calibration target or threshold
- a metric definition or normalization
- the sample size or statistical procedure
- the go/no-go decision rule
- the prompt template or decoding parameters

In contrast, an amendment is **not** needed for:

- a bug fix that brings code into line with the documented spec
- a throughput improvement that doesn't change outputs
- adding more documentation
- adding more tests
- adding ablations or follow-ups that don't replace the primary run

When uncertain, prefer to amend. A small amendment is cheap; an
undocumented design drift is a research credibility cost.

---

## 8. When LAB-NOTES.md must be appended

Every time we observe a number — a throughput, a calibrated value, a
sample size, a sanity-check result, a debugging finding. Even if it's
not surprising. The log is for our future selves and for reviewers.

A new LAB-NOTES entry must be present in the same commit that produces
the observation, when the observation comes from running code that gets
committed.

---

## 9. The point of all of this

When the paper writes "we ran X and got Y," a reviewer (or a future you)
should be able to walk from that claim back to:

1. the `RESULTS.md` that recorded Y, with its CIs and decision
2. the commit SHA that produced Y
3. the `PRECOMMIT.md` (with its amendments) that locked in everything
   about *how* X was run before Y was visible
4. the `LAB-NOTES.md` entries that record the empirical observations
   along the way
5. the tests that verified the math primitives Y depends on
6. the `pyproject.toml` and `uv.lock` that fix every dep version
7. the `BENCHMARKS.md` (or equivalent) that justifies the throughput-
   relevant configuration

If any link in that chain is missing, we have a hole in the research
record. The discipline above exists to keep that chain whole.
