# Lab Notes — Experiment 1 Slice

A chronological, timestamped record of observations, surprises, and
parameter decisions made during the build. Distinguishes from the other
two documents in this directory:

- **`PRECOMMIT.md`** — *binding* design decisions, frozen before code runs.
  Changes only via dated amendment blocks.
- **`README.md`** — operator-facing summary for someone running this slice.
- **`LAB-NOTES.md`** *(this file)* — *non-binding* observations. What we
  tried, what we saw, what surprised us, what we calibrated and to what
  value. This is the research log; nothing here changes the binding spec,
  but PRECOMMIT.md amendments cite entries here when they are triggered by
  what we observed.

Entries appended in chronological order; oldest at top.

---

## 2026-05-10 — Contriever encoding setup

### Throughput sweep (`scripts/03_encode_corpus.py --limit 10000` + ad-hoc sweep)

Recorded in [`BENCHMARKS.md`](./BENCHMARKS.md). Summary: `max_len` dominates
throughput on M4 Pro MPS, batch size plateaus around 256. Chose
**batch=256, max_len=128 → 140 p/s** (extrapolating to ~10.4 h for the full
5.2M-passage corpus).

**Reason for `max_len=128` (versus contriever's published 256):** the
BeIR/hotpotqa corpus is one-passage-per-Wikipedia-article at a median of
~63 words (~100 tokens). The 95th-percentile passage is around 150 tokens
in our spot checks. Truncating at 128 affects the long tail only; both
baseline and Vesica-RAG share the encoder and truncation, so the slice's
comparison metric is invariant. Recorded the trade in `BENCHMARKS.md` and
in the script comments.

### Sanity check — contriever cosines are meaningful

A pre-encode 3-passage sanity check:

```
cos(apollo, lunar)  = 0.8263   (related)
cos(apollo, banana) = 0.2216   (unrelated)
cos(lunar,  banana) = 0.2169   (unrelated)
```

Confirms contriever produces sensible semantic geometry on the MPS path.
Test code is inline (not committed); commit `ab49ca6` references the
result for future readers.

### Encoder kicked off

PID 6579, started 2026-05-10 ~22:48 local, logging to `/tmp/mandorla_encode.log`.
Steady state at **163 p/s** (slightly better than the 140 benchmark, presumably
because the live encode runs on an idle machine). ETA ~8.5 hours from launch.

---

## 2026-05-10 — Wikipedia corpus audit (`scripts/02_pull_wiki_corpus.py`)

The BeIR/hotpotqa corpus is **one-passage-per-Wikipedia-article**, 5,233,329
passages, schema `{_id, title, text}`. Critically:

- **All 5.23M titles are unique** in the corpus (one passage per article)
- **Zero unmatched supporting-fact titles** across all 233,689 supporting
  facts in HotpotQA dev + train (audited by script 02)

This is the best possible setup for vesica-coverage: title-based matching
is unambiguous, and we don't need to caveat the metric for title-aliasing
drift.

---

## 2026-05-10 — α calibration finding (smoke test on shard 0, 200k passages)

### What happened

Ran `scripts/05_build_box_index.py` against the first encoded shard (200k of
5.2M passages) to validate the pipeline end-to-end before the full encode
finishes. PRECOMMIT.md B2 specifies α calibrated such that

> the median pairwise expected intersection volume over a random 10k-chunk
> sample is ~0.05 of the median single-box volume

With the original grid `(0.1, 0.25, ..., 10.0)` and 200 random pairs, the
chosen α was **10.0 (grid top)** with achieved ratio **0.054** — close to
the target but on the upper boundary. Widening to
`(0.5, 1.0, ..., 50.0)` and 1000 pairs, the chosen α was **50.0 (still grid
top)** with achieved ratio **0.040** — *not* converging to 0.05 even with
massive α.

### Why this happens (geometric explanation)

In 64-D with isotropic boxes, the volume of a single box is
`(2·half_width)^64` and the volume of a pairwise intersection scales as
the product across dimensions of per-dim hard-overlap side. The
*intersection-to-single-box volume ratio* is therefore the product across
dimensions of `(per-dim overlap fraction)`. To achieve a global ratio of
0.05 you need a per-dimension overlap of `0.05^(1/64) ≈ 0.954`: each
dimension's overlap must be ~95% of the box width on average.

This requires either:

1. Inter-chunk per-dim center gaps that are tiny compared to box widths
   (which means α must be enormous relative to kNN distances), or
2. A much lower box dimensionality.

Mandorla §3.3 F4 flags this exact failure mode ("high-dimensional
intersection sparsity"). The 64-D isotropic-box construction is more
susceptible to it than the paper's `~0.05` target anticipated.

### Practical implication

The α calibration is a *target*, not a *guarantee*. The chosen α is the
closest achievable on the grid; the achieved ratio is reported alongside.
For the screening slice, what matters is that α produces non-trivial
Vesicas that selectively retrieve relevant chunks — not that the absolute
intersection-to-volume ratio hits exactly 0.05.

PRECOMMIT.md Amendment 1 (dated below) re-frames the calibration so this
isn't a "miss against spec" but an explicit, principled choice. See that
amendment for the binding change.

### Recommended follow-ups (out of scope for the slice)

These belong to the broader Experiment 1, but worth recording:

- Use **anisotropic** per-dim half-widths derived from per-dim local
  variance, rather than isotropic from a scalar kNN distance. This breaks
  the 95%-per-dim requirement because dimensions with low signal variance
  contribute trivially.
- Try a **lower box dimensionality** (e.g., 32-D, 16-D) and compare
  vesica-coverage. Lower dim trades information for tractable intersection
  geometry.
- Try a **trained box head** (PRECOMMIT.md B3) where intersection volume
  is supervised by hyperlink co-occurrence — boxes wouldn't be derived
  from kNN at all.

### Shard-0 box index numbers

For traceability:

```
n             = 200000 chunks (shard 0 only)
d_box         = 64
k_nn          = 10
projection    = Gaussian random, seed=1337
kNN distances:  min=0.083, median=0.738, mean=0.727, max=1.015
α chosen      = 50.0 (grid top)
ratio target  = 0.05
ratio actual  = 0.040
```

These will be overwritten when the script is re-run against the full 5.2M-
passage encode. Full-corpus numbers will be added as a separate
LAB-NOTES.md entry then.

---

## 2026-05-11 — `PRECOMMIT.md` Amendment 2: §3.4 Build Path mapping

### Trigger

Audit-readiness review: a reviewer (peer review, hiring panel, future-self)
walking from a `RESULTS.md` claim back to "what was the original 12-week
program in `mandorla.md` §3.4, and how does this run fit into it?" should
find a single, frozen, structured answer. Prior to this amendment, the
mapping was implicit — scattered across the D1–D4 deferral decisions in
`PRECOMMIT.md`'s body, paper §3.4 itself, and a "What this slice tests /
does not test" subsection. None of those gave the W1-by-W12 enumeration
that a reviewer would scan for.

### What the amendment did

Adds a "Mapping to `mandorla.md` §3.4 'The Build Path'" table to
`PRECOMMIT.md` enumerating, week-by-week, what the slice does, defers, or
declares out-of-scope, with citations to the existing decisions that
explain why. Also calls out the **one numerical drift** from §3.4 — the
slice's 5.2M-passage `BeIR/hotpotqa` corpus is finer-grained than §3.4's
"~1M Wikipedia chunks" — and lays out three reasons that drift is
deliberate (canonical retrieval-formatted corpus, title-unique 1-to-1
mapping for vesica-coverage, hardware-feasible at the slice's scale).

### What did not change

No binding decision (D1–D4, A–F, decision rule, architecture spec, baselines,
sample size, ship criteria) is altered. The amendment is *informational* in
the strict sense — its purpose is to consolidate existing scope facts into
one auditable block, not to redirect the slice.

### Why this is a `PRECOMMIT.md` amendment rather than a `LAB-NOTES.md`-only entry

`CLAUDE.md` §1 specifies that any content change to `PRECOMMIT.md` goes
through a dated amendment block, regardless of whether the change is
binding-altering or informational. Adding a section to the body — even
one that doesn't reframe any decision — is still a content change. The
amendment block preserves a clean diff in the audit chain.

---

## 2026-05-11 — Pipeline code complete (scripts 01–09, src modules, 92 tests)

While the corpus encode runs (≈24% through 5.2M at this entry), the rest
of the pipeline was written so it's ready the moment the indices exist:

- **`src/`** — `regions.py` (Region / Vesica / BoxExtent per §2.2),
  `box.py` (GumbelBox closed-form intersect per Dasgupta 2020 Lemma 1),
  `projection.py` (seeded Gaussian random projection 768→64),
  `calibration.py` (α grid search), `retrieve.py` (DenseRetriever
  Protocol + numpy `BaselineRetriever` + production `FaissDenseRetriever`
  + `VesicaRetriever`), `eval.py` (HotpotQA F1/EM, vesica-coverage,
  bootstrap CIs), `generate.py` (Ollama wrapper, decoding pinned in code),
  `index_io.py` (artifact loaders + `QueryEncoder`), `runner.py` (the
  shared resumable eval loop), `data.py` (HotpotQA / corpus / shard
  loaders).
- **`scripts/`** — `01` HotpotQA pull, `02` corpus verify+audit, `03`
  contriever encode (resumable), `04` FAISS build, `05` box index + α
  calibration, `06` τ_v calibration, `07` baseline run, `08` Vesica-RAG
  run, `09` score → RESULTS.md.
- **`tests/`** — 92 unit tests across box math, projections, regions,
  calibration, retrieval (incl. FAISS-vs-numpy ranking agreement), eval
  metrics, generation prompt assembly, and the runner loop. All green.

Scripts 04–09 can't run end-to-end until `03` finishes and `04`+`05` are
re-run against the full corpus (the shard-0 smoke tests already validated
`04` and `05` plumbing). The slice README's "Reproduce" command sequence
was corrected in the same commit to match the actual script filenames
(it had stale placeholders `02_pull_wiki_dump.py` / `03_chunk_and_encode.py`
from when it was first drafted).

### Two interpretation choices made while writing scripts/09

1. **"vesica-coverage uplift" = raw Vesica-RAG coverage.** The PRECOMMIT.md
   decision table phrases the diagnostic as an "uplift" with a "+5 pp" bar.
   The contriever baseline forms no Vesicas, so its coverage is identically
   zero; the uplift over that baseline is just the raw Vesica-RAG coverage.
   `scripts/09` therefore checks `coverage ≥ 0.05` (GO) / `≥ 0.03` (WEAK).
   Recorded here so it's not a silent reinterpretation.
2. **Split-result handling.** The table requires *both* conditions for GO
   and *both* for WEAK GO; it doesn't explicitly say what to do when one
   metric clears GO and the other is below WEAK. `scripts/09` buckets such
   a result at the lower tier (per the literal "both required" reading) but
   attaches a "SPLIT RESULT — human adjudication" note in RESULTS.md rather
   than silently deciding. If a real split occurs, it's a candidate for a
   PRECOMMIT.md amendment to sharpen the rule.
