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
