# Mandorla — Project Lab Notes

Project-level chronological log. Per `CLAUDE.md` §1, each experiment also
keeps its own `LAB-NOTES.md` inside `experiments/{NN}-{slug}/`. This file
exists for events that affect the whole repository — infrastructure shifts,
identity / attribution changes, large dependency upgrades, repo-wide
refactors. Experiment-specific findings (throughput numbers, calibration
values, debugging) go in the per-experiment file.

Entries appended in chronological order; oldest at top.

---

## 2026-05-11 — Git identity rewrite (all commits to date)

### What happened

Every commit on `main` from the project's first commit (`fd7bcf5 MANDORLA
v1.0: position paper, three pre-registered experiments`) through
`bc7d5c3 Add CLAUDE.md` had author + committer set to
`Jacob Patterson <runascode@protonmail.com>`. That email is associated with the
`runascode` GitHub account, so the GitHub UI rendered every commit's
author tag as `runascode`.

The canonical project contact is `Jacob Patterson <runascode@protonmail.com>`
— matching `CITATION.cff`, `LICENSE-CODE`'s copyright line, and the
correspondence email in the paper's title block (`tex/mandorla.tex`).
Commit attribution should match.

### Procedure

```
git config user.name  "Jacob Patterson"
git config user.email "runascode@protonmail.com"
git rebase --root --exec 'git commit --amend --no-edit --reset-author'
git push --force-with-lease
```

`--force-with-lease` (not `--force`) so the push fails safely if the
remote has changed unexpectedly. 11 commits were rewritten.

### What changed

- **All commit SHAs.** New HEAD is `6d9b459`; the prior HEAD was `bc7d5c3`.
  The old SHAs no longer exist on the remote.
- **Author + committer** on every commit on `main` is now
  `Jacob Patterson <runascode@protonmail.com>`.
- **Commit content, messages, and order** are unchanged.

### Why this is a `LAB-NOTES.md` entry, not a `PRECOMMIT.md` amendment

This change does not alter any binding *design* decision about any
experiment — no dataset changes, no metric changes, no baseline changes.
The slice's `RESULTS.md` (when written) will still cite commit SHAs;
those SHAs will simply be the new ones. The audit chain
(`CLAUDE.md` §9) is intact: any reader walking from a future
`RESULTS.md` back to the producing commits will find them under the new
SHAs and under the correct attribution.

### Effect on the outer `runascode` super-repo

The outer repo at `~/Desktop/Projects/runascode` carries `mandorla` as a
submodule under `content/mandorla`. Its staged submodule pointer was
already uncommitted (held since the initial submodule add), so no
force-push of the outer repo is required. The local submodule clone was
reset to `origin/main` (now `6d9b459`) and re-staged in the outer repo's
index.

### Repo identity going forward

Repo-local git config now pins:

```
user.name  = Jacob Patterson
user.email = runascode@protonmail.com
```

Set per-repo (not globally), so the user's `runascode` identity on other
projects is unaffected.

---

## 2026-05-12 — Hardware / performance lessons (carry forward to future experiments)

From the Experiment 1 slice runs on the dev machine (Apple M4 Pro, 48 GB).
Detailed accounts are in `experiments/exp1-vesica-rag/LAB-NOTES.md`; the
takeaways for *any* future experiment on this class of hardware:

1. **LLM inference is GPU-bound and does not parallelize on a single GPU.**
   Ollama running an 8B Q5 model already saturates the Metal GPU with one
   request ("100 % GPU" in `ollama ps`). Setting `OLLAMA_NUM_PARALLEL > 1`
   and issuing concurrent requests just splits the GPU's fixed token
   throughput across sequences — same aggregate tokens/s, N× per-request
   latency, more in-flight work to lose on a crash. The fix isn't
   concurrency; it's a smaller/faster model, fewer prompt tokens, or a
   faster GPU. Plan wall-clock as `n_questions × tokens_per_question ÷
   GPU_tokens_per_sec`, not `÷ n_workers`. Freeing RAM/CPU helps general
   machine health but won't move this number.

2. **Brute-force array scans dominate when they're O(N) per query.** The
   slice's per-Vesica box-containment step scanned the full 5.23M × 64
   box-centers array (`(centers >= min) & (centers <= max)).all(axis=1)`)
   ~5×/question — ~4–5 s/question of numpy, half the Vesica-RAG run's
   per-question time. Anything that does a linear scan over millions of
   rows inside a per-item loop should use a spatial index (FAISS over the
   relevant vectors, then filter) instead — it's the *same* computation,
   done in ms instead of seconds. Build the helper indices up front.

3. **`uv run`-launched env vars don't reach a running service.**
   `OLLAMA_NUM_PARALLEL` (and similar) are read by `ollama serve` at
   start-up; setting them on the client process does nothing. If you need
   a server-side knob changed, restart the server.

4. **macOS bundled-OpenMP conflict (`OMP Error #15`).** `faiss-cpu` and
   `torch` each ship their own `libomp` on macOS arm64; importing faiss
   *before* torch crashes the process. Load torch first (e.g. via the
   module that imports it), faiss later, and set FAISS thread counts after
   both are loaded. Don't `import faiss` at module top in a script that
   also pulls in torch.

5. **Per-worker OpenMP oversubscription.** If you ever do run a worker
   pool, pin FAISS/BLAS to 1 OpenMP thread per worker
   (`faiss.omp_set_num_threads(1)`, `OMP_NUM_THREADS=1`); otherwise each
   worker fans out across all cores and the machine thrashes.

6. **A multi-GB resident index scanned per query is a single-machine
   memory wall.** The slice's 16 GB `IndexFlatIP` (5.23M × 768, exact
   brute-force search) is touched in full on every query. One worker → it
   stays resident, fine. Two concurrent workers searching it → the OS
   pages it in and out from disk per search and throughput collapses (we
   saw ~7× *slower* than single-worker). You can't parallelize over such
   an index without N× the RAM. A second machine doesn't help either —
   the bottleneck (retrieval, holding the big index) is exactly the part
   that can't be replicated cheaply; only generation offloads, and only
   if it's *pipelined* off the retrieval thread (one scan in flight at a
   time → index stays resident → a separate pool of generation threads,
   some pointed at the remote host, consumes a queue). The cheaper fixes:
   shrink the index (IVF/PQ/HNSW instead of flat, accepting some recall
   loss) so it fits with headroom, or accept single-worker throughput.
   Decide this *before* the index is built.

---

## 2026-05-13 — Experiment 01 complete; branching decision

### What happened

The Experiment 01 screening slice ran end-to-end on HotpotQA dev
(7,405 questions). Headline F1 lift **−1.64** (95% CI −2.34, −0.93);
vesica-coverage **4.36%** (CI 3.90, 4.82). Per the locked
`exp1-vesica-rag/PRECOMMIT.md` decision rule (F1 lift bar +1.0 weak /
+2.0 GO **and** coverage bar +3 / +5 pp, both required for any GO
outcome), this is **NO-GO**. Full numbers and provenance in
`exp1-vesica-rag/results/RESULTS.md`; commit `5a3b34a`.

### Diagnostic on the raw outputs (not part of the binding decision)

Three hypotheses on offer for why a +4.36 pp vesica-coverage signal
didn't move F1:

- **H1 — coverage is the bottleneck.** Falsified. On the n=323
  coverage-hit subset (gold pair surfaced as a candidate Vesica),
  F1 lift is **−0.01 (CI −0.04, +0.02)**. Bridge & covered (n=190):
  −0.02 (CI −0.06, +0.02). Comparison & covered (n=133): 0.00 (CI
  −0.06, +0.06). At these n's a real ≥+1 F1 lift would have been
  detectable; it is not present.
- **H2 — Vesicas don't help the LLM.** Supported by the same evidence:
  on the perfect-coverage subset, no lift.
- **H3 — Vesicas displace useful baseline context.** Quantified. Mean
  baseline context: 25.0 chunks. Mean Vesica-RAG context: 23.8 chunks.
  Mean overlap: 8.3. Mean baseline-only chunks dropped: 16.7.
  Gold-pair-in-context drops from **41.39%** (baseline) to **30.74%**
  (Vesica-RAG) — an 11 pp cost from displacement. But, per H1, even
  preserving the gold pair would not have moved F1.

The screening was effectively an **LLM-saturation test**: Llama-3.1-8B
at top-25 dense retrieval on HotpotQA dev already extracts essentially
all the signal that the intersection primitive (B2 density-extent
boxes over a 64-D random projection of contriever) could deliver. The
slice's design confounded two separable questions (does the primitive
retrieve different evidence; does the LLM benefit) and answered the
second well but couldn't speak to the first.

Full diagnostic at `exp1-vesica-rag/results/DIAGNOSTIC.md`.

### Lesson #7 — pre-commits must isolate the variable they intend to test

The screening slice's `PRECOMMIT.md` was correct for the question it
asked ("does Vesica-RAG produce better answers on HotpotQA dev"). But
the *thesis-level* claim it was implicitly testing ("does the
intersection primitive surface better evidence") required isolating
retrieval from downstream consumption. The slice produced a clean
answer to the wrong question.

Going forward: every future PRECOMMIT must include an explicit
sub-section on **what the experiment confounds with what** and an
explicit decision about whether to isolate or accept the confound.
If a future experiment's downstream model is plausibly saturated at
the baseline retrieval, the experiment is *not* testing the retrieval
claim and that must be stated.

### Branching decision (the queue now)

- **Experiment 02 — retrieval-isolation** is locked
  (`experiments/02-retrieval-isolation/PRECOMMIT.md`, 2026-05-13). It
  tests the primitive on Pair-Recall@25 across three multi-hop
  datasets (HotpotQA fullwiki, 2WikiMultiHop, MuSiQue-Ans dev) with
  *no LLM in the loop*. Reuses the FAISS + box indices bit-for-bit
  from Exp 01. Decision rule: ≥+5 pp on all three for GO, ≥+3 pp on
  ≥2 of 3 (no dataset worse by >1 pp) for WEAK GO, else NO-GO. Budget:
  one week.
- **Experiment 03 — Hex-Vote** and **Experiment 04 — Mandorla
  Curriculum** are scaffolded as design sketches
  (`experiments/03-hex-vote/README.md` and
  `experiments/04-curriculum/README.md`) with the blocking decisions
  that must be resolved before each gets its PRECOMMIT lock. Neither
  depends on Experiment 02's verdict for *correctness*, but Experiment
  02's verdict will drive *prioritization*:
  - If Exp 02 is GO → the retrieval-form of the thesis survives →
    follow up with B3 (learned box head) before committing to Exp 03
    or 04.
  - If Exp 02 is NO-GO → the in-query intersection primitive is dead
    in this projection → Exp 03 (Hex-Vote) becomes the highest-
    priority surviving falsifiable test.

### Note on the published slice post

The post at `runascode.com/results/vesica-rag-slice` (and the slice's
`RESULTS.md`) remains binding and unchanged per `PRECOMMIT.md` ship
discipline. The diagnostic and the next-experiment decision live in
the repo, not in the slice's `RESULTS.md`, so the audit chain stays
clean: the slice's numbers are what they are; what we *did with* those
numbers is its own document.

---

## 2026-05-15 — Experiment 02 complete; retrieval form of Thesis 2 closed

### What happened

Experiment 02 (retrieval-isolation) ran to completion: 22,398 dev
questions across HotpotQA (7,405), 2WikiMultiHop (12,576), and
MuSiQue-Ans (2,417); single worker; ~33 h wall; retrieval-only, no
LLM. Decisive **NO-GO**.

Pair-Recall@25 lift (Vesica-augmented − contriever baseline):

| Dataset | Lift | 95% CI |
|---|---|---|
| HotpotQA | −10.65 pp | (−11.41, −9.91) |
| 2Wiki | −6.99 pp | (−7.52, −6.48) |
| MuSiQue | −3.23 pp | (−4.01, −2.48) |

Every CI is clear of zero on the negative side; every secondary
metric (Pair-Recall@10, Any-Gold-Recall@25, RR-of-first-pair) is
negative on all three. The intersection primitive doesn't fail
neutrally — it *actively degrades* gold-pair retrieval, because the
Vesica contained-chunk sets evict higher-value contriever hits under
the 25-chunk budget. This is H3 from the slice diagnostic, now
confirmed at the pure-retrieval level with no LLM confound.

Internal-consistency check passed: HotpotQA baseline Pair-Recall@25
(41.39%) equals the slice diagnostic's baseline gold-pair-in-context
(41.39%) to the digit — two independently written pipelines agree.

### Lesson #8 — a confounded NO-GO is worth de-confounding before pivoting

Exp 01's NO-GO was real but appealable ("LLM saturation hid the
signal"). The cheap, decisive move was not another full RAG variant
but a *retrieval-only* test on more datasets — same indices, no LLM,
~1 week of work. It converted an appealable negative into an
unappealable one. When a screening result is confounded, the next
experiment should remove the confound, not add scope. Doing this
first prevented sinking weeks into B3 / formal-Exp-1 on a premise
(the primitive retrieves better evidence) that was false all along.

### Branching decision (the queue now)

The retrieval form of Thesis 2 is **closed for this operationalization**
(B2 density-extent boxes, contriever-derived 64-D random projection,
in-query, no store). Two screening experiments, three datasets, two
pipelines, headline + four secondary metrics — all negative,
decisively so in Exp 02.

- **Not pursued:** formal Vesica-RAG (full RAG on MuSiQue/2Wiki), B3
  as a drop-in box swap. B3 *could* still be revisited, but only with
  a redesigned retrieval-assembly that fixes the displacement
  mechanism (greedy contained-chunk sets under a finite budget) — that
  is a new experiment with a new PRECOMMIT, not a box swap, and it now
  carries a heavier burden of proof (beat "actively harmful," not
  "no signal").
- **Highest-priority surviving test:** Experiment 03 (Hex-Vote).
  Different projection of the thesis (multi-agent topology, Thesis 1),
  entirely different infrastructure (LLM message-graph, no FAISS, no
  16 GB index, no single-machine memory wall). Its blocking decisions
  are enumerated in `experiments/03-hex-vote/README.md`; the next
  PRECOMMIT lock should be there.
- **Independently citable byproduct:** the LLM-saturation finding from
  the Exp 01 diagnostic. It limits what HotpotQA-style RAG benchmarks
  can teach about *any* retrieval intervention, not just Vesica-RAG.
  Worth a standalone short writeup decoupled from the MANDORLA thesis;
  it is the result here most likely to be cited by people who don't
  share the MANDORLA priors.

### Publication

Per the published-results discipline, Exp 02's NO-GO is published the
same way the slice's was. The slice blog post at
`runascode.com/results/vesica-rag-slice` gets a second clearly-labeled
follow-up section (the post text is additive-only, like the diagnostic
section before it; the slice's binding numbers are never edited). The
new Exp 02 numbers are their own `RESULTS.md` with their own producing
commit, so the audit chain stays intact.

---

## 2026-05-15 — Curriculum (Thesis 3) de-risked: pilot Q1 PASS, Q2 NO

### What happened

Before committing the 8–12-week, multi-GPU Experiment 04, a cheap
synthetic pilot (`experiments/04-curriculum/pilot/`, local MPS, ~3 h
total including a falsified first design) de-risked its load-bearing
assumption. The pilot's pre-stated rule: Q1 = do the box losses train
without collapse; Q2 = does the curriculum produce held-out
compositional transfer beyond plain CLM *and* beyond a capacity-matched
generic-auxiliary control. Outcome: **Q1 PASS, Q2 NO** (3 conditions ×
3 seeds × 6000 steps).

- Q1: curriculum vesica/parent losses → 0.000, bit-acc 1.000,
  intersection log-volume finite/well-dispersed (~80±18, no collapse).
  Fully trainable inside a from-scratch transformer.
- Q2: comp-OOD F1@G — baseline 0.588±0.014, generic_aux 0.580±0.012,
  curriculum 0.570±0.017. curriculum − generic_aux = −0.009
  (−0.65 pooled SD); curriculum marginally *worst*. No transfer.

### Lesson #9 — the most informative negative is "trains perfectly, inert"

The first pilot world (R0) was falsified by its own baseline in ~30
min (held-out *pairs* + individually-observed entities = trivial
recall+AND; baseline ceilinged). The redesign (R1, 2-hop latent
composition, entity-level held-out, generic-aux control) produced a
*much* stronger result than a Q1 failure would have: the mechanism
trains flawlessly and is **inert** for the capability it was posited
to induce. "It won't optimize" is weak evidence (maybe you tuned it
wrong). "It optimizes perfectly and changes nothing downstream" is
strong evidence about the *idea*, not the implementation. Design
screening experiments so the informative failure mode is the latter,
not the former — and de-risk the load-bearing assumption *before*
pre-registering the expensive run, never after.

### Branching decision — the queue after three negatives

Of the three falsifiable projections cheap enough to screen, all three
are now negative:

- **Retrieval (Thesis 2):** Exp 01 (confounded NO-GO) + Exp 02
  (decisive, unconfounded NO-GO). Closed for this operationalization.
- **Recursive construction (Thesis 3):** Exp 04 pilot — assumption
  fails in clean isolation; full Exp 04 not pre-registered.

Remaining: **Experiment 03 (Hex-Vote)**, Thesis 1 (geometric primacy
in multi-agent topology). Genuinely independent infrastructure (LLM
message-graph, no FAISS, no training run) so it does not inherit the
other two's fate. But it is now the *last* cheap falsifiable test, and
its prior should be set honestly: the two more-directly-testable
projections of the same underlying thesis both failed *mechanistically*
(displacement; inert-transfer), not marginally. Recommended sequencing,
unchanged from the post-Exp-02 read and now reinforced:

1. Write up the **LLM-saturation finding** as a standalone artifact —
   highest value/effort, true independent of the thesis, the result
   most likely to be cited by non-believers.
2. Treat **Exp 03** as a real but low-prior bet, not momentum. If it
   is run, its PRECOMMIT must include the same "isolate the variable"
   and "capacity-matched control" discipline the curriculum pilot
   added (Lesson #8/#9), or a positive result won't be attributable.
3. The honest standing for the position paper is now: a pre-registered
   program with **two disconfirmed projections, one assumption-failed
   projection, one untested projection, and one portable methodology
   result** — a stronger and rarer artifact than an unfalsified
   manifesto, but it is no longer a paper whose central empirical bet
   is open. That reframing is the user's call; the record supports it.

### Publication

Same discipline as Exp 01/02: the pilot is pre-PRECOMMIT and binds
nothing, but the outcome is published, not buried. The slice blog post
gets a third additive follow-up section (slice numbers never edited);
the pilot's own numbers live in `pilot/LAB-NOTES.md` + `pilot/PILOT.md`
"OUTCOME" with their producing commit, audit chain intact.

---

## 2026-05-15 — Operator pilot: last cheap escape closed; program complete

The curriculum pilot's negative had one residual rescue: intersection
was an *auxiliary* loss with a CLM bypass, so maybe
intersection-as-the-actual-computation would help. A second pilot
(`experiments/04-curriculum/pilot2-operator/`, rule locked in PILOT2.md
before any number) put the intersection operator **on the critical
path, no bypass, capacity-matched** against a point bottleneck, on an
intersective AND task *and* a non-intersective control.

Result: **DECISIVE NEGATIVE.** comp-OOD F1@G — AND: POINT 0.587±0.016
vs INTERSECTION 0.573±0.008 (Δ −0.013); NONINT: POINT 0.540±0.011 vs
INTERSECTION 0.525±0.011 (Δ −0.015). Negative on both tasks, all three
seeds, capacity matched (int/point params 0.965). The intersection
bottleneck loses to point composition *even on the task whose ground
truth is literally a set intersection*.

### Lesson #10 — the purity ratchet bottoming out IS the result

The full series: every disconfirmation was met with a plausible "not
tested fairly" rescue, and every rescue, once its condition was
satisfied, produced another negative.

| Negative | Rescue offered | Rescue satisfied by | Outcome |
|---|---|---|---|
| Exp 01 (RAG) | "LLM saturated" | Exp 02 (no LLM) | still negative |
| Exp 02 (retrieval) | "borrowed point geometry" | curriculum pilot (trained from scratch) | still negative |
| curriculum pilot | "only an auxiliary loss / bypass" | operator pilot (critical path, no bypass, capacity-matched) | still negative |
| operator pilot | "architecture isn't intersection-native" | — (multi-year program, not a pilot) | not paid |

The remaining rescue is not an experiment. And the *pattern* — an
ever-purer, ever-costlier precondition demanded after each negative,
each one failing the moment it is met — is the canonical behavior of a
false thesis, not a mistested true one. The disciplined terminus is to
name the pattern and stop, which is what this entry does.

### Branching — terminal state of the cheap program

- **Retrieval (Thesis 2):** closed (Exp 01 + 02).
- **Construction (Thesis 3):** closed (curriculum pilot + operator
  pilot), no PRECOMMIT, full Exp 04 not undertaken.
- **Geometric primacy (Thesis 1 / Hex-Vote):** still untested,
  independent infrastructure, but prior is now very low and it is no
  longer "the momentum direction" — running it is only justified if the
  goal is to fully resolve the thesis for the paper's sake, with the
  same locked-rule + capacity-control discipline, and a cheap pilot
  first. Not running it is equally defensible and is the
  recommendation's default.
- **Thesis-independent assets:** `findings/reader-saturation-hotpotqa.md`
  (the portable result; ready to publish standalone) and the
  methodology itself (a fully audited, pre-registered, repeatedly-
  disconfirmed program — the rarer artifact).

The recommended next action remains the saturation note's standalone
publication and repositioning the position paper from manifesto to "a
pre-registered program and its disconfirmation." The record now fully
supports that reframing; it is the user's editorial call. No further
experiment is the disciplined default.
