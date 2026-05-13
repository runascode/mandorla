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
