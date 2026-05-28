# Stage A — Substrate-Rethinks Literature Survey: Implementation Plan

> **For agentic workers:** REQUIRED — use [superpowers:subagent-driven-development](https://github.com/superpowers) (if subagents available) or [superpowers:executing-plans](https://github.com/superpowers) to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce v1 of [`findings/substrate-rethinks-survey.md`](../../findings/substrate-rethinks-survey.md) — a thesis-independent literature survey of non-point / non-token / non-autoregressive substrate attempts. Two functions: (a) inform Gate G1 (does Stage B of Exp 06 need to run at all?), (b) stand on its own as a citable artifact independent of MANDORLA.

**Architecture:** Iterative reading-and-synthesis. Five lineages, surveyed in parallel reading streams. Each lineage gets one matrix row, one prose section, and a "what was attempted / at what scale / what failed / why" assessment. After the matrix is complete, the load-bearing claim of Stage B is sharpened from a paragraph (current [`PILOT-DESIGN.md` §5.1](PILOT-DESIGN.md)) to a single PRECOMMIT-ready sentence. Gate G1 decision recorded last.

**Tech Stack:** Markdown only. No code. Working bibliography tracked inline in `READING-LIST.md` (project-internal; BibTeX not required for the survey itself). Existing [`PILOT-DESIGN.md`](PILOT-DESIGN.md) is the working spec.

**Budget:** 1–2 person-weeks reading, ~50–100 papers shortlisted, ~15–25 read deeply.

**Master seed**: N/A (no code).

**Spec reference:** [`experiments/06-substrate-pilot/PILOT-DESIGN.md`](PILOT-DESIGN.md) §4 (Stage A scope, deliverable, output that B needs).

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| [`experiments/06-substrate-pilot/READING-LIST.md`](READING-LIST.md) | new | Paper shortlist by lineage; one row per paper with citation, scale, axes-moved-flag |
| [`experiments/06-substrate-pilot/READING-NOTES.md`](READING-NOTES.md) | new | Per-paper deep-read notes by lineage; one §-section per paper read deeply |
| [`experiments/06-substrate-pilot/LAB-NOTES.md`](LAB-NOTES.md) | new | Chronological progress log per [`CLAUDE.md`](../../CLAUDE.md) §1 (per-experiment LAB-NOTES) |
| [`findings/substrate-rethinks-survey.md`](../../findings/substrate-rethinks-survey.md) | stubbed | The deliverable; populated as tasks complete |
| [`experiments/06-substrate-pilot/PILOT-DESIGN.md`](PILOT-DESIGN.md) | exists | Amendment block at bottom records the sharpened load-bearing claim (per [`CLAUDE.md`](../../CLAUDE.md) §7) |

Working notes (`READING-LIST.md`, `READING-NOTES.md`) are committed for the audit chain — a reader walking from the deliverable back through to source papers must be able to do so without missing links. They are *not* polished prose.

---

## Task 0: Scaffold

**Files:**
- Create: `experiments/06-substrate-pilot/READING-LIST.md`
- Create: `experiments/06-substrate-pilot/READING-NOTES.md`
- Create: `experiments/06-substrate-pilot/LAB-NOTES.md`

- [ ] **Step 1: Create `READING-LIST.md`** with the 5 lineage headings (SDM/HDC; Box/region embeddings; Topological/geometric DL; Predictive coding/FEP for language; Non-AR seq / latent diffusion LMs) and an empty paper table under each:

```markdown
| # | Citation | Year | Scale | Axes moved (I/R/O) | Shortlist tier | Deep-read? |
|---|---|---|---|---|---|---|
```

Tiers: T1 = canonical / load-bearing for the lineage; T2 = significant; T3 = adjacent.

- [ ] **Step 2: Create `READING-NOTES.md`** with the same 5 lineage headings, empty bodies, and a per-paper note template at top:

```markdown
### [Citation, Year]

**Claim:**
**Method (in one paragraph):**
**Scale demonstrated:**
**Axes moved:** input ☐ representation ☐ objective ☐
**Failure mode / limitation (if any, at what scale):**
**Why it does / does not bear on "all three together":**
**Relation to MANDORLA (if any):**
```

- [ ] **Step 3: Create `LAB-NOTES.md`** with the header pattern from the project's other experiment LAB-NOTES files (read [`01-vesica-rag/LAB-NOTES.md`](../01-vesica-rag/LAB-NOTES.md) for reference). First entry:

```markdown
## 2026-05-28 — Stage A begins

Reading begins per [`STAGE-A-PLAN.md`](STAGE-A-PLAN.md). Scaffolding (this file, `READING-LIST.md`, `READING-NOTES.md`) created. No reading done yet.
```

- [ ] **Step 4: Verify acceptance criteria for Task 0**

- `READING-LIST.md`, `READING-NOTES.md`, `LAB-NOTES.md` all present in `experiments/06-substrate-pilot/`.
- Each has the 5 lineage headings.
- `READING-NOTES.md` carries the per-paper template.
- `LAB-NOTES.md` has the opening entry dated 2026-05-28.

- [ ] **Step 5: Commit**

```bash
git add experiments/06-substrate-pilot/READING-LIST.md \
        experiments/06-substrate-pilot/READING-NOTES.md \
        experiments/06-substrate-pilot/LAB-NOTES.md
git commit -m "Exp 06 Stage A: scaffold READING-LIST / READING-NOTES / LAB-NOTES"
```

---

## Task 1: Shortlist papers per lineage

Five lineages, each shortlisted independently. Target: ~10–20 papers per lineage, tiered T1 / T2 / T3. The T1 set + a sampling of T2/T3 is what gets deep-read in Task 2.

**Files modified:**
- `experiments/06-substrate-pilot/READING-LIST.md`
- `experiments/06-substrate-pilot/LAB-NOTES.md` (one chronological entry per lineage completed)

### Task 1.1: SDM / HDC

- [ ] **Step 1**: Seed citations from MANDORLA paper (`mandorla.md` references Kanerva; Plate; the Hawkins / Thousand Brains line). Include them as T1.
- [ ] **Step 2**: Add the Schlegel et al. comparative review of HDC/VSA architectures.
- [ ] **Step 3**: Search Google Scholar / Semantic Scholar for citing papers of (a) Kanerva 1988 (SDM); (b) Plate 1995 (HRRs); (c) Schlegel et al. ~2022 review. Stop at ~30 candidates.
- [ ] **Step 4**: Triage into T1/T2/T3 by load-bearingness — does the paper claim a *substrate-level* contribution (input / representation / objective replacement at scale), or only mathematical / cognitive-modeling? T1 only for substrate claims.
- [ ] **Step 5**: For each paper, fill the table row with citation, year, claimed scale, *provisional* axes-moved flags, tier.
- [ ] **Step 6**: Append a `LAB-NOTES.md` entry: "SDM/HDC shortlist complete: N candidates, T1 = m, T2 = k, T3 = j. Surprises: ..."
- [ ] **Step 7**: Commit.

```bash
git add experiments/06-substrate-pilot/READING-LIST.md \
        experiments/06-substrate-pilot/LAB-NOTES.md
git commit -m "Exp 06 Stage A: SDM/HDC shortlist"
```

**Acceptance**: Table row populated for ≥10 SDM/HDC papers, ≥3 of them T1. LAB-NOTES entry written.

### Task 1.2: Box / region embeddings

- [ ] **Step 1**: Seed from MANDORLA's citations (Vilnis & McCallum; Li, Patel, Boratko; Lai & Domeniconi).
- [ ] **Step 2**: Follow McCallum group's recent publications up through 2024–2025 (Boratko's PhD line).
- [ ] **Step 3**: Include any non-McCallum region/set embedding work (Gumbel boxes, density boxes, hyperbolic / probabilistic alternatives).
- [ ] **Step 4–6**: Triage, populate, LAB-NOTES entry, as in Task 1.1.
- [ ] **Step 7**: Commit.

```bash
git add experiments/06-substrate-pilot/READING-LIST.md \
        experiments/06-substrate-pilot/LAB-NOTES.md
git commit -m "Exp 06 Stage A: box/region embeddings shortlist"
```

**Acceptance**: ≥10 papers shortlisted, ≥3 of them T1; LAB-NOTES entry written. Particular attention to scale — most box-embedding work is small. The empty-cell observation in the matrix is likely strongest here.

### Task 1.3: Topological / geometric deep learning

- [ ] **Step 1**: Seed from MANDORLA's citations (Bronstein–Bruna–Cohen–Veličković 2021 "Geometric Deep Learning"; Bodnar's cell-complex line).
- [ ] **Step 2**: Add the simplicial-network and message-passing-on-complexes lineage (Bodnar et al., the GDL workshop papers).
- [ ] **Step 3**: Note carefully which works *replace the representation primitive* (the relevant axis for our matrix) vs. which only add structure on top of point embeddings.
- [ ] **Step 4–6**: Triage, populate, LAB-NOTES.
- [ ] **Step 7**: Commit.

```bash
git add experiments/06-substrate-pilot/READING-LIST.md \
        experiments/06-substrate-pilot/LAB-NOTES.md
git commit -m "Exp 06 Stage A: topological/geometric DL shortlist"
```

**Acceptance**: ≥10 papers shortlisted, ≥3 of them T1; LAB-NOTES entry written. The "axes moved" classification is the value-add — most TDL papers move the representation axis in a graph/complex sense, not in the region-vs-point sense. The matrix row should record both observations.

### Task 1.4: Predictive coding & active inference for language

- [ ] **Step 1**: Seed from Friston's recent active-inference / FEP-for-language work; Whittington–Behrens TEM; Bogacz's modernizations of predictive coding.
- [ ] **Step 2**: Add the predictive-coding-as-backprop-alternative line (Millidge, Tschantz, Whittington).
- [ ] **Step 3**: Add any active-inference-for-LLMs / for-sequence-modeling work that has appeared in the last 24 months.
- [ ] **Step 4–6**: Triage, populate, LAB-NOTES.
- [ ] **Step 7**: Commit.

```bash
git add experiments/06-substrate-pilot/READING-LIST.md \
        experiments/06-substrate-pilot/LAB-NOTES.md
git commit -m "Exp 06 Stage A: predictive coding / active inference shortlist"
```

**Acceptance**: ≥10 papers shortlisted, ≥3 of them T1; LAB-NOTES entry written. This is the lineage where "objective primitive" claims most overlap with substrate-level claims; flag carefully.

### Task 1.5: Non-AR sequence models & latent diffusion LMs

- [ ] **Step 1**: Seed: LeCun's JEPA papers; Bakhtin et al. on energy-based LMs; Lou & Ermon SEDD; Gulrajani et al. diffusion-LM line; Sahoo et al. masked-diffusion LMs.
- [ ] **Step 2**: Include the byte-level / token-free line (ByT5, Mamba/MambaByte, RWKV) as adjacent — they move the *input* axis but not the others.
- [ ] **Step 3–5**: Triage with extra care — this is the *largest-scale* substrate-rethink lineage. Note demonstrated scale carefully; "the largest non-AR LM trained is X-B params on Y-T tokens" is the headline.
- [ ] **Step 6**: LAB-NOTES entry should highlight the demonstrated-scale numbers for this lineage specifically.
- [ ] **Step 7**: Commit.

```bash
git add experiments/06-substrate-pilot/READING-LIST.md \
        experiments/06-substrate-pilot/LAB-NOTES.md
git commit -m "Exp 06 Stage A: non-AR / latent diffusion LM shortlist"
```

**Acceptance**: ≥10 papers, ≥3 T1. Demonstrated-scale numbers recorded explicitly per T1 paper.

---

## Task 2: Deep reads + notes

For each lineage, deep-read the T1 set plus a sampling of T2/T3 selected for what they reveal about *axes moved* and *scale demonstrated*. Target: 15–25 papers total, written up using the per-paper template in `READING-NOTES.md`.

**Files modified:**
- `experiments/06-substrate-pilot/READING-NOTES.md` (one § per deep-read paper)
- `experiments/06-substrate-pilot/LAB-NOTES.md` (chronological — one entry per ~5 papers or per surprise)

### Task 2.1–2.5: One per lineage

Each lineage's deep-read task has the same shape:

- [ ] **Step 1**: From `READING-LIST.md`, identify the deep-read set: all T1, plus ~1–3 T2 that fill specific gaps (e.g., "newest", "largest-scale", "most-cited").
- [ ] **Step 2**: For each paper in order: read; complete the per-paper template in `READING-NOTES.md`; flag axes moved (I / R / O). One commit per paper.

```bash
git add experiments/06-substrate-pilot/READING-NOTES.md
git commit -m "Exp 06 Stage A: deep-read [Author Year]"
```

- [ ] **Step 3**: After the lineage's deep reads are done, append a `LAB-NOTES.md` entry summarizing: how many papers deep-read, axes-moved tally across the lineage, surprises, lineage-level scale demonstrated.

**Acceptance per lineage**: ≥3 deep-read § sections, each completing the template fully. Axes-moved flags set on the basis of paper content, not the shortlist guess.

**Acceptance overall**: 15–25 deep-read § sections in `READING-NOTES.md` across the five lineages.

---

## Task 3: Fill the matrix

The matrix is the central structural artifact. It lives in [`findings/substrate-rethinks-survey.md`](../../findings/substrate-rethinks-survey.md) and is what readers from outside MANDORLA will engage with first.

**Files modified:**
- `findings/substrate-rethinks-survey.md`

- [ ] **Step 1**: Open `findings/substrate-rethinks-survey.md`. Replace the empty matrix at "## Matrix (empty — to be filled by v1)" with a populated version. Each row pulls from the `READING-NOTES.md` lineage section and the `READING-LIST.md` table.

```markdown
| Lineage | Input axis moved? | Representation axis moved? | Objective axis moved? | Demonstrated scale |
|---|---|---|---|---|
| SDM / HDC | [Y/N — with one-line justification] | [Y/N — ...] | [Y/N — ...] | [largest-scale demonstrated; cite] |
| ... | | | | |
```

- [ ] **Step 2**: For each cell, the justification is ≤1 sentence and cites a specific paper from the deep-read set. Vague "yes, in principle" justifications are not acceptable.
- [ ] **Step 3**: The "demonstrated scale" cell cites the single largest credible scale a working model in that lineage has achieved (e.g., "SEDD at ~1B params", "predictive-coding nets at MNIST-CIFAR", etc.).
- [ ] **Step 4**: Commit.

```bash
git add findings/substrate-rethinks-survey.md
git commit -m "Exp 06 Stage A: substrate-rethinks survey matrix populated"
```

**Acceptance**: 5 rows, all cells filled, each cell with citation. Empty cells (no, axis not moved at credible scale by this lineage) are *valuable* — they motivate Stage B — and must be explicitly justified, not skipped.

---

## Task 4: Write lineage sections

One prose section per lineage in `findings/substrate-rethinks-survey.md`, between the matrix and the existing "Provenance and independence" section. Each section ~400–800 words.

**Files modified:**
- `findings/substrate-rethinks-survey.md`

### Task 4.1–4.5: One per lineage, same shape

- [ ] **Step 1**: Draft the lineage section. Required structure:

```markdown
### Lineage N: [Name]

**Claim.** [What does this lineage assert about the substrate?]

**Method.** [How does it operationalize that claim — in 2–3 sentences. No more.]

**Demonstrated scale.** [Largest credible working system, with citation. Bare numbers: params, tokens, benchmark.]

**What it actually moves.** [Input / representation / objective — with the matrix row's justification expanded to a paragraph.]

**Failure modes at scale (if reported).** [Where the lineage hits a wall, with citation; or "no credible scale-up has been attempted in the open literature."]

**Bearing on 'all three together'.** [Does this lineage's evidence inform — or rule out — the joint claim in PILOT-DESIGN §5.1? One paragraph.]
```

- [ ] **Step 2**: Cite specific papers, not "the literature." Every claim that could be wrong needs a citation. Style: light, in-line citations with author–year.
- [ ] **Step 3**: Verify against the lineage's `READING-NOTES.md` section that nothing important is omitted.
- [ ] **Step 4**: Commit.

```bash
git add findings/substrate-rethinks-survey.md
git commit -m "Exp 06 Stage A: survey — [Lineage Name] section"
```

**Acceptance per lineage**: Section follows the structure; every cell of the matrix row is reflected in the section's "What it actually moves" paragraph; the "Bearing on 'all three together'" paragraph is not hedged into uselessness.

---

## Task 5: Sharpen the load-bearing claim

Stage B's PRECOMMIT cannot be locked until Stage A has produced a single PRECOMMIT-ready sentence that operationalizes M, Δ_warrant, and task family. This task produces it.

**Files modified:**
- `findings/substrate-rethinks-survey.md` (new section "Operationalized load-bearing claim for Exp 06 Stage B")
- `experiments/06-substrate-pilot/PILOT-DESIGN.md` (amendment block at bottom, per [`CLAUDE.md`](../../CLAUDE.md) §7)

- [ ] **Step 1**: Re-read the matrix and the five lineage sections. Identify what effect sizes the literature reports as "meaningful" on the relevant tasks. Pick a Δ_warrant numeric grounded in those reports (default 0.05 if reports are silent).

- [ ] **Step 2**: Choose the toy task *family* within region-prediction (intersection / union / hierarchical containment per [`PILOT-DESIGN.md`](PILOT-DESIGN.md) §5.5). The specific generative process — distribution over (centre, extent) pairs in ℝ^d, sequence length, train/val/test sizes — is **selected here as a default** (e.g., intersection of two input regions in d=4 or 8) **and locked in PRECOMMIT** after the headroom check (§5.5) passes. The chosen task must satisfy the design constraint quoted from `PILOT-DESIGN.md` §5.5: *"the region representation of inputs and outputs is not optional information — point representations of either strictly lose what the task requires."* Document this constraint and how the chosen task satisfies it, in 1–3 sentences. Document the provisional metric M (default: containment-IoU on the predicted output region; alternatives reported alongside).

- [ ] **Step 3**: Draft the operationalized claim as a *single sentence* in `findings/substrate-rethinks-survey.md`:

```markdown
## Operationalized load-bearing claim for Exp 06 Stage B

> On the [task family] with [data spec], a region-native
> (input × representation × objective) model outperforms every
> "drop-one-axis" variant at matched parameter count and matched
> per-step FLOPs by ≥ [Δ_warrant] on [M] with 95% CI clear of zero
> across [3+] seeds.
```

- [ ] **Step 4**: Verify the sentence is actually falsifiable — read it adversarially. Could a reasonable second researcher lock a PRECOMMIT from this sentence alone? If not, sharpen.

- [ ] **Step 5**: Append an amendment to `PILOT-DESIGN.md` (after §13 Sign-off):

```markdown
## Amendments

### Amendment 1 — 2026-MM-DD — Load-bearing claim sharpened from Stage A

**Trigger:** Stage A completion (`findings/substrate-rethinks-survey.md` v1).

**Original passage** (§5.1):

> [quote the original provisional claim, paragraph form]

**New passage** (replaces §5.1):

> [quote the sharpened single-sentence operationalized claim]

**Why:** Per the sequencing in §7, PRECOMMIT cannot be locked without Stage A's
operationalized form. This amendment records the operationalization.
**Scope:** §5.1 only; §5.2–§5.7 unchanged.
```

- [ ] **Step 6**: Commit.

```bash
git add findings/substrate-rethinks-survey.md \
        experiments/06-substrate-pilot/PILOT-DESIGN.md
git commit -m "Exp 06 Stage A: load-bearing claim sharpened; PILOT-DESIGN Amendment 1"
```

**Acceptance**: The single sentence is read aloud and stands on its own. Δ_warrant, M, task family, and seed count are all numeric/specific. Amendment block follows [`CLAUDE.md`](../../CLAUDE.md) §1 format.

---

## Task 6: "What's left to test" — Gate G1 decision

The gate from `PILOT-DESIGN.md` §7: does Stage B need to run at all, or has prior literature already closed the question? This is the place to decide *and to record the reasoning even if the decision is "B runs."*

**Files modified:**
- `findings/substrate-rethinks-survey.md` (new section "What's left to test")
- `experiments/06-substrate-pilot/LAB-NOTES.md` (Gate G1 decision entry)

- [ ] **Step 1**: Re-read the matrix. Identify whether any cell — or interaction across cells — already constitutes a credible direct test of the operationalized claim. Two failure-of-test patterns are common: (a) a lineage moved all three axes but on a non-discriminating task; (b) a lineage tested on a discriminating task but moved fewer than three axes.

- [ ] **Step 2**: Write the "What's left to test" section:

```markdown
## What's left to test

Of the surveyed lineages, none [or: [Lineage X]] has tested the
joint claim on a task where region representations of inputs and
outputs strictly add information — i.e., where point representations
of either lose what the task requires — at matched capacity. The
prior work either [...] or [...]. Stage B of Exp 06 is the
[smallest / closest / etc.] test that fills this gap.

[OR: if prior work closes the question — e.g., a credible negative
already exists at matched capacity on a discriminating task — write
that explicitly and recommend closing 06 without B running.]
```

- [ ] **Step 3**: Append a `LAB-NOTES.md` Gate G1 entry:

```markdown
## 2026-MM-DD — Gate G1 decision

After Stage A v1, the decision is **[PROCEED / CLOSE]** for Stage B.

Reasoning: [one paragraph; cite specific lineages and matrix cells]
```

- [ ] **Step 4**: If PROCEED, the next program work is locking PRECOMMIT.md for B (separate task, separate plan).
       If CLOSE, write a brief "06 closed by prior art" note in `findings/`, update root [`README.md`](../../README.md) status table, and update root [`LAB-NOTES.md`](../../LAB-NOTES.md). The literature survey *still ships* as a thesis-independent artifact.

- [ ] **Step 5**: Commit.

```bash
git add findings/substrate-rethinks-survey.md \
        experiments/06-substrate-pilot/LAB-NOTES.md
git commit -m "Exp 06 Stage A: Gate G1 decision recorded — [PROCEED|CLOSE]"
```

**Acceptance**: Section is concrete (cites specific cells/papers, not "in general"). Gate G1 decision is binary. Reasoning is written so a reviewer can disagree with it on the merits, not just shrug at vagueness.

---

## Task 7: Intro and intended summary

Done *last*, after the body. Writing it earlier risks anchoring the survey on a position that the matrix would have rebutted.

**Files modified:**
- `findings/substrate-rethinks-survey.md` (replace the placeholder "Intended summary" section; revise the "Why this is worth publishing on its own" if needed)

- [ ] **Step 1**: Draft the "Summary" section (replaces "Intended summary"). 1–2 short paragraphs. The first paragraph is the headline: matrix N-of-K observation, the empty-cell insight, the standalone claim. Example phrasing for the matrix observation (computed from Task 3's actual cells): *"Of the 5 lineages surveyed, N move ≤2 of the 3 primitive axes (input × representation × objective) at the demonstrated scales; the remaining K move all 3 but on tasks where the joint claim does not discriminate."* The second paragraph is what this survey adds to the existing literature.
- [ ] **Step 2**: Re-read "Why this is worth publishing on its own"; revise if the matrix changed the framing.
- [ ] **Step 3**: Verify the survey's opening 2 paragraphs land independently of MANDORLA. A reader who has never heard of intersection-as-primitive should be able to read the first page without confusion.
- [ ] **Step 4**: Commit.

```bash
git add findings/substrate-rethinks-survey.md
git commit -m "Exp 06 Stage A: survey intro + summary"
```

**Acceptance**: The opening 2 paragraphs name no MANDORLA-internal concept that isn't defined inline. The survey reads first-class — a finding that survives MANDORLA, same shape as [`findings/reader-saturation-hotpotqa.md`](../../findings/reader-saturation-hotpotqa.md).

---

## Task 8: Independent-read review

Read the whole survey as if you had never seen MANDORLA. Mark passages where prior context is assumed but not supplied.

**Files modified:**
- `findings/substrate-rethinks-survey.md` (small revisions)

- [ ] **Step 1**: Read top to bottom in one sitting. Note any passage where understanding requires knowledge from `mandorla.md` or `PILOT-DESIGN.md`.
- [ ] **Step 2**: For each such passage, either supply the context inline (preferred) or remove the reference. Survey must be *self-contained*.
- [ ] **Step 3**: Verify every citation in the prose appears in the matrix or vice versa — the two should be a closed set.
- [ ] **Step 4**: Verify the "Provenance and independence" section accurately states the relationship to MANDORLA without making the survey contingent on MANDORLA.
- [ ] **Step 5**: Commit if revisions made.

```bash
git add findings/substrate-rethinks-survey.md
git commit -m "Exp 06 Stage A: independent-read review pass"
```

**Acceptance**: A reader cold to the project can read the survey straight through and understand every claim it makes. No "see §X of MANDORLA" handoffs except in the explicitly-marked "Provenance" section.

---

## Task 9: Final sign-off

- [ ] **Step 1**: Update root [`README.md`](../../README.md) Status table to reflect Stage A completion. If Gate G1 = PROCEED, status entry for 06 changes from "Pre-PRECOMMIT design (2026-05-28)" to "Stage A complete (YYYY-MM-DD), Stage B PRECOMMIT pending." If Gate G1 = CLOSE, status changes to "Closed by prior art (YYYY-MM-DD); survey published."
- [ ] **Step 2**: Update root [`LAB-NOTES.md`](../../LAB-NOTES.md) with a brief entry recording Stage A completion and Gate G1 outcome.
- [ ] **Step 3**: Update the stub [`findings/substrate-rethinks-survey.md`](../../findings/substrate-rethinks-survey.md) header from "STUB" to "v1 (YYYY-MM-DD)".
- [ ] **Step 4**: Final commit and push.

```bash
git add README.md LAB-NOTES.md findings/substrate-rethinks-survey.md
git commit -m "Exp 06 Stage A: complete — [Gate G1 outcome]"
git push
```

**Acceptance**:
- Survey is published as v1 at `findings/substrate-rethinks-survey.md`.
- Root README and LAB-NOTES reflect Stage A completion.
- Gate G1 outcome is recorded with reasoning.
- If PROCEED, next program work item is "Lock PRECOMMIT.md for Stage B" (separate plan, separate session).
- If CLOSE, 06 is formally closed in the README status table.

---

## Out of scope for this plan

- **Stage B PRECOMMIT lock.** Separate session, separate plan. Cannot be locked until Stage A completes Task 5.
- **Stage B implementation.** Coding, training, eval. Out of scope entirely; only entered on Gate G1 = PROCEED *and* PRECOMMIT locked.
- **Reading lineages outside the five named in PILOT-DESIGN §4.1.** If during reading a sixth lineage emerges as load-bearing, that is a `LAB-NOTES.md` observation and a possible `PILOT-DESIGN.md` amendment, not an unilateral expansion of the plan.

## Risks during execution

- **Time inflation.** Lit work expands to fill time. The plan's 1–2 person-week budget is held by Task-1 + Task-2 cumulative effort, not by per-task deadlines. If after 2 weeks the matrix is not complete, write `LAB-NOTES.md` honestly, finish with what's deep-read, and ship.
- **Anchoring.** Reading the friendliest lineage first biases the matrix. Mitigation: do the shortlists for all 5 lineages (Task 1) before any deep reads (Task 2).
- **Citation churn.** A paper read deeply may be deprecated by a later read. Mitigation: tolerate small revisions to `READING-NOTES.md`; commit them; do not let one paper's deprecation block forward motion.
- **"Just one more paper" rescue.** If the matrix's empty cells feel uncomfortable, the temptation is to keep reading until one is filled. The empty cells are the result. Resist.
