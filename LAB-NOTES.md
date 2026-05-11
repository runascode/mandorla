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
