# Branch register

Every branch that exists, what it holds, and what it is waiting on. A branch missing from here is a
branch nobody owns.

**Who keeps it current:** the [Integrator](integrator.md) when one is running — and **when one is not,
whoever creates or merges a branch updates this file in the same action.** It went a week stale
listing branches that had already been merged and deleted, because the Integrator was deliberately
not started and the duty therefore belonged to nobody.

**Update this file in the same action that creates, merges or deletes a branch.** The point is to
stop branches accumulating unnoticed — one sat unpushed with five commits of real work for a week,
and another was closed by mistake while holding the config the box was actively running against.

Naming: `feat/<scope>-<what>`, `chore/<what>`, `docs/<what>`. Never work directly on `main`.

Last reconciled: 2026-08-19.

---

## Live

| Branch | PR | Ahead | Holds | Waiting on |
|---|---|---|---|---|
| `feat/data-loop-player-field` | **#80** | +2 | D1 — the side to move recorded on every stored position, 33 files, tests updated. Touches `corpus.py`, flagged in the title | Review, then merge |
| `analysis/h1-and-colour-check` | — | +1 | The H1 second-mover diagnosis and the F12 colour-check verdict, plus gotchas 23–24. **Rescued from the shared checkout** | Merge |
| `runner/i3-book-delta-log` | — | +1 | The I3 book-delta run log and the `twogtp` harness facts. **Rescued from the shared checkout** | Merge |
| `chore/fix-protocol-workspace-gap` | — | +1 | This file, and the charter fixes for the workspace gap | Merge |

## Rules that have been learned the hard way

1. **Never close a PR without checking every commit on it.** #73 was closed as "superseded" when only
   one of its six commits was — orphaning the eval config the box was mid-run against and the
   evaluation spec.
2. **Check `.gitignore` differences between branches before `git add -A`.** A folder ignored on `main`
   but not on a feature branch got swept into that branch's commit, and switching back to `main` then
   deleted it from disk. Nothing was lost only because it had been committed.
3. **Docs-only branches do not need a codex review.** Say so rather than burning ten minutes
   reviewing markdown. They still need a human read.
4. **A branch more than ~10 commits behind main needs rebasing before its code can be trusted** —
   a review against stale main produces noise.
5. **`ruff check` is not the lint gate — `ruff format --check src tests scripts` is the other half.**
   A branch passed the first and failed CI on the second.
6. **Merge means merge.** Raising a PR and reporting it as done is not the same thing; if the
   instruction was to merge, merge it or say why you cannot.
7. **Leave the shared checkout on `main`.** It was left on a feature branch, so the next two sessions
   that opened the repo at the default path landed on someone else's branch and worked there. Whoever
   finishes with a branch returns the shared checkout to `main`.

## Merge order when several are open

Oldest-first by dependency, not by age: if two branches touch the same file, merge the one whose
content the other builds on. Right now: **#73 → `chore/track-agents-protocol` → `feat/benchmark-f9-result`**,
because #73 and the F9 record both edit `docs/plans/fair-pentobi-benchmark.md`.

## Merged recently

| PR | Branch | What |
|---|---|---|
| #75 | `feat/benchmark-f9-result` | F9's fair-fight result and its colour split |
| #74 | `chore/track-agents-protocol` | The agents protocol, tracked; five review findings fixed |
| #73 | `feat/eval-config-and-fair-fight` | Eval config, evaluation spec, F8 calibration, plan corrections |
| #72 | `docs/roadmap-and-stream-names` | ROADMAP, workstream names, investigation rescued into `docs/research/` |
| #71 | `feat/fair-pentobi-benchmark` | Benchmark rework: book activation, condition separation, draw scoring, colour-aware Elo |
| #70 | `audit/fable-bug-sweep-stream-c` | Exact tests for the Gumbel path and optimizer continuation |
| #69 | `stream-a-plan-2026-08-04` | Instrument repairs before the pilot |
| #68 | `docs/width-and-capacity-findings` | Width and capacity probe findings |
