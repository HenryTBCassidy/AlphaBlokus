# Branch register

Every branch that exists, what it holds, and what it is waiting on. The
[Integrator](integrator.md) keeps this current; a branch missing from here is a branch nobody owns.

**Update this file in the same action that creates, merges or deletes a branch.** The point is to
stop branches accumulating unnoticed — one sat unpushed with five commits of real work for a week,
and another was closed by mistake while holding the config the box was actively running against.

Naming: `feat/<scope>-<what>`, `chore/<what>`, `docs/<what>`. Never work directly on `main`.

Last reconciled: 2026-08-11.

---

## Live

| Branch | PR | Ahead | Holds | Waiting on |
|---|---|---|---|---|
| `feat/eval-config-and-fair-fight` | **#73** | +8 | Eval config the box runs against, `docs/10-EVALUATION-SPEC.md`, F8 parity record, pilot-bar restatement, N6 banner fix, M2/F9 de-duplication | Codex review, then merge |
| `chore/track-agents-protocol` | — | +2 | `agents/` protocol tracked instead of ignored (this file, the charters, the queue, the results log) | Henry's read, then merge. **Docs only — no code to review** |
| `feat/benchmark-f9-result` | — | +1 | F9's result recorded in the plan and roadmap, including the colour split | Merge after #73 (touches the same plan file) |
| `feat/width-shadow-probe` | — | +1, **−30 behind** | `scripts/width_shadow_probe.py` — 445 lines, **never reviewed, never run**. The deferred width experiment (M4) | Codex review in flight. Needs rebasing before it could run |

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

## Merge order when several are open

Oldest-first by dependency, not by age: if two branches touch the same file, merge the one whose
content the other builds on. Right now: **#73 → `chore/track-agents-protocol` → `feat/benchmark-f9-result`**,
because #73 and the F9 record both edit `docs/plans/fair-pentobi-benchmark.md`.

## Merged recently

| PR | Branch | What |
|---|---|---|
| #72 | `docs/roadmap-and-stream-names` | ROADMAP, workstream names, investigation rescued into `docs/research/` |
| #71 | `feat/fair-pentobi-benchmark` | Benchmark rework: book activation, condition separation, draw scoring, colour-aware Elo |
| #70 | `audit/fable-bug-sweep-stream-c` | Exact tests for the Gumbel path and optimizer continuation |
| #69 | `stream-a-plan-2026-08-04` | Instrument repairs before the pilot |
| #68 | `docs/width-and-capacity-findings` | Width and capacity probe findings |
