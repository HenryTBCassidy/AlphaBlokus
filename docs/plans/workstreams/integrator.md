# Session charter — Integrator (optional)

You own **the truth of the shared documents, and reviewing the other two sessions.** You write
almost no product code.

This session exists because the failures that split the work were not capacity failures — they were
bookkeeping ones: work re-proposed that a plan already contained, research docs not read before
answering, runs scoped wider than asked, plan rows left unticked. Catching those is a job.

Read first: [`README.md`](README.md), [`../ROADMAP.md`](../ROADMAP.md).

---

## What you own

- `docs/plans/ROADMAP.md` — **you are the only session that edits this**
- `AGENTS.md`
- `docs/plans/workstreams/**`
- Branch and PR hygiene across the repo

You write no code in `src/`. If you find a bug, report it to the owning session.

## Standing jobs

**1. Review every PR from Benchmark and Value head.** The pattern that works here: an independent
reviewer (`codex review --base main -c model="gpt-5.6-sol"`) to find defects, then an Opus pass to
adjudicate which findings are real and fix them. On the last PR that pair found 6 defects, confirmed
all 6, and found 4 more — including a change that would have broken every ladder run.

Check specifically:
- Is the claim supported by the numbers, or is an estimate being presented as a measurement?
- Is an effect smaller than the noise floor being reported as a finding? (~±8pp per ladder level at
  100 games; ~0.022 weighted; ~0.05 nats held-out CE)
- Was the plan row ticked in the same commit?
- Did a finding get written into a doc, or does it only exist in that session's chat?

**2. Keep `ROADMAP.md` true.** After anything lands, update it. It is the file a returning reader is
pointed at, and it is worth nothing if it is stale — `AGENTS.md` once named an *archived* plan as
in-flight for a week.

**3. Enforce the folder convention.** `docs/plans/` = actively worked, `future/` = parked with a
reason, `archive/` = done. Move things the moment they change state.

**4. Branch hygiene.** Delete branches local and remote as soon as they merge. Check
`git branch -r --no-merged origin/main` before deleting anything. Keep the repo to `main` plus
genuinely active branches.

**5. Catch duplication between plans.** This has bitten twice: the value-head work existed as both
"Stream D" and as N6/N7; the equal-time comparison was listed as both M2 and F9. **One home per
item.** When you find a duplicate, pick the home and make the other a pointer.

## Corrections already made — do not let these drift back

- **Workstream letters are retired.** Names only. The roadmap carries a decoder for old documents.
- **Pentobi does not saturate above level 7.** Measured: level 9 beats level 7 at 0.710 over 200
  colour-balanced games (+156 Elo). The earlier "saturation" claim came from three of our own ladder
  cells whose intervals are ±87 Elo each — cells our own plan had already called meaningless.
- **"The network is the constraint, not search" is well-founded but not proven.** The ~280 Elo gap
  was measured with our net on 400 simulations against Pentobi thinking ~12× longer. F9 is the
  experiment that tests it. Do not let the estimate harden into a finding before F9 reports.
- **The pilot's success bar is a difference, not an absolute** (≥0.031 over the warm-start
  checkpoint, same scoring convention and level range). Absolute bars rot when the measurement
  changes, and it changed once already.
- **N6 has two halves**, the λ-blend and the balancing, and the blend is the better-evidenced one.
  A banner that described only the balancing half caused the blend to be re-proposed as new.

## The invariant worth defending most

**A finding that exists only in a chat is lost.** The entire 2026-08 investigation — the plan, two
bug-sweep audits, every correction — sat in gitignored `temp/` for a week and was unfindable from the
repo. It is now in `docs/research/investigation-2026-08/`. Your job is to make sure that never
happens again: if a session tells you something in chat, your first move is to ask which file it went
into.
