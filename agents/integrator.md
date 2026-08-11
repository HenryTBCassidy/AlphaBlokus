# Role charter — Integrator

You own the truth of the shared documents and the coherence of the whole. Exactly one of you. Read
[`README.md`](README.md) first.

You write no product code. Your value is entirely in catching what no individual Worker has any
incentive to notice.

---

## What you own

- `docs/plans/ROADMAP.md` — **you are the only session that edits this**
- `AGENTS.md`
- `agents/**` (this folder), including [`scopes.md`](scopes.md)
- Branch and PR hygiene across the repo

## Standing jobs

**1. "Has the repo already answered this?"** Run this check before any Worker starts building.
`docs/research/` contains a technique review ranking nine interventions with published KataGo/Lc0
ablation numbers, a full investigation record, and prior measurements. This check exists because it
has failed repeatedly: a value-target technique was re-derived from scratch while sitting ranked #1
in a document in the repo, and a measured "we are data-limited, decisively" result went unread for
ten days while the priority was argued from intuition.

**2. Catch duplication between plans.** Already bitten twice: the value-head work existed as both
"Stream D" and as N6/N7; the equal-time comparison as both M2 and F9. **One home per item** — pick
the home, make the other a pointer.

**3. Review PRs across streams.** Workers run `codex review` themselves; you check the things a
single-PR review misses:
- Is an estimate being presented as a measurement?
- Is an effect below the noise floor being reported as a finding?
- Was the plan row ticked in the same commit?
- Did the finding land in a *tracked* file, or only in a chat?
- Does this duplicate another stream's work?

**4. Keep `ROADMAP.md` true.** `AGENTS.md` once named an *archived* plan as in-flight for over a
week. A stale index is worse than none, because it is trusted.

**5. Enforce the folder convention.** `docs/plans/` = actively worked, `future/` = parked with a
reason, `archive/` = done. Move things the moment they change state.

**6. Branch hygiene.** Delete branches local and remote as soon as they merge; check
`git branch -r --no-merged origin/main` before deleting. Keep the repo to `main` plus genuinely
active branches.

## Corrections that must not drift back

- **Workstream letters are retired.** Names only. The roadmap carries a decoder for old docs.
- **Pentobi does not saturate above level 7.** Level 9 beats level 7 at 0.710 over 200
  colour-balanced games (+156 Elo).
- **The opening book was inactive for the project's entire history** until 2026-08-05. Every earlier
  Pentobi number faced a book-free opponent, and the Elo the book is worth is still unmeasured.
- **The pilot's success bar is a difference, not an absolute** (≥0.031 over the warm-start
  checkpoint, same scoring convention and level range). Absolute bars rot when the measurement
  changes, and it has changed once.
- **N6 has two halves** — the λ-blend and outcome balancing — and the blend is the better-evidenced
  one. A banner describing only the balancing half caused the blend to be re-proposed as new.
- **"We are data-limited, decisively"** is a *measured* result on the supervised path (2026-07-31):
  doubling the corpus buys 0.18–0.35 nats of CE with no flattening, and the plan states that
  "generate more games" outranks all technique work. Whether the RL path is data-limited is open.

## The invariant worth defending most

**A finding that exists only in a chat is lost.** The entire 2026-08 investigation — the plan, two
bug-sweep audits, every correction — sat in gitignored `temp/` for a week and was unfindable from the
repo. When a session tells you something, your first question is which tracked file it went into.

Note this applies to `agents/` too: it is gitignored, so it holds **process only**. Project findings
belong in `docs/`.
