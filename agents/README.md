# How we work: roles, hand-offs and rules

This folder is the operating protocol for running AlphaBlokus across **several Claude Code sessions
at once**. Every session reads this file first, then its own charter.

**This folder is tracked in git**, so every worktree has it and any session can be pointed at a
relative path. One rule holds regardless:

> Anything a session learns that belongs to the *project* — a measurement, a defect, a corrected
> assumption — goes into `docs/plans/**` or `AGENTS.md`, **not** here. This folder holds **process**;
> `docs/` holds **findings**. A result recorded only in `box-results.md` is invisible to anyone
> reading the project's documentation.

---

## Why we split

One session running everything kept losing the thread: re-proposing work a plan already contained,
missing evidence already measured and written down, scoping runs wider than asked, leaving plan rows
unticked. None of those were capacity problems — they were bookkeeping and attention problems, and
they are fixed by making each session's surface small and putting the coordination in files.

---

## The four roles

| Role | How many | Owns | Touches the box? |
|---|---|---|---|
| **[Worker](worker.md)** | one per scope | A worktree, a branch, an exclusive set of files, one plan | No — submits jobs to the queue |
| **[GPU-runner](gpu-runner.md)** | exactly one | The box. Drains the queue, verifies completion, reports raw numbers | **Exclusively** |
| **[Analyst](analyst.md)** | one | Turning raw results into conclusions, and writing them into the plans | No |
| **[Integrator](integrator.md)** | one | `ROADMAP.md`, `AGENTS.md`, cross-stream review, branch hygiene | No |

**Why Analyst is separate from GPU-runner.** Every serious error in this project has been an
interpretation error, not a production error: a "Pentobi saturates above level 7" claim drawn from
cells the plan itself called meaningless; an eval-set lineage count asserted as verified and wrong; a
96.3% figure whose raw data no longer exists shaping three runs of next steps; "moved strength by
0.002" quoted against a noise floor of 0.022. The runner's job is *did it complete*. Drawing
conclusions is a different skill and needs someone whose only job it is.

**Why Integrator is separate from everyone.** No Worker has any incentive to notice that two streams
are building the same thing. That has already happened twice — the value-head work existed as both
"Stream D" and N6/N7; the equal-time comparison as both M2 and F9.

---

## The hand-offs — four files, and nothing else counts

Separate sessions **cannot talk to each other**. Every hand-off is a file. If it is not in one of
these, it did not happen.

| From → To | File | Contains |
|---|---|---|
| Worker → GPU-runner | [`box-queue.md`](box-queue.md) | A job request: what, why, cost, kill condition |
| GPU-runner → Analyst | [`box-results.md`](box-results.md) | Raw numbers + completion evidence, **no interpretation** |
| Analyst → everyone | the plan doc + `ROADMAP.md` | The conclusion, with its interval and the rule it was judged against |
| Worker → Integrator | the PR | |

---

## Rules that apply to every role

**1. One session owns the box.** There is one GPU. Only the GPU-runner starts box jobs. Two jobs on
one GPU corrupt each other's timings and can OOM the machine — both have happened.

**2. File ownership is exclusive, with three named exceptions.** Your charter lists what you own.
Do not edit another role's files; raise it instead.

The exceptions exist because the hand-offs require them:

| File | Who writes it |
|---|---|
| [`box-queue.md`](box-queue.md) | **Workers append** job requests; the **GPU-runner** moves entries to Done |
| [`box-results.md`](box-results.md) | **GPU-runner** only. Everyone else reads |
| `src/alphablokus/config.py` | Any scope — but flag it in the PR title |

Everything else in `agents/` — the charters, [`scopes.md`](scopes.md), [`branches.md`](branches.md) —
is the Integrator's.

**3. Never commit to `main`, and never work in the shared checkout.** The checkout at
`/Users/henrycassidy/code/personal projects/AlphaBlokus` stays on `main`, always — it is the one place
every session reads from, so a session that switches its branch silently moves the ground under
everyone else. Make your own worktree:

```bash
git worktree add ~/ab-<slug> -b <branch> origin/main
```

Branch naming: `feat/<scope>-<what>` for Workers, `runner/<what>`, `analysis/<what>`,
`chore/<what>`. Never check out another session's branch.

**This applies to every role, including the ones whose only output is markdown.** The first time these
sessions ran, the GPU-runner and Analyst both edited the shared checkout — which was sitting on a
feature branch belonging to neither of them — and left finished work uncommitted, one `git checkout`
away from being destroyed. Their charters said nothing about where to work; only the Worker's did.
That gap is what this rule closes.

**4. Findings go into a tracked file in the same commit as the work.** Not later, not batched. The
entire 2026-08 investigation — the plan, two audits, every correction — sat in gitignored `temp/` for
a week and was unfindable from the repo. That is the failure this rule exists for.

**5. Tick the plan row in the same action that finishes the work.**

**6. Never report an effect smaller than the noise floor.** ~±8pp per ladder level at 100 games;
~0.022 on the weighted ladder score; ~0.05 nats held-out policy CE; ~±87 Elo on a single ladder rung.

**7. Before building anything, check whether the repo already answers it.** `docs/research/` holds a
ranked technique review with published ablation numbers, a full investigation record, and prior
measurements. Re-deriving what is already written down has cost this project real time.

---

## Starting a session

**Every launch prompt must tell the session to make its own worktree.** The first round of prompts
said so for the two Workers and omitted it for the GPU-runner and the Analyst — and those two are
exactly the ones that went wrong. The instruction is not optional boilerplate; it is the load-bearing
line.

Point the new chat at its charter by absolute path and state its scope. For example:

> Read `/Users/henrycassidy/code/personal projects/AlphaBlokus/agents/worker.md` and
> `/Users/henrycassidy/code/personal projects/AlphaBlokus/agents/README.md`. You are the Worker for
> the **value head** scope. Your plan is `docs/plans/supervised-network-improvements.md`, items
> N6 and N7.

The Integrator keeps the list of live scopes in [`scopes.md`](scopes.md) and the state of every
branch in [`branches.md`](branches.md).
