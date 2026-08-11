# Workstreams — who owns what

The project runs as **separate Claude Code sessions**, one per workstream. This file is the
contract between them. Each session is started by pointing it at its own charter in this folder.

Why: a single session kept losing track — re-proposing work already in a plan, missing context
already in the repo, and scoping runs wider than asked. Splitting by ownership makes each session's
surface small enough to hold, and puts the coordination in files instead of in one conversation's
memory.

Read [`../ROADMAP.md`](../ROADMAP.md) first for what the project is doing and why.

---

## The sessions

| Session | Charter | Owns | Needs the box? |
|---|---|---|---|
| **Benchmark** | [`benchmark.md`](benchmark.md) | Measuring how strong we are, and keeping the instruments honest | **Yes — exclusively** |
| **Value head** | [`value-head.md`](value-head.md) | Fixing the network's "who's winning?" estimate, and the data plumbing it needs | Only when Benchmark hands it over |
| **Integrator** *(optional)* | [`integrator.md`](integrator.md) | Keeping the roadmap true, reviewing the other two, branch hygiene | No |

---

## The rules that make this work

**1. One session owns the box.** There is a single GPU. **Benchmark owns it.** No other session
starts a box job without asking in the shared channel (i.e. Henry). Two jobs on one GPU corrupt
each other's timings and can OOM the machine.

**2. File ownership is exclusive.** Each charter lists the files its session owns. Do not edit
files another session owns — raise it instead. Ownership is what stops two sessions fighting over
the same PR.

| Path | Owner |
|---|---|
| `scripts/pentobi_benchmark.py`, `scripts/measure_move_times.py`, `scripts/mini_ladder.py` | Benchmark |
| `src/alphablokus/games/blokusduo/pentobi/**` | Benchmark |
| `src/alphablokus/evaluation/ladder_*.py` | Benchmark |
| `docs/plans/fair-pentobi-benchmark.md`, `docs/10-EVALUATION-SPEC.md` | Benchmark |
| `src/alphablokus/selfplay/**`, `src/alphablokus/storage/selfplay_store.py` | Value head |
| `src/alphablokus/training/**`, `src/alphablokus/games/base_wrapper.py` | Value head |
| `src/alphablokus/cli.py` | Value head |
| `docs/plans/supervised-network-improvements.md` | Value head |
| `docs/plans/ROADMAP.md`, `AGENTS.md`, `docs/plans/workstreams/**` | Integrator (or Henry) |
| `src/alphablokus/config.py` | **Shared** — say so in the PR title when you touch it |

**3. Every session works on its own branch, in its own worktree.** Never commit to `main`. Never
check out another session's branch. Branch naming: `feat/<session>-<what>`, e.g.
`feat/benchmark-search-scaling`, `feat/valuehead-player-plumbing`.

**4. Findings go in files, not in chat.** If a session learns something the others need — a
measurement, a defect, a corrected assumption — it goes in the relevant plan doc or `AGENTS.md`
gotchas **in the same commit as the work**. A finding that exists only in one chat's history is
lost the moment that chat ends. This is the specific failure that produced this folder.

**5. Tick the plan row in the same action that finishes the work.** Not later, not in a batch.

---

## Current state (2026-08-11)

- **Running:** F9, the fair fight — gen-40 vs Pentobi level 9 at equal thinking time, 100 games,
  6 workers on the box. Owned by Benchmark.
- **Open branch, no PR yet:** `feat/eval-config-and-fair-fight` — the eval config, the evaluation
  spec, the F8 calibration record. Benchmark's first job is to raise that PR.
- **Not started:** everything in the Value head charter.
- **Parked:** [`../future/`](../future/) — corpus v2, score head.

## The one dependency between sessions

**Value head's data-plumbing must land before any new self-play data is generated.** Positions do
not currently record whose turn it is, and both historical replay buffers were deleted — so the
schema can be changed almost for free right now. Generate data first and that window shuts: the one
dataset everything downstream depends on would lack the column, and we would have to regenerate.

So: **Value head goes first on anything that produces data.** Benchmark's runs consume existing
checkpoints and are unaffected.
