# Session charter — Benchmark

You own **how strong we are, and whether the measurement is honest.** You also own the box
exclusively.

Read first, in order: [`README.md`](README.md) (the inter-session contract),
[`../ROADMAP.md`](../ROADMAP.md), [`../../10-EVALUATION-SPEC.md`](../../10-EVALUATION-SPEC.md),
[`../fair-pentobi-benchmark.md`](../fair-pentobi-benchmark.md).

---

## The question you exist to answer

> In a fair fight — equal thinking time, Pentobi as shipped with its opening book — does our best
> net win the majority of games?

That is Henry's actual goal. The weighted ladder score is a progress-tracking instrument, not the
goal. Do not substitute one for the other.

## What you own

- **The box, exclusively.** `ssh gpu-anywhere`. Serialise every job; there is one GPU.
- `scripts/pentobi_benchmark.py`, `scripts/measure_move_times.py`, `scripts/mini_ladder.py`
- `src/alphablokus/games/blokusduo/pentobi/**`
- `src/alphablokus/evaluation/ladder_selection.py`, `ladder_elo.py`
- `run_configurations/blokus_cloud_v3_eval.json`
- `docs/plans/fair-pentobi-benchmark.md`, `docs/10-EVALUATION-SPEC.md`

Do not edit `selfplay/`, `storage/`, `training/`, `base_wrapper.py` or `cli.py` — Value head owns
those.

## Work queue

| # | Item | State |
|---|---|---|
| 0 | **Raise the PR** for `feat/eval-config-and-fair-fight` (4 commits, pushed, no PR) | ⬜ do this first |
| F9 | Fair fight: gen-40 vs L9, 4,096 sims, book on, 100 games | 🔄 running |
| F10 | Search-scaling slope: 400 / 1,600 / 6,400 sims at L8 and L9 | ⬜ |
| F11 | Book-on vs book-off delta at L9 — converts every historical number onto the "as shipped" scale | ⬜ |
| M5 | bfloat16 vs float32 self-play A/B (`scripts/validate_jax_search.py --dtype`) — short, free, has slipped twice | ⬜ |

After F9 lands: report the win rate with a Wilson interval, and **say plainly whether we win the
majority.** If it lands near 50%, extend to 300 games before concluding.

## Hard-won facts you must not rediscover

1. **Pentobi's level is a hardcoded simulation count**, `counts_duo = {3, 21, 77, 213, 861, 7280,
   221867, 1109339, 5546695}` in `libpentobi_mcts/Player.cpp`. The 6→7 step is **30×** while every
   other step is 3–8×. Realised effort is lower than the table: L9/L7 measures ~14×, not 25×.
2. **The opening book was inactive for the project's entire history** until 2026-08-05. It works
   now via symlinks beside the binary. A book hit = instant return with an **empty `move_values`**.
   `PentobiPlayer` requires `nobook` explicitly — keep it that way.
3. **Parity is a moving target.** More time for our net lengthens games, and Pentobi's per-move
   budget grows with ply, so its average rises too. Always re-measure *at* the computed budget
   (`--verify-sims`); a single pass under-budgets us ~20%.
4. **Match the mean, not the median.** Pentobi's mean and median differ ~3× at L9. The mean is
   equivalent to total time per game, which is the convention adopted.
5. **Our search is CPU-bound, not GPU-bound.** At 4,096 sims the GPU sits at 0–2%; ~11 of ~12.5 s
   per move is Python tree work. So **many workers are fine** (6 fits in RAM), a faster GPU would
   barely help, and contention does not invalidate a fixed-simulation-budget run — it only inflates
   the clock.
6. **Level 9 needs `--workers 6`, not 16, for book-free ladder work.** Each worker costs ~3.4 GB
   (1.9 GB Pentobi tree + 1.4 GB python). 16 workers OOM'd the box and silently lost four cells.
7. **Never invert a pooled ladder score with a plain logistic.** Half-and-half colours flatten the
   curve; use `evaluation/ladder_elo.py`. At 100 games near a 0.2 score, one rung's interval spans
   **±87 Elo**, so adjacent-level differences are not resolvable.
8. **A fair fight below level 6 is meaningless.** Equal time would give our net <16 simulations, and
   below that MCTS plays a uniform random legal move.
9. **Two ladder conditions must never mix.** `--condition fair-fight` writes to a separate
   directory and `is_longitudinal()` filters it out of promotion decisions. Do not weaken either.
10. **The box tunnel drops regularly.** Fix the **Mac side first**:
    `launchctl kickstart -k gui/501/com.henrycassidy.devtunnel-ssh`. It has worked every time.

## Reporting contract

When a run lands: record the numbers in `fair-pentobi-benchmark.md` **in the same commit**, tick the
row, and if the result changes a project-level belief, update `AGENTS.md` gotchas and tell the
Integrator session (or Henry) so `ROADMAP.md` gets updated. A result that lives only in this chat is
lost.

State results with intervals, and never report an effect smaller than the noise floor: ~±8pp per
ladder level at 100 games, ~0.022 on the weighted score.
