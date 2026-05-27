# Parallel Self-Play (F1 of full-cycle-optimisation)

Sub-plan for the first optimisation in [`full-cycle-optimisation.md`](full-cycle-optimisation.md). Replaces the current sequential `for _ in range(num_eps): execute_episode()` loop in `Coach` with a worker-pool implementation that runs N self-play games concurrently across CPU cores. Arena and Elo eval get the same treatment as a natural extension once the worker pool is in place.

Expected wall-clock impact: **5–7× on the self-play + arena + Elo phases combined**, which together account for ~95% of total per-gen time. Bottleneck moves from "Python-bound serial loop" to "Python-bound but parallel across cores."

Companion docs:
- [`docs/plans/full-cycle-optimisation.md`](full-cycle-optimisation.md) — master plan and the F1 row this sub-plan implements.
- [`docs/02-ALGORITHMS.md`](../02-ALGORITHMS.md) — self-play / arena / Elo phase definitions.
- [`docs/07-DATA-STORAGE.md`](../07-DATA-STORAGE.md) — how training examples flow back into the per-gen replay buffer (matters for the result-aggregation step).

---

## Why this is the highest-priority optimisation

Three facts make parallel self-play the cheapest big win:

1. **Self-play games are embarrassingly parallel.** Each game is fully independent — separate MCTS tree, separate seed, separate state. No coordination needed except at the boundaries (gather results, sync weights).
2. **The 3060 Ti's host CPU has 8+ cores sitting idle.** A single Python process can use only one core meaningfully (because of the GIL and because move generation is pure Python). The other 7 cores are wasted.
3. **The change is contained.** Touches `Coach`, `Arena`, and the `Player` abstraction — none of the game-logic core. Worker boundaries are well-defined (one episode per task) and serialisation is simple.

Estimated speedup factor is roughly `min(num_workers, num_eps)` — capped by the number of work units, not the number of cores. With `num_eps=80` and 8 workers, we'd expect close to 8× on the self-play phase alone.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| P0 | Establish baseline: run `scripts/benchmark_phases.py --config run_configurations/profile_baseline.json` on the PC; record numbers in the master plan's progress tracker. See [P0 detail](#p0-baseline-benchmark) | 30 min | High | partial — script + config landed; PC run pending |
| P1 | Decide worker model: process pool (chosen — see [P1 detail](#p1-worker-model)) vs thread pool. Document the decision and trade-offs in this plan. | 30 min | High | ✓ |
| P2 | Design per-worker setup: how each worker gets its own game instance, nnet, MCTS, and seed | 1 hr | High | |
| P3 | Implement `core/coach.py` parallel-aware self-play loop, gated by `num_parallel_workers` config field | 2-3 hr | High | |
| P4 | Determinism tests: parallel run with seed=K and serial run with seed=K must produce **the same set of training examples** (up to ordering) | 1.5 hr | High | |
| P5 | Aggregate per-worker profiling stats (sims/sec, search time) back into the metrics collector — currently per-process, needs to be summed/averaged | 1 hr | Medium | |
| P6 | Replace per-episode `tqdm` with a coordinator that shows overall progress across workers | 30 min | Medium | |
| P7 | First measurement: re-run the baseline profile with parallelism on, on the PC; expect ~5–7× wall-clock improvement | 30 min | High | |
| P8 | Apply the same worker-pool pattern to arena games (currently 50/gen, sequential) | 1-2 hr | High | |
| P9 | Apply the same worker-pool pattern to Elo eval games (currently 20/gen, sequential) | 1 hr | Medium | |
| P10 | End-to-end test run on PC with parallel self-play + arena + Elo, comparing wall-clock to baseline. Update master plan progress tracker. | 1 hr | High | |
| P11 | Archive this plan: `git mv` to `docs/plans/archive/`. Append a Scope additions section if anything landed beyond this row table. Update master plan F1 status to "Done". | 5 min | Low | |

Total: ~10–12 hours focused work.

---

## P0. Baseline benchmark

The benchmark lives at `scripts/benchmark_phases.py`, with the fixed
config at `run_configurations/profile_baseline.json` (16 self-play eps,
10 arena games, 10 Elo games, 300 sims, 64f×4b net, `cuda: true`).
Validated on the Mac with TTT (`/tmp/benchmark_ttt_test.json`) and with a
reduced-shape Blokus config (`/tmp/profile_baseline_mac_quick.json`,
50 sims, 6 games) — both produced clean parquets and a working HTML
report. Numbers on the Mac aren't useful for F1; they only confirm the
script runs end-to-end.

**To capture the real baseline, run on the PC over SSH:**

```bash
# On the PC, from the repo root
./scripts/run_benchmark.sh --config run_configurations/profile_baseline.json
```

The wrapper sets `PYTHONUNBUFFERED=1`, tees combined stdout/stderr to
`temp/<run_name>_benchmark/run.log`, and forwards args to
`scripts.benchmark_phases`. To check progress mid-run from another
shell session: `tail -f temp/profile_baseline_benchmark/run.log` —
each per-game heartbeat line shows progress, average per-game time, and
ETA.

Output lands at `temp/profile_baseline_benchmark/`:

- `report.html` — top: wall-clock summary + bar chart, run-time
  estimator table for 4 scale points. Bottom: per-phase MCTS
  drill-down (component pie, per-move timing, search characteristics
  — same drill-down `scripts/mcts_profiling.py` produces, but one
  section per phase).
- `phase_summary.parquet` — one row per phase (wall-clock + mean/game).
- `per_game.parquet` — every game's MCTS stats.
- `<phase>_episode_stats.csv` / `<phase>_move_stats.csv` — ad-hoc
  inspection.

Expected wall-clock on the 3060 Ti: roughly **15–20 minutes** total
(`profile_baseline.json` is sized to be cheap enough for a coffee break
but representative — 16 self-play games at ~55s each ≈ 15 min for that
phase, with arena/Elo adding a bit on top).

After it runs:

1. Copy the key numbers from `phase_summary.parquet` into the
   [progress tracker](full-cycle-optimisation.md#progress-tracker) row
   for F1 — that becomes the "Measured before" entry.
2. Spot-check the drill-down report; confirm the move-gen vs inference
   split is roughly **43% move-gen / 50% inference** at production net
   size (`docs/08-TRAINING-ESTIMATES.md` — the older "65–72% move-gen"
   numbers were from a small profiling net and don't apply here).
3. Tick this row.

The same script is re-run at P7 (after parallelism lands) with
`--num-workers 8` to capture the "Measured after" number.

---

## P1. Worker model

Decision: **process pool (`multiprocessing` / `concurrent.futures.ProcessPoolExecutor`)**.

Rationale:

- **Move generation is pure Python and GIL-bound.** A thread pool wouldn't help — threads would serialise on the GIL exactly the way a single-threaded loop already does. Move-gen is ~43% of self-play time at production net size (see `docs/08-TRAINING-ESTIMATES.md`) — smaller than originally believed but still big enough that parallelising it across cores is worthwhile. We need true OS-level parallelism, not threads.
- **Each self-play game is independent** so the cross-process IPC cost (pickle workers' return values back to the main process) is paid once per game, not per move. 80 games × ~50 KB of training examples ≈ 4 MB of IPC traffic — negligible.
- **PyTorch handles process-level parallelism cleanly.** Each worker can load its own copy of the network weights into its own process's memory. CPU inference is fine for small nets (we're at 64f×4b); GPU inference per process is also viable on the 3060 Ti's 8 GB if needed (8 × ~50 MB net ≈ 400 MB).

What we explicitly rule out:

- **Thread pool**: GIL-bound on move gen → no real speedup.
- **`asyncio` workers**: same GIL constraint plus higher cognitive overhead.
- **Shared inference server** (one nnet on GPU, workers send positions for evaluation): more complex, defers naturally to F4 (batched inference) where it'll be needed. Out of scope here.

**Reconfirmed 2026-05-26** after the production-net benchmark showed inference is ~50% of search time (not ~17%). Decision still holds:

- Threads are still wrong — the 43% move-gen slice is still GIL-bound, so threads still can't parallelise it. The bigger inference slice doesn't change that.
- Processes are still right — each worker holds its own net copy in its own GPU context (memory pressure is fine: 8 × ~50 MB ≈ 400 MB on the 8 GB 3060 Ti).
- The downside the new finding *did* surface is GPU contention between workers on the 50% inference slice — that caps F1's single-axis speedup at ~1.8× (instead of the original 5–7× estimate). The fix is F4 (batched inference server) in the master plan, not a worker-model change here.

---

## P2. Per-worker setup

What each worker needs at task start:

1. A game instance (`BlokusDuoGame(pieces_config_path)` — non-trivial to construct because it builds the orientation codec + initial-action cache). Building once per worker, not per task, is important; otherwise we pay seconds per game.
2. A neural-network wrapper with the latest weights.
3. An MCTS instance bound to that nnet.
4. A deterministic seed derived from `(generation, episode_index, base_seed)`.

Two-phase worker lifecycle:

```python
# Worker initializer (runs once per worker process at pool start)
def _worker_init(game_kwargs, nnet_state_dict_path, mcts_config):
    global _GAME, _NNET, _MCTS
    _GAME = BlokusDuoGame(**game_kwargs)
    _NNET = NNetWrapper(_GAME, ...)
    _NNET.load_checkpoint(nnet_state_dict_path)
    _MCTS = MCTS(_GAME, _NNET, mcts_config)

# Worker task (runs once per episode)
def _worker_play_episode(seed, episode_idx, generation):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Reset MCTS tree
    _MCTS.reset()
    # Run the existing execute_episode logic, return training examples + profiling stats
    ...
```

`Pool(initializer=_worker_init, initargs=(...))` gives us cheap per-worker setup amortised across many tasks.

**On weight syncing**: the network changes once per generation (after training). At the start of each gen, the main process saves the current weights to a temp file, then re-spawns the pool with the new initargs (or sends the path via a queue). The first gen uses the random-init weights.

---

## P3. Coach refactor

Current code at `core/coach.py:_learn_loop`:

```python
iteration_examples = deque([], maxlen=self.config.max_queue_length)
for _ in tqdm(range(self.config.num_eps), desc="Self Play"):
    self.mcts = MCTS(self.game, self.nnet, self.config.mcts_config)
    iteration_examples.extend(self.execute_episode())
```

Refactored:

```python
iteration_examples = deque([], maxlen=self.config.max_queue_length)
n_workers = self.config.num_parallel_workers or 1
if n_workers > 1:
    iteration_examples.extend(
        self._execute_episodes_parallel(self.config.num_eps, generation, n_workers),
    )
else:
    for _ in tqdm(range(self.config.num_eps), desc="Self Play"):
        self.mcts = MCTS(self.game, self.nnet, self.config.mcts_config)
        iteration_examples.extend(self.execute_episode())
```

`_execute_episodes_parallel` constructs the pool, dispatches `num_eps` tasks via `imap_unordered`, aggregates results as they complete, and reports progress via a single tqdm in the main process.

Critical: when `num_parallel_workers <= 1`, behaviour is **bit-for-bit identical** to today (no pool, no IPC, no serialisation). This is the backward-compat path and the basis of the determinism test in P4.

---

## P4. Determinism tests

The single most important test in this whole sub-plan. Any change to the order in which `np.random.*` is called perturbs MCTS tie-breaks and shifts every downstream training example. With parallelism, ordering is by definition non-deterministic — but we can still seed each *episode* deterministically.

Strategy:

1. Each episode `i` in gen `g` gets seed `f(base_seed, g, i)` (a stable hash, e.g. `base_seed + g * 10000 + i`).
2. Inside the episode, the worker calls `np.random.seed(seed)` and `torch.manual_seed(seed)` before running MCTS.
3. With the same `(base_seed, g, i)` triple, episode `i` of gen `g` is bit-for-bit identical regardless of which worker runs it or in what order.

Test:

```python
def test_parallel_self_play_matches_serial(blokus_game, test_config):
    # Run with 1 worker (sequential codepath)
    cfg_serial = replace(test_config, num_parallel_workers=1, num_eps=4)
    coach_serial = Coach(blokus_game, ..., cfg_serial)
    serial_examples = coach_serial._collect_self_play_examples(generation=1)

    # Run with 4 workers
    cfg_parallel = replace(test_config, num_parallel_workers=4, num_eps=4)
    coach_parallel = Coach(blokus_game, ..., cfg_parallel)
    parallel_examples = coach_parallel._collect_self_play_examples(generation=1)

    # Same set (order may differ between workers, but set membership matches)
    assert _example_set(serial_examples) == _example_set(parallel_examples)
```

`_example_set` hashes (board_bytes, policy_bytes, value) tuples for set-comparison.

If this test passes, we have *provably* introduced no statistical drift — the parallel run produces identically the same training data as the serial run. If it fails, the seeding is broken.

---

## P5. Profiling aggregation

`MCTSProfiler` (in `core/mcts.py`) currently accumulates stats into a single instance. With workers, each has its own profiler. The metrics collector needs to sum / average across workers.

Approach:

- Each worker's `_play_episode` returns its profiler stats alongside training examples.
- Main process accumulates: total sims, total search time → final sims/sec. Per-stage time means averaged across workers.
- These are logged once per generation as before — no change to W&B / parquet schema.

---

## P6. Progress reporting

Currently `tqdm(range(num_eps), desc="Self Play")` shows per-episode progress in the main process. With workers, the main process never iterates the inner loop.

Switch to:

```python
with tqdm(total=num_eps, desc="Self Play") as pbar:
    for examples in pool.imap_unordered(_worker_task, tasks):
        iteration_examples.extend(examples)
        pbar.update(1)
```

Single tqdm in the main process, updates as each worker finishes an episode. No per-worker tqdm (would interleave chaotically).

---

## P7. First measurement

After P1–P6 land, run:

```
uv run python scripts/mcts_profiling.py --config run_configurations/profile_baseline.json
```

Compare per-game wall-clock against baseline. Expect:

- Single-worker config: ~same as baseline (overhead of pool setup adds ~1-2s startup, negligible per game)
- 4-worker config: ~3.5–4× speedup
- 8-worker config: ~6–7× speedup (some efficiency loss to scheduler / IPC)

Beyond ~8 workers diminishing returns — Python's pickle and process-startup cost stops being free.

Update the master plan's [progress tracker](full-cycle-optimisation.md#progress-tracker) with both the per-game and per-gen numbers.

---

## P8. Parallel arena

Arena games are equally embarrassingly parallel — each game is one network vs another, no cross-game state. Same worker-pool pattern, different "play one episode" function (`Arena.play_game(...)` instead of `Coach.execute_episode(...)`).

One subtlety: arena alternates which player starts each game (player1_was_white alternates). The dispatcher needs to assign tasks with the right `player1_was_white` flag.

Game records are returned by each worker; main process serialises them into the arena replays parquet as today.

Acceptance decision (`_should_accept_new_network`) runs in the main process after all games complete — no change there.

---

## P9. Parallel Elo eval

Same pattern again. 20 games per gen, sequential today, parallel after this row. Smaller win in absolute time (~15-20 min/gen → ~3-5 min/gen) but free given the worker pool exists.

---

## P10. End-to-end validation

Kick off a 1-gen Blokus run with the new parallelism enabled. Compare wall-clock to the equivalent serial baseline (which is `blokus_pc_second`'s per-gen number: ~2h 20min). Expect a 4–6× reduction. Update the master plan's progress tracker.

If results look sensible:
- Update the example run configs to set `num_parallel_workers` to a reasonable default (likely 8 or `os.cpu_count()`).
- Commit.

If results look broken (e.g. acceptance rate drops, loss curves diverge from serial baseline at same seed): P4's determinism test should have caught any data-level regression, so any divergence is likely a stats-aggregation bug in P5 or a tqdm/progress-reporting issue. Roll back, fix, retry.

---

## P11. Archive

`git mv docs/plans/parallel-self-play.md docs/plans/archive/`. Append a "Scope additions" section to this file first if anything landed outside the row table.

Tick the F1 row in [`full-cycle-optimisation.md`](full-cycle-optimisation.md) and update its progress tracker with the final measured speedup.

---

## Out of scope

- **F4 (batched inference)**: per the master plan's compatibility matrix, F1 + F4 needs a shared inference server. Defer.
- **Parallel training step**: training within an epoch already uses PyTorch's data-parallel DataLoader pattern; not a self-play concern.
- **Worker autoscaling**: keep `num_parallel_workers` as a static config field for now. Dynamic adjustment based on CPU load is over-engineering at this stage.
- **Cross-machine distributed self-play**: single-PC only. Revisit when the master plan hits M3.

---

## Notes for whoever picks this up

- Test the determinism check (P4) **first**, before any of the perf work. A failing determinism test means subsequent perf claims are unreliable — you might be getting "free" speedup at the cost of running different (worse) training data.
- The `BlokusDuoGame` constructor builds the orientation codec from `pieces.json` — non-trivial. Make sure workers construct it once via `initializer`, not once per task.
- Watch for `pickle` errors when first running the pool. PyTorch nnet wrappers, frozen dataclasses, and numpy arrays all pickle fine; if anything in the codebase has captured a closure or a lambda, that won't.
- The main process's `nnet` is the source of truth for "the current latest weights" — workers always re-load from a checkpoint file at gen boundaries, not by copying the live `nnet` object.
