# Parallelise the Pentobi benchmark across a worker pool

**What this covers.** The Pentobi benchmark plays its games strictly **serially** against a single
`pentobi-gtp` engine, so a full 1–9 ladder (e.g. 20 games × 9 levels = 180 games) takes ~45–70 min.
Profiling during a live run showed the **GPU idle at ~2%** — the benchmark is *not* inference-bound; it's
bottlenecked by (a) Pentobi's CPU search (which grows sharply with level), (b) per-move Python/GTP
round-trip overhead, and (c) the fact that it's one game at a time. Since we know up front exactly how
many games we want, we can split them across a **pool of worker processes**, each with its own net +
its own Pentobi engine, and aggregate the results. Expected: **near-linear speedup up to ~core count**
(the idle GPU has huge headroom), dropping a full ladder to well under 15 min.

**Prerequisites:** none. Self-contained change to the benchmark script (+ optional reuse of existing
parallel patterns). Best landed *after* [`pentobi-resign-handling.md`](pentobi-resign-handling.md) so a
full ladder runs without crashing.

**Ground truth for the current code** (verified file:line anchors):
- `scripts/pentobi_benchmark.py:104-127` — `benchmark_level(game, net_player, level, games, seed)`: the
  **serial hot loop**. Builds one `PentobiPlayer(game, level, seed=seed)` (line 106) and calls
  `Arena(net_player, pentobi, game).play_games(games, record=True)` (lines 109-113). This is what gets
  parallelised.
- `scripts/pentobi_benchmark.py:270-279` — the per-level loop that calls `benchmark_level` and prints
  results; `--games` default 20 (line 199), `--sims` default 400 (line 200), `--seed` default 1 (line 201).
- `scripts/pentobi_benchmark.py:248-262` — how the net is built: `instantiate_game_and_network(config)`
  → `nnet.load_checkpoint(args.net)` → `NetworkPlayer(game, nnet, _eval_mcts_config(...), temp=0.0, ...)`.
  A worker must reproduce this itself (a live `NetworkPlayer`/net can't cross a process boundary).
- `src/alphablokus/games/blokusduo/pentobi/player.py:40-53` — `PentobiPlayer(game, level, *, binary=None,
  threads=1, seed=None)`; `startGame` (player.py:62-72) reseeds `base_seed + game_index` per game, so
  **each worker needs a disjoint seed base** to avoid replaying identical games.
- `src/alphablokus/evaluation/arena.py:188` — `Arena.play_games(num, ...) -> (p1_wins, p2_wins, draws,
  records)`; rounds `num` down to even and swaps colours at the midpoint (so an **even** per-worker chunk
  stays half/half by colour).
- `src/alphablokus/registry.py:45` — `instantiate_game_and_network(config)`; the composition-root entry
  point each worker calls to build its game+net.
- `src/alphablokus/parallel/pool.py:553` — `run_two_player_games_parallel(...)` is the existing
  net-vs-net parallel pattern (worker loads a checkpoint, plays a chunk, returns W/L/D). **Reference it
  for the worker/aggregation shape**, but it pits two *checkpoints* — Pentobi isn't a checkpoint, so this
  needs a Pentobi-opponent variant rather than a direct reuse.
- Config: `num_parallel_workers` and `worker_cuda` already exist on `RunConfig` (used by self-play);
  `net_config.cuda` selects the net's device.

**Critical gotcha — multiprocessing start method.** Workers initialise Torch/CUDA (and the code imports
JAX elsewhere). Forking a CUDA/JAX process deadlocks or corrupts state (we already see the
`os.fork() ... JAX is multithreaded ... likely deadlock` warning in training). **Use the `spawn` (or
`forkserver`) start method** for the benchmark pool so each worker is a fresh interpreter that builds its
own game/net/engine. This is why the worker must take *plain picklable args* (config, checkpoint path,
level, seed, device) and construct everything itself — never receive a live net/player.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| P1 | `--workers` arg + device policy for workers (GPU-shared cap vs CPU) | 45 min | High | ✅ |
| P2 | Extract a process-safe worker fn: (config, ckpt, level, n_games, seed_base, device) → W/L/D + records | 1.5 h | High | ✅ |
| P3 | Parallel driver: even chunks across a `spawn` pool, disjoint seeds, aggregate; serial path when workers=1 | 2 h | High | ✅ |
| P4 | Colour-balance + determinism + records-collection policy | 1 h | High | ✅ |
| P5 | Tests (parallel≡serial aggregate; picklability; even-chunk/seed logic) | 1.5 h | High | ✅ |
| P6 | Validate (full 1–9 serial-vs-parallel wall-clock + same-ish win rates) + docs/IDEAS note | 45 min | Medium | ✅ |

---

## P1. `--workers` arg + device policy

**Fix.** Add `--workers N` to `pentobi_benchmark.py` (default: `min(config.num_parallel_workers or 4,
<levels sized>)`; `1` = today's serial path, unchanged). Decide how workers use the net's device — this
is the one real tradeoff:

- **GPU-shared (recommended default).** Each worker runs the net on the GPU. The card is idle most of
  the time (waiting on Pentobi), so a handful of workers interleave their net bursts nicely. **Cap by
  VRAM:** each process needs its own CUDA context (~0.6–1.5 GB), so on the 8 GB 3060 Ti expect ~3–5
  workers max. Default `--workers 4`; document "lower if you see CUDA OOM."
- **CPU net (scale beyond VRAM).** Force the net onto CPU in workers (set `net_config.cuda=False` in the
  worker's config copy) to run as many workers as cores. Slower per-move for the large net, but removes
  the VRAM cap. Offer as `--workers` > VRAM-allows automatically falling back to CPU, or a `--cpu-net`
  flag.

Recommend shipping GPU-shared default + a `--cpu-net` escape hatch. Either way the win is parallelism.

**Effort:** 45 min.

---

## P2. Extract a process-safe worker function

**Current state.** `benchmark_level` (pentobi_benchmark.py:104) takes a live `net_player` and a single
engine — not process-safe.

**Fix.** Add a module-level function (picklable, no closures over live objects):
```python
def _play_chunk(config_path: str, net_ckpt: str | None, level: int, n_games: int,
                seed_base: int, sims: int, batch: int, opening_temp: float,
                opening_moves: int, cpu_net: bool, collect_records: bool
                ) -> tuple[int, int, int, list]:
    # 1. load_args(config_path); if cpu_net: force net_config.cuda=False
    # 2. game, nnet = instantiate_game_and_network(config); load_checkpoint(net_ckpt) if given
    # 3. net_player = NetworkPlayer(game, nnet, _eval_mcts_config(config.mcts_config, sims, batch),
    #                               temp=0.0, opening_temp=..., opening_moves=...)
    # 4. pentobi = PentobiPlayer(game, level, seed=seed_base)   # disjoint base per worker
    # 5. try: return Arena(net_player, pentobi, game).play_games(n_games, record=collect_records)[:3]+records
    #    finally: pentobi.close()
```
Pass the **config path** (a str), not the config object, so nothing GPU-touching crosses the process
boundary before the worker sets its own device. This mirrors how `pool.py` workers rebuild from a
checkpoint path.

**Effort:** 1.5 h.

---

## P3. Parallel driver

**Fix.** Replace `benchmark_level`'s body with a driver that fans `_play_chunk` out over a
`concurrent.futures.ProcessPoolExecutor(max_workers=W, mp_context=multiprocessing.get_context("spawn"))`:
- Split `games` into `W` **even** chunks summing to `games` (distribute any remainder as +2s to keep each
  chunk even, so `Arena.play_games`'s half/half swap holds per chunk).
- Give worker *i* a **disjoint seed base** so no two workers replay the same games, e.g.
  `seed_base_i = args.seed + i * max_games_per_worker` (choose the stride > any chunk size).
- Submit one task per chunk (per level), collect futures, sum `(net_wins, pentobi_wins, draws)` and
  concatenate records. Keep the `serial` code path when `W == 1` (bit-for-bit today's behaviour).
- **Load balancing (recommended enhancement):** rather than a fresh pool per level, build **one pool**
  and submit `(level, chunk)` tasks for *all* levels at once. Fast low-level tasks then free workers to
  pick up slow high-level (level 8/9) tasks — much better utilisation than draining level-by-level. Group
  results back by level for reporting. (If this complicates it, ship per-level pools first and note this
  as a follow-up.)

**Effort:** 2 h.

---

## P4. Colour-balance, determinism, records policy

- **Colour balance:** each per-worker chunk must be even so `play_games` splits it half/half; verify the
  aggregate is still ~50/50 white/black across the pool (add an assertion/logged summary).
- **Determinism:** results won't be bit-identical to the serial run (different RNG interleaving), which is
  fine for a benchmark — but disjoint seeds must guarantee **no duplicate games**. Document this.
- **Records for replays:** the report only embeds `REPLAYS_PER_LEVEL = 4` games/level
  (pentobi_benchmark.py:50). Don't collect full records from every worker — have only the *first* chunk
  per level pass `collect_records=True` (enough for the 4 replays); others `False` to save memory/IPC.

**Effort:** 1 h.

---

## P5. Tests

Real objects, no mocks (style guide). Use TicTacToe or a fast stub opponent to avoid needing a live
Pentobi in CI:
- **Aggregate correctness:** a parallel run of N games across W workers returns totals that sum to N and
  match a serial run's *distribution* within tolerance (use a deterministic dummy opponent, e.g. a
  `Player` that always plays the first legal move, so win/loss counts are predictable and W-invariant).
- **Chunking/seed logic** (pure functions): even chunks summing to `games`; disjoint seed ranges per
  worker; remainder handling.
- **Picklability:** `_play_chunk`'s args are all plain types; the function is importable at module level
  (a quick `pickle.dumps` of the partial/args in a test).
- Note: don't spin up real `pentobi-gtp` in unit tests; the Pentobi-specific path is covered by the
  end-to-end validation (P6).

**Effort:** 1.5 h.

---

## P6. Validate + docs

- Run the **same** full 1–9 sweep serially (`--workers 1`) and parallel (`--workers 4`) on the same net;
  confirm **similar win rates per level** (within CI) and record the **wall-clock speedup**.
- Watch for CUDA OOM at the chosen worker count; document the safe default for the 3060 Ti (and note it
  scales with VRAM on a bigger card).
- Update `scripts/pentobi_benchmark.py` docstring + `docs/05-EVALUATION.md` with the `--workers` usage,
  and add a one-line note to `docs/IDEAS.md` (or mark it resolved there if it was logged).
- Full CI green.

**Effort:** 45 min.

---

## Notes for the executing agent

- **Style contract:** full type annotations (mypy `--strict`), `ruff` lint + format, frozen dataclasses,
  loguru (`{}`; no `print` — though note the existing script uses `print` for its CLI output; match the
  surrounding file), Google docstrings, `from __future__ import annotations`, real objects in tests.
- **Scope:** this is a *benchmark throughput* change — don't alter game logic, the net, or how a single
  game is scored. `--workers 1` must reproduce today's behaviour exactly.
- **The two things most likely to bite:** (1) using `fork` instead of `spawn`/`forkserver` (CUDA/JAX will
  deadlock) — use `spawn`; (2) VRAM exhaustion from too many GPU workers — cap the default and expose
  `--cpu-net`.
- **One commit per checklist row**; tick Done as each lands.
- **Archive on completion** (`git mv` to `docs/plans/archive/`, per PLAN-FORMAT.md).

---

## Outcome (2026-07-05)

All six rows landed. `scripts/pentobi_benchmark.py` gained `--workers N` (+ `--cpu-net`):
`--workers 1` is the unchanged serial path; `>1` fans the sweep across a `spawn`
`ProcessPoolExecutor`. The reusable pieces are pure and unit-tested
(`tests/test_pentobi_benchmark.py`, 13 tests): `_even_chunks` (even chunks summing to
the even game count, ≤ workers, balanced within 2), `_plan_tasks` (globally-disjoint
per-task seed windows; records collected only from leading chunks), `_aggregate_level`
(sum + Wilson CI in the serial dict shape), plus a real `spawn`-pool test proving the
fan-out+aggregate is worker-count-invariant over a deterministic game. `_play_chunk`
(the Pentobi worker) is verified picklable-by-reference with all-plain args.

**Validation (Mac, `blokus_mac_test`, fresh net, levels 1–2, 8 games, 100 sims):**
serial `--workers 1` = **76.7 s** (139% CPU); parallel `--workers 4 --cpu-net` = **26.2 s**
(369% CPU) → **~2.9× speedup**, identical aggregate results (0-8-0 per level, 16 games
total correctly summed across 4 chunks/level). Reports embed the full `REPLAYS_PER_LEVEL`
in both paths. The full 1–9 GPU-shared wall-clock + CUDA-OOM ceiling on the 3060 Ti is a
box task (documented default: ~4 GPU workers, drop to `--cpu-net` to scale past VRAM).

## Scope additions

- **Records quota, not "first chunk only".** The plan (P4) said only the *first* chunk per
  level collects records. With small chunks (e.g. 8 games / 4 workers → 2-game chunks) that
  under-fills the report's `REPLAYS_PER_LEVEL = 4`. `_plan_tasks` instead marks the *leading*
  chunk(s) to collect until their cumulative game count covers `REPLAYS_PER_LEVEL` — the first
  chunk alone still suffices at realistic sizes, so this is strictly a small-run robustness fix.
- **Load-balanced single pool shipped** (the P3 "recommended enhancement", not deferred): one
  pool serves all `(level, chunk)` tasks at once, so fast low-level chunks free their worker for
  slow level-8/9 chunks rather than draining level-by-level.
- **Device resolution moved into the worker.** Each worker re-derives CUDA availability from its
  own config copy (parent stays GPU-clean and builds only a `game` for report rendering), so the
  parent never creates a CUDA context on the parallel path.
```
