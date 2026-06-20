# Self-Play Quick Wins + Standardised Benchmark

Two things, driven by the [Linux profiling](../research/linux-performance-findings.md):
(1) land the cheap self-play speedups the profiling surfaced, and (2) build a **single
standardised benchmark command** that emits **one self-contained HTML report** (time +
memory, charts) so we stop specifying benchmark runs ad hoc and can do clean before/after.

Scope note: this is **self-play speed + benchmarking only**. The training-step RAM fix
(lazy/streamed boards, for ≥10k games/gen) is tracked separately in the findings doc — not
here. Disk is a non-issue (248 GB free, ~26 bytes/example on disk), so **no data
compression** is in scope.

**Guard:** move-gen changes must stay bit-identical — the F2 equivalence tests
(`tests/test_blokusduo/test_movegen_equivalence.py`, `test_movegen_determinism.py`) and the
MCTS golden/parallel-determinism tests are the safety net.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| 1 | **Standardise the benchmark** — one command, fixed suite, one self-contained HTML report (time pie + top-function bars + worker-sweep games/s + memory charts) | ~0.5 day | High | ✅ |
| 2 | **Baseline run** — run the new benchmark on current `main`-equivalent code; save the "before" report | ~0.5 hr | High | ✅ |
| 3 | **Quick win A — game-ended via F2** — replace the old-generator existence check in `get_game_ended` with an F2 early-exit `has_any_move`; bit-identical | ~0.5 day | High | ✅ |
| 4 | **Quick win B — retire the old generator from production** — move `_generate_valid_moves`/`_valid_moves`/`_placements_at_point`/`_all_cells_valid` to a reference module used only as the test oracle; production routes through F2 everywhere | ~0.5 day | Medium | ⏸ Deferred |
| 5 | **Re-benchmark + compare** — rerun the standard suite; before/after report; decide if the Cython move-gen loop is worth pursuing (separate plan) based on the measured gain + GPU-utilisation headroom | ~0.5 hr | High | ✅ |

> **Step 3 result (2026-06-20):** routing `get_game_ended` through F2's early-exit
> `has_any_move` (commit `07a199d`) gave **single-worker 5.13 → 4.21 s/game (1.22×)** and
> **16-worker 1.449 → 1.646 games/s (1.14×)** on the standardised benchmark — bit-identical
> (guarded by the new `has_any_move` equivalence test). Reports: `temp/benchmarks/baseline_6cfa88e.html`
> and `after_step3_07a199d.html`.
>
> **Step 4 deferred:** step 3 already removed the old generator from the *hot path* (the win
> above). Fully retiring it to a reference module is blocked on reimplementing the **first-move**
> path in F2 (`_fill_first_move_mask` + the first-move branch of `has_any_move` still use the old
> generator) — a **cold, one-off-per-game path with no performance value**. Deferred as low-value
> hygiene; the old generator is now confined to (a) the cold first-move path and (b) the
> equivalence-test oracle. Revisit only if we want the full cleanup.

---

## 1. Standardise the benchmark

One committed entry point — e.g. `scripts/benchmark.py` (wrapping the existing
`profile_self_play` machinery) — run with no ad-hoc args:

```bash
uv run python -m scripts.benchmark --config <cfg>   # sensible default config
```

**The report conveys DATA ONLY — no recommendations / "things to do".** Its job is to show
where time and memory go (charts + tables) as clearly as possible; interpretation and next
steps are discussed in the terminal, not baked into the report. Err on the side of *more*
helpful charts.

It runs a **fixed suite** and writes **one self-contained HTML** (charts inline as base64
so it's portable), fetched to the laptop automatically:

- **Time:** single-worker `s/game` (N repeats, averaged ± spread); a **pie** of the
  self-play time split (move-gen / UCB-select / inference / backprop / board); **bar** of
  top-N own-time functions (cProfile).
- **Throughput:** **worker sweep {1, 2, 4, 8, 16}** games/s (bar/line) — shows the
  parallel scaling + where it saturates.
- **Memory:** in-episode tree peak; training-step RSS-vs-buffer-size ramp (**line**, with
  the 32 GB line); examples/game.
- **Header:** config, net size, commit, date — so every report is self-describing and
  reproducible.

Design goals: deterministic (fixed seed), self-labelling, re-runnable for before/after, no
hand-editing. This replaces the ad-hoc `bench_sweep.py` / `run_allprof.sh` scratch scripts.

## 2. Baseline run

Run #1 of the standard suite on the current code → the "before" report. Establishes the
reference the quick wins are measured against.

## 3. Quick win A — game-ended via F2

`get_game_ended` → `_has_valid_moves` → `_generate_valid_moves` is the **old, non-F2**
generator (no F2 branch exists), and it runs per new leaf — ~16% of single-worker self-play
and fully redundant with the F2 enumeration done at expansion. Add an **F2 early-exit
existence check** (`has_any_move(board, player)` — stop at the first piece that fits, reusing
the F2 forbidden/lookup machinery) and route `get_game_ended` through it. Bit-identical
(same legality logic as F2 enumeration); guarded by the equivalence + determinism tests.

## 4. Quick win B — retire the old generator from production

With A done, the old generator has no production caller. Move it (and its helpers) into a
**reference module** (`movegen_reference.py` or similar) whose sole job is to be the
correctness oracle the F2 equivalence tests check against. Production code paths
(`valid_move_masking`, `get_game_ended`) go through F2 exclusively. This is the clean state
the F2 migration should have left — one production generator, one test oracle.

## 5. Re-benchmark + compare

Rerun the standard suite → "after" report. Compare single-worker `s/game`, the time pie,
and the worker-sweep games/s. Then decide — *from the measured numbers + a GPU-utilisation
check at 16 workers* — whether the **Cython move-gen inner loop** (`_fill_legal_actions_for_anchor`,
~25%) is worth a separate plan, or whether we've banked enough and should move to Pentobi
benchmarking + a serious run.
