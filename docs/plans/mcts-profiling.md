# AlphaBlokus — MCTS Profiling & Instrumentation

Add detailed timing instrumentation to MCTS so we can identify bottlenecks in Blokus Duo self-play. Build a standalone profiling script that runs MCTS games and produces a visual report of where time is spent.

Prerequisites: Valid move generation (M1–M3 complete). Move count analysis script exists for reference.

---

## Checklist

| # | Item | Effort | Done |
|---|------|--------|------|
| P1 | Add `profiling_level` to `MCTSConfig` | 15 min | ✅ |
| P2 | Add move generation and game-ended timing to `MCTS.search()` | 30 min | ✅ |
| P3 | Add per-move stats collection to `MCTS.get_action_prob()` | 45 min | ✅ |
| P4 | Extend `MCTSEpisodeStats` with new fields | 20 min | ✅ |
| P5 | Update `MetricsCollector` / Coach to flush new fields | 20 min | ✅ |
| P6 | Build `scripts/mcts_profiling.py` with report generation | 1.5 hours | ✅ |
| P7 | Tests for new profiling functionality | 30 min | |

**Estimated total: ~4 hours**

---

## P1. Add `profiling_level` to `MCTSConfig`

**Current state:** `MCTSConfig` has `num_mcts_sims` and `cpuct`. No control over profiling granularity.

**Fix:** Add `profiling_level: str = "standard"` to `MCTSConfig`.

| Level | What's collected | Use case |
|-------|-----------------|----------|
| `"none"` | Nothing — skip all `perf_counter` calls | Maximum performance (benchmarking) |
| `"standard"` | Episode-level aggregates (current behaviour) | Normal training runs |
| `"detailed"` | Everything in standard + per-move breakdown + move-gen/game-ended timing | Profiling script, debugging |

**Files:** `core/config.py`

---

## P2. Add move generation and game-ended timing to `MCTS.search()`

**Current state:** `search()` times neural net inference (`_total_inference_time_s`) but `valid_move_masking` and `get_game_ended` calls are untracked — their time is lumped into the overall search time.

**Fix:** When profiling level is `"detailed"`, wrap `valid_move_masking` and `get_game_ended` calls with `perf_counter` and accumulate into new counters:

- `_total_valid_moves_time_s` — time in `valid_move_masking` (called once per leaf expansion)
- `_total_game_ended_time_s` — time in `get_game_ended` (called once per `search()` call)
- `_num_valid_moves_calls` — count of `valid_move_masking` calls (should equal `num_leaf_expansions`)
- `_num_game_ended_calls` — count of `get_game_ended` calls (should equal `total_sims`)

When profiling level is `"standard"` or `"none"`, these are zero (no timing overhead).

**Files:** `core/mcts.py`

---

## P3. Add per-move stats collection to `MCTS.get_action_prob()`

**Current state:** `get_action_prob()` runs N simulations and returns a policy. All timing is aggregated across the entire episode — we can't see which moves were expensive.

**Fix:** When profiling level is `"detailed"`, after each `get_action_prob()` call, record a per-move snapshot:

```python
@dataclass(frozen=True)
class MCTSMoveStats:
    move_number: int
    num_sims: int                    # Always config.num_mcts_sims
    search_time_s: float             # Wall time for this move's sim loop
    num_leaf_expansions: int         # New leaves added during this move
    num_valid_moves: int             # len(valid_moves) at this position
    inference_time_s: float          # Neural net time during this move
    valid_moves_time_s: float        # Move generation time during this move
    game_ended_time_s: float         # Game-ended check time during this move
```

Store these in a list `_move_stats: list[MCTSMoveStats]` on the MCTS instance. This requires taking deltas of the cumulative counters before and after each `get_action_prob()` call.

**Files:** `core/mcts.py`

---

## P4. Extend `MCTSEpisodeStats` with new fields

**Current state:** `MCTSEpisodeStats` has 6 fields (num_moves, total_sims, search/inference time, leaf expansions, tree size).

**Fix:** Add the new aggregate fields and optionally the per-move list:

```python
@dataclass(frozen=True)
class MCTSEpisodeStats:
    # Existing
    num_moves: int
    total_sims: int
    total_search_time_s: float
    total_inference_time_s: float
    num_leaf_expansions: int
    tree_size: int

    # New (detailed profiling)
    total_valid_moves_time_s: float = 0.0
    total_game_ended_time_s: float = 0.0
    num_valid_moves_calls: int = 0
    num_game_ended_calls: int = 0
    move_stats: tuple[MCTSMoveStats, ...] = ()
```

The new fields default to zero so existing code that constructs `MCTSEpisodeStats` without them still works. `move_stats` is a tuple (frozen) of per-move snapshots, empty unless detailed profiling is on.

**Files:** `core/mcts.py`

---

## P5. Update `MetricsCollector` / Coach to flush new fields

**Current state:** `Coach.learn()` calls `mcts.get_episode_stats()` and logs to `MetricsCollector.log_self_play_profiling()`. The parquet schema has the 6 existing fields.

**Fix:** Add the new aggregate fields to `log_self_play_profiling()`. Per-move stats are NOT written to parquet during training (too granular) — they're only used by the profiling script. The aggregate fields (`total_valid_moves_time_s`, etc.) are written if non-zero.

**Files:** `core/storage.py`, `core/coach.py`

---

## P6. Build `scripts/mcts_profiling.py` with report generation

Standalone script that plays N Blokus Duo games using MCTS with detailed profiling enabled, then produces an HTML report.

**Inputs:**
- `--num-games` (default 5 — MCTS games are slow)
- `--num-sims` (default 50 — override MCTS sim count)
- `--checkpoint` (optional — path to a `.pth.tar` file. If omitted, uses a fresh random net)

**Report contents:**
- **KPI cards:** avg game length, avg time per game, avg time per move, avg sims per second
- **Time breakdown pie chart:** inference vs move generation vs game-ended vs other (UCB selection, tree management)
- **Per-turn timing curves:** line charts showing how inference time, move generation time, and game-ended time vary by turn number
- **Per-turn move count curve:** legal moves per turn (from the MCTS perspective, not random play)
- **Histogram:** time per `get_action_prob()` call

**Output:** `temp/mcts_profiling/report.html` + raw data CSV/JSON.

**Files:** `scripts/mcts_profiling.py`

---

## P7. Tests for new profiling functionality

- `MCTSEpisodeStats` new fields are populated when `profiling_level="detailed"`
- `MCTSEpisodeStats` new fields are zero when `profiling_level="standard"`
- `move_stats` list length equals `num_moves` when detailed
- `move_stats` is empty when standard
- Per-move times sum to approximately the episode totals
- Existing TicTacToe MCTS tests still pass (backwards compatible)

**Files:** `tests/test_core/test_profiling.py`
