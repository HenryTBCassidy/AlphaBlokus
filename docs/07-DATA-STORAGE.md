# Data Storage Reference

All persistent data for a training run lives under a single run directory:

```
{root_directory}/{run_name}/
```

Both paths are set in `RunConfig` (`core/config.py`). Every property below hangs off `RunConfig.run_directory`.

---

## Directory Layout

```
{run_directory}/
│
├── SelfPlayHistory/              Raw training data (board, policy, value); flat, one file per gen
│   ├── self_play_0.parquet
│   └── ...
│
│   # ── Hive-partitioned metrics (all written by MetricsCollector.flush) ──
├── TrainingData/                 Per-batch loss (pi/v/total)
├── ArenaData/                    Arena W/L/D + accept decision (per gen)
├── Timings/                      Phase durations (per gen)
├── SelfPlayProfiling/            MCTS performance (per episode)
├── ResourceUsage/                Process + GPU memory snapshots (per phase)
├── TrainingThroughput/           Samples/second (per epoch)
├── TrainingEntropy/              Network policy entropy on the eval set (per epoch)
├── PolicyAccuracy/               Network top-1/top-5 vs eval-set targets (per epoch)
├── ValueCalibration/             Value-head reliability buckets (per epoch)
├── EloRatings/                   Elo vs frozen gen-0 baseline (per gen)
├── MinimaxResults/               Results vs perfect-play minimax — TTT only (per gen)
├── SymmetryDiagnostic/           Policy-symmetry KL per reference position (per gen)
├── ArenaReplays/                 Recorded arena games: moves + top-K policies (per gen)
│   └── generation=N/<file>.parquet   (each of the above follows this layout)
│
├── EvalSet/                      Frozen held-out positions (boards/policies/values .npy + marker)
├── Nets/                         Model checkpoints (.pth.tar)
└── Logs/                         Log files
```

Each hive-partitioned directory contains `generation=N/<name>.parquet` subdirectories; `pd.read_parquet(dir)` reconstructs the `generation` column from the path. `EvalSet/` is the one non-parquet metrics directory — three `.npy` arrays plus a `targets_kind.txt` marker (`minimax_v1` for TTT, `selfplay_v1` otherwise).

---

## Two Storage Systems

All parquet I/O lives in `core/storage.py`. Data is written by two independent subsystems with different conventions:

| System | Directories | Partitioning | Read pattern |
|--------|-------------|-------------|--------------|
| **`SelfPlayStore`** | `SelfPlayHistory/` | Flat numbered files | `store.load()` / `store.load_window()` |
| **`MetricsCollector`** | All other hive-partitioned directories (TrainingData, ArenaData, Timings, profiling, resources, throughput, entropy, accuracy, calibration, Elo, minimax, symmetry, arena replays) | Hive-partitioned (`generation=N/`) | `pd.read_parquet(directory)` |

The split exists because self-play data contains numpy arrays serialised as raw bytes (opaque binary blobs), which need a custom deserialiser to reconstruct. The metrics tables are all plain tabular data. Coach retains thin wrapper methods that delegate to `SelfPlayStore`. (`EvalSet/` is written directly by `Coach` as `.npy`, not by either class.)

---

## SelfPlayHistory — Raw Training Data

**Writer:** `SelfPlayStore.save()` in `core/storage.py` (called via `Coach.save_self_play_history()`)
**Path:** `SelfPlayHistory/self_play_{generation}.parquet`
**Partitioning:** None — flat files, one per generation
**Granularity:** One row per training example (board position from a self-play game)

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `board` | `bytes` | Flattened numpy array of the board state |
| `policy` | `bytes` | Flattened numpy array of the MCTS-improved policy vector |
| `value` | `float` | Game outcome from this position's perspective (+1 win, -1 loss) |

### File-Level Metadata

Shape and dtype information is stored in the parquet file's schema metadata (not as columns), so the deserialiser knows how to reconstruct the numpy arrays:

| Key | Example Value | Description |
|-----|---------------|-------------|
| `board_shape` | `"44,14,14"` (Blokus) / `"2,3,3"` (TicTacToe) | Comma-separated dimensions |
| `board_dtype` | `"float32"` | Numpy dtype string (board planes are float32) |
| `policy_size` | `"17837"` (Blokus) / `"10"` (TicTacToe) | Length of the policy vector |
| `policy_dtype` | `"float32"` | Numpy dtype string (policy cast to float32 in self-play to halve buffer footprint) |

### Notes

- One file per generation, not one file per episode. All episodes within a generation are concatenated.
- Data includes symmetry-augmented positions (rotations/reflections added by `IGame.get_symmetries()`).
- Old generations are pruned from memory by a sliding window (`Coach._generation_window_size()`), but parquet files on disk are never deleted.
- Cannot be read with a plain `pd.read_parquet()` — use `SelfPlayStore.load()` or `SelfPlayStore.load_window()` from `core.storage`.
- The Blokus board encoding is `44×14×14` (44-channel per-piece spatial planes). TicTacToe uses `2×3×3` (2-channel player split). Both are built by `board.as_multi_channel(player)`.

---

## Metrics Tables (Hive-Partitioned)

The remaining 6 datasets are all written by `MetricsCollector.flush()` using the same pattern:

1. Components call `log_*()` methods during execution, buffering records in memory.
2. At the end of each generation, `Coach` calls `flush(config, generation)`.
3. `flush()` writes each buffer to `{directory}/generation={N}/{filename}`, dropping the `generation` column from the data (it's encoded in the directory name).
4. Buffers are cleared after writing.

All 6 can be read back with:

```python
df = pd.read_parquet(config.training_data_directory)  # or any of the 6
# 'generation' column is automatically reconstructed from directory names
```

---

### TrainingData — Neural Network Loss

**Path:** `TrainingData/generation=N/data.parquet`
**Granularity:** One row per training batch
**Logged by:** `NNetWrapper.train()` via `MetricsCollector.log_training()`

| Column | Type | Description |
|--------|------|-------------|
| `epoch` | `int` | Epoch index within the generation |
| `batch_number` | `int` | Batch index within the epoch |
| `pi_loss` | `float64` | Policy-head loss for this batch (`F.kl_div`, batch-mean) |
| `v_loss` | `float64` | Value head MSE loss for this batch |
| `total_loss` | `float64` | `pi_loss + v_loss` for this batch |

> Earlier versions also wrote `average_pi_loss` / `average_v_loss` / `average_loss` running-mean columns. These were **removed** — they reset every epoch and produced misleading start-of-epoch spikes; the reporting layer smooths the raw per-batch losses instead.

---

### ArenaData — Model Evaluation

**Path:** `ArenaData/generation=N/arena.parquet`
**Granularity:** One row per generation
**Logged by:** `Coach.learn()` via `MetricsCollector.log_arena()`

| Column | Type | Description |
|--------|------|-------------|
| `wins` | `int` | Games won by the new (candidate) network |
| `losses` | `int` | Games won by the previous (incumbent) network |
| `draws` | `int` | Drawn games |
| `accepted` | `bool` | Whether the new network passed the acceptance test (persisted so the report never recomputes it) |

The new network is accepted if `(wins + 0.5·draws) / (wins + losses + draws) >= config.update_threshold` — the score-based rule in `core/acceptance.py`. The decision is computed once there and stored in the `accepted` column.

---

### Timings — Phase Durations

**Path:** `Timings/generation=N/timings.parquet`
**Granularity:** One row per phase per generation (typically 4 rows)
**Logged by:** `Coach.learn()` via `MetricsCollector.log_timing()`

| Column | Type | Description |
|--------|------|-------------|
| `cycle_stage` | `str` | `"SelfPlay"`, `"Training"`, `"Arena"`, or `"WholeCycle"` |
| `time_elapsed` | `float` | Wall-clock seconds (via `time.perf_counter()`) |

`WholeCycle` is the total generation time, roughly equal to the sum of the other three.

---

### SelfPlayProfiling — MCTS Performance

**Path:** `SelfPlayProfiling/generation=N/profiling.parquet`
**Granularity:** One row per self-play episode (game)
**Logged by:** `Coach.learn()` reading from `MCTS.get_episode_stats()`

| Column | Type | Description |
|--------|------|-------------|
| `episode` | `int` | Episode index within the generation |
| `num_moves` | `int` | Moves played in the game (`get_action_prob()` calls) |
| `total_sims` | `int` | Total MCTS simulations across all moves |
| `total_search_time_s` | `float` | Wall-clock time in MCTS simulation loops |
| `total_inference_time_s` | `float` | Wall-clock time in `nnet.predict()` calls |
| `num_leaf_expansions` | `int` | New nodes added to the search tree |
| `tree_size` | `int` | Unique states in the tree at episode end |
| `sims_per_second` | `float` | **Derived:** `total_sims / total_search_time_s` |
| `inference_fraction` | `float` | **Derived:** `total_inference_time_s / total_search_time_s` |

MCTS is recreated per episode (`Coach` line 174), so counters are naturally per-game with no reset needed.

---

### ResourceUsage — Memory Snapshots

**Path:** `ResourceUsage/generation=N/resources.parquet`
**Granularity:** One row per phase per generation (3 snapshots: after SelfPlay, Training, Arena)
**Logged by:** `Coach.learn()` via `_get_memory_snapshot()`

| Column | Type | Description |
|--------|------|-------------|
| `cycle_stage` | `str` | `"SelfPlay"`, `"Training"`, or `"Arena"` |
| `process_rss_bytes` | `int` | Process RSS via `psutil` (bytes on all platforms) |
| `gpu_memory_bytes` | `float` / `null` | GPU allocation (CUDA or MPS), null if no GPU |

Memory is measured at the Python process level (not system-wide). GPU memory covers only the tensor allocator tracked by PyTorch, not driver-level overhead.

---

### TrainingThroughput — Samples Per Second

**Path:** `TrainingThroughput/generation=N/throughput.parquet`
**Granularity:** One row per epoch per generation
**Logged by:** `NNetWrapper.train()` via `MetricsCollector.log_training_throughput()`

| Column | Type | Description |
|--------|------|-------------|
| `epoch` | `int` | Epoch index within the generation |
| `num_examples` | `int` | Total training examples processed in this epoch |
| `epoch_time_s` | `float` | Wall-clock seconds for the full epoch |
| `samples_per_second` | `float` | **Derived:** `num_examples / epoch_time_s` |

---

### TrainingEntropy / PolicyAccuracy / ValueCalibration — per-epoch network diagnostics

These three are logged by `BaseNNetWrapper.train()` after each epoch when an `EvalSet` is supplied — they forward-pass the network (no MCTS) over the frozen held-out positions.

**TrainingEntropy** (`entropy.parquet`) — one row per epoch:

| Column | Type | Description |
|--------|------|-------------|
| `epoch` | `int` | Epoch within the generation |
| `mean_entropy` / `std_entropy` | `float` | Network policy entropy over the eval set |
| `eval_set_size` | `int` | Number of held-out positions |

**PolicyAccuracy** (`accuracy.parquet`) — one row per epoch:

| Column | Type | Description |
|--------|------|-------------|
| `epoch` | `int` | Epoch within the generation |
| `top1_accuracy` / `top5_accuracy` | `float` | Fraction of eval positions where the net's top-1 / top-5 hits a target-optimal action |
| `eval_set_size` | `int` | Number of held-out positions |

**ValueCalibration** (`calibration.parquet`) — one row per reliability bucket per epoch (10 buckets):

| Column | Type | Description |
|--------|------|-------------|
| `epoch` | `int` | Epoch within the generation |
| `bucket_idx` | `int` | Bucket index 0–9 over predicted v ∈ [-1, 1] |
| `bucket_center` | `float` | Bucket centre |
| `bucket_mean_actual` | `float`/`null` | Mean actual outcome of positions in this bucket (null if empty) |
| `bucket_count` | `int` | Positions in this bucket |

---

### EloRatings — strength vs the frozen gen-0 baseline

**Path:** `EloRatings/generation=N/elo.parquet` · **Granularity:** one row per generation · **Logged by:** `Coach` via `MetricsCollector.log_elo()`

| Column | Type | Description |
|--------|------|-------------|
| `elo_rating` | `float` | Absolute display rating = `baseline_rating + elo_diff` |
| `elo_diff` | `float` | `400·log10(score_rate/(1−score_rate))` vs the frozen baseline |
| `baseline_rating` | `int` | Display anchor (default 400) |
| `score_rate` | `float` | `(wins + 0.5·draws) / games` |
| `wins` / `losses` / `draws` / `games` | `int` | Results vs the baseline |

---

### MinimaxResults — vs perfect play (TicTacToe only)

**Path:** `MinimaxResults/generation=N/minimax.parquet` · **Granularity:** one row per generation · **Logged by:** `Coach` via `MetricsCollector.log_minimax()`. Only written when `game == "tictactoe"` and `minimax_games_per_gen > 0`.

| Column | Type | Description |
|--------|------|-------------|
| `wins` / `losses` / `draws` / `games` | `int` | Results vs the minimax oracle |
| `win_rate` / `draw_rate` / `loss_rate` | `float` | Derived rates. Target: `draw_rate → 1`, `loss_rate → 0` |

---

### SymmetryDiagnostic — policy equivariance

**Path:** `SymmetryDiagnostic/generation=N/symmetry.parquet` · **Granularity:** one row per (reference position, symmetry) · **Logged by:** `Coach` via `MetricsCollector.log_symmetry_diagnostic()`

| Column | Type | Description |
|--------|------|-------------|
| `position_idx` | `int` | Reference-position index (stable across generations) |
| `symmetry_idx` | `int` | Which non-identity symmetry (Blokus has 1: the transpose) |
| `kl_divergence` | `float` | KL between the net's policy on the symmetric board and the symmetric image of its policy. 0 = equivariant |
| `top1_match` | `bool` | Whether the argmax matched under the symmetry |

---

### ArenaReplays — recorded games for the replay viewer

**Path:** `ArenaReplays/generation=N/games.parquet` · **Granularity:** one row per move · **Logged by:** `Coach` via `MetricsCollector.log_arena_game()`. Bulky structured data — **not** mirrored to W&B.

| Column | Type | Description |
|--------|------|-------------|
| `game_idx` / `move_idx` | `int` | Game within the generation / move within the game |
| `player` | `int` | +1 or −1 — who moved |
| `action` | `int` | Action index chosen |
| `top_k_actions` / `top_k_probs` | `list` | Top-K visited actions and their MCTS visit fractions |
| `played_prob` | `float` | Visit fraction of the played action (surfaced even when it falls outside top-K) |
| `outcome` | `float` | Game outcome (denormalised onto every move row) |
| `player1_was_white` | `bool` | Which side player1 played (denormalised) |

---

## Source Code Reference

| Concern | File | Key functions/classes |
|---------|------|-----------------------|
| Directory paths | `core/config.py` | `RunConfig` properties |
| All parquet I/O | `core/storage.py` | `MetricsCollector`, `SelfPlayStore` |
| Coach thin wrappers | `core/coach.py` | `save_self_play_history()`, `load_self_play_history()` |
| Memory snapshots | `core/coach.py` | `MemorySnapshot`, `_get_memory_snapshot()` |
| MCTS counters | `core/mcts.py` | `MCTSEpisodeStats`, `get_episode_stats()` |
| Training loss + throughput logging | `games/base_wrapper.py` | `BaseNNetWrapper.train()` |
| Wiring (profiling + resources) | `core/coach.py` | `Coach.learn()` |

---

## Weights & Biases (optional, additive)

W&B is an **additive** reporting layer — it does not replace any of the parquet writes above. If a run's `RunConfig.wandb` is set, every `log_*` call on `MetricsCollector` is mirrored to `wandb.log({...})` in addition to its existing buffer append. Parquet, HTML reports, and Nets behaviour are unchanged.

### When to use which

| Use case | Best tool |
|----------|-----------|
| Watch a long unattended run from another machine / phone | **W&B dashboard** (live, browser-based, no `rsync` needed) |
| Retrospective deep-dive after a run finished | **HTML report** (rendered from local parquets, fully offline, rich Plotly visuals) |
| Compare runs against each other | W&B (built-in run comparison) |
| Archival / reproducibility | Parquets in `temp/<run_name>/` (source of truth) |
| Air-gapped / no network | Set `wandb.mode: "disabled"` or omit the `wandb` block entirely |

### What gets logged to W&B

Every `MetricsCollector.log_*` call has a corresponding `wandb.log` payload, namespaced by topic. Each namespace is registered with a `step_metric` so its panels plot against a meaningful x-axis (generation / cumulative episode / cumulative batch / wall-clock) instead of W&B's internal step counter.

| Namespace | Source method | Granularity | x-axis (step_metric) | Series |
|-----------|---------------|-------------|----------------------|--------|
| `progress/*` | `log_progress`, auto-augmented in `_publish` | Per log call | `progress/wall_clock_seconds` (self) | `generation`, `epoch`, `episode`, `batch`, `generation_fraction`, `eta_seconds`, `wall_clock_seconds` |
| `self_play/*` | `log_self_play_profiling` | Per self-play episode | `global_episode` | `num_moves`, `total_sims`, `search_time_s`, `inference_time_s`, `sims_per_second`, `inference_fraction`, `leaf_expansions`, `tree_size`, `policy_entropy` |
| `self_play_per_gen/*` | `_publish_self_play_per_gen` (flush) | Per generation | `generation` | aggregates: `policy_entropy_mean/std`, `num_moves_mean`, `tree_size_mean`, `sims_per_second_mean`, `inference_fraction_mean` |
| `training/*` | `log_training` | Per training batch | `global_batch` | `pi_loss`, `v_loss`, `total_loss`, plus `network_policy_entropy`, `network_top1_accuracy`, `network_top5_accuracy`, `value_calibration_error` (per epoch) |
| `training_per_gen/*` | `_publish_training_per_gen` (flush) | Per generation | `generation` | aggregates: `pi_loss`, `v_loss`, `total_loss`, `network_policy_entropy`, `network_top1_accuracy`, `network_top5_accuracy`, `value_calibration_error` |
| `arena/*` | `log_arena` | Per generation | `generation` | `wins`, `losses`, `draws`, `win_rate`, `accepted`, `acceptance_rate` (running) |
| `elo/*` | `log_elo` | Per generation | `generation` | `rating`, `diff_vs_baseline`, `baseline_rating`, `score_rate`, `wins`, `losses`, `draws` |
| `minimax/*` | `log_minimax` (TTT only) | Per generation | `generation` | `win_rate`, `draw_rate`, `loss_rate`, `wins`, `losses`, `draws` |
| `throughput/*` | `log_training_throughput` | Per training epoch | `generation` | `num_examples`, `epoch_time_s`, `samples_per_second` |
| `timing/*` | `log_timing` | Per phase per generation | `generation` | `{SelfPlay,Training,Arena,WholeCycle}_s` |
| `resources/*` | `log_resource_usage` | Per phase per generation | `generation` | `<phase>_rss_mb`, `<phase>_gpu_mb` (in MB) |

The bare counter names (`generation`, `epoch`, `episode`, `batch`) are registered with `hidden=True` so they don't auto-chart as standalone panels; they're mirrored into `progress/*` in `_publish` for explicit display.

Plus W&B captures the full `RunConfig` (flattened via a Path→str helper) at run init, so hyperparameters appear alongside the metrics in the dashboard's config panel.

### Recommended workspace layout

The dashboard reads top-to-bottom in the same order as a single training cycle, so a glance tells you both where you are and what's happening.

| Order | Section | Headline panels | What it answers |
|-------|---------|-----------------|-----------------|
| 1 | **Run progress** | `progress/generation`, `progress/eta_seconds`, `progress/wall_clock_seconds`, plus sawtooth `progress/epoch` / `progress/episode` / `progress/batch` | Where am I, when will it finish? |
| 2 | **Self-play** | `self_play_per_gen/*` (per-gen means up top), `self_play/*` (per-episode raw, collapsible) | Is MCTS exploring sensibly? Game length plausible? |
| 3 | **Training loss** | `training_per_gen/{total,pi,v}_loss` combined on one panel, per-batch `training/*` collapsible | Is loss decreasing? |
| 4 | **Learning quality** | `training_per_gen/network_top1_accuracy` (largest), then `top5_accuracy`, `value_calibration_error`, `network_policy_entropy` | Is the net internalising correct play? |
| 5 | **Arena** | `arena/win_rate` (with 0.55 horizontal threshold), `arena/acceptance_rate`, stacked bar of wins/losses/draws | Did the new net win the head-to-head? |
| 6 | **Strength** | `elo/rating` (largest, dashed baseline at 400), `elo/diff_vs_baseline`, `minimax/win_rate`/`draw_rate`/`loss_rate` (TTT only) | Is the model actually getting stronger? |
| 7 | **Operational** | resource snapshots, sims/sec, GPU memory | Is the PC OK? |

The W&B workspace itself (panel placement, chart types, axis ranges) is configured once per project in the UI: open the project, drag panels into sections, set chart type via gear icon, rename display titles. Persists across runs.

### Configuration

A `wandb` block in any run config enables it:

```json
"wandb": {
  "project": "alphablokus-poc",
  "entity": null,
  "tags": ["ttt", "smoke", "mac"],
  "mode": "online"
}
```

- `project` — W&B project name (auto-created on first run).
- `entity` — team/user; `null` uses the default for the logged-in account.
- `tags` — free-text tags surfaced in the W&B UI for filtering.
- `mode` — `"online"` (sync to cloud), `"offline"` (write to local `wandb/` dir, sync later), or `"disabled"` (no-op, useful for tests).

Omit the `wandb` block entirely (or set it to `null`) to disable W&B without code changes — the existing `test_run.json`, `full_run.json`, and `smoke_test.json` would all behave identically to before the W&B integration if the block were removed.

### Authentication

W&B uses the API key in `~/.netrc` (written by `wandb login`). The project's API key for Henry's account lives at `local/wandb_api_key.txt` (gitignored, never committed). On a fresh machine, copy it across via `scp` and run `wandb login --relogin` once.

### Lifecycle

`MetricsCollector.__init__` calls `wandb.init(...)` if `config.wandb` is set. `Coach._learn_loop` runs the training inside a `try/finally` so `MetricsCollector.close()` (which calls `wandb.finish()`) always fires, including on a crash mid-run. Local `wandb/run-*` directories are gitignored and safe to delete at any time — the cloud has authoritative copies of synced runs.
