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
├── SelfPlayHistory/              Raw training data (board, policy, value)
│   ├── self_play_0.parquet
│   ├── self_play_1.parquet
│   └── ...
│
├── TrainingData/                 Neural network loss metrics
│   ├── generation=1/data.parquet
│   └── generation=2/data.parquet
│
├── ArenaData/                    Model evaluation results
│   ├── generation=1/arena.parquet
│   └── generation=2/arena.parquet
│
├── Timings/                      Phase durations per generation
│   ├── generation=1/timings.parquet
│   └── generation=2/timings.parquet
│
├── SelfPlayProfiling/            MCTS performance per episode
│   ├── generation=1/profiling.parquet
│   └── generation=2/profiling.parquet
│
├── ResourceUsage/                Process and GPU memory snapshots
│   ├── generation=1/resources.parquet
│   └── generation=2/resources.parquet
│
├── TrainingThroughput/           Samples/second per epoch
│   ├── generation=1/throughput.parquet
│   └── generation=2/throughput.parquet
│
├── Nets/                         Model checkpoints (.pth.tar)
└── Logs/                         Log files
```

---

## Two Storage Systems

All parquet I/O lives in `core/storage.py`. Data is written by two independent subsystems with different conventions:

| System | Directories | Partitioning | Read pattern |
|--------|-------------|-------------|--------------|
| **`SelfPlayStore`** | `SelfPlayHistory/` | Flat numbered files | `store.load()` / `store.load_window()` |
| **`MetricsCollector`** | All other 6 directories | Hive-partitioned (`generation=N/`) | `pd.read_parquet(directory)` |

The split exists because self-play data contains numpy arrays serialised as raw bytes (opaque binary blobs), which need a custom deserialiser to reconstruct. The metrics tables are all plain tabular data. Coach retains thin wrapper methods that delegate to `SelfPlayStore`.

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
| `board_dtype` | `"float64"` | Numpy dtype string |
| `policy_size` | `"17837"` (Blokus) / `"10"` (TicTacToe) | Length of the policy vector |
| `policy_dtype` | `"float64"` | Numpy dtype string |

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
| `pi_loss` | `float64` | Policy head cross-entropy loss for this batch |
| `v_loss` | `float64` | Value head MSE loss for this batch |
| `total_loss` | `float64` | `pi_loss + v_loss` for this batch |
| `average_pi_loss` | `float` | Running mean of policy loss up to this batch |
| `average_v_loss` | `float` | Running mean of value loss up to this batch |
| `average_loss` | `float` | **Derived at flush:** `average_pi_loss + average_v_loss` |

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

The new network is accepted if `wins / (wins + losses) >= config.update_threshold`.

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
