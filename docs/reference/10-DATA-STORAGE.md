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
| `board_shape` | `"14,18"` (Blokus) / `"3,3"` (TicTacToe) | Comma-separated dimensions |
| `board_dtype` | `"float64"` | Numpy dtype string |
| `policy_size` | `"17837"` (Blokus) / `"10"` (TicTacToe) | Length of the policy vector |
| `policy_dtype` | `"float64"` | Numpy dtype string |

### Notes

- One file per generation, not one file per episode. All episodes within a generation are concatenated.
- Data includes symmetry-augmented positions (rotations/reflections added by `IGame.get_symmetries()`).
- Old generations are pruned from memory by a sliding window (`Coach._generation_window_size()`), but parquet files on disk are never deleted.
- Cannot be read with a plain `pd.read_parquet()` — use `SelfPlayStore.load()` or `SelfPlayStore.load_window()` from `core.storage`.
- The current Blokus board is `14×18` (the 14×14 board with 2-column piece-inventory strips on each side). This may change to `44×14×14` when multi-channel encoding is implemented (see `09-BOARD-ENCODING-OPTIONS.md`).

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
