# Data Storage Reference

All persistent data for a training run lives under a single run directory:

```
{root_directory}/{run_name}/
```

Both paths are set in `RunConfig` (`alphablokus/config.py`). Every property below hangs off `RunConfig.run_directory`.

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

All parquet I/O lives in `alphablokus/storage/` (`selfplay_store.py` + `metrics.py`). Data is written by two independent subsystems with different conventions:

| System | Directories | Partitioning | Read pattern |
|--------|-------------|-------------|--------------|
| **`SelfPlayStore`** | `SelfPlayHistory/` | Flat numbered files | `store.load()` / `store.load_games()` / `store.load_recent_games()` |
| **`MetricsCollector`** | All other hive-partitioned directories (TrainingData, ArenaData, Timings, profiling, resources, throughput, entropy, accuracy, calibration, Elo, minimax, symmetry, arena replays) | Hive-partitioned (`generation=N/`) | `pd.read_parquet(directory)` |

The split exists because self-play data contains numpy arrays serialised as raw bytes (opaque binary blobs), which need a custom deserialiser to reconstruct. The metrics tables are all plain tabular data. Coach retains thin wrapper methods that delegate to `SelfPlayStore`. (`EvalSet/` is written directly by `Coach` as `.npy`, not by either class.)

---

## SelfPlayHistory — Raw Training Data

**Writer:** `SelfPlayStore.save()` in `alphablokus/storage/selfplay_store.py` (called via `ReplayBuffer.save_fresh()` in `alphablokus/training/replay_buffer.py`)
**Path:** `SelfPlayHistory/self_play_{generation}.parquet`
**Partitioning:** None — flat files, one per generation
**Granularity:** One row per training example (board position from a self-play game)

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `board` | `bytes` | Flattened numpy array of the **compact** board (`IBoard.to_compact()`) |
| `policy_indices` | `bytes` | `int32` action ids of the MCTS-improved policy's nonzero entries (**sparse**, see `storage/sparse_policy.py`) |
| `policy_values` | `bytes` | `float32` probabilities at those action ids, aligned with `policy_indices` |
| `value` | `float` | Game outcome from this position's perspective (+1 win, -1 loss) |

### File-Level Metadata

Shape and dtype information is stored in the parquet file's schema metadata (not as columns), so the deserialiser knows how to reconstruct the numpy arrays:

| Key | Example Value | Description |
|-----|---------------|-------------|
| `board_kind` | `"compact_v1"` | Storage-format marker. Its **absence** means a legacy file holding the dense `(C,N,N)` encoding — `load()` refuses these rather than misreading the bytes. |
| `policy_kind` | `"sparse_v1"` | Storage-format marker. Its **absence** means a legacy file holding one dense full-action-space `policy` blob per row — `load()` refuses these too. |
| `board_shape` | `"14,14"` (Blokus) / `"3,3"` (TicTacToe) | Comma-separated dimensions of the compact board |
| `board_dtype` | `"int8"` | Numpy dtype string (compact placement board is int8) |
| `policy_size` | `"17837"` (Blokus) / `"10"` (TicTacToe) | Length of the dense action space the sparse policies index into (for on-demand densify) |
| `game_sizes` | `"61,57,…"` | Per-game position counts (in row order). Lets `load_games()` split the flat rows back into per-game lists so the games-sized replay buffer can be reconstructed on resume. |

### Notes

- One file per generation, holding that generation's **fresh** games only (not the whole buffer). All episodes within a generation are concatenated, in game order, with boundaries recorded in `game_sizes`.
- Boards are stored compact (e.g. Blokus's 196-byte int8 placement board) and re-encoded to the dense `(C,N,N)` network input lazily at batch time via `IGame.encode_compact` — ~175× less buffer RAM than storing the dense planes.
- Policies are stored sparse (`(indices, values)` — the same form the live buffer holds), so save and resume never materialise a dense policy vector; densification happens per training batch in `_LazyPolicyDataset`. On-disk size is ~40× smaller than the old dense format (~1–2 KB vs ~71 KB per Blokus position).
- Data includes symmetry-augmented positions (rotations/reflections added by `IGame.get_symmetries()`).
- The rolling replay buffer keeps the last `replay_buffer_games` games in memory (oldest auto-evict); parquet files on disk are never deleted. On resume, `SelfPlayStore.load_recent_games()` refills the buffer newest-first.
- Cannot be read with a plain `pd.read_parquet()` — use `SelfPlayStore.load()` / `load_games()` / `load_recent_games()` from `alphablokus.storage.selfplay_store`.
- **Backend-agnostic:** the JAX self-play backend's harvester (`games/blokusduo/jax/harvest.py`) assembles its examples in the *exact* representation the python episode loop produces (compact canonical boards, sparse policies, same draw-sign convention), so jax-generated generations write, load, and resume through this same format with no marker or schema difference.
- **Legacy files** are refused loudly, never misread: pre-`compact_v1` files (dense board encodings, e.g. run2's parquets) and pre-`sparse_v1` files (dense policy blobs, all runs before oom-hardening O1) both fail `load()` with an explanatory error. Resume such runs from their checkpoints instead.

---

## Pentobi Distillation Corpus — Expert Training Data

**Writer:** `write_shard()` in `alphablokus/games/blokusduo/pentobi/corpus.py` (driven by `scripts/pentobi_corpus.py generate`)
**Path:** `<corpus_dir>/corpus_{shard:05d}.parquet`
**Partitioning:** Fixed-size shards (`--games-per-shard`, default 10 games) — the resume and parallelism unit
**Granularity:** One row per **expert ply** (position before each Pentobi move; random opening-prefix plies are never harvested)

### Schema

A strict **superset** of the SelfPlayHistory schema: the first four columns are byte-identical in meaning (and carry the same `board_kind`/`policy_kind` markers, asserted equal to `SelfPlayStore`'s in tests), so a trainer can decode either source with shared code.

| Column | Type | Description |
|--------|------|-------------|
| `board` | `bytes` | Canonical compact board (int8 14×14 placement grid, side-to-move perspective) |
| `policy_indices` | `bytes` | `int32` — exactly one entry: the action index Pentobi played |
| `policy_values` | `bytes` | `float32` — exactly `[1.0]` (**one-hot** behavioural-cloning target; label smoothing is applied at training time, not stored) |
| `value` | `float` | Game outcome from the side to move: +1 win / −1 loss / 0 draw |
| `margin` | `int32` | Final score margin from the side to move (`(white_score − black_score) × player`) |
| `player` | `int8` | Side to move: +1 White, −1 Black |
| `game_id` | `int64` | Globally unique game id (deterministic: game `g` of a run always has id `g`) |
| `ply` | `int32` | 0-based ply index within the full game (opening plies count, so harvested plies start at `opening_random_plies`) |
| `action` | `int32` | The played action index (denormalised copy of `policy_indices` for cheap analysis) |

### File-Level Metadata

`board_kind`/`board_shape`/`board_dtype`/`policy_kind`/`policy_size`/`game_sizes` exactly as SelfPlayHistory, plus:

| Key | Example Value | Description |
|-----|---------------|-------------|
| `dataset_kind` | `"pentobi_distill_v1"` | Corpus format marker (`read_shard_meta` refuses files without it) |
| `level` | `"9"` | Pentobi level both sides played at |
| `opening_random_plies` | `"4"` | Random opening-prefix length `k` (0 = seed variation only) |
| `games_meta` | JSON | Per-game provenance: `game_id`, `pentobi_seed`, `opening_actions`, `white_score`, `black_score` |

### Notes

- Read with `iter_corpus_examples()` (yields the same `(board, (indices, values), value)` tuples as the self-play pipeline) or `read_shard_meta()`/`analyze_corpus()` for provenance and diversity metrics; `validate_shard()` replays every game through the rules engine and checks every stored row.
- Symmetry augmentation is **not** stored — apply `IGame.get_symmetries` at training time (the stored board rebuilds via `board_from_compact`, and the one-hot policy transposes to the one-hot of `transpose_action`).
- Shards are written atomically (`.tmp` → rename), so any file matching the final name is complete; `generate` skips existing shards on rerun (resume) and each game's seeds are a pure function of `(--seed, game_id)`.
- Design + generation strategy: `docs/plans/pentobi-distillation.md`.

---

## Pentobi Distillation Corpus v2 — Soft Targets + an Opening Dataset

**Writers:** `write_game_shard()` / `export_opening()` in `alphablokus/games/blokusduo/pentobi/corpus_v2.py`
**Store:** `alphablokus/games/blokusduo/pentobi/store.py` (`SearchSpaceStore`) — the SQLite search-space DAG that *directs* generation and is the source of truth for the opening dataset
**Path:** `<corpus_dir>/games/corpus_{shard:05d}.parquet`, `<corpus_dir>/opening/opening_{shard:05d}.parquet`, `<corpus_dir>/store.sqlite`
**Granularity:** one row per harvested game ply (games) / one row per searched DAG node (opening)

v2 keeps v1's column *format* (`board_kind = "compact_v1"`, `policy_kind = "sparse_v1"`) and replaces its
*content*: the policy is Pentobi's whole preference distribution rather than a one-hot of the played move,
and openings are planned, labelled and stored instead of being random unharvested noise. Design:
`docs/plans/pentobi-corpus-v2.md` and `docs/plans/corpus-search-space-store.md`.

### `games/` schema

v1's columns survive with identical meaning (`value`, `margin`, `player`, `game_id`, `ply`, `action`), plus:

| Column | Type | Description |
|--------|------|-------------|
| `policy_indices` / `policy_values` | `bytes` | **Soft target**: top-32 children by visits, renormalised to sum 1 (v1 stored `[action]` / `[1.0]`) |
| `child_values` | `bytes` | `float32`, aligned to `policy_indices` — Pentobi's per-child value |
| `tail_mass` | `float32` | Visit mass dropped by the top-32 truncation (≈ 0.036 at ply 1, 0.017 at ply 2) |
| `search_value` | `float32` | Pentobi's backed-up value for the side to move = the **top child's** value (GTP `get_value` is a constant 0 and is never used) |
| `top_action` | `int32` | `argmax(visits)`; equal to `action` on a full-strength continuation, so any mismatch is visible |

### `opening/` schema

| Column | Type | Description |
|--------|------|-------------|
| `board` | `bytes` | The node's **key-frame** compact board (symmetry-canonical orientation; policy indices share the frame) |
| `policy_indices` / `policy_values` / `child_values` / `tail_mass` / `search_value` | — | As above |
| `depth` | `int32` | Plies from the empty board (= pieces placed) |
| `reach_weight` | `float32` | Product of ancestor visit shares, summed over DAG parents |
| `budget_share` / `planned_games` | `float32` / `int32` | The active plan's allocation at this node |
| `node_id` / `parent_id` | `int64` | Graph structure (`parent_id` = the first discovered parent) |
| `player` | `int8` | Side to move |
| `outcome_mean` / `outcome_count` | `float32` / `int32` | Empirical outcome of subtree playouts, filled by the `link` pass (count 0 until then) |

### File-Level Metadata

`board_kind`/`board_shape`/`board_dtype`/`policy_kind`/`policy_size`/`level` as v1, plus:

| Key | Example Value | Description |
|-----|---------------|-------------|
| `dataset_kind` | `"pentobi_distill_v2"` | Format marker for both datasets |
| `dag_hash` | `"3f9c…"` | SHA-256 over the searched DAG — a stale export is detectable, not merely suspected |
| `plan` | JSON | `plan_id`, `budget`, `temperature`, `min_replicas` of the generating allocation |
| `games_meta` | JSON | Per game: `game_id`, `node_id`, start `board_key` (hex), `replica`, `engine_seed`, `witness_actions`, scores, `plies` |

### Notes

- **Identity is positional.** A game is "replica *r* of position *P*", seeded `hash64(board_key ‖ replica)`, so re-planning at a bigger budget can never regenerate a game we already hold and never invalidates one.
- **Shards are self-describing**: `iter_shard_playouts()` rebuilds or verifies the store's `playouts` table from footers alone (`SearchSpaceStore.reconcile`) — the crash repair for a run that died between a shard rename and its DB transaction. **Sync the store DB with the shards**; it is the map of what they mean.
- Read with `distill.load_corpus_games_v2()` / `distill.load_opening_examples()` — the single reader for both datasets, yielding the pipeline's `(board, (indices, values), value)` tuples with an optional load-time target temperature τ, asserting `support ⊆ legal`, and returning each row's opening-subtree holdout unit; validate with `validate_game_shard()` (full replay; asserts the target sums to 1, `support ⊆ legal` — never equality, since Pentobi searches 315 of 414 first moves — `action ∈ support`, and `top_action == argmax`) and `validate_opening_shard()` (structural, plus a witness-path replay when handed the store).
- Opening rows are stored in their node's key frame, so they are *not* interchangeable with game rows byte-for-byte; the trainer's order-2 augmentation regenerates the mirror of either.

---

## Metrics Tables (Hive-Partitioned)

The remaining **13 datasets** (TrainingData, ArenaData, Timings, SelfPlayProfiling, ResourceUsage, TrainingThroughput, TrainingEntropy, PolicyAccuracy, ValueCalibration, EloRatings, MinimaxResults, SymmetryDiagnostic, ArenaReplays) are all written by `MetricsCollector.flush()` using the same pattern:

1. Components call `log_*()` methods during execution, buffering records in memory.
2. At the end of each generation, `Coach` calls `flush(config, generation)`.
3. `flush()` writes each buffer to `{directory}/generation={N}/{filename}`, dropping the `generation` column from the data (it's encoded in the directory name).
4. Buffers are cleared after writing.

All 13 can be read back with:

```python
df = pd.read_parquet(config.training_data_directory)  # or any of the 13
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

The new network is accepted if `(wins + 0.5·draws) / (wins + losses + draws) >= config.update_threshold` — the score-based rule in `alphablokus/evaluation/acceptance.py`. The decision is computed once there and stored in the `accepted` column.

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

MCTS is recreated per episode (`selfplay/generate.py::_generate_serial`; the worker pool does the same per episode in `parallel/pool.py`), so counters are naturally per-game with no reset needed.

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
| Directory paths | `alphablokus/config.py` | `RunConfig` properties |
| Metrics parquet I/O + W&B mirroring | `alphablokus/storage/metrics.py` | `MetricsCollector` |
| Self-play parquet I/O | `alphablokus/storage/selfplay_store.py` | `SelfPlayStore` |
| Buffer + persistence orchestration | `alphablokus/training/replay_buffer.py` | `ReplayBuffer.save_fresh()`, `load_recent()`, `load_for_resume()` |
| Memory snapshots | `alphablokus/training/diagnostics.py` | `MemorySnapshot`, `get_memory_snapshot()` |
| MCTS counters | `alphablokus/search/stats.py` + `search/mcts.py` | `MCTSEpisodeStats`, `get_episode_stats()` |
| Training loss + throughput logging | `alphablokus/games/base_wrapper.py` | `BaseNNetWrapper.train()` |
| Wiring (profiling + resources) | `alphablokus/training/coach.py` | `Coach.learn()` |

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
  "tags": ["ttt", "pipeline-check", "mac"],
  "mode": "online"
}
```

- `project` — W&B project name (auto-created on first run).
- `entity` — team/user; `null` uses the default for the logged-in account.
- `tags` — free-text tags surfaced in the W&B UI for filtering.
- `mode` — `"online"` (sync to cloud), `"offline"` (write to local `wandb/` dir, sync later), or `"disabled"` (no-op, useful for tests).

Omit the `wandb` block entirely (or set it to `null`) to disable W&B without code changes — the existing `test_run.json`, `full_run.json`, and `pipeline_check.json` would all behave identically to before the W&B integration if the block were removed.

### Authentication

W&B uses the API key in `~/.netrc` (written by `wandb login`). The project's API key for Henry's account lives as the `WANDB_API_KEY` entry in `local/secrets.env` (gitignored, never committed; see `docs/guides/CLOUD-TRAINING.md` → "Secrets"). On a fresh machine, copy that file across via `scp` and run `wandb login --relogin` once.

### Lifecycle

`MetricsCollector.__init__` calls `wandb.init(...)` if `config.wandb` is set. `Coach._learn_loop` runs the training inside a `try/finally` so `MetricsCollector.close()` (which calls `wandb.finish()`) always fires, including on a crash mid-run. Local `wandb/run-*` directories are gitignored and safe to delete at any time — the cloud has authoritative copies of synced runs.
