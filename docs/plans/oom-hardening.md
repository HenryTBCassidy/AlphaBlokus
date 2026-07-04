# OOM Hardening — Kill the Dense-Policy Materializations

> **Status (2026-07-03): in flight, execution deferred.** The box is unavailable for the verification this plan needs (peak-RSS at scale), so [refactor-repo-architecture](archive/refactor-repo-architecture.md) runs first; that plan's R18 re-maps the file/line citations below onto the post-refactor layout, and its R31 performs the `load_window` deletion mentioned under O2. Until O1–O2 land, keep the interim mitigation: `num_eps ≤ 8000`.

This plan fixes the recurring out-of-memory crashes on the 32 GB box (most recently `blokus_gumbel_overnight_20260703`, which OOM-killed the whole box at the gen-1 self-play→train transition and lost the night). It is the output of a full-codebase memory-materialization audit (four parallel sweeps: self-play/buffer/persistence, JAX backend, training/inference/MCTS, eval/arena/reporting/scripts).

**Root cause (single fact):** the on-disk self-play format stores the **policy densely** (17,837-dim float32 ≈ 71 KB per position). Boards are *already* compact on disk (`BOARD_KIND = "compact_v1"`, 196 B). So every save densifies a whole generation of policies, and every resume rehydrates a whole buffer of them. At 10k games/gen that's ~25 GB on save (≈57 GB with symmetry doubling); a 50k-game resume is ~125–285 GB. Storing the policy **sparse** on disk (it is *already sparse in the live buffer*) eliminates the top four CRITICAL findings at once.

**Interim mitigation (already communicated):** keep `num_eps ≤ 8000` until O1–O2 land. After O1–O2, 10k+ games/gen is safe on the box.

**Companion docs:** [PLAN-FORMAT](../guides/PLAN-FORMAT.md), [STYLE-GUIDE](../guides/STYLE-GUIDE.md). Suggested branch: `fix/oom-sparse-selfplay`.

**Audit map (severity → step):**

| Finding | Where | Severity | Fixed by |
|---|---|---|---|
| Save densifies whole generation | `training/replay_buffer.py::save_fresh`, `storage/selfplay_store.py::save` | CRITICAL | O1, O3 |
| Load rehydrates whole file dense (`to_pandas`/`iterrows`) | `storage/selfplay_store.py::load` (`to_pandas`/`iterrows`) | CRITICAL | O2 |
| Resume rehydrates whole buffer dense (live=sparse, resume=dense) | `storage/selfplay_store.py::load_recent_games` | CRITICAL | O2 |
| JAX harvester holds dense 17,837 policy per open position | `games/blokusduo/jax/harvest.py` (dense per-position policy) | MODERATE | O4 |
| No JAX VRAM fraction cap on shared 8 GB card | `games/blokusduo/jax/__init__.py` | MODERATE | O5 |
| Whole generation accumulated in host RAM before buffer | `parallel/pool.py::run_self_play_episodes_parallel` (result accumulation), `selfplay/generate.py` (per-backend fresh_games accumulation), `games/blokusduo/jax/backend.py::generate_self_play_games` (wave loop) | MODERATE | O6 |
| Report loads entire ArenaReplays history then samples 16×6 | `reporting/training.py::_make_arena_replays_section`; `scripts/replay.py` (whole-history read) | MODERATE | O7 |
| Repeated OOMs undetected until crash | — | process gap | O8 |
| Eval-set dense (safe at 200, scales badly); MCTS transient churn | `games/base_wrapper.py::_compute_eval_set_diagnostics`, `search/mcts.py::get_action_prob` / `_simulate_batch` transients | MINOR | O9 |

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| O1 | Store self-play **policy sparse** on disk; drop the densify in `save_self_play_history`; add `policy_kind` marker | 2-3 h | High | ✅ |
| O2 | Sparse read + resume: `load`/`load_recent_games` return sparse (match live buffer); drop `to_pandas`/`iterrows` | 2-3 h | High | ✅ |
| O3 | Stream the parquet write in row-group chunks (`ParquetWriter`) — no whole-generation table in RAM | 1-1.5 h | Medium | ✅ |
| O4 | Sparsify JAX harvester policy at append (kill per-position dense 17,837 + `np.zeros` churn) | 1-1.5 h | Medium | ✅ |
| O5 | Cap JAX VRAM via `XLA_PYTHON_CLIENT_MEM_FRACTION` on the shared 8 GB card | 30-45 m | Medium | ✅ |
| O6 | Stream completed games into the buffer instead of accumulating the whole generation | 2-3 h | Medium | ✅ |
| O7 | Partition-filtered ArenaReplays reads (report + `scripts/replay.py`) | 1-1.5 h | Medium | ✅ |
| O8 | Guardrails: startup RAM-budget check + peak-RSS logging at phase transitions + regression test | 2 h | High | ✅ |
| O9 | Minor trims: eval-set cap assert, MCTS `int32`/list transients | 45 m | Low | ✅ |

> **O6 note:** implemented as a per-game `sink` callback (`generate_games(..., sink=replay_buffer.add_game)`) rather than a generator. Game order is unchanged in all three backends — the pool path streams off `pool.map`'s submission-order iterator (no as-completed reordering), so the determinism tests pass verbatim; the jax path streams examples per harvested game but still returns its stats at the end (their timing apportionment needs the generation's total wall clock).

---

## O1. Store self-play policy sparse on disk

**Current state.** `ReplayBuffer.save_fresh` (`training/replay_buffer.py`) turns the generation's in-RAM **sparse** policies into **dense** ones purely to persist them:

```python
dense = deque((board, as_dense(pi, action_size), value) for board, pi, value in flat)
self._self_play_store.save(dense, file_index, game_sizes=game_sizes)
```

`SelfPlayStore.save` (`storage/selfplay_store.py`) then serializes each dense policy to bytes (`p.tobytes()`) in a single DataFrame. The dense deque alone is ~25 GB at 10k games; the `tobytes` list and `from_pandas` copy stack more on top. This is the exact allocation that OOM-killed the box.

**Fix.** Never densify to persist. The live buffer policy is already `(indices, values)` (see `storage/sparse_policy.py`). Persist those directly:
- `ReplayBuffer.save_fresh`: pass the sparse examples straight through — delete the `as_dense` densify.
- `SelfPlayStore.save`: write two policy columns (`policy_indices` int32 bytes, `policy_values` float32 bytes) instead of one dense `policy` blob. Add a `policy_kind = "sparse_v1"` schema marker alongside the existing `board_kind`.
- Keep `policy_size` (action space) in metadata so the loader can densify on demand.

On-disk size drops ~40× (71 KB → ~1–2 KB/position) and the save-time RAM spike is gone.

**Migration.** Mirror the existing legacy-refusal (`storage/selfplay_store.py::load`): a file missing `policy_kind` (i.e. old dense format) is refused with a clear message. No auto-migration — self-play parquets are per-run artifacts and we're starting fresh scaled runs.

---

## O2. Sparse read + resume path

**Current state.** `SelfPlayStore.load` (`storage/selfplay_store.py::load` (`to_pandas`/`iterrows`)) reads the whole file, calls `table.to_pandas()`, then `for _, row in df.iterrows()` reshaping each policy to dense 17,837 — three full copies (Arrow + pandas + deque) and `iterrows` is pathologically slow at 800k rows. `load_recent_games` (`storage/selfplay_store.py::load_recent_games`) loops this newest-first up to `replay_buffer_games`, so **resume rebuilds the entire buffer dense** — ~125 GB at 50k games. This is a latent landmine: a run that trained fine (sparse live buffer) OOMs the instant it is `--resume`d (dense reload).

**Fix.**
- `load`: read the sparse columns directly off the Arrow table (`table.column(...)`, no `to_pandas`/`iterrows`), rebuild `(board_compact, (indices, values), value)` tuples — the **same sparse form the live buffer holds**. Densification stays where it belongs: per-batch in `_LazyPolicyDataset.__getitem__` via `as_dense`.
- `load_recent_games`/`load_games`: unchanged logic, now carrying sparse policies → resumed buffer RAM equals live buffer RAM (~6–8 GB at 50k, not 125 GB).
- ~~Delete the dead `load_window`~~ — done by refactor R31.

**Depends on:** O1 (format).

---

## O3. Stream the parquet write in chunks

**Current state.** Even with sparse policies, `save` builds the whole generation as one `pd.DataFrame` → `pa.Table.from_pandas` (`storage/selfplay_store.py::save`) before writing — ~1.5–2 GB transient at 10k games.

**Fix.** Write incrementally with `pq.ParquetWriter`, appending row-groups of N positions (e.g. per-game or fixed 50k-row chunks) from a generator, so only one chunk is ever in RAM. Schema metadata (`board_kind`, `policy_kind`, `game_sizes`, `policy_size`) written on the first row-group. Defense-in-depth once O1 makes the data sparse.

**Depends on:** O1.

---

## O4. Sparsify the JAX harvester policy at append

**Current state.** `games/blokusduo/jax/harvest.py` (per-position dense append) scatters each position's top-K trace into a fresh dense `np.zeros(17837)` and appends it to the open game's `slot.policies` (`harvest.py::_OpenGame`), holding it dense until the game ends and only sparsifying in `_finish_game`. That's ~760 MB resident at `batch_size=256` and one `np.zeros(17837)` per ply×game (8,192/wave) of alloc/GC churn — a partial regression of the sparse-policy fix.

**Fix.** Keep the `(topk_ids, weights)` pair (already ≤64 nonzeros) at append time; densify only transiently inside `_finish_game` where entropy/transpose needs it. ~1 KB instead of 71 KB per open position, and the per-ply `np.zeros` churn disappears. Scales with `batch_size` (not `num_eps`), so this also unblocks raising batch size for throughput.

---

## O5. Cap JAX VRAM fraction on the shared 8 GB card

**Current state.** `XLA_PYTHON_CLIENT_PREALLOCATE=false` is correctly set (`games/blokusduo/jax/__init__.py` (XLA preallocate guard), `games/blokusduo/jax/__init__.py` (XLA preallocate guard)), but there is no `XLA_PYTHON_CLIENT_MEM_FRACTION` cap. JAX growing on demand and fragmenting against torch's caching allocator on one 8 GB card is a classic mixed-run OOM (and directly relevant to any future inference-GPU / training-GPU split).

**Fix.** Set an explicit `XLA_PYTHON_CLIENT_MEM_FRACTION` (start ~0.4) in the same gateways, before first `import jax`, so JAX and torch each get a bounded slice. Document the per-wave VRAM budget (≈ `batch_size × num_mcts_sims × top_k`) and keep `top_k = 64`. Does not scale with `num_eps` — raising games/gen stays safe; raising `batch_size`/`sims`/`top_k` is what to watch.

---

## O6. Stream completed games into the buffer

**Current state.** Self-play collects the **whole generation** in the main process before it reaches the buffer: `run_self_play_episodes_parallel` appends every game to `per_episode_examples` (`parallel/pool.py::run_self_play_episodes_parallel` (result accumulation)); the JAX backend builds `records` then `examples = [record.examples for ...]` (`games/blokusduo/jax/backend.py::generate_self_play_games` (wave loop)); the Coach then re-lists them (`selfplay/generate.py` (per-backend fresh_games accumulation)) and calls `replay_buffer.extend`. That's a ~1.5–2.5 GB sparse whole-generation copy coexisting with the buffer. Sparse, so not the trigger — but a real multi-GB spike that compounds with everything else.

**Fix.** Push each completed game into `replay_buffer` (and optionally persist it) as it arrives, rather than materializing the full generation then extending. Touches the self-play return contract (yield/callback per game instead of return-all), `Coach._run_self_play_*`, and pairs naturally with O3's streamed write. More invasive than O1–O5 — keep it a distinct commit; not required to relaunch safely at 10k once O1–O2 land.

---

## O7. Partition-filtered ArenaReplays reads

**Current state.** `reporting/training.py::_make_arena_replays_section` loads the **entire** ArenaReplays history (`_load_metrics` → `pd.read_parquet` over the whole dir, `reporting/training.py::_load_metrics`) then renders only `_REPLAY_MAX_GENERATIONS=16 × _REPLAY_MAX_GAMES_PER_GEN=6` games — ~430 MB–1 GB discarded. `scripts/replay.py` (whole-history read) reads the whole dir to print one game. (The report runs once end-of-run and is exception-wrapped, so this is not a run-killer — but it grows unbounded with generations.)

**Fix.** Compute the evenly-sampled generation list first, then read with parquet `filters=[("generation","in",sampled)]` (and push `game_idx < 6` into the read). `scripts/replay.py`: `filters=[("generation","=",gen)]` — hive partitioning makes it a single-file read.

---

## O8. Guardrails so this stops recurring

**Current state.** Every OOM so far was discovered only when the box crashed. There is no pre-flight check and no peak-memory visibility. This is the process gap behind "we keep hitting this."

**Fix (three cheap, independent guards):**
1. **Startup RAM-budget check** — at run start, estimate peak buffer + persistence RAM from config (`num_eps`, `replay_buffer_games`, positions/game, per-example bytes) and **refuse/loudly warn** if it exceeds a fraction (e.g. 0.8) of `psutil.virtual_memory().total`. Turns a 3 a.m. OOM into an instant config error.
2. **Peak-RSS logging at phase transitions** — extend the existing `_get_memory_snapshot` (`training/coach.py` (memory snapshot after self-play)) to log peak RSS after self-play, after save, and after train, to console + W&B. Makes the next spike visible in the run, not post-mortem.
3. **Regression test** — assert the save/load round-trip never densifies a whole generation: e.g. monkeypatch `as_dense` and assert it is not called during `save_self_play_history`, plus a sparse round-trip equality test. Locks in O1–O2.

Pairs with the separate box-hardening work (heartbeat alert, GRUB-default-Ubuntu, auto-power-on) tracked elsewhere.

---

## O9. Minor transient trims

Low priority; bundle into one commit.
- **Eval set** (`training/coach.py::__init__`, `games/base_wrapper.py::_compute_eval_set_diagnostics`): add an assert/comment pinning `_eval_set_size` — dense `(n, 17837)` is fine at 200 (~14 MB) but would OOM if scaled to the buffer. Optionally store the eval set sparse too.
- **MCTS** (`search/mcts.py::_simulate_batch`): `root_visit_counts` as `int32` (halves the per-move 143 KB); the per-move `.tolist()` / list-comprehensions at `:242,262-264` build full-action-space Python lists (~0.5 MB/move, ~99% zeros) — optional to trim, transient and bounded.
- **`predict_encoded`** (`games/base_wrapper.py::_compute_eval_set_diagnostics`): full-policy GPU→CPU copy is bounded by batch (~18 MB) — leave as-is, noted for completeness.

---

## Confirmed safe (checked, no action)

- **Training loop** — `_LazyPolicyDataset` + `train()` (`games/base_wrapper.py::_LazyPolicyDataset` / `train`): compact boards, densified one DataLoader batch at a time. This is the pattern O1–O3 make the disk path mirror.
- **MCTS tree** — sparse per-node arrays sized to legal moves, never dense 17,837 (`search/mcts.py::_Node` / `get_action_prob`).
- **Inference channel/server** — fixed-size shared buffers `(N,K,…)`, allocated once (`parallel/inference_channel.py::ChannelSpec` buffers, `parallel/inference_server.py::InferenceServer.serve_forever`).
- **JAX device trace** — compact int8 boards + top-K policy per wave, streamed to host and dropped; no `(num_eps,…)` or `(…,17837)` device array (`jaxplay/actors.py`, `jaxenv/search.py`).
- **HTML report** — reads only scalar aggregate parquets (one row per gen/epoch/batch), never the dense self-play store (`reporting/training.py`).
- **Per-game self-play** — dense `pi` (incl. symmetry) held for one in-flight game only, sparsified at game end (`selfplay/episode.py::play_self_play_episode`).
