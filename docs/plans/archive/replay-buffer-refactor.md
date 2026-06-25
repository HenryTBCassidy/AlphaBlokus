# Replay-buffer refactor

Replaces the generation-keyed training buffer with a single **rolling replay buffer sized in games**,
trained the way we already train — **full-pass epoch training over the whole buffer (use all the
data)** — plus a compact on-disk/in-RAM board representation. This is the pragmatic first step of
[IDEAS I4](../IDEAS.md#i4-continuous-non-gated-training) (continuous training) and it fixes three
problems at once:

> **Direction change (2026-06-25).** An earlier draft of R5/R6 introduced a `target_reuse` knob with
> `RandomSampler(replacement=True)` to *decouple* reuse from buffer size. We reversed that after
> discussion: in a **data-poor** regime we want to **use all our data each generation**, not sample a
> subset. The sampling machinery's only unique benefit (a large buffer trained at sub-`B/F` reuse) is
> not needed here, and its only evidence-backed reuse target (run2 ≈ 5, all data used) is what full-pass
> training over a game-sized buffer gives for free. So training stays as plain `epochs` full passes;
> reuse becomes the *emergent* quantity `epochs × B/F` (logged, not a knob). The worker built the
> sampling version faithfully from the older draft — R5/R6 below are the corrected target. R1–R4 are
> unaffected.

1. **The `max_generations_lookback` bug.** `_manage_training_window` pops at most one generation per
   call while self-play appends exactly one, so the window can never *shrink* below the value it
   reached by gen 5 (always 5). Setting `max_generations_lookback: 1` is silently a no-op — every
   scaled run so far trained on a rolling 5-generation window regardless. Deleting the
   generation-window machinery removes the bug by construction.
2. **The `max_generations_lookback` no-op meant reuse was uncontrollable downward.** Reuse is
   `epochs × window`: `run1_taper` ran at **2 × 7 = 14** (each MCTS target — from a weaker past net —
   replayed ~14×), while `run2_bignet` at **1 × 5 = 5** with 3× more fresh games/gen learned far faster
   and more smoothly (Elo 0.56→0.97 in 14 gens vs noisy 0.5→0.8 over 150). The over-reuse was a *config
   choice* (`epochs=2`, `lookback=7`); the bug only blocked going the other way (you couldn't shrink
   below a 5-gen window). Sizing the buffer honestly in **games** restores that control — set a small
   `B` and `epochs=1` for low reuse, or run2's `B=5000, epochs=1` for reuse 5.
3. **The training-step OOM.** The in-RAM buffer holds every board as a dense `(44,14,14)` float32
   array (~ 34.5 KB). `as_multi_channel(1)` is a pure function of the board's 196-byte int8
   `_piece_placement_board` (verified — it reads nothing else), so storing the **compact placement
   array** and re-encoding lazily cuts buffer RAM **~175×** (run2's ~12 GB buffer → ~70 MB). This is
   the "shrink the buffer itself" work explicitly deferred in
   [`lazy-board-encoding.md`](lazy-board-encoding.md) — this plan supersedes that deferral.

**In scope:** rolling game-sized buffer, compact board storage + lazy re-encode, sample-based
training with an explicit reuse dial, removal of the generation-window machinery.

**Out of scope (deliberately):** removing the arena/Elo accept-gate, and true async actor/learner
continuous training. Both are the wrong move for a game-limited single-GPU project (the GPU is never
contended today — self-play and training are temporally separate — and async would steal cycles from
the self-play that is the actual bottleneck). The gate stays as cheap insurance in this noisy
small-scale regime. See [IDEAS I4](../IDEAS.md#i4-continuous-non-gated-training) for the full case.

**Relationship to [`lazy-board-encoding.md`](lazy-board-encoding.md):** that plan's L1/L2 (per-item
densify of the *dense* buffer, no second copy) already landed and independently unblock resuming
`run2` at `lookback: 5`. Its L3 (resume run2) can proceed before or independently of this refactor —
they don't conflict. This refactor then replaces the dense buffer with the compact one, obsoleting
that plan's deferred "Out of scope" section.

---

## Config model and run dynamics

The new design exposes **two knobs** for the data regime (plus the familiar `epochs`/`batch_size`):

| Knob | Field | Controls |
|---|---|---|
| Buffer size **B** (games) | `replay_buffer_games` | How much history is held; sets **staleness** (oldest game is `B/F` gens old). |
| Fresh games / gen **F** | `num_eps` (existing name) | How fast the buffer turns over. |
| Epochs **E** | `epochs` (existing name) | Full passes over the buffer per generation. |

**Per-generation loop** (steady state): generate `F` fresh games with the current net → push into the
buffer (oldest `F` games auto-evicted by a game-keyed `deque(maxlen=B)`) → train **`E` full,
shuffled-minibatch passes over the whole buffer** — i.e. use all the data, exactly as the existing
`BaseNNetWrapper.train` already does. No sampling, no `train_steps` count.

**Reuse is emergent, not a knob.** Each position is trained on `E × (B/F)` times over its life
(it sits in the buffer for `B/F` generations and sees `E` passes per generation). We **log** this
effective reuse and the staleness (`B/F`) every generation so it's visible, but you set it indirectly
by choosing `B` and `E`:

- **Staleness** = `B/F` generations. e.g. `B=5000, F=1000` ⇒ nothing older than 5 net-versions.
- **Reuse** = `E × B/F`. e.g. `B=5000, F=1000, E=1` ⇒ reuse 5 (run2's regime); `B=2000, F=1000, E=1`
  ⇒ reuse 2.

**Warmup needs no special-casing.** The buffer starts empty; gen 1 generates `F` games, trains `E`
passes over whatever is present, and the buffer self-fills over the first `B/F` generations before it
begins sliding.

**`num_eps` is NOT auto-derived from `B`.** Forcing `F = B` would give every game a one-generation
life — no old/new mixing, which defeats the buffer's stability benefit. Keep `F < B`.

**Worked example** (run2's winning regime): `B=5000, F=1000, E=1` → buffer holds the last 5,000 games
(~5 generations), trains one full pass over all of them each generation → reuse ≈ 5, staleness ≤ 5
generations.

**Deferred to a fast follow-up (NOT in v1):** making `B` *grow* over training (KataGo-style: small
early when the net drifts fast and old games are genuinely stale, large late when the net has
converged and old games are nearly free and aid stability) and/or *ramping `F`* up over generations
for compute efficiency. Both are just turning `B`/`F` into functions of generation — trivial once the
fixed-value machinery below exists, and best tuned against a clean fixed-value baseline first. Note
the *direction*: standard practice **grows** the window over training — the freshness premium is
smallest exactly when the net has converged, because old data is then almost as good as fresh.

> **Why not sampling (the reversed R5).** Drawing a fixed number of mini-batches (`RandomSampler`)
> would let `B` and reuse move independently — a large buffer trained at *less* than `B/F` reuse. That
> only matters in a data-rich regime; here we're game-limited and want to use every game we generate.
> Sampling would also *skip* part of the buffer each generation (with replacement, ~`1−e^{−R/(B/F)}`
> of it), which is the opposite of "use all the data." It slots back in as a ~10-line sampler swap if a
> future data-rich run ever needs it.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| R1 | Add `IBoard.to_compact()` / `IGame.encode_compact()` seam; implement for Blokus (free `encode_planes_from_placement`) **and** TTT. Exact-equality test both games | 1.5-2 hr | High | ✅ |
| R2 | Store compact array in the in-RAM buffer (via `to_compact`); lazy re-encode in `_LazyPolicyDataset.__getitem__` (via `game.encode_compact`). self-play stops pre-encoding. Keep generation-window structure for now. Validate equivalence + measure RAM drop | 2-3 hr | High | ✅ |
| R3 | `SelfPlayStore` parquet stores compact boards (versioned schema metadata); loader reconstructs. Update resume path. Equivalence + round-trip test | 2 hr | High | ✅ |
| R4 | Replace generation-window machinery with a single rolling buffer sized in games (`replay_buffer_games`). Delete `_generation_window_size`/`_manage_training_window`/`max_generations_lookback`. Update save + resume | 3-4 hr | High | ✅ |
| R5 | **Full-pass epoch training over the whole buffer** (use all the data): keep `epochs`, `DataLoader(shuffle=True)` looped `epochs` times, scheduler `T_max = num_generations × epochs`. Log emergent reuse (`E × B/F`) + staleness. **Revert the `target_reuse`/`RandomSampler` version** | 1-1.5 hr | High | ✅ |
| R6 | Update run_configurations + docs (`02-ALGORITHMS`, `07-DATA-STORAGE`); report emergent reuse/buffer/staleness | 1-2 hr | Medium | ✅ |
| R7 | Parity/learning validation run on the box (small config), confirm RAM + learning curve; archive plan; flip IDEAS I4 → Promoted | 2 hr + run | High | ⏳ plan archived + IDEAS I4 promoted; box parity run pending (Henry's) |

Total: ~14-18 focused hours plus one validation run.

---

## Implementation notes (read before starting)

Cross-cutting gotchas a first-time reader of this codebase will hit. Each names the file:line and the
fix. The shared modules (`core/self_play.py`, `games/base_wrapper.py`, `core/coach.py`) serve **both**
TTT and Blokus — keep everything game-agnostic (that's what R1's interface seam is for).

**G1 — The buffer must hold per-*game* lists, not flat positions.** Today both
`Coach._run_self_play_serial` (`coach.py:415-417`) and `_run_self_play_parallel` (`coach.py:461-462`)
`iteration_examples.extend(...)` into a flat `deque(maxlen=max_queue_length)` (`coach.py:245`), which
**discards game boundaries**. Eviction "by game" (R4's `deque(maxlen=replay_buffer_games)`) needs those
boundaries. The parallel path already returns per-game lists (`per_episode_examples:
list[list[ProcessedExample]]`, `parallel_self_play.py:413, 505`); the serial path's
`execute_episode()` returns one game's list. Collect a `list[list[...]]` (one inner list per game) for
the generation and push each game in as a unit. The per-gen `max_queue_length` cap becomes meaningless
under the games-sized buffer — drop it (and its config field) rather than leave a dead knob.

**G2 — `nnet.train` stops being "one generation".** `coach.py:286` calls `train(train_examples, …)`
where `train_examples` was the concatenated window. Now pass the **whole buffer** (flattened to
positions) plus a new `fresh_positions_this_gen: int` (count of positions this generation's self-play
produced — known from G1's collection). `train` derives `train_steps` from it (R5). Also: the
data-validation asserts in `base_wrapper.train` (`base_wrapper.py:164-173`) check
`sample_board.shape == (channels, board_rows, board_cols)` — with compact storage `ex[0]` is now the
compact array (e.g. `(14,14)`), so that board-shape assert must move to *after* encode, or assert the
compact shape instead. The policy/value asserts are unchanged.

**G3 — The eval set must encode compact boards.** `_ensure_eval_set` (`coach.py:710`, see line 765)
builds `boards_all = np.array([ex[0] …])` and stores it as `EvalSet.boards`, which is fed **straight to
the network** (`base_wrapper.py:280-284` — `torch.tensor(boards_chunk)` → `self.nnet(tensor)`). So those
boards must be the *encoded* `(C,N,N)` planes, not compact. Fix: `np.array([game.encode_compact(ex[0])
for ex in train_examples])`. Relatedly, the TTT minimax path `_minimax_targets_for_eval_set`
(`coach.py:813-816`) currently decodes a position from the 2-channel encoding (`encoded[0]-encoded[1]`);
with compact storage it receives the 3×3 grid directly — decode from that instead (simpler).

**G4 — Resume path calls a method R4 deletes.** `load_self_play_history_for_resume` (`coach.py:392`)
and `load_self_play_history` (`coach.py:908`) compute a window via `_generation_window_size` (deleted in
R4). Replace with: load per-generation parquet files **newest-first**, accumulating games until
`replay_buffer_games` is reached, to reconstruct the buffer the next generation would hold. The
`progress.json` marker logic (`coach.py:369-390`) and `main.py`'s resume wiring stay as-is — only the
buffer-refill changes. Check `main.py` for the `--resume` call site.

**G5 — Scheduler `T_max` and step cadence.** `_create_scheduler` (`base_wrapper.py:115-124`) uses
`num_generations × epochs`; `scheduler.step()` is called per-epoch (`base_wrapper.py:214-215`). With
`epochs` gone, step **once per generation** (after the per-gen training loop) and set
`T_max = num_generations`.

**G6 — Config migration is low-risk (`dataclass_wizard.fromdict`).** `RunConfig` is a frozen dataclass
loaded via `dataclass_wizard.fromdict` (`config.py:6, 379`), which **ignores unknown JSON keys** by
default. So: add `replay_buffer_games: int = 5000` and `target_reuse: float = 2.0` **with defaults** —
existing JSONs keep loading through R1-R5. Remove `max_generations_lookback` (RunConfig, `config.py:133`)
and `epochs` (NetConfig, `config.py:68`) from the dataclasses whenever convenient; leftover keys in
not-yet-updated JSONs are harmless. Scrub the dead keys from `run_configurations/*.json` in R6. (R4/R5
should not *read* the doomed fields once the new ones exist.)

**G7 — Run the determinism test after R2.** `tests/test_core/test_parallel_self_play.py` pins
serial≡parallel training examples. Storing compact boards changes the stored array but both paths
identically, so it should stay green — but it compares `ex[0]` arrays, so run it explicitly once R2
lands to confirm the compact arrays match across paths.

**G8 — `save_self_play_history` saves "this gen's fresh games".** It currently reads
`self.train_examples_history[-1]` (`coach.py:895`). Under the flat buffer there's no "last generation"
slot — save the generation's fresh games tracked in G1, not `buffer[-1]`.

---

## R1. Compact-board interface seam (`to_compact` / `encode_compact`)

**Why this is an interface, not a Blokus function.** `core/self_play.py` and `games/base_wrapper.py`
are **shared by both games**. The compact-board change must go through the `IBoard`/`IGame` protocol —
hardcoding Blokus's `_piece_placement_board` in shared code breaks TicTacToe. `state_key` is already a
compact form but it's untyped `bytes` with no decode, so add a typed pair instead.

**Add to `core/interfaces.py`:**
- `IBoard.to_compact() -> NDArray` — the minimal canonical (player-1-perspective) int8 array that
  `as_multi_channel(1)` can be rebuilt from.
- `IGame.encode_compact(compact: NDArray) -> NDArray` — rebuilds the full `(C, N, N)` float32 planes
  from that array; must equal `board.as_multi_channel(1)`.

**Blokus** (`games/blokusduo/board.py:168` is the current encoder; it reads *only*
`self._piece_placement_board` + `current_player`):
- `to_compact()` returns the int8 14×14 `self._piece_placement_board`.
- Pull the encode body into a module-level free function and have `as_multi_channel(1)` and
  `BlokusDuoGame.encode_compact` both call it:

```python
def encode_planes_from_placement(ppb: NDArray) -> NDArray:
    """44-channel float32 encoding of a canonical (player-1) placement board.
    Identical to BlokusDuoBoard.as_multi_channel(1). ppb is the int8 (14,14) array.
    """
    rep = np.zeros((44, BlokusDuoBoard.N, BlokusDuoBoard.N), dtype=np.float32)
    for piece_id in range(1, 22):
        rep[piece_id - 1] = (ppb == piece_id)
        rep[21 + piece_id - 1] = (ppb == -piece_id)
    rep[42] = ppb > 0
    rep[43] = ppb < 0
    return rep
```

`as_multi_channel(self, current_player)` keeps its signature; for `current_player == 1` it delegates to
the free function, otherwise it multiplies `ppb` by `current_player` first (today's non-canonical path).

**TTT** (`games/tictactoe/board.py:38` — `as_multi_channel` is a pure function of `as_2d`): mirror
trivially. `to_compact()` returns `as_2d` (3×3 int8); `encode_compact(grid)` returns
`np.stack([grid == 1, grid == -1]).astype(np.float32)` (≡ `as_multi_channel(1)`).

**Validate.** For *both* games, over a spread of positions:
`game.encode_compact(board.to_compact())` is exactly `board.as_multi_channel(1)` (`np.array_equal`).
No behaviour change anywhere else yet.

---

## R2. Compact boards in the in-RAM buffer + lazy re-encode

**Current state.** `play_self_play_episode` (`core/self_play.py:103`) stores
`x[0].as_multi_channel(1)` — a dense 34.5 KB array per position. `_LazyPolicyDataset.__getitem__`
then materialises a tensor from that already-dense array (L1/L2 of `lazy-board-encoding.md` removed
the *second* copy but the buffer itself is still dense).

**Fix.** Store the **compact array** (`to_compact()`, game-agnostic) instead, and encode in
`__getitem__`:

- `play_self_play_episode` (`core/self_play.py:101-108`) returns `(x[0].to_compact(), sparse_pi, value)`
  — drop the `x[0].as_multi_channel(1)` call. (Symmetry-augmented boards are real board objects —
  `get_symmetries` returns `(BlokusDuoBoard, pi)` pairs, `game.py:197-199` — so `.to_compact()` works on
  both the identity and transposed entries; verify in the test.)
- `_LazyPolicyDataset` takes an `encode_fn` (= `game.encode_compact`, passed from the wrapper which
  holds `self.game`); `__getitem__` calls `encode_fn(self._boards[idx])` → `torch.from_numpy(...)`.
  One batch of boards is dense at a time, same as today. **Do not** import a Blokus symbol here.

Keep the list-of-deques-per-generation structure for this step — only the stored *representation*
changes. This alone takes the buffer from ~12 GB to ~70 MB and removes the OOM ceiling.

**Two things break the instant boards go compact — fix them in this same step:** the `train()`
board-shape assert (**G2**) and the eval-set board encoding in `_ensure_eval_set` (**G3**). Both
currently assume `ex[0]` is the `(C,N,N)` dense array.

**Validate.** `__getitem__` output identical to the pre-change dense path (same tensors).
Construct a run2-sized synthetic buffer and confirm RSS is ~175× smaller. Existing training/coach
tests green, and run the serial≡parallel determinism test (**G7**).

---

## R3. Compact boards in the parquet store

**Current state.** `SelfPlayStore.save` (`core/storage.py:1055`) writes the dense board bytes +
`board_shape=(44,14,14)`, `board_dtype=float32` metadata; `load` reconstructs the dense array.

**Fix.** Store the compact array bytes (`to_compact()` output — `(14,14)` int8 for Blokus, `(3,3)` for
TTT); record `board_kind=compact_v1` in the schema metadata alongside the existing (already-generic)
`board_shape`/`board_dtype` keys. `load` reconstructs the compact array into the buffer (the Dataset
re-encodes downstream via `game.encode_compact`). `load_window`/`load_self_play_history_for_resume`
need no logic change — they just carry compact arrays now.

A `board_kind` marker (absent ⇒ legacy dense) lets the loader refuse or migrate old files explicitly
rather than silently misreading 196 bytes as a `(44,14,14)` array. **Old run2 parquets are not
loadable into the new buffer** — acceptable (run2 is a one-off; resume it via the L3 path *before*
this lands if you want it salvaged). Note this in the loader.

**Validate.** Round-trip: save → load → `game.encode_compact(loaded)` equals the original
`as_multi_channel(1)`. Resume test reconstructs a buffer and trains a step.

---

## R4. Rolling game-sized buffer (delete the window machinery)

**Current state.** `Coach.train_examples_history: list[deque]`, one deque per generation;
`_generation_window_size` + `_manage_training_window` trim it by generation count — and the single-pop
throttle is the `max_generations_lookback` bug.

**Fix.** Replace with a single rolling buffer holding the last *N games* worth of positions:

- New config field `replay_buffer_games: int` (replaces `max_generations_lookback`). Sizing in games
  (not generations or positions) is the honest unit — it's invariant to `num_eps` and to
  positions-per-game variance, and matches how AlphaZero/MuZero/KataGo describe buffers.
- Internally hold a `deque` of per-*game* example-lists with `maxlen = replay_buffer_games`; oldest
  games auto-evict. Each generation extends it with the gen's fresh games (**G1** — preserve game
  boundaries through collection), then flattens to a position list for training.
- Delete `_generation_window_size` and `_manage_training_window`. The bug is gone by construction.
  Update the now-broken resume refill (**G4**) and `save_self_play_history` (**G8**) in the same step.
- `save_self_play_history` still writes one parquet per generation (fresh games only — track them
  separately before pushing into the buffer). Resume fills the buffer newest-first from recent
  generation files until `replay_buffer_games` is reached.

**Default.** `replay_buffer_games ≈ 5000` reproduces run2's healthy regime (5 gens × 1000 eps).
`num_eps` stays an independent knob (do **not** derive it from `B`).

**Validate.** Buffer length caps at `replay_buffer_games`; eviction is FIFO by game; resume
reconstructs the same buffer a fresh run would hold at that generation; warmup (buffer not yet full)
trains without special-casing.

---

## R5. Full-pass epoch training over the whole buffer

> **This section was reversed (2026-06-25).** The worker landed a `target_reuse`/`RandomSampler`
> version from the older draft; this is the corrected target — **use all the data via full-pass epoch
> training**, which is what `BaseNNetWrapper.train` already did before the refactor. The task here is
> largely to *un-do* the R5 sampling diff while keeping R1–R4.

**Target state.** `BaseNNetWrapper.train` runs `epochs` full, shuffled-minibatch passes over the
**whole buffer** — `DataLoader(dataset, batch_size, shuffle=True)` looped `epochs` times. Every
position in the buffer is trained on exactly `epochs` times per generation. No `RandomSampler`, no
`train_steps`, no `target_reuse`.

**Revert checklist:**
- Restore the `epochs` config field (on `NetConfig`); remove `target_reuse`.
- Replace the `RandomSampler(replacement=True, num_samples=…)` loader with the plain
  `DataLoader(..., shuffle=True)` epoch loop.
- Scheduler: step **per epoch** with `T_max = num_generations × epochs` (the original behaviour) —
  *not* once per generation.
- Keep the change that `train`'s `examples` argument is now the **whole buffer** (it always was the
  concatenated window — that's unchanged in spirit) (**G2** still applies for the compact board-shape
  assert).
- Keep logging, but log the **emergent** reuse `epochs × B/F` and staleness `B/F` (computed from
  config, not from a sampler). `fresh_positions_this_gen` is no longer needed for step-count math; drop
  that plumbing if it was only added for R5.

**Default.** `epochs = 1` with `B=5000, F=1000` ⇒ reuse 5 — run2's winning regime.

**Validate.** Each generation does exactly `epochs` full passes over the buffer (batch count ≈
`epochs × buffer_positions / batch_size`); loss curve sane on a short TTT run; logged reuse equals
`epochs × B/F`.

---

## R6. Configs, docs, reporting

- Update `run_configurations/*.json`: drop `max_generations_lookback`, add `replay_buffer_games`,
  **keep `epochs` and `num_eps`** (remove any `target_reuse` the R5 draft added). Refresh the blokus
  scaled/run configs to the healthy defaults (`replay_buffer_games=5000, num_eps=1000, epochs=1` ⇒
  reuse 5).
- `docs/02-ALGORITHMS.md` (training-data / replay section: the buffer model — `B`/`F`/`epochs`,
  staleness=`B/F`, emergent reuse=`epochs × B/F`; **not** a `target_reuse` sampling knob) and
  `docs/07-DATA-STORAGE.md` (parquet schema: `board_kind=compact_v1`, `game_sizes` metadata).
- Add emergent reuse (`epochs × B/F`), buffer fill, and staleness (`B/F` gens) to the HTML report and
  W&B so the data regime is visible at a glance.

---

## R7. Validation run, archive, promote

- Short parity run on the box (small net, e.g. the `linux_15` shape) at `epochs=1` with a buffer matching
  the old effective window: confirm it learns at least as well as the buggy-window equivalent, RSS stays
  flat, no OOM.
- Update the [full-cycle-optimisation baseline table](archive/full-cycle-optimisation.md) only if a
  throughput number changes (it shouldn't — this is a data-regime change, not a speed change).
- `git mv` this plan to `docs/plans/archive/`; add a Scope additions section if anything landed beyond
  the checklist.
- Flip [IDEAS I4](../IDEAS.md#i4-continuous-non-gated-training) status to **Promoted**, pointing at
  this plan (and the archived location once moved).

> **Status (2026-06-25).** R1–R6 landed and the full suite is green (an end-to-end TTT run confirmed
> epoch training, buffer fill/eviction, and the report render). Plan archived and IDEAS I4 promoted in
> the same commit. **The box parity run remains Henry's** — run it before relying on the new defaults.

---

## Scope additions

Work that landed beyond the literal checklist (captured here so the archived plan tells the full story):

- **`game_sizes` parquet metadata + game-aware loaders.** To reconstruct the *per-game* rolling buffer
  on resume (G4 implied per-game accumulation but didn't specify how boundaries survive the flat
  parquet), `SelfPlayStore.save` records each generation's per-game position counts, and new
  `load_games` / `load_recent_games` split the flat rows back into games and refill the buffer
  newest-first. Approved during review.
- **Dev-script fixups.** Four out-of-pipeline scripts referenced removed config fields; updated to the
  new schema (and to compact boards where they build synthetic training data):
  `scripts/benchmark_phases.py`, `scripts/benchmark_predict_batch.py`, `scripts/mcts_profiling.py`,
  `scripts/profile_self_play.py`.
- **`MetricsCollector.log_training_dynamics`** — a W&B-only per-generation publish of emergent reuse
  (`epochs × B/F`), staleness (`B/F`), and buffer fill; the HTML config table shows the governing knobs
  (`replay_buffer_games`, `epochs`, staleness, emergent reuse).
- **Direction change (2026-06-25).** R5 first landed as a `target_reuse` / `RandomSampler` sampling
  design, then was reverted to full-pass epoch training after the data-poor-regime discussion (see the
  banner at the top). R1–R4 were unaffected by the reversal.
- **New tests:** `tests/test_core/test_compact_encoding.py` (R1 seam exact-equality) and
  `tests/test_core/test_full_pass_training.py` (R5 batch-count = `epochs × ceil(buffer/batch)`).
