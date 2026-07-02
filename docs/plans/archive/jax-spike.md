# JAX De-Risk Spike — GPU-Native Legality, Step, and Throughput Ceiling

> **Completed 2026-07-02.** All rows ✅. Results and the go/no-go verdict live in
> [`docs/research/jax-spike-findings.md`](../../research/jax-spike-findings.md):
> parity exact everywhere; env solved (570k steps/s); pipeline purely inference-bound at
> ~3× production on the 3060 Ti — criterion 2 not met on current hardware; **conditional
> go** (JAX/mctx + Gumbel low-sim search + RTX 5070 Ti clears 10× multiplicatively).

Before committing to a full JAX/mctx rewrite of self-play, prove (or kill) the core hypothesis: **Blokus Duo legality and board transitions can be expressed as fixed-shape JAX array ops that are bit-identical to the Python engine and, batched on the RTX 3060 Ti, fast enough to project ≥10× self-play throughput.** The spike produces parity-validated kernels, throughput measurements, and a written go/no-go decision — no mctx integration, no training, no changes to the production pipeline.

Context: `docs/research/numba-hot-path-results.md` (the numba plan landed 2.33× single-worker per-sim but only **1.26× production** — 0.947 games/s / ~10.5k sims/s at 16 workers — because inference is now ~44% of the wall and the **GPU is the new bottleneck**; CPU micro-opt is exhausted), `docs/research/deepmind-run-configs.md` (we are data/net/sims-poor vs every reference config), and the run3 Pentobi benchmarks (best 25% at level 1; no visible gain gen 30 → gen 82). A GPU-native pipeline attacks both remaining walls at once: dense large-batch inference *and* zero CPU-side game logic. The full-rewrite plan is deliberately *not* written yet — it depends on this spike's numbers.

Prerequisites: none (independent of the training pipeline; run3 can keep running except during J5/J6 GPU benchmarks).

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| J1 | Add `jax` optional dependency group + `experiments/jax_spike/` skeleton + import probe test | 1 h | High | ✅ |
| J2 | Static table builder: derive `(17837, 196)` cover/edge/corner matrices + per-action metadata from `build_move_tables()` | 3 h | High | ✅ |
| J3 | JAX legality kernel `legal_mask(state, player)`; exact parity vs reference on all 5,000 dev-cache positions | 1 day | High | ✅ |
| J4 | JAX board step, game-end detection, and scoring; parity by replaying the dev-cache action sequences | 1 day | High | ✅ |
| J5 | Batched random-rollout throughput benchmark (Mac CPU + box GPU), batch-size sweep, HTML report | 0.5–1 day | High | ✅ |
| J6 | Dummy 128f×8b JAX net; mask+forward+sample "pseudo-self-play" step throughput on the box | 0.5–1 day | Medium | ✅ |
| J7 | Findings report `docs/research/jax-spike-findings.md` + go/no-go decision against the pre-registered criteria | 2 h | High | ✅ |

Total: ~4–5 working days. J1→J4 are strictly sequential; J5/J6 depend on J4; J6 can be dropped if J5 already settles the decision either way.

---

## Pre-registered go/no-go criteria

Written down *before* measuring, so the decision isn't post-hoc:

1. **Parity (hard requirement):** exact legal-mask equality with the reference generator on 5,000/5,000 dev-cache positions, and bit-identical replayed boards + game-end values. Any unexplained mismatch = no-go until explained.
2. **Throughput (go threshold):** projected self-play node-throughput ≥ **10×** the current pipeline's, with a 2× safety margin for mctx/tree overhead the spike doesn't measure. Concretely: J5/J6 measure `mask+step` steps/s and net forwards/s at large batch; projected sims/s = `1 / (1/steps_per_s + 1/forwards_per_s)`. The Python baseline is the fresh post-numba N6 measurement (`docs/research/numba-hot-path-results.md`): **~10.5k sims/s / 0.947 games/s at 16 workers** on the run2/run3 config. So the bar is ~105k projected sims/s, i.e. ~210k+ measured raw, given the 2× margin.
3. **No dynamic-shape escape hatches:** the kernels must be `jit`/`vmap`-clean (no host callbacks, no per-state Python). If correctness forces one, that's a finding against the approach — report it.

If go: write the full rewrite plan (env + mctx + net + replay/checkpoint plumbing) as a separate doc. If no-go: the numba path is already exhausted (archived), so the fallback is the RTX 5070 Ti (~1.5–2× under the current architecture now that inference is ~44% of the wall) plus incremental inference work (bf16, torch.compile).

---

## Design (shared by J2–J6)

**The whole rules engine reduces to three precomputed static matrices.** For each of the 13,729 geometrically-possible placements (already enumerated by `build_move_tables()` in `games/blokusduo/movegen_tables.py`), scatter its cell lists into rows of three `(action_size, 196)` int8 matrices indexed by **action id** (0–17,836; rows for the 4,107 never-legal action ids and pass stay all-zero and are excluded via a static `placeable` row mask):

- `COVER` — footprint cells (`MoveTables.cells`, ≤5 per row)
- `EDGE` — edge-adjacent halo (`MoveTables.adj_cells`)
- `CORNER` — diagonal, non-edge attach cells (`MoveTables.attach_cells`)

Plus static vectors: `piece_of_action` (int8, 1–21), `placeable` (bool), `piece_sizes` (int8[22]), and the two first-move start cells in flat array-index form via `CoordinateIndexDecoder.to_idx` — White (4,4) → cell 130, Black (9,9) → cell 65.

**State** (all fixed-shape, one struct-of-arrays per batched game):
`ppb` int8 (196,) signed placement board (same convention as `BlokusDuoBoard._piece_placement_board`), `remaining` bool (2, 21), `last_piece` int8 (2,), `current_player` int8, `done` bool. Note the JAX state deliberately does **not** carry placement points or side-danger — both are consequences of `ppb` that the matmul formulation recomputes implicitly.

**Legality** for player `p` with `own = (sign(ppb) == p)`, `occ = (ppb != 0)` (float or int8 vectors of length 196):

```python
overlap_free = COVER @ occ == 0
no_edge      = EDGE  @ own == 0
corner_ok    = CORNER @ own > 0
first_move   = remaining[p].all()                      # scalar
corner_ok    = jnp.where(first_move, COVER[:, start_cell[p]] > 0, corner_ok)
available    = remaining[p][piece_of_action - 1]
mask         = placeable & available & overlap_free & no_edge & corner_ok
mask         = mask.at[PASS_IDX].set(~mask.any())
```

Three `(17837, 196) @ (196,)` int8 matvecs per state (~10M MACs), `vmap`-ped over thousands of states. Semantics to match exactly: pass (17,836) is legal iff no placement is; opponent edge contact is allowed; the first-move rule replaces the corner rule (footprint must cover the player's start cell).

**Step:** `ppb += p * piece_of_action[a] * COVER[a]`; clear `remaining[p, piece]`; set `last_piece[p]`; flip player. Pass leaves the board unchanged (mirrors `get_next_state`, `games/blokusduo/game.py:112-113`).

**Game end / scoring** (mirrors `game.py:139-162, 413-432`): terminal when neither player has any legal placement (two mask reductions); score per player = `-Σ piece_sizes[remaining]`, `+15` if all placed, `+5` more if the monomino was placed last; result `±1` / `1e-4` draw.

**Correctness anchors:** building the matrices *from* `MoveTables` (rather than re-deriving geometry) means the JAX kernel inherits F2's enumeration, and any parity failure isolates to the rule-condition translation, not the geometry. The one rule encoded differently is corner-contact: F2 checks it via incrementally-maintained placement points (`board.py:455-517`); the matmul form checks the same condition directly from the footprint's diagonal halo. The 5,000-position equivalence run is exactly the test of that translation.

---

## J1. Dependency group + skeleton

Add to `pyproject.toml` `[project.optional-dependencies]`: `jax = ["jax>=0.4.35"]` (CPU wheel on the Mac; on the box install with `uv sync --extra jax` plus the CUDA variant `"jax[cuda12]"` — pin whatever resolves against the box's driver; record the exact versions in J7). Create `experiments/jax_spike/` (`__init__.py`, `tables.py`, `kernels.py`, `benchmark.py`) and `tests/test_jax_spike/` with `pytest.importorskip("jax")` at module level so the suite stays green without the extra. Spike code still follows `STYLE-GUIDE.md` (typed, documented) — it's throwaway-*scoped*, not throwaway-*quality*, since J2/J3 kernels are the seed of the real rewrite.

## J2. Static table builder (`experiments/jax_spike/tables.py`)

`build_jax_tables() -> JaxTables` (frozen dataclass of numpy arrays; conversion to `jnp` happens at kernel setup). Source everything from `build_move_tables()`: scatter `cells`/`adj_cells`/`attach_cells` (NULL_CELL-padded, `movegen_tables.py:138-182`) into the three dense matrices using `action_id`, ignoring NULL_CELL (196) entries; derive `piece_of_action` from `MoveTables.piece`, `placeable` from `action_to_move_id >= 0`, `piece_sizes` from the piece manager.

Tests (`tests/test_jax_spike/test_tables.py`): `placeable.sum() == 13_729`; row sums match `n_cells`/`n_adj`/`n_attach`; `COVER` rows disjoint from `EDGE`/`CORNER` rows; spot-check one hand-computed placement per rule (e.g. monomino at a corner); start cells equal `CoordinateIndexDecoder.to_idx((4,4))`/`to_idx((9,9))` flattened.

## J3. Legality kernel + parity (`experiments/jax_spike/kernels.py`)

Implement `legal_mask` as in the design, `jit`-compiled, plus `legal_mask_batch = vmap(legal_mask)`. Include a `has_any_move(state, player)` reduction (needed by game-end and cheaper than a full mask — a single `any` over the fused conditions).

Parity test (`tests/test_jax_spike/test_parity.py`), mirroring `tests/test_blokusduo/test_movegen_equivalence.py`: for each of the 5,000 cached positions (`tests/fixtures/blokus_positions.py::iter_cached_positions`), rebuild the Python board, convert to JAX state via `board.to_compact()` + remaining/last-piece fields, and assert `np.array_equal(jax_mask, reference_mask)` — exact, all 5,000, including empty-board (first-move) and late-game (pass) strata. Also run the F2 generator as a second oracle (three-way agreement). Mark the full run `slow` if needed, but it must run in CI-equivalent local `uv run pytest -m "not slow"` on at least a 500-position subsample.

## J4. Step, game-end, scoring + replay parity

`step(state, action) -> state` and `game_result(state) -> (done, value)` per the design, `jit`/`vmap`-clean (pure `jnp.where` branching — no Python conditionals on traced values). Parity test: replay every dev-cache action sequence move-by-move through `step`, asserting after each ply that `ppb` equals the Python board's `to_compact()`, remaining-piece sets match, and `game_result` equals `get_game_ended` (including the `1e-4` draw convention and both bonus rules). This doubles as the pass-semantics test (sequences include forced passes in the late/end strata).

## J5. Random-rollout throughput benchmark

`experiments/jax_spike/benchmark.py`, invoked `uv run python -m experiments.jax_spike.benchmark --device {cpu,gpu} --out temp/benchmarks/jax_spike_{sha}.html`, following the `scripts/benchmark.py` conventions (HTML with inlined matplotlib via the `_fig_b64` pattern; header records commit, device, jax version).

Measure, with proper warm-up (discard the first jitted call) and `block_until_ready()`:

1. `legal_mask_batch` alone — masks/s vs batch ∈ {64, 256, 1024, 4096, 8192}.
2. Full random self-play: `lax.fori_loop` over a fixed 50-ply horizon of mask → categorical sample (fixed PRNG key handling) → step, `vmap`-ped; report board-steps/s and completed games/s.
3. Python baseline for the ratio: time the F2/numba `valid_move_mask` per-call on the same positions (`scripts/benchmark_movegen.py`), and take the production sims/s baseline from the post-numba N6 measurement (`docs/research/numba-hot-path-results.md`, ~10.5k sims/s at 16 workers) rather than re-running it.

Run on the Mac (CPU) for sanity, then on the box GPU. **Constraint: run3 owns the GPU** — either wait for it to finish or accept a marked-as-dirty contended number first and re-measure clean after; the report must say which. Record VRAM for the table set + state batch (expected trivial: the three matrices are ~10 MB as int8).

## J6. Dummy-net pseudo-self-play step (optional if J5 is decisive)

Hand-rolled pure-JAX (flax not required) ResNet matching run3's shape: 44→128 conv trunk, 8 residual blocks, conv policy head to 17,837 logits + value head. Random weights — this measures *throughput*, not play. Benchmark on the box: (a) standalone forward/s vs batch; (b) fused loop step = `legal_mask` → forward → masked categorical sample → `step`, i.e. the inner loop of a batched self-play actor minus tree search. Report projected sims/s per the go/no-go formula, and fp32 vs bf16/fp16 variants.

## J7. Findings report + decision

`docs/research/jax-spike-findings.md` (+ index row in `docs/research/README.md`): environment/versions, parity outcome, all throughput tables, projected sims/s vs the ~25–30k/s Python baseline, the go/no-go verdict against the pre-registered criteria, and — if go — the scope sketch for the full rewrite plan (env API, mctx policy choice, net port/checkpoint conversion, eval parity via the existing Pentobi GTP harness staying CPU-side). Archive this plan per `PLAN-FORMAT.md` lifecycle.

---

## Risks / open questions

- **Corner-rule translation** is the main parity risk (F2 reaches it via placement points; we check the diagonal halo directly). If parity fails, diff the disagreeing positions' anchor sets first — the dev cache's stratification makes small repro cases easy.
- **First-move semantics**: F2 defers first moves to the reference generator (`movegen_runtime.py:138-143`); the JAX path must reproduce that from the start-cell condition alone. Covered by the empty-board stratum (5% of the cache).
- **Sampling inside `jit`** (J5/J6): masked categorical over 17,837 logits is standard (`jnp.where(mask, logits, -inf)`), but PRNG key threading through `vmap`+`fori_loop` is a known footgun — budget an hour, not a day.
- **GPU contention with run3** — see J5. Clean numbers matter; this is the whole point of the spike.
- **What the spike deliberately does not answer**: mctx tree overhead, replay/training pipeline throughput, and multi-GPU. The 2× margin in the go threshold is the hedge on the first; the rewrite plan owns the rest.
