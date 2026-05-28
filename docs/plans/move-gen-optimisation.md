# Move-gen optimisation (F2 of full-cycle-optimisation)

Sub-plan for the second optimisation in [`full-cycle-optimisation.md`](full-cycle-optimisation.md). Replaces the current array-based legal-move generation in `games/blokusduo/board.py` with a Pentobi-style precomputed-move-list approach: at runtime, looking up a short candidate list keyed by `(anchor cell, 6-bit adj_status, piece)` and verifying each candidate with a 5-cell unrolled byte check.

Expected wall-clock impact at production net + 4 workers: **~1.5× on per-game cost.** That speedup is bounded by Amdahl — move generation is ~43% of MCTS search time on the current implementation, the rest is GPU inference + tree bookkeeping which F2 doesn't touch.

Companion docs:
- [`docs/research/bitboards.md`](../research/bitboards.md) — generic bitboard concept (background only; we are *not* using bitboards here).
- [`docs/research/pentobi/move-generation.md`](../research/pentobi/move-generation.md) — Pentobi's actual move-gen design, citation by citation. **Required reading** before implementing anything in this plan.
- [`docs/research/pentobi/python-port-gotchas.md`](../research/pentobi/python-port-gotchas.md) — C++ → Python translation notes, where Pentobi's tricks don't carry over and what to do instead.
- [`docs/research/pentobi/architecture.md`](../research/pentobi/architecture.md) — Pentobi's code layout, for context.
- [`docs/plans/full-cycle-optimisation.md`](full-cycle-optimisation.md) — master plan and the F2 row this sub-plan implements.

---

## Design decision: Pentobi-style precomputed move-list table

We had two candidate designs going in. This plan locks in **Option 1**.

### Option 1 (chosen): precomputed move-list table

- Number every legal placement at startup with a unique move ID, 0..13,728 for Blokus Duo.
- For each move ID, store the list of cells it occupies, the cells it would mark as forbidden, and the cells it would mark as new attach points. All as fixed-size arrays of cell indices.
- Build a 4D lookup table `(anchor=0..195, adj_status=0..63, piece=0..20) → list of move IDs` that pre-filters candidates by the locality of forbidden cells around the anchor.
- At runtime: byte arrays for `forbidden[c][cell]` and `is_attach_point[c][cell]`. Move generation iterates attach points, computes `adj_status`, looks up the candidate list, then verifies each candidate by 5 byte reads + an OR.

### Option 2 (rejected): true bitboard implementation

- Pack the 14×14 board into ~196 bits per state plane. Legality via AND of precomputed footprint/contact/corner masks.
- Speculative for Blokus — no published implementation we could find.

### Why Option 1

1. **Pentobi works**, and uses exactly this design. Strongest open-source Blokus engine. Their C++ runs an order of magnitude faster than any naive implementation while never touching a bitwise trick. If they evaluated bitboards and rejected them, that's strong evidence.
2. **Lower implementation risk in Python.** The per-move check is "5 byte reads and an OR." No 196-bit integer ops, no hand-rolled bit-ordering decisions, no symmetry-via-bit-shuffle. Each sub-component is independently testable against a slow reference.
3. **Easier equivalence tests.** Easy to compare new-implementation `valid_moves()` against current array-based `valid_moves()` on thousands of random positions. Bitboards would add an extra layer (bitboard ↔ array conversion) to every test, multiplying the failure surface.
4. **No license risk.** Pentobi is GPL-3.0. We're implementing the same *algorithm*, in our own words, in Python. Algorithms aren't copyrightable; verbatim code is. We have detailed research notes (with file:line citations) but no copy-pasted source.

### When to revisit

If after F2 lands the move-gen slice still dominates the wall-clock (i.e. it didn't shrink as expected), the bitboard variant becomes worth measuring. Open a new sub-plan at that point — don't try to retrofit bitboards into this F2 implementation mid-flight.

---

## Why this is the right thing to do next

Three reasons:

1. **Move-gen is the largest remaining CPU-bound slice.** Post-F1 the production-net cost profile is ~50% GPU inference + ~43% Python move-gen + ~7% other. F1 made the move-gen slice parallelise across cores but didn't make it cheaper per game. F2 makes it cheaper per game, which compounds.
2. **F4 (batched inference) and F2 are independent.** They attack different slices. We can land F2 without making F4's work harder; in fact, by shrinking the move-gen share, F4's later inference improvements become a larger fraction of the remaining wall-clock.
3. **The infrastructure is ready.** F1 brought reliable parallel-execution scaffolding and the `benchmark_phases.py` script gives clean before/after measurements. F2 inherits all of that.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| P0 | Establish baseline. Re-run `scripts/run_benchmark.sh --config run_configurations/profile_baseline.json --num-workers 4` to confirm the post-F1 numbers haven't drifted, then record component split (move-gen %, inference %, other %) | 30 min | High | ✓ (recent benchmark, no perf changes since) |
| P1 | **Verify the move count agrees with Pentobi.** Write `scripts/count_onboard_placements.py` that enumerates every `(cell, piece-orientation)` pair from our 17,836-action grid, computes the cells the piece would occupy at each, and counts those that fit entirely on the 14×14 board. Compare against Pentobi's hard-coded constant of **13,729** for Duo (`Move.h:32`). If the numbers disagree, **stop and investigate** — see [P1 detail](#p1-verify-the-move-count) | 1 hr | **Highest** | ✓ (13,729 ↔ 13,729 exact match) |
| P2 | Build random-position generator + position cache. Stratified across early/mid/late game (more weight on early/mid since they have the most legal moves and hardest invariants). Cache positions to disk so subsequent test runs are fast | 1.5 hr | High | ✓ (`dev_5000.npz` built, distribution verified) |
| P3 | Write the equivalence test scaffolding. Compares `valid_moves()` set membership between current implementation and (eventually) the new one across the generated positions. Tests must run **before** the new impl exists — they'll initially compare current vs current (a no-op pass) and become real when P6 lands | 1 hr | High | ✓ (4 tests passing, 1 gauntlet skip pending P8) |
| P4 | Build the geometry tables (the static `MoveInfo` equivalent). One numpy array per piece encoding cell offsets per orientation. Function `compute_all_placements()` enumerates every `(piece, orientation, anchor)` triple, drops off-board ones, assigns sequential move IDs 0..13,728. Verify count matches the number P1 produced (which should also match Pentobi's `Move.h` constant) | 2-3 hr | High | ✓ (13,729 moves, 80ms build, 686 KB) |
| P5 | Build the (anchor, adj_status, piece) lookup table. For each `(anchor=0..195, adj_status=0..63, piece=0..20)` cell, precompute the list of move IDs from the geometry table that satisfy "uses this piece, includes anchor, and doesn't occupy any of the forbidden-marked neighbour cells". Store as `begin`/`size` numpy arrays + a flat move-ids buffer | 3-4 hr | High | ✓ (indexed by every footprint cell, not just bounding-box top-left — required runtime dedup. See [P5 detail](#p5-the-lookup-table)) |
| P6 | Implement the runtime move generator. New class `BlokusDuoBoardOptimised` (or method on the existing board) that uses the precomputed tables. Maintains `forbidden[c]`, `is_attach_point[c]`, `pieces_remaining[c]` incrementally. `valid_moves()` iterates attach points → computes adj_status → looks up candidates → byte-verifies. Initially shadows the existing implementation behind a config flag | 4-6 hr | High | ✓ (`movegen_runtime.F2MoveGenerator`, equivalence test passes on 5,000 stratified positions) |
| P7 | Cache the precomputed tables to disk (`games/blokusduo/precomputed/move_tables.npz` or similar). Workers in `core/parallel_self_play.py` load from disk rather than rebuild, so spawn-based pool start stays sub-second | 1 hr | High | **Deferred** — 200ms build is already fast enough. Revisit if profiling shows worker spawn dominated by it. |
| P8 | Switch over. Once the equivalence test passes on the **full gauntlet of 50,000+ stratified positions** (see P2), flip the default to the new implementation. Old code stays available behind a config flag for one generation of runs, then removed | 30 min | Medium | **Deferred** — 5K dev cache passes. Build the 50K gauntlet + flip the default before any real training. |
| P9 | Measurement. Two parts: **(a)** wall-clock benchmark with the new impl using `scripts/run_benchmark.sh --num-workers 4 --output-dir temp/profile_baseline_f2_w4_<date>` — capture before/after numbers and component split. **(b)** Training-determinism check: short self-play run with old impl + same with new impl at the same seed; assert the produced training-example arrays are bit-identical. Catches RNG-state drift that the position-level equivalence test (P3) can't see | 1-2 hr | High | ✓ **9.06× per-call speedup** via `scripts/benchmark_movegen.py`. Determinism check: 5 seeds × 30 moves each, F2 produces bit-identical trajectories to current impl. Full-pipeline benchmark not yet run on PC. |
| P10 | Archive plan. `git mv` to `docs/plans/archive/`. Append a Scope additions section if anything landed beyond this checklist. Update master plan F2 row to **Done** ✓ | 5 min | Low | |

Total: ~15-19 focused hours.

---

## P1. Verify the move count

Before sinking days into precomputing 13,729-element tables, **independently verify that our action-space encoding agrees with Pentobi's**. The number 13,729 comes from Pentobi's `Move.h:32` and is the count of legal `(piece, orientation, anchor)` placements that fit entirely on the 14×14 board. If our action-space encoding produces a different on-board count, either:

- our `pieces.json` definitions differ from Pentobi's,
- our "anchor cell" convention for each piece-orientation differs (Pentobi uses the lowest-leftmost cell of the piece's bounding box; we might use something else), or
- Pentobi's number we read is for a different variant or includes/excludes pass differently.

Any of those mean the F2 plan's tables would be sized wrong from the start. Catch it at the cheapest possible point — before any other F2 work.

Concretely, write `scripts/count_onboard_placements.py`:

```python
# Pseudo-code outline
def count_onboard_placements() -> int:
    """For our 17,836-element placement-action space (14×14 × 91), return
    the count that fit entirely on the 14×14 board (i.e. exclude
    placements where any cell of the piece would lie off-board)."""
    game = BlokusDuoGame(...)
    onboard = 0
    for action in range(0, 17_836):  # excludes the pass action
        piece, orientation, anchor = game.action_codec.decode(action)
        cells = game.compute_cells(piece, orientation, anchor)  # returns list of (row, col)
        if all(0 <= r < 14 and 0 <= c < 14 for r, c in cells):
            onboard += 1
    print(f"On-board placements: {onboard}")
    print(f"Pentobi's count for Duo: 13,729 (from Move.h:32)")
    assert onboard == 13_729, f"MISMATCH: got {onboard}, expected 13,729"
```

Expected runtime: seconds (it's 17,836 cheap geometry checks). Output: a single number printed, plus an assertion failure if it disagrees.

If the assertion fails, **stop**. Don't proceed to P2-P10. Investigate one of:
- Compare a handful of `(piece, orientation, anchor)` triples between our `pieces.json` and Pentobi's `libpentobi_base/PieceTransforms*.cpp` to see where geometry disagrees.
- Try re-running with a different "anchor cell" interpretation (we might be using top-left of bounding box where Pentobi uses something else).
- Worst case: dump both sets of `(piece, orientation, anchor) → cells` mappings to a CSV and diff them.

The script lives forever in `scripts/` as a regression check — running it after any change to `pieces.json` confirms nothing drifted.

---

## P2. Random position generator

The single most important test infrastructure piece. Must produce realistic mid-game `BlokusDuoBoard` states efficiently, **stratified across game phases**.

### Why stratify

Late-game positions have few legal moves (often 1-3 per player) and tend to be easy to get right — the constraints are so tight that "any legal move" is roughly the same as "the legal move." **Early- and mid-game positions have hundreds of legal moves each**, with subtle constraints from the corner-touch rule and edge-adjacency rule. Bugs in `valid_moves()` are most likely to show up there. So we deliberately oversample the regime where the test is hardest to pass.

### Suggested distribution

For Duo (~28 moves per full game), stratify by *moves played so far*:

| Phase | Moves played | Fraction of positions | Reason |
|---|---|---|---|
| Empty | 0 | 5% | Tests the starting-cell rule |
| Early | 1-8 | 35% | Lots of attach points, lots of legal moves |
| Mid | 9-18 | 40% | The most active region with the most subtle constraints |
| Late | 19-26 | 15% | Few moves but the edge cases (forced passes, near-end-of-game) live here |
| End | 27+ | 5% | Game-end + force-pass conditions |

### Generator interface

```python
def random_positions(
    n: int, *, seed: int = 42,
) -> Iterator[BlokusDuoBoard]:
    """Yield N random Blokus Duo positions, stratified by game phase
    according to PHASE_DISTRIBUTION (see above).

    Each position is reached by initialising a fresh board and playing
    `target_moves` random legal moves, where `target_moves` is sampled
    from PHASE_DISTRIBUTION. The 'random legal move' sampling uses the
    *current* valid_moves() implementation — so the new implementation
    is tested on positions IT didn't pick.

    Seeded RNG: same `seed` produces the same set of positions.
    """
    rng = np.random.default_rng(seed)
    game = ...
    for _ in range(n):
        phase = rng.choice(PHASES, p=PHASE_WEIGHTS)
        target_moves = rng.integers(phase.min_moves, phase.max_moves + 1)
        board = game.initialise_board()
        for _ in range(target_moves):
            valids = board.valid_moves()
            if not valids.any():
                break  # game ended early; yield current state anyway
            action = rng.choice(np.where(valids)[0])
            board = board.with_move(action)
        yield board
```

### How many positions?

Two tiers:

- **During development** (fast iteration loop, run on every commit / change): **5,000 positions**. Catches the obvious bugs in seconds-to-minutes.
- **Before P8 switch-over** (one-time gauntlet): **50,000 positions**. The "we're really confident" pass before flipping the default. Maybe 10-30 minutes of CPU.

Anything fewer than 5,000 during dev risks bugs hiding in the long tail of unusual positions. Anything more than 50,000 has diminishing returns — at that point bugs that haven't been hit will likely manifest as obvious failures in real training runs.

### Caching to disk

Generating 50,000 positions takes meaningfully longer than just running the test against them (each position needs the slow `valid_moves` 0-26 times to *get there*). So generate once, cache to disk, reuse on every test run:

```
tests/fixtures/blokus_duo_positions/
├── dev_5000.npz       # 5,000 positions, seed=42, stratified
├── gauntlet_50000.npz # 50,000 positions, seed=42, stratified
```

Each `.npz` stores a 2D `state_key` array (one row per position; each row is the 196-byte int8 representation). Loading + reconstructing a board from its state_key is fast. Regeneration only needed if the stratification weights or the seed changes (in which case bump the filename: `dev_5000_v2.npz`).

This makes the equivalence test fast enough to run in the regular `pytest` loop with a `pytest.mark.slow` marker on the 50,000-gauntlet test.

---

## P3. Equivalence test scaffolding

```python
def test_valid_moves_matches_reference(positions):
    """For each position, the new and old implementations must agree
    on the *set* of legal moves. We don't care about ordering."""
    old_board = positions  # the existing BlokusDuoBoard
    new_board = OptimisedBlokusDuoBoard.from_legacy(old_board)

    old_valids = set(np.where(old_board.valid_moves())[0].tolist())
    new_valids = set(np.where(new_board.valid_moves())[0].tolist())

    assert old_valids == new_valids, (
        f"Mismatch on position {old_board.state_key.hex()[:16]}...: "
        f"only in old: {old_valids - new_valids}, "
        f"only in new: {new_valids - old_valids}."
    )
```

Run this test against **10,000 positions minimum** before declaring the new implementation correct. The cost is a few minutes of CPU time, dwarfed by the cost of finding a bug in production.

Edge cases the position generator should cover:
- Empty board (only the diagonal-start move is legal).
- First move played, both colours about to make second move.
- Positions where one colour has no legal moves but the game continues.
- Positions where many pieces are exhausted (only the small ones left).
- Symmetric positions (test both serial and parallel symmetries don't drift).

---

## P4. Geometry tables (static placement data)

The static "every legal placement" enumeration. Inputs:
- The piece definitions from `games/blokusduo/pieces.json` (21 pieces, each with shape).
- The board geometry (14×14, cell indexing).

Output: numpy arrays sized by total move count (13,729 for Duo, verified against Pentobi's `Move.h:32`):

```python
# Per-move static data, all numpy arrays indexed by move_id 0..13,728:
move_piece:           np.uint8[13_729]            # piece id, 0..20
move_anchor:          np.uint8[13_729]            # the anchor cell index, 0..195
move_cells:           np.uint8[13_729, 5]         # the cells the piece occupies; padded with NULL_CELL=255
move_n_cells:         np.uint8[13_729]            # how many cells are actually used (≤5)
move_adj_cells:       np.uint8[13_729, 16]        # edge-adjacent cells (would become forbidden); NULL_CELL padded
move_n_adj:           np.uint8[13_729]            # how many adj cells used (≤16)
move_attach_cells:    np.uint8[13_729, 16]        # diagonal-adjacent cells (would become attach points)
move_n_attach:        np.uint8[13_729]            # how many attach cells used
```

The `NULL_CELL = 255` (or similar) sentinel is the Python equivalent of Pentobi's `Point::null()` — see `docs/research/pentobi/move-generation.md` §2. Sized one past the last valid cell index (196 valid, so 197 array slots for `forbidden`, with index 196 = the sentinel that's always `False`).

Build procedure:
1. For each piece (0..20), enumerate its unique orientations (the 91 total for Blokus).
2. For each orientation, enumerate every anchor cell (0..195) the piece could be anchored at.
3. For each `(piece, orientation, anchor)`, compute the piece's footprint cells. Drop the placement if any cell goes off-board.
4. Compute the adj and attach cell lists from the footprint.
5. Number the surviving placements 0..N-1 sequentially.
6. Assert `N == 13_729` (the Pentobi constant for Duo).

Verification beyond "count matches Pentobi":
- For a 1-cell piece (the smallest), exhaustively check `cells == [anchor]`.
- For each piece, manually verify one orientation's cell offsets against the pieces.json definition.

---

## P5. The lookup table

This is the headline data structure — Pentobi's `PrecompMoves::m_moves_range` equivalent.

For every `(anchor=0..195, adj_status=0..63, piece=0..20)` combination, precompute the list of move IDs that:
1. Use this `piece`,
2. Are anchored at this `anchor`,
3. Do *not* occupy any cell flagged as forbidden by `adj_status`. The `adj_status` is the 6-bit pack of "is this neighbour cell forbidden?" for the 6 specific cells around the anchor — see [`pentobi/move-generation.md`](../research/pentobi/move-generation.md) §4f for the diagram of which 6 cells and in what bit order.

Storage: two numpy arrays:

```python
# Range table: where each (anchor, adj_status, piece) candidate list
# lives in the flat move_ids buffer. 4 bytes per entry, ~2 MB total.
table_begin:  np.uint32[196, 64, 21]
table_size:   np.uint16[196, 64, 21]

# Flat backing store of move IDs. ~5 MB at uint16.
move_ids:     np.uint16[~2_600_000]   # exact size determined at build time
```

To get the candidate list for `(anchor, adj_status, piece)`:
```python
begin = table_begin[anchor, adj_status, piece]
size  = table_size[anchor, adj_status, piece]
candidates = move_ids[begin : begin + size]
```

Build algorithm:
1. For each `(anchor, piece)` pair, collect all move IDs in the geometry table matching that pair. Sort them in some stable order.
2. For each move ID, determine which `adj_status` bits **would prevent** that move — i.e. which of the 6 adj_status cells the move's cells/adj_cells overlap with. Call this `incompat_mask` per move ID.
3. For each `(anchor, adj_status, piece)`: the candidate list is the subset of move IDs from step 1 such that `(incompat_mask & adj_status) == 0`.
4. Pack: assign each list a contiguous slice in the flat `move_ids` buffer; fill `table_begin`/`table_size` accordingly.

Steps 1–4 are pure numpy / Python at build time. **Expected build wall-clock**: under a minute even unoptimised. If it takes longer, vectorise.

Storage check: ~196 × 64 × 21 = ~260K entries in the range table, ~2.6M total move IDs in the flat buffer. Both fit comfortably in memory.

---

## P6. Runtime move generator

The hot-path implementation. Structure:

```python
class OptimisedBlokusDuoBoard:
    """Move-gen-optimised Blokus Duo board. Same public API as
    BlokusDuoBoard (valid_moves, with_move, current_player, etc) so it
    can drop in behind the same protocol."""

    # Shared static tables (built once, cached to disk):
    _GEOMETRY: GeometryTables           # the P4 output
    _LOOKUP: LookupTable                # the P5 output

    # Per-instance dynamic state:
    forbidden: np.ndarray   # shape (2, 197), uint8 0 or 1, last slot is the NULL sentinel
    is_attach_point: np.ndarray  # shape (2, 196), uint8
    pieces_remaining: np.ndarray  # shape (2, 21), uint8

    def valid_moves(self) -> np.ndarray:
        """Return a bool array of shape (action_size,) — True for each
        legal action."""
        out = np.zeros(self.ACTION_SIZE, dtype=bool)
        c = self.current_player_index  # 0 or 1
        forbidden_c = self.forbidden[c]
        attach_c = self.is_attach_point[c]
        pieces_c = self.pieces_remaining[c]

        # Iterate attach points
        attach_cells = np.where(attach_c)[0]
        for anchor in attach_cells:
            if forbidden_c[anchor]:
                continue
            adj_status = self._compute_adj_status(anchor, forbidden_c)
            for piece in np.where(pieces_c)[0]:
                begin = self._LOOKUP.begin[anchor, adj_status, piece]
                size = self._LOOKUP.size[anchor, adj_status, piece]
                candidates = self._LOOKUP.move_ids[begin : begin + size]
                for move_id in candidates:
                    if self._move_is_legal(move_id, forbidden_c):
                        action_id = self._move_to_action_id(move_id)
                        out[action_id] = True
        return out

    @staticmethod
    def _move_is_legal(move_id: int, forbidden_c: np.ndarray) -> bool:
        """The 5-cell unrolled byte check."""
        cells = OptimisedBlokusDuoBoard._GEOMETRY.move_cells[move_id]
        # cells is a numpy slice of length 5, padded with NULL_CELL
        # for shorter pieces. forbidden_c[NULL_CELL] is always 0 so
        # the OR works correctly without bounds-checking.
        return not (
            forbidden_c[cells[0]] |
            forbidden_c[cells[1]] |
            forbidden_c[cells[2]] |
            forbidden_c[cells[3]] |
            forbidden_c[cells[4]]
        )
```

The Python overhead per move-id check will be much higher than C++'s; the win is on the *fewer move IDs to check* (the precomputed table pre-filters them) and the lack of allocation in the inner loop. Profile after P9 — if `valid_moves()` is still slow, candidates for further optimisation:
- Vectorise the inner loop across the candidate list (numpy fancy indexing into `forbidden_c`).
- Move the inner check to Cython if the Python interpreter overhead per candidate dominates.

Neither is required to land F2. They are deferred speedups.

---

## P7. Disk caching

The precomputed tables are pure functions of the piece definitions and board size. Build once, save to disk, never rebuild unless `pieces.json` changes.

Suggested layout:

```
games/blokusduo/precomputed/
├── move_tables.npz      # geometry + lookup tables packed together
├── pieces_hash.txt      # SHA-256 of pieces.json at build time
```

At import time:
```python
def load_or_build_tables() -> tuple[GeometryTables, LookupTable]:
    cache_path = PRECOMPUTED_DIR / "move_tables.npz"
    pieces_hash = sha256_of_pieces_json()
    if cache_path.exists() and (PRECOMPUTED_DIR / "pieces_hash.txt").read_text().strip() == pieces_hash:
        return _load_npz(cache_path)
    tables = build_all_tables()
    _save_npz(cache_path, tables)
    (PRECOMPUTED_DIR / "pieces_hash.txt").write_text(pieces_hash)
    return tables
```

Important: each worker in `core/parallel_self_play.py` calls this. With cache hit, it's a sub-100-ms load. Without cache, the first import builds it; subsequent workers see the now-cached version.

The `pieces_hash` check is paranoia — without it, an edit to pieces.json would silently use stale tables. The build is fast enough that recomputing on hash-miss is fine.

---

## P8. Switch-over criteria

The new implementation goes live only when **all** of these are true:

1. The P3 equivalence test passes on the 50,000-position gauntlet from P2 (see P2 for the stratified-sampling rationale).
2. The full existing test suite (`uv run pytest tests/`) still passes.
3. The MCTS profiling output (`scripts/mcts_profiling.py`) shows move-gen time dropping (sanity-check that we're actually exercising the new code).
4. A short end-to-end run (TTT with parallel=2) produces identical training data to a serial run at the same seed (the F1 determinism test extended to the F2-active codepath).

Until these all pass, the new implementation lives behind a config flag (`use_optimised_movegen: bool`, defaulting to False) and the old code is preserved unchanged.

---

## P9. Measurement

Two parts.

### P9.a: Wall-clock benchmark

Re-run `scripts/run_benchmark.sh --config run_configurations/profile_baseline.json --num-workers 4 --output-dir temp/profile_baseline_f2_w4_<YYYY-MM-DD>` after the switch lands. Compare per-game numbers against the F1-after baseline (~12s/game with 4 workers, 7.9 min total benchmark wall-clock).

Naming convention for benchmark output directories — encodes which optimisations are active, worker count, and date:

```
temp/profile_baseline_f1_w4_2026-05-27/   ← F1 only, 4 workers, baseline reference
temp/profile_baseline_f1_w8_2026-05-28/   ← F1 only, 8 workers (backfill measurement)
temp/profile_baseline_f2_w4_<date>/        ← F1+F2, 4 workers (this measurement)
temp/profile_baseline_f2_w8_<date>/        ← F1+F2, 8 workers (optional)
```

Expectations for P9.a:
- Move-gen slice of MCTS time shrinks from ~43% to **~15-20%** (still significant, but much smaller than before).
- Per-game wall-clock with 4 workers drops from ~12s to **~8-10s**.
- Overall benchmark wall-clock drops from 7.9 min to **~5-6 min**.

Anything outside ±20% of those expectations is worth investigating before claiming F2 done. The expected ranges come from Amdahl on the measured component shares; significantly different means either the move-gen wasn't actually replaced (still using the old code somehow) or our cost model is missing something.

### P9.b: Training-determinism check

The equivalence test (P3) verifies position-level `valid_moves()` equality. That's necessary but not sufficient: it doesn't catch bugs where the new impl computes the *right answer* but mutates global RNG state or otherwise causes downstream divergence in self-play.

So we additionally verify: **a self-play run with the old impl at fixed seed produces bit-identical training examples as the same run with the new impl at the same seed.** This is a stricter test, and it's the one that confirms F2 won't silently perturb training data.

Test recipe:

```python
def test_training_trajectory_identical():
    config = load("run_configurations/profile_baseline.json")
    # Same seed, both impls
    examples_old, stats_old = run_self_play_episodes(
        config, num_episodes=5, use_new_movegen=False,
    )
    examples_new, stats_new = run_self_play_episodes(
        config, num_episodes=5, use_new_movegen=True,
    )
    assert_examples_bit_identical(examples_old, examples_new)
    assert_stats_match(stats_old, stats_new)
```

Why this should hold: `valid_moves()` returns a binary mask. `np.where(mask > 0)[0]` returns the legal action IDs in monotonically increasing order regardless of how the mask was computed. Self-play picks moves via `np.random.choice(legal_action_ids, p=pi)` — same input array + same RNG state → same action choice. So the two implementations must walk the same trajectory.

If this test fails when the equivalence test (P3) passes, there's an interaction between the new impl and global state (RNG, dict ordering, etc) that we missed. That's a real correctness bug, not just a numeric noise issue.

Record both P9.a and P9.b results into the master plan progress tracker as the F2 row.

---

## P10. Archive

`git mv docs/plans/move-gen-optimisation.md docs/plans/archive/`. Append a "Scope additions" section first if anything landed outside the P0-P10 checklist. Update [`full-cycle-optimisation.md`](full-cycle-optimisation.md)'s F2 row to **Done ✓** with measured speedup.

---

## Out of scope for this plan

- **True bitboard implementation.** Considered, rejected. See "Design decision" above.
- **Cython or C extensions for the hot loop.** Possible follow-up if profiling shows Python overhead in the candidate-list inner loop dominates. Not required to land F2.
- **MCTS-side caching of valid_moves results.** A separate optimisation orthogonal to F2.
- **Game variants other than Blokus Duo.** Pentobi supports Classic (20×20), Trigon, etc. We don't. Plan stays Duo-specific.
- **Symmetric move detection** (Pentobi's `symmetric_move` field). Used in their early-game arena to handle symmetric duos as ties. Our MCTS doesn't use this; out of scope.
- **Action-space trimming** (the 17,837 → 13,729 question). Discussed in earlier conversations; conclusion was *not* to trim because our policy head is a giant FC that would need re-architecting, and the move-id space inside this F2 doesn't need to match the action-id space the network outputs.

---

## Notes for whoever picks this up

- **Move-count verification before anything else.** P1 (the `count_onboard_placements.py` script) runs before P2-P10. If the count doesn't match Pentobi's 13,729, every downstream step is sized wrong from the start. The check is cheap (a few seconds); the cost of skipping it and discovering the disagreement mid-implementation is huge.
- **Test infrastructure before optimisation.** P2 and P3 (position generator + equivalence-test scaffolding) land before any new move-gen code is written. The equivalence test running against the current implementation (a trivial pass at that point) means we have a working oracle before we change anything. If you skip ahead to P4+ and only write tests later, you have no way to detect regressions during development.
- **The `NULL_CELL=255` sentinel** is critical to the unrolled 5-cell check. Make sure `forbidden_c[NULL_CELL]` is always 0 by allocating `forbidden_c` with shape `(197,)` (one past the last valid cell index) and never touching `forbidden_c[NULL_CELL]`. This is the Python equivalent of Pentobi's `Point::null()`.
- **Don't rely on bit operations for any of this.** The whole point of choosing Option 1 is that the runtime path is byte-array reads. If you find yourself reaching for `int.bit_count()` or packing bits in inventive ways during P6, you've drifted from the design.
- **`pieces.json` is the source of truth for piece geometry.** Read shapes from there at build time; don't hardcode them. Pentobi's piece definitions are slightly different (they handle multiple variants); we have a single fixed pieces.json.
- **Watch the move-ID space.** Pentobi uses 0..13,728 for Duo (13,729 move IDs). Our action space is 0..17,836 (17,837 actions) because it's a clean `14×14 × 91 + 1` cartesian product. These are different numbering schemes. P5's `_move_to_action_id()` converts between them. Get this conversion right or every legal move gets pointed at the wrong policy slot.
- **The `adj_status` bit ordering** is documented in [`pentobi/move-generation.md`](../research/pentobi/move-generation.md) §4f. Match it byte-for-byte (N=0, W=1, E=2, S=3, NW=4, SE=5). The exact same bit ordering is what makes the precomputed table coherent — a mismatch between table-build and table-read would silently corrupt move generation.
- **Don't skip the disk caching (P6).** First-build time on a slow machine could be 30-60s. With 4-8 spawn-based workers each rebuilding it, that's 4-8 minutes of pure setup before the first game starts. The cache turns that into milliseconds.
- **You will write a bug somewhere.** The bit ordering, the move-ID conversion, the adj_status computation, the NULL_CELL sentinel — one of them. The equivalence test on 10,000 positions exists to catch your bugs cheaply. Don't relax that to 100 positions to make tests fast; the bugs hide in the long tail.
