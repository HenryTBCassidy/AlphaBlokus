# Move Generation — Further Optimisation

Deferred optimisation candidates for Blokus Duo move generation. These are high-effort changes that should only be pursued after an end-to-end training run reveals whether further speed is needed.

**Current state:** 3.4x speedup achieved over the naive implementation (6.5s → 1.8s per MCTS game at 25 sims). Move generation at ~2ms per leaf expansion is 70% of MCTS search time. Low-effort optimisations (O5–O8) were benchmarked and rejected. The remaining bottleneck is the Python-level per-cell validation loop in `_all_cells_valid`.

**When to revisit:** After completing a Minimal Blokus Duo training run. If training time is acceptable, these optimisations may not be needed. If move generation is still the dominant cost, start with O10 (Cython) as it's the most straightforward.

---

## O9. Cache valid moves per placement point

**What:** Precompute and store all valid Actions for each placement point when the point is first discovered. `_valid_moves` becomes a cache lookup instead of a full enumeration. When a new piece is placed, incrementally invalidate cached moves that are now blocked.

**How:**
1. When `_update_placement_points` discovers a new point at (i,j), compute all piece orientations that can legally touch that point and store them in the `PlacementDict` value (currently empty `{}`).
2. `_valid_moves` flattens and deduplicates the cached moves across all points.
3. When a piece is placed, invalidate cached moves at nearby points: any move that overlaps the new piece, touches a new friendly side, or uses the placed piece_id.

**The hard part:** Invalidation requires reverse lookups — "which cached moves have a cell at position X?" Building and maintaining these secondary indices is complex and error-prone.

**Files:** `board.py` (PlacementDict population and invalidation), `game.py` (`_valid_moves` becomes cache read)

**Effort:** 4–6 hours. Invalidation logic is the bulk of it.

**Expected speedup:** ~2x on `_valid_moves`, but adds work to `with_piece`. Net effect depends on the ratio of query to placement calls in MCTS (~1:1), so could be a wash.

**Risk:** Medium — invalidation bugs could cause missed or phantom moves.

---

## O10. Cython for `_all_cells_valid`

**What Cython is:** A Python-to-C compiler. You write Python-like code with C type annotations, and it compiles to a C extension module. Loops over typed variables become raw C loops. Array indexing on numpy arrays becomes direct C pointer arithmetic.

**What it would look like:**
```cython
# _fast_validation.pyx
cimport numpy as np

def all_cells_valid(
    np.ndarray[np.int64_t, ndim=2] filled_cells,
    int ins_i, int ins_j,
    np.ndarray[np.int8_t, ndim=2] board_2d,
    np.ndarray[np.uint8_t, ndim=2] side_danger,
) -> bint:
    cdef int k, ri, rj
    cdef int n_cells = filled_cells.shape[0]
    for k in range(n_cells):
        ri = ins_i + filled_cells[k, 0]
        rj = ins_j + filled_cells[k, 1]
        if board_2d[ri, rj] != 0:
            return False
        if side_danger[ri, rj]:
            return False
    return True
```

The current Python version is identical logic but each `board_2d[ri, rj]` goes through Python's `__getitem__` protocol (dozens of instructions) instead of a direct C memory access (one instruction).

**Files:** New `games/blokusduo/_fast_validation.pyx`, build config in `pyproject.toml`, `game.py` imports with Python fallback.

**Effort:** 2–3 hours. Cython code is ~30 lines. Build tooling configuration is the annoying part (platform-specific compilation, CI integration, fallback import).

**Expected speedup:** 5–10x on `_all_cells_valid` (C loops vs Python loops). Since that function is ~70% of search time, overall MCTS speedup: ~2–3x. Would bring "Serious" training from 8.5 → ~3–4 days on RTX 3060 Ti (re-profiling pending — see `docs/plans/gpu-training-poc.md`).

**Risk:** Low correctness risk (same algorithm, just compiled). Medium build risk (platform portability, dependency on Cython).

---

## O11. Bitboard representation

**What a bitboard is:** Represent the 14×14 board as a 196-bit integer. Each bit is one cell. Piece orientations are also bitmasks. Placement validation becomes two bitwise AND operations instead of a per-cell loop.

**How it works:**

Bit `i*14 + j` represents cell (i, j). Board state is three bitmasks:
- `occupied`: all cells with any piece
- `white_mask`: cells with white pieces
- `black_mask`: cells with black pieces

Side danger zone is computed by bit-shifting:
```python
COL_0_MASK = sum(1 << (i * 14) for i in range(14))       # leftmost column
COL_13_MASK = sum(1 << (i * 14 + 13) for i in range(14)) # rightmost column

danger = (player_mask << 14) | (player_mask >> 14) \
       | ((player_mask << 1) & ~COL_0_MASK) \
       | ((player_mask >> 1) & ~COL_13_MASK)
```

Each piece orientation has a precomputed base bitmask (at position 0,0). To place at (i,j), shift by `i*14+j`.

Validation:
```python
def is_valid(piece_mask, occupied, side_danger):
    return (piece_mask & occupied) == 0 and (piece_mask & side_danger) == 0
```

Two bitwise ANDs, two comparisons. No loops. Python's native `int` supports arbitrary precision and the operations are implemented in C inside CPython.

**What would change:** The existing numpy board stays (needed for neural net input). The bitboard is an additional representation for fast validation. `BlokusDuoBoard` gains `_occupied: int`, `_white_mask: int`, `_black_mask: int` fields. `_all_cells_valid` is replaced entirely.

**Files:** `board.py` (bitboard fields, update in `__init__`/`_from_state`/`with_piece`/`canonical`), `pieces.py` (precompute piece bitmasks), `game.py` (`_all_cells_valid` → bitwise check).

**Effort:** 4–6 hours. 196 bits doesn't fit in standard integer types (but Python's arbitrary-precision int handles it). Column boundary masking for side danger shifts needs careful implementation.

**Expected speedup:** 10–50x on validation (two C-level bitwise ANDs vs ~20 Python-level operations). Overall MCTS: ~3–5x. Would bring "Serious" training from 8.5 → ~2 days on RTX 3060 Ti (re-profiling pending — see `docs/plans/gpu-training-poc.md`).

**Risk:** Medium — unfamiliar technique for Blokus, 196-bit boundary handling, significant refactor of board internals. Well-established in chess engines but Blokus pieces are larger and more varied.

---

## Compatibility

| | O9 (cache) | O10 (Cython) | O11 (bitboard) |
|---|---|---|---|
| O9 | — | Compatible | Compatible but less needed |
| O10 | Compatible | — | Replaced by O11 |
| O11 | Compatible but less needed | Replaces O10 | — |

---

## Also consider: Batched MCTS inference

Not a move generation optimisation, but the single biggest performance win available. Currently each MCTS simulation calls `nnet.predict()` on one board at a time. Batching 16+ leaf evaluations into one GPU call would make inference ~12x faster, dropping it from ~33% to ~3% of search time on GPU. This would make move generation even more dominant (97% of time), increasing the ROI of O9/O10/O11.

This is a significant MCTS architecture change (leaf-parallel MCTS with virtual loss). See the AlphaZero paper for the approach. Should be implemented after the training pipeline is validated.
