# AlphaBlokus — Blokus Duo Valid Move Generation

Implement `BlokusDuoGame.valid_moves()` and `valid_move_masking()` — the main blocker for Blokus Duo self-play.

Prerequisites: Placement point cache validated (13 tests passing). Board/Game separation complete.

**NOTE:** Once the algorithm is finalised and implemented, update `docs/02-ALGORITHMS.md` with a section on move generation (algorithm description, complexity, caching strategy).

---

## Checklist

| # | Item | Effort | Done |
|---|------|--------|------|
| M1 | Implement naive correct `_valid_moves` | 1.5 hours | ✅ |
| M2 | Implement `valid_move_masking` using `_valid_moves` + `ActionCodec` | 15 min | ✅ |
| M3 | Implement `get_game_ended` using `_valid_moves` | 15 min | ✅ |
| M4 | Write comprehensive tests for `valid_moves` | 1.5 hours | |
| M5 | Profile: log move counts per turn in MCTS, measure `valid_moves` call time | 1 hour | |
| M6 | Optimise based on profiling data | TBD | |
| M7 | Update `docs/02-ALGORITHMS.md` with move generation section | 30 min | |

**Estimated total: ~5 hours** (excluding M6 which depends on profiling results)

---

## M1. Implement naive correct `valid_moves`

The algorithm revolves around placement points — board coordinates that are diagonally adjacent to a friendly piece and not side-adjacent to one. These are already cached and maintained incrementally by `BlokusDuoBoard._update_placement_points()`.

### Algorithm

```
valid_moves(board, player):
    if player has placed no pieces:
        return initial_actions[player]

    moves = set()
    points = board.placement_points(player)
    remaining = board.remaining_piece_ids(player)
    board_2d = board.as_2d

    for (pi, pj) in points:
        for piece_id in remaining:
            for orientation in piece.basis_orientations:
                piece_array = get_piece_orientation_array(piece_id, orientation)
                for (di, dj) in filled_cells(piece_array):
                    # This filled cell should land on the placement point
                    insertion_i = pi - di
                    insertion_j = pj - dj
                    if not fits_on_board(piece_array, insertion_i, insertion_j):
                        continue
                    if all_cells_valid(piece_array, insertion_i, insertion_j, player, board_2d):
                        action = to_action(piece_id, orientation, insertion_i, insertion_j)
                        moves.add(action)

    return list(moves)
```

**Key checks in `all_cells_valid`:** For every filled cell of the piece at its candidate position:
- Cell is on the board (bounds check)
- Cell is empty (`board_2d[i, j] == 0`)
- Cell is not side-adjacent to a friendly piece (`_no_sides(i, j, player, board_2d)`)

The corner check is NOT needed per-cell — the fact that one cell lands on a placement point guarantees at least one diagonal touch.

**Coordinate translation:** Placement points use array indices. `Action` uses board coordinates. The `CoordinateIndexDecoder` converts between them.

**Deduplication:** The same Action can be reached from multiple placement points (if the piece has multiple cells that each touch a different point). Using a `set[Action]` handles this since `Action` is a frozen dataclass.

### Complexity estimate

Per call: `|placement_points| × |remaining_pieces| × |orientations_per_piece| × |cells_per_piece|` candidates, each requiring `|cells_per_piece|` validity checks.

Rough numbers mid-game: ~10 points × ~15 pieces × ~4 orientations × ~4 cells = ~2,400 candidates, each checking ~4 cells = ~10,000 array lookups. Should be fast enough for initial implementation.

**Files:** `games/blokusduo/game.py`

---

## M2. Implement `valid_move_masking`

Trivial once `valid_moves` works — encode each Action to an index and set the mask bit.

```python
def valid_move_masking(self, board, player):
    moves = self.valid_moves(board, player)
    mask = np.zeros(self.get_action_size())
    for action in moves:
        mask[self.action_codec.encode(action)] = 1
    if not moves:
        mask[self.action_codec.pass_action_index] = 1
    return mask
```

**Files:** `games/blokusduo/game.py`

---

## M3. Implement `get_game_ended`

Already stubbed in `BlokusDuoGame.get_game_ended`. Currently calls `self.valid_moves(board, player)` which returns `[]`. Once M1 is done, this will work correctly — game ends when neither player has legal moves.

**Files:** `games/blokusduo/game.py` (already in place, just needs M1)

---

## M4. Write comprehensive tests

Test categories:
- First move returns cached initial actions
- Single piece placed, verify expected legal moves for next turn
- Piece near edge — moves that would go off-board are excluded
- Side adjacency — moves that would touch friendly side are excluded
- Opponent pieces block squares but don't create side violations
- Late game with few remaining pieces
- No legal moves → pass action in masking
- Round-trip: every move returned by `valid_moves` can be applied via `get_next_state`
- `valid_move_masking` has correct number of 1-bits matching `len(valid_moves)`

**Files:** `tests/test_blokusduo/test_valid_moves.py`

---

## M5. Profile move generation

- Log `len(valid_moves)` per turn during MCTS self-play
- Measure wall-clock time of `valid_moves` calls
- Identify whether the bottleneck is candidate enumeration or validity checking
- Get distributions across game phases (opening/mid/endgame)

This data informs M6.

---

## M6. Optimise based on profiling

Potential optimisations (implement only if profiling shows they're needed):

- **Pre-compute piece corners:** For each piece orientation, cache which filled cells are "exterior" (have a diagonal that extends outside the piece). Only these can touch placement points, reducing the inner loop.
- **Cache piece fits per placement point:** Store all valid actions per point. Incrementally invalidate when a new piece is placed. Complex but eliminates re-enumeration.
- **Numpy vectorisation:** Replace Python loops with array operations for the validity checks.
- **Bitboard representation:** Represent board and pieces as bitmasks for fast overlap/adjacency checks.

---

## M7. Update algorithms documentation

Add a "Move Generation" section to `docs/02-ALGORITHMS.md` covering:
- Algorithm description and pseudocode
- Placement point cache maintenance
- Complexity analysis
- Optimisation strategies applied (from M6)
