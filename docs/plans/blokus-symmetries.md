# Blokus Duo Symmetries

Implement `BlokusDuoGame.get_symmetries()` so self-play training data is doubled by reflecting each (board, policy) pair across the main diagonal — the only non-identity symmetry the game admits given its fixed starting squares.

Before writing the symmetries code itself we also align our colour ↔ starting-square assignment with Pentobi's canonical Blokus Duo convention. That's a near-trivial swap but it has to happen first so the symmetry analysis is correct (and so eventual Pentobi benchmarking lines up cleanly).

Companion docs: `docs/04-BLOKUS-DUO.md` (game rules, pieces, orientations) and `docs/plans/archive/reporting-overhaul.md` (the reporting + W&B + diagnostic infrastructure this plan extends).

## Why two symmetries, not eight

Pentobi's source (`libpentobi_base/StartingPoints.cpp`) defines Blokus Duo's starting squares with `add_colored_starting_point` (vs `add_colorless_starting_point` used elsewhere) — meaning each square is **fixed to a specific colour**:

- Color(0) = "Purple", starts at (4, 4), moves first
- Color(1) = "Orange", starts at (9, 9), moves second

Both squares lie on the main diagonal. Of the 8 rigid motions of a 14×14 square, only **identity** and **reflection across the main diagonal** keep both starting squares in place without swapping them. Everything else (rotations, anti-diagonal reflection, horizontal/vertical flips) either moves a starting square off the diagonal or swaps them with each other. Colour-swap workarounds for the swapping variants don't recover validity because Pentobi's rule fixes who moves first — Purple at (4,4) — so a transformed game where Orange is at (4,4) is a different game, not the same game viewed differently.

So the symmetry group is order 2: `{identity, main-diagonal reflection}`. Self-play augmentation gets a 2× multiplier (vs TTT's 8×).

A "perspective flip" (swap the canonical-form +1/-1 channels) would be a further augmentation independent of the geometric symmetry. It's not a board symmetry per se and requires inverting the value target and the policy interpretation, so it's deferred — noted at the bottom of this plan.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| S0 | Swap starting-square convention to match Pentobi (Purple/White → (4,4); Orange/Black → (9,9)) | 30 min | High | |
| S1 | Build the orientation-transpose lookup table: for each of the 91 `orientation_id`s, compute the transposed shape and find the matching `orientation_id`. Verify round-trip for all 91. | 2 hr | High | ✅ |
| S2 | Implement `BlokusDuoBoard.transposed()` returning a board whose placement grid and internal caches reflect a main-diagonal transpose | 1.5 hr | High | ✅ |
| S3 | Implement `transpose_action(action_id: int) -> int` that decodes (cell, orientation), transposes both, re-encodes. Pass stays as itself. | 1 hr | High | ✅ |
| S4 | Wire `BlokusDuoGame.get_symmetries(board, pi)` to return `[(board, pi), (board.transposed(), transpose_pi(pi))]`. Replace the `NotImplementedError`. | 30 min | High | ✅ |
| S5 | **Equivariance tests**: for a hand-built mid-game position, verify `transpose(get_next_state(b, p, a)) == get_next_state(transpose(b), p, transpose_action(a))` across all legal `a`. | 1.5 hr | High | ✅ |
| S6 | **Invariance tests**: `valid_move_masking(transpose(b), p)` has the same legal-action count as `valid_move_masking(b, p)`, and the legal-action set transposes cleanly. | 30 min | High | ✅ |
| S7 | **Round-trip tests**: `transpose(transpose(b)) == b`, `transpose_action(transpose_action(a)) == a` on a random sample of states + actions. | 30 min | Medium | ✅ |
| S7.5 | **Property-based fuzz** (optional, hypothesis-style): random move sequences of varying length × random transpose-insertion points. Validates T2.b/T5.b hold for any rollout, not just hand-built ones. | 1 hr | Low | |
| S8 | **Visual sanity check**: render a mid-game board via the Blokus renderer, transpose it, render again — confirm pieces flip across the main diagonal by eye. Single test, snapshot HTML. | 30 min | Medium | |
| S9 | **Position-symmetry diagnostic** for the post-run report. Given a trained net + a fixed pair of symmetric positions, print the raw NN policy on both and a "symmetry KL divergence" score. New row in HTML report's diagnostics section. Same code works for TTT too (different reference position). | 2 hr | Medium | |
| S10 | **End-to-end Blokus test run**: small config (2 gens × 10 episodes × 50 sims, 64f×4b), executed on the PC via systemd-user. Verifies `get_symmetries()` is wired correctly into the self-play augmentation path. Not for training quality — just for breakage. | 1 hr | High | |
| S11 | `git mv docs/plans/blokus-symmetries.md docs/plans/archive/` in the final commit before PR. Add a "Scope additions" section first if anything landed beyond the row table. | 5 min | Low | |

Total: ~13 hours focused work (12 without the optional fuzz layer). Mappable to 12 reviewable commits.

---

## S0. Swap starting-square convention to match Pentobi

Pentobi's `libpentobi_base/StartingPoints.cpp` puts Color(0) (Purple, first to move) at (4, 4) and Color(1) (Orange, second to move) at (9, 9). Our code currently has the inverse:

```python
# games/blokusduo/game.py:38-39 (current)
self.white_start = (9, 9)
self.black_start = (4, 4)
```

White is our `+1` player (first to move), so White corresponds to Pentobi's Purple — and should start at **(4, 4)**, not (9, 9). The swap is one line each, but a few callsites narrate the old convention and need updating to stay accurate.

**Changes:**

1. `games/blokusduo/game.py:38-39` — swap constants.
2. `games/blokusduo/game.py:26-27` — class docstring currently says "White starts in the bottom-right corner / Black starts in the top-left corner". Both halves were already wrong by board-notation standards (top-left = (0,0)); update to reflect the Pentobi-aligned positions explicitly.
3. `tests/test_blokusduo/test_valid_moves.py:354-369` — two tests narrate "White monomino at start (9, 9)". They use `with_piece` directly (no starting-rule enforcement), so they pass either way mechanically. Update the coordinates + comments to (4, 4) for narrative correctness.
4. `docs/04-BLOKUS-DUO.md:30` — "White starts at (9, 9) and Black starts at (4, 4)" — invert.
5. `docs/02-ALGORITHMS.md:202` — "White: (9,9), Black: (4,4)" — invert.

`docs/01-BACKGROUND.md:215` lists "(4,4) and (9,9)" without assigning to colours and is fine as-is.

**Verification:** existing test suite passes (173/173). The starting-square swap is a behaviour change for first-move legality but the rest of move generation is position-agnostic, so failures here would be a sign of something baked-in we missed.

---

## S1. Orientation-transpose lookup table

The action space encodes `(cell_index × 91 orientations) + 1 pass`. Reflecting an action across the main diagonal requires reflecting BOTH the cell (`(r, c) → (c, r)`) AND the orientation: a piece in orientation `o` at cell `(r, c)` becomes a piece in *some other* orientation at `(c, r)`. The orientation transformation is the tricky bit — it's a lookup `orientation_id → orientation_id`.

**How to build it:**

For each `orientation_id` in `[0, 91)`:
1. Decode to `(piece_id, orientation_name)` via `OrientationCodec.decode`.
2. Get the piece's grid representation in that orientation via `PieceManager.get_piece_orientation_array`.
3. Transpose the grid (`np.transpose` or equivalent — `(r, c) → (c, r)`).
4. Find which of the same piece's basis orientations matches the transposed grid. Round-trip via `OrientationCodec.encode((piece_id, matched_orientation))`.
5. Store `transpose_orientation[orientation_id] = new_orientation_id`.

Edge cases to test exhaustively:
- **Monomino**: 1×1 shape, all orientations equivalent. Lookup should be identity for all entries.
- **Pieces with internal symmetry**: e.g. the 2×2 square has only 1 distinct orientation; the I-pentomino has 2 (horizontal vs vertical). Transposing these should land on a member of the same equivalence class.
- **Chiral pieces** (L, J, S, Z, F shapes): transposed shape might equal a different chirality. The map should always find a match within the same piece's basis orientations — if not, that's a piece-orientation-table bug.

**Where it lives:** new module method `PieceManager.orientation_transpose_id(orientation_id) -> int`, with the table built lazily on first call. Alternatively a standalone module `games/blokusduo/symmetry.py` if it grows.

### Tests for S1

Two layers; both run over the full 91-entry action space.

**T1.a — bijection + involution on IDs.** Property check at the integer level. Doesn't depend on the piece-grid representation being right, just that the lookup is a well-formed self-inverse permutation.

```python
def test_transpose_orientation_id_is_involution_and_bijection():
    table = piece_manager.transpose_orientation_table()  # length 91
    # Involution
    for o in range(91):
        assert table[table[o]] == o
    # Bijection
    assert len(set(table)) == 91
```

**T1.b — grids match the lookup (per-piece, exhaustive).** This is the test that actually validates the *meaning* of the table. For every piece × every basis orientation, the table's claimed transpose must equal the numerical transpose of the piece's grid in that orientation.

```python
def test_per_piece_transpose_lookup_matches_grid_transpose():
    for piece_id in range(1, 22):  # 21 pieces
        for orientation in piece_manager.pieces[piece_id].basis_orientations:
            o_id = piece_manager.get_piece_orientation_id((piece_id, orientation))
            new_o_id = piece_manager.orientation_transpose_id(o_id)
            new_piece_id, new_orientation = piece_manager.get_piece_orientation(new_o_id)
            # Transpose must stay within the same piece (never crosses)
            assert new_piece_id == piece_id, (
                f"piece {piece_id} transposed to a different piece {new_piece_id}"
            )
            grid = piece_manager.get_piece_orientation_array(piece_id, orientation)
            new_grid = piece_manager.get_piece_orientation_array(new_piece_id, new_orientation)
            assert np.array_equal(np.transpose(grid), new_grid), (
                f"piece {piece_id} orientation {orientation}: "
                f"transpose lookup claims {new_orientation} but grids don't match"
            )
```

T1.a is fast; T1.b is the substantive correctness check. If T1.b passes for all 91 entries the table is provably correct.

---

## S2. `BlokusDuoBoard.transposed()`

A method on `BlokusDuoBoard` returning a new immutable board representing the main-diagonal transpose. Internally:

1. **Piece placement grid** (`_piece_placement_board`): transpose the 14×14 int8 array directly. Piece IDs and signs (positive = White, negative = Black) carry through unchanged.
2. **Per-piece spatial channels** — these are computed lazily by `as_multi_channel`, so we don't need to materialise them; just confirm the lazy computation returns the transposed channels correctly once the underlying placement board is transposed.
3. **Internal caches** that need explicit reconstruction (NOT just transposition):
   - `_placement_points` (per-player corner positions for next piece) — recompute from the transposed placement board.
   - `_remaining_piece_ids` (which pieces each player still has) — invariant under board transposition; carry through as-is.
   - `_last_piece_played` — invariant; carry through.
   - `_side_danger_zone` if cached — recompute.
4. `_initial_actions` reference — same `BlokusDuoBoard._initial_actions` dict is shared; we re-bind to it on the new board.

Implementation should call existing constructor helpers rather than directly manipulating fields where possible — keeps the immutability invariants intact.

### Tests for S2

The board transpose is the foundation S3–S10 all stand on, so this section is intentionally exhaustive. Three layers, each catching a different class of bug.

**T2.a — direct equality on the raw placement grid.** Anchor: if this fails nothing else can be trusted.

```python
def test_transposed_placement_grid_equals_numpy_transpose(positions):
    for position in positions:  # empty, early, mid, late, near-end
        assert np.array_equal(
            position.transposed()._piece_placement_board,
            np.transpose(position._piece_placement_board),
        )
```

**T2.b — cache consistency via reconstruction (the gold-standard test).** Builds a fresh board from the transposed placement grid and compares it field-by-field against `position.transposed()`. The two must be byte-identical. This is the test that catches half-stale caches — the most common bug shape in immutable-board transformations.

```python
def test_transposed_board_caches_match_fresh_reconstruction(positions):
    for position in positions:
        transposed = position.transposed()
        # Public path: build a board from the *transposed* placement grid
        # using whatever factory the codebase exposes. This is what the
        # transpose method should be equivalent to.
        rebuilt = BlokusDuoBoard.from_placement_grid(
            np.transpose(position._piece_placement_board),
            piece_manager=blokus_game.piece_manager,
            ...,  # other constructor deps
        )
        for player in [1, -1]:
            assert transposed.placement_points(player) == rebuilt.placement_points(player)
            assert transposed.remaining_piece_ids(player) == rebuilt.remaining_piece_ids(player)
            assert transposed.last_piece_played(player) == rebuilt.last_piece_played(player)
            assert np.array_equal(
                transposed.side_danger_zone(player),
                rebuilt.side_danger_zone(player),
            )
        # …and every other cached field exposed on the public board API
```

If `BlokusDuoBoard` doesn't already expose a `from_placement_grid` factory, S2 includes adding one (or an equivalent test-only constructor) — the rebuild-from-truth path needs *some* mechanism. Worst case the helper lives in a test-only module.

**T2.c — invariants preserved across the transform.** Cheap smoke checks that fail fast if any high-level quantity drifted. Run alongside T2.a/T2.b on the same positions:

```python
def test_transposed_board_preserves_invariants(positions):
    for position in positions:
        t = position.transposed()
        # Total placed cells identical
        assert (t._piece_placement_board != 0).sum() == (position._piece_placement_board != 0).sum()
        # Per-player invariants
        for player in [1, -1]:
            assert len(t.remaining_piece_ids(player)) == len(position.remaining_piece_ids(player))
            assert t.last_piece_played(player) == position.last_piece_played(player)
            assert len(t.placement_points(player)) == len(position.placement_points(player))
```

### Choice of positions to test against

Each layer runs over a fixture list spanning representative game states:

1. **Empty board** — degenerate, every cache is empty
2. **First-move-only** — a single piece on each starting square
3. **Mid-game** — 4-6 pieces per player, mixed orientations including off-diagonal placements
4. **Near-end** — 15+ pieces per player, dense board
5. **Diagonal-symmetric position** — a hand-built board that's its own transpose (every piece sits on a symmetric placement). Transpose should be a no-op; great forcing function.

This gives 5 positions × 3 layers = 15 distinct assertions per test, exhaustive enough to catch any half-baked cache update.

---

## S3. `transpose_action(action_id: int) -> int`

Pure function in `games/blokusduo/symmetry.py` (or wherever S1's lookup ended up). Given an `action_id`:

```python
def transpose_action(self, action_id: int) -> int:
    if action_id == self.pass_action_id:
        return action_id  # pass is invariant
    cell, orientation_id = self.action_codec.decode(action_id)
    new_cell = (cell[1], cell[0])  # transpose row/col
    new_orientation_id = self.piece_manager.orientation_transpose_id(orientation_id)
    return self.action_codec.encode(new_cell, new_orientation_id)
```

Exact signature TBD — `action_codec`'s API is already in `board.py` (`ActionCodec`). Reuse it; don't reinvent.

**Vector form:** `transpose_pi(pi: NDArray) -> NDArray` builds a permutation of shape `(17_837,)` via the per-action map, then re-indexes. Compute the permutation once at game init, then it's a single numpy gather per call.

**Test:** for every legal action in a sample of mid-game boards, the transposed action is also legal on the transposed board, and applying transposed action to transposed board produces the transpose of the original next-state.

---

## S4. `BlokusDuoGame.get_symmetries(board, pi)`

The wiring step. Replace the `NotImplementedError`:

```python
def get_symmetries(
    self, board: BlokusDuoBoard, pi: NDArray,
) -> list[tuple[BlokusDuoBoard, NDArray]]:
    """Return identity + main-diagonal reflection.

    Blokus Duo's symmetry group is order 2: only the main-diagonal
    reflection preserves the fixed starting-square assignments at (4,4)
    and (9,9). See docs/plans/archive/blokus-symmetries.md for the
    derivation.
    """
    return [
        (board, pi),
        (board.transposed(), self._symmetry.transpose_pi(pi)),
    ]
```

Where `self._symmetry` is the helper object holding the precomputed orientation transpose map + the action permutation built in S3. Constructed lazily on first call to avoid slowing `BlokusDuoGame.__init__`.

**Test:** smoke test calling `get_symmetries` on a fresh board returns 2 tuples; verify the second tuple's board is `np.transpose` of the first.

---

## S5. Equivariance tests

The core correctness test: applying a move then transposing must equal transposing then applying the transposed move. This is what makes symmetry augmentation a *valid* augmentation rather than a corrupted training signal.

Two flavours: **single-move** (catches bugs in one step) and **full-replay** (catches bugs that only manifest over a sequence — e.g. a cache that's "wrong on the second move but right on the first").

**T5.a — single-move equivariance.** Run over every legal action from a hand-built mid-game position; both players.

```python
def test_get_next_state_equivariant_under_transpose(blokus_game, positions):
    for position in positions:
        for player in [1, -1]:
            valids = blokus_game.valid_move_masking(position, player)
            for action in np.flatnonzero(valids):
                # Path A: move then transpose
                post_move, _ = blokus_game.get_next_state(position, player, int(action))
                path_a = post_move.transposed()
                # Path B: transpose then move
                transposed_action = blokus_game.transpose_action(int(action))
                transposed_board = position.transposed()
                path_b, _ = blokus_game.get_next_state(transposed_board, player, transposed_action)
                assert path_a == path_b
```

**T5.b — full-replay equivariance (the killer test).** Apply the same game sequence in two ways and verify the final boards match.

```python
def test_transpose_commutes_with_game_replay(blokus_game, seeded_action_sequence):
    actions, players = seeded_action_sequence
    # Path A: play the sequence on the original board, then transpose at the end
    path_a = blokus_game.initialise_board()
    for action, player in zip(actions, players):
        path_a, _ = blokus_game.get_next_state(path_a, player, action)
    final_a = path_a.transposed()

    # Path B: transpose the starting board, then play the *transposed* actions
    path_b = blokus_game.initialise_board().transposed()
    for action, player in zip(actions, players):
        t_action = blokus_game.transpose_action(action)
        path_b, _ = blokus_game.get_next_state(path_b, player, t_action)

    assert final_a == path_b
```

Note that "transposed starting board" is the same as the starting board for Blokus (empty board is its own transpose), but writing it explicitly makes the symmetry obvious and protects against future regressions if `initialise_board()` ever places pre-game markers.

Run T5.b across multiple seeded rollouts (e.g. 10 random move sequences of length 20). Failures here point at a bug in S1, S2, or S3 — combined with T2.b passing, they specifically isolate S3.

---

## S6. Invariance tests

The valid-move masking must be preserved under transpose:

```python
def test_valid_move_count_invariant_under_transpose(blokus_game, mid_game_board):
    mask_orig = blokus_game.valid_move_masking(mid_game_board, player=1)
    mask_transposed = blokus_game.valid_move_masking(mid_game_board.transposed(), 1)
    assert mask_orig.sum() == mask_transposed.sum()

    # Stronger: the transposed action set equals the action set on the
    # transposed board
    legal_orig = set(np.where(mask_orig)[0])
    legal_transposed = set(np.where(mask_transposed)[0])
    transposed_legal_set = {blokus_game.transpose_action(a) for a in legal_orig}
    assert transposed_legal_set == legal_transposed
```

---

## S7. Round-trip tests

Idempotence and bijection checks:

```python
def test_transpose_action_is_involution(blokus_game):
    for action_id in range(blokus_game.get_action_size()):
        assert blokus_game.transpose_action(blokus_game.transpose_action(action_id)) == action_id

def test_board_transpose_is_involution(blokus_board):
    assert blokus_board.transposed().transposed() == blokus_board
```

Fast, exhaustive over the action space.

---

## S7.5. Property-based fuzz (optional)

Hypothesis-style randomised tests covering the same invariants as T2.b and T5.b but over the space of all reachable boards, not just hand-built fixtures. Cheap to add (one or two `hypothesis.given` decorators) and dramatically widens the bug-finding net.

```python
from hypothesis import given, strategies as st

@given(st.integers(0, 50), st.integers(0, 2**32))
def test_replay_equivariance_random_rollouts(rollout_length, seed):
    rng = np.random.default_rng(seed)
    actions, players = generate_legal_rollout(blokus_game, rollout_length, rng)
    # T5.b body, parameterised
    ...
```

Optional because T2.b + T5.b on the hand-built fixture suite already cover the high-leverage cases. Worth doing if any fuzz test surfaces a regression that wasn't caught by the deterministic suite — at that point it pays for itself.

If `hypothesis` is added as a dev dependency for this, mention it in `STYLE-GUIDE.md`'s testing section so the convention is documented.

---

## S8. Visual sanity check

A single test that:
1. Builds a recognisable mid-game position (e.g. White's L-pentomino at (2,2), Black's T-tetromino at (10,10))
2. Renders both the original and transposed boards via `reporting.display_blokusduo.BlokusDuoRenderer`
3. Snapshots both HTML fragments to `tests/snapshots/blokusduo/`
4. Manual inspection step — open the snapshot, confirm by eye

This catches subtle bugs that pure-numerical tests miss (e.g. piece appears correctly transposed but the renderer's coordinate convention undoes it).

---

## S9. Position-symmetry diagnostic

A post-run diagnostic that shows whether a trained net learned to play symmetrically. Generalises beyond Blokus — TTT will use the same diagnostic to track whether the asymmetric-preference issue Henry spotted improves over generations.

**Mechanism:**

1. Pick a small set of reference positions per game (3-5 mid-game boards with non-trivial symmetric pairs).
2. After training completes (or each gen, optionally), run `nnet.predict(board)` and `nnet.predict(board.transposed())` for each reference.
3. Compute KL divergence between the two raw-policy distributions (after transposing one to align with the other's coordinates).
4. Add a section to the HTML report and a `learning_quality/symmetry_kl_divergence` W&B metric.

Trained nets should drive this toward zero. Non-zero residuals reveal directional biases (the "favourite corner" effect).

**Where it lives:**
- `reporting/symmetry_diagnostic.py` — new module
- New parquet directory `temp/<run>/SymmetryDiagnostic/`
- Logged once per gen (or once per run, configurable)
- Game-aware via the renderer / game-symmetries plumbing already in place

---

## S10. End-to-end Blokus test run

Tiny configuration to validate the augmentation path works end-to-end on the PC:

```jsonc
{
  "game": "blokusduo",
  "run_name": "blokus_pc_test",
  "seed": 42,
  "num_generations": 2,
  "num_eps": 10,
  "temp_threshold": 6,
  "update_threshold": 0.55,
  "num_arena_matches": 10,
  "minimax_games_per_gen": 0,  // no minimax for Blokus
  "mcts_config": { "num_mcts_sims": 50, "cpuct": 1.0 },
  "net_config": {
    "learning_rate": 0.001, "dropout": 0.3, "epochs": 5,
    "batch_size": 64, "cuda": true,
    "num_filters": 64, "num_residual_blocks": 4
  }
}
```

Purpose is **breakage detection**, not quality:
- Self-play games complete without crashing
- Training examples include 2× the position count (proof that `get_symmetries` returned 2 tuples)
- Arena games complete
- Report renders
- W&B dashboard populates the new game's section

Wall-clock expectation: ~10-15 minutes on the 3060 Ti given the small config.

---

## S11. Archive

`git mv docs/plans/blokus-symmetries.md docs/plans/archive/`. Before the move, append a "Scope additions beyond original plan" section if anything landed outside the row table (per [[feedback_plan_scope_additions]]).

---

## Deferred — not in this PR

- **Perspective flip augmentation** (swap +1/-1 canonical channels + invert value sign + reinterpret policy). Adds another 2× data, total 4× with the geometric symmetry. Skipped because it's a different *kind* of augmentation that requires inverting target values and reconstructing policies from the opposite player's POV — meaningful complexity. Worth revisiting if Blokus training stalls on data quantity.
- **Profiling impact of symmetry augmentation on self-play wall-clock.** The augmentation roughly doubles the training data per gen, so training time is now non-trivial relative to self-play. We may want to revisit batch_size / epochs after a few real Blokus runs.
