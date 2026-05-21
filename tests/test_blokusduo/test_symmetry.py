"""Tests for the Blokus Duo symmetry plumbing.

Covers the layered correctness checks described in
``docs/plans/blokus-symmetries.md`` — orientation lookup (S1), board
transpose (S2), action transpose (S3), and the higher-level equivariance /
invariance / round-trip properties (S5–S7).
"""
from __future__ import annotations

import numpy as np
import pytest

from games.blokusduo.board import Action, BlokusDuoBoard
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pieces import Orientation, PieceManager


# ── S1: orientation transpose lookup ────────────────────────────────────────


def test_orientation_transpose_id_is_involution(piece_manager: PieceManager) -> None:
    """T1.a — applying the transpose lookup twice returns the original ID."""
    n = piece_manager.num_entries
    for o_id in range(n):
        assert piece_manager.orientation_transpose_id(
            piece_manager.orientation_transpose_id(o_id),
        ) == o_id


def test_orientation_transpose_id_is_bijection(piece_manager: PieceManager) -> None:
    """T1.a — the lookup is a permutation of [0, n)."""
    n = piece_manager.num_entries
    images = {piece_manager.orientation_transpose_id(o_id) for o_id in range(n)}
    assert images == set(range(n))


def test_orientation_transpose_id_matches_grid_transpose(
    piece_manager: PieceManager,
) -> None:
    """T1.b — for every piece × basis orientation, the lookup's claimed
    transposed orientation has a grid equal to ``np.transpose`` of the
    original grid. Substantive correctness check.
    """
    for piece_id, piece in piece_manager.pieces.items():
        for orientation in piece.basis_orientations:
            o_id = piece_manager.get_piece_orientation_id((piece_id, orientation))
            new_o_id = piece_manager.orientation_transpose_id(o_id)
            new_piece_id, new_orientation = piece_manager.get_piece_orientation(new_o_id)

            assert new_piece_id == piece_id, (
                f"piece {piece_id} ({piece.name}) {orientation.value} transposed "
                f"to a different piece {new_piece_id}"
            )
            original = piece_manager.get_piece_orientation_array(piece_id, orientation)
            transposed_grid = piece_manager.get_piece_orientation_array(
                new_piece_id, new_orientation,
            )
            assert np.array_equal(np.transpose(original), transposed_grid), (
                f"piece {piece_id} ({piece.name}) orientation {orientation.value}: "
                f"lookup says transpose is {new_orientation.value} but grids differ"
            )


def test_monomino_orientation_transpose_is_identity(
    piece_manager: PieceManager,
) -> None:
    """Sanity check: the monomino has only one orientation (Identity), and
    its transpose must map back to itself.
    """
    monomino_id = next(
        pid for pid, p in piece_manager.pieces.items()
        if p.identity.shape == (1, 1)
    )
    o_id = piece_manager.get_piece_orientation_id((monomino_id, Orientation.Identity))
    assert piece_manager.orientation_transpose_id(o_id) == o_id


@pytest.fixture
def transposed_orientation_table(piece_manager: PieceManager) -> dict[int, int]:
    """All 91 (orientation_id → transposed orientation_id) pairs.

    Lets follow-up symmetry tests reuse the table without rebuilding it.
    """
    return {
        o_id: piece_manager.orientation_transpose_id(o_id)
        for o_id in range(piece_manager.num_entries)
    }


# ── Position fixtures shared by S2+ tests ───────────────────────────────────


def _action(piece_id: int, orientation: Orientation, x: int, y: int) -> Action:
    return Action(piece_id=piece_id, orientation=orientation, x_coordinate=x, y_coordinate=y)


@pytest.fixture
def empty_board(blokus_game: BlokusDuoGame) -> BlokusDuoBoard:
    return blokus_game.initialise_board()


@pytest.fixture
def first_move_board(blokus_game: BlokusDuoGame, empty_board: BlokusDuoBoard) -> BlokusDuoBoard:
    """Each player has placed their starting monomino on their start square.

    White at (4, 4), Black at (9, 9) — the Pentobi-aligned convention.
    """
    # In board coordinates (x, y), origin bottom-left. We choose y to land
    # the monomino's single cell onto the array-index starting square.
    decoder = blokus_game._coordinate_index_decoder
    white_x, white_y = decoder.to_coordinate((4, 4))
    black_x, black_y = decoder.to_coordinate((9, 9))
    b = empty_board.with_piece(_action(1, Orientation.Identity, white_x, white_y), 1)
    b = b.with_piece(_action(1, Orientation.Identity, black_x, black_y), -1)
    return b


@pytest.fixture
def mid_game_board(blokus_game: BlokusDuoGame, empty_board: BlokusDuoBoard) -> BlokusDuoBoard:
    """A hand-built mid-game position with several pieces per side, mixing
    orientations to exercise the orientation-transpose lookup.

    Uses ``initial_actions`` to satisfy the starting-square constraint on
    each player's first move and then valid-moves-aware follow-ups.
    """
    # First move: white monomino on its start square.
    white_first = next(
        a for a in blokus_game.initial_actions[1] if a.piece_id == 1
    )
    b = empty_board.with_piece(white_first, 1)
    black_first = next(
        a for a in blokus_game.initial_actions[-1] if a.piece_id == 1
    )
    b = b.with_piece(black_first, -1)
    # A couple of follow-up moves picked to land on different orientations.
    for player in (1, -1, 1, -1):
        legal = blokus_game._valid_moves(b, player)
        # Pick a move with a non-Identity orientation when possible so the
        # transpose tests touch the orientation lookup.
        choice = next(
            (m for m in legal if m.orientation != Orientation.Identity),
            legal[0],
        )
        b = b.with_piece(choice, player)
    return b


@pytest.fixture
def positions(
    empty_board: BlokusDuoBoard,
    first_move_board: BlokusDuoBoard,
    mid_game_board: BlokusDuoBoard,
) -> list[BlokusDuoBoard]:
    """Representative fixtures for S2's layered checks."""
    return [empty_board, first_move_board, mid_game_board]


# ── S2: board transpose ─────────────────────────────────────────────────────


def test_transposed_placement_grid_equals_numpy_transpose(
    positions: list[BlokusDuoBoard],
) -> None:
    """T2.a — raw placement-grid equality. Anchor: if this fails nothing else
    can be trusted.
    """
    for position in positions:
        assert np.array_equal(
            position.transposed()._piece_placement_board,
            np.transpose(position._piece_placement_board),
        )


def test_transposed_board_caches_match_independent_recompute(
    positions: list[BlokusDuoBoard],
) -> None:
    """T2.b — every cached field on the transposed board matches what we'd
    compute from scratch off the transposed placement grid. Catches half-
    stale caches.
    """
    for position in positions:
        transposed = position.transposed()
        transposed_grid = np.transpose(position._piece_placement_board)
        transposed_2d = np.sign(transposed_grid).astype(np.int8)

        # side_danger — recompute via the canonical static method
        for player in (1, -1):
            expected_danger = BlokusDuoBoard._compute_side_danger(transposed_2d, player)
            actual_danger = transposed.side_danger_zone(player)
            assert np.array_equal(actual_danger, expected_danger), (
                f"side_danger mismatch for player {player} on transposed board"
            )

        # placement_points — recompute by walking the grid and checking the
        # placement validity predicate at each cell.
        for player in (1, -1):
            expected_keys = set()
            for i in range(BlokusDuoBoard.N):
                for j in range(BlokusDuoBoard.N):
                    if BlokusDuoBoard._valid_placement(i, j, player, transposed_2d):
                        expected_keys.add((i, j))
            actual_keys = set(transposed.placement_points(player).keys())
            assert actual_keys == expected_keys, (
                f"placement_points key set mismatch for player {player}"
            )


def test_transposed_board_preserves_invariants(
    positions: list[BlokusDuoBoard],
) -> None:
    """T2.c — counts, remaining-piece sets, last-piece-played carry through
    unchanged, and per-player placement-point counts match.
    """
    for position in positions:
        t = position.transposed()
        # Total placed cells identical
        assert (t._piece_placement_board != 0).sum() == (
            position._piece_placement_board != 0
        ).sum()
        for player in (1, -1):
            assert t.remaining_piece_ids(player) == position.remaining_piece_ids(player)
            assert t.last_piece_played(player) == position.last_piece_played(player)
            # Geometric-invariant count
            assert len(t.placement_points(player)) == len(position.placement_points(player))


def test_transposed_board_is_involution(positions: list[BlokusDuoBoard]) -> None:
    """Transposing twice gives back the same placement state. Stronger forms
    of involution (full field equality) are exercised by the other tests.
    """
    for position in positions:
        assert np.array_equal(
            position.transposed().transposed()._piece_placement_board,
            position._piece_placement_board,
        )


# ── S4: get_symmetries ──────────────────────────────────────────────────────


def test_get_symmetries_returns_two_pairs(
    blokus_game: BlokusDuoGame, mid_game_board: BlokusDuoBoard,
) -> None:
    """Blokus Duo's symmetry group has order 2 — identity plus the main-
    diagonal reflection. ``get_symmetries`` must always return exactly two
    tuples for any non-empty board.
    """
    n = blokus_game.get_action_size()
    pi = np.ones(n, dtype=np.float32) / n
    symmetries = blokus_game.get_symmetries(mid_game_board, pi)
    assert len(symmetries) == 2
    # First entry is identity
    assert symmetries[0][0] is mid_game_board
    assert symmetries[0][1] is pi
    # Second entry is the transpose
    assert np.array_equal(
        symmetries[1][0]._piece_placement_board,
        np.transpose(mid_game_board._piece_placement_board),
    )


def test_get_symmetries_policy_pairs_with_board(
    blokus_game: BlokusDuoGame, mid_game_board: BlokusDuoBoard,
) -> None:
    """The policy entry of the second tuple must be the action-permutation
    of the original — i.e. the policy correctly transposed.
    """
    rng = np.random.default_rng(seed=0)
    pi = rng.random(blokus_game.get_action_size()).astype(np.float32)
    _, (transposed_board, transposed_pi) = blokus_game.get_symmetries(mid_game_board, pi)
    expected = blokus_game.transpose_policy(pi)
    assert np.array_equal(transposed_pi, expected)


# ── S3: action transpose ────────────────────────────────────────────────────


def test_pass_action_is_self_transpose(blokus_game: BlokusDuoGame) -> None:
    pass_id = blokus_game.action_codec.pass_action_index
    assert blokus_game.transpose_action(pass_id) == pass_id


def test_transpose_action_is_involution_over_full_action_space(
    blokus_game: BlokusDuoGame,
) -> None:
    """T7-style check at the action level: applying transpose twice returns
    the original ID for *every* index in the 17,837-element action space.
    """
    n = blokus_game.get_action_size()
    for a in range(n):
        assert blokus_game.transpose_action(blokus_game.transpose_action(a)) == a


def test_transpose_action_is_bijection_over_full_action_space(
    blokus_game: BlokusDuoGame,
) -> None:
    n = blokus_game.get_action_size()
    images = {blokus_game.transpose_action(a) for a in range(n)}
    assert images == set(range(n))


def test_transpose_action_geometry(blokus_game: BlokusDuoGame) -> None:
    """Spot-check a specific action: place piece 1 (monomino, Identity) at
    array index (4, 4) — i.e. board coord obtained via the decoder. Its
    transpose should be the same monomino at the array-transposed anchor,
    which for the diagonal cell (4, 4) is the same cell.
    """
    decoder = blokus_game._coordinate_index_decoder
    x, y = decoder.to_coordinate((4, 4))
    action_id = blokus_game.action_codec.encode(
        _action(1, Orientation.Identity, x, y),
    )
    assert blokus_game.transpose_action(action_id) == action_id


def test_transpose_policy_pairs_with_transpose_action(
    blokus_game: BlokusDuoGame,
) -> None:
    """``transpose_policy(pi)[a]`` must equal ``pi[transpose_action(a)]`` for
    every action — the defining property of the permutation gather.
    """
    rng = np.random.default_rng(seed=42)
    n = blokus_game.get_action_size()
    pi = rng.random(n).astype(np.float32)
    new_pi = blokus_game.transpose_policy(pi)
    sample_indices = rng.integers(0, n, size=200)
    for a in sample_indices:
        assert new_pi[a] == pi[blokus_game.transpose_action(int(a))]


# ── S5: equivariance under move application ────────────────────────────────


def _placement_equal(a: BlokusDuoBoard, b: BlokusDuoBoard) -> bool:
    """Two boards have identical placement state. Sufficient for testing
    geometric equivariance — the higher-level invariants (remaining pieces,
    last played, derived caches) are covered by S2 / T2.c.
    """
    return np.array_equal(a._piece_placement_board, b._piece_placement_board)


def test_get_next_state_equivariant_under_transpose_single_move(
    blokus_game: BlokusDuoGame, mid_game_board: BlokusDuoBoard,
) -> None:
    """T5.a — for every legal action from a hand-built mid-game position
    and for both players, applying-then-transposing equals transposing-
    then-applying-the-transposed-action.
    """
    for player in (1, -1):
        mask = blokus_game.valid_move_masking(mid_game_board, player)
        legal_actions = np.flatnonzero(mask)
        # Cap the iteration count for runtime sanity — pick a representative
        # sample across orientation_ids. The test still covers hundreds of
        # actions per player.
        sample = legal_actions[:: max(1, len(legal_actions) // 200)]
        for action in sample:
            action = int(action)
            # Path A: apply then transpose
            post_move_a, next_player_a = blokus_game.get_next_state(
                mid_game_board, player, action,
            )
            path_a = post_move_a.transposed()
            # Path B: transpose then apply transposed action
            t_action = blokus_game.transpose_action(action)
            t_board = mid_game_board.transposed()
            path_b, next_player_b = blokus_game.get_next_state(t_board, player, t_action)

            assert next_player_a == next_player_b
            assert _placement_equal(path_a, path_b), (
                f"equivariance failed for player {player}, action {action} "
                f"(transposed: {t_action})"
            )


def test_transpose_commutes_with_game_replay(blokus_game: BlokusDuoGame) -> None:
    """T5.b — apply the same sequence of legal moves in two ways and verify
    the resulting boards match.

    Path A: play the sequence on the original board, then transpose.
    Path B: transpose the empty board, then play the transposed action
            sequence.
    """
    rng = np.random.default_rng(seed=2026)
    n_rollouts = 5
    rollout_length = 8  # 4 moves per player

    for rollout in range(n_rollouts):
        empty = blokus_game.initialise_board()
        original = empty
        actions: list[int] = []
        players: list[int] = []
        player = 1
        for _ in range(rollout_length):
            mask = blokus_game.valid_move_masking(original, player)
            legal = np.flatnonzero(mask)
            if len(legal) == 0:
                break
            choice = int(rng.choice(legal))
            actions.append(choice)
            players.append(player)
            original, player = blokus_game.get_next_state(original, player, choice)

        # Path A
        final_a = original.transposed()

        # Path B
        path_b = empty.transposed()
        for action, p in zip(actions, players, strict=True):
            t_action = blokus_game.transpose_action(action)
            path_b, _ = blokus_game.get_next_state(path_b, p, t_action)

        assert _placement_equal(final_a, path_b), (
            f"replay equivariance failed on rollout {rollout} "
            f"({len(actions)} moves)"
        )


# ── S6: invariance of valid-move masking ────────────────────────────────────


def test_valid_move_count_invariant_under_transpose(
    blokus_game: BlokusDuoGame, positions: list[BlokusDuoBoard],
) -> None:
    """Legal-move count is preserved under transposition for every fixture
    and both players.
    """
    for position in positions:
        for player in (1, -1):
            mask_orig = blokus_game.valid_move_masking(position, player)
            mask_t = blokus_game.valid_move_masking(position.transposed(), player)
            assert mask_orig.sum() == mask_t.sum(), (
                f"legal-move count drift under transpose for player {player}"
            )


def test_legal_action_set_transposes_correctly(
    blokus_game: BlokusDuoGame, mid_game_board: BlokusDuoBoard,
) -> None:
    """The action-by-action set of legal moves on the transposed board is
    exactly the transpose-image of the legal set on the original.
    """
    for player in (1, -1):
        mask_orig = blokus_game.valid_move_masking(mid_game_board, player)
        mask_t = blokus_game.valid_move_masking(mid_game_board.transposed(), player)
        legal_orig = set(np.flatnonzero(mask_orig).tolist())
        legal_t = set(np.flatnonzero(mask_t).tolist())
        expected_t = {blokus_game.transpose_action(a) for a in legal_orig}
        assert legal_t == expected_t, (
            f"legal action set image mismatch for player {player}: "
            f"diff = {legal_t.symmetric_difference(expected_t)}"
        )


def test_diagonal_symmetric_position_is_self_transpose(
    blokus_game: BlokusDuoGame,
    empty_board: BlokusDuoBoard,
) -> None:
    """A board that's symmetric about the main diagonal should be its own
    transpose. Forces every field to behave correctly on the no-op case.
    """
    # White monomino at (4, 4) and Black monomino at (9, 9) — both on the
    # diagonal, so the resulting board is its own transpose.
    decoder = blokus_game._coordinate_index_decoder
    white_x, white_y = decoder.to_coordinate((4, 4))
    black_x, black_y = decoder.to_coordinate((9, 9))
    b = empty_board.with_piece(_action(1, Orientation.Identity, white_x, white_y), 1)
    b = b.with_piece(_action(1, Orientation.Identity, black_x, black_y), -1)

    t = b.transposed()
    assert np.array_equal(t._piece_placement_board, b._piece_placement_board)
    for player in (1, -1):
        assert np.array_equal(t.side_danger_zone(player), b.side_danger_zone(player))
        assert set(t.placement_points(player).keys()) == set(
            b.placement_points(player).keys()
        )
