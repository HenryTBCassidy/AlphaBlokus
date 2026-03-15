import numpy as np
import pytest

from games.blokusduo.board import BlokusDuoBoard, CoordinateIndexDecoder
from games.blokusduo.game import BlokusDuoGame


@pytest.fixture
def decoder() -> CoordinateIndexDecoder:
    return CoordinateIndexDecoder(14)


def test_coordinate_to_idx_roundtrip(decoder: CoordinateIndexDecoder):
    """to_coordinate(to_idx(c)) should equal c for valid coordinates."""
    test_points = [(0, 0), (0, 13), (13, 0), (13, 13), (7, 7), (3, 10)]
    for coord in test_points:
        idx = decoder.to_idx(coord)
        recovered = decoder.to_coordinate(idx)
        assert recovered == coord, f"Roundtrip failed for {coord}: got {recovered}"


def test_coordinate_corners(decoder: CoordinateIndexDecoder):
    """Corner coordinates should map to correct array indices."""
    assert decoder.to_idx((0, 0)) == (13, 0)
    assert decoder.to_idx((13, 13)) == (0, 13)
    assert decoder.to_idx((0, 13)) == (0, 0)
    assert decoder.to_idx((13, 0)) == (13, 13)


# -- Placement validation tests ------------------------------------------------


def test_valid_placement_empty_board():
    """Centre of an empty board has no corners touching, so should not be valid."""
    board = np.zeros((14, 14), dtype=np.int8)
    assert not BlokusDuoBoard._valid_placement(7, 7, 1, board)


def test_no_sides_empty_board():
    """On an empty board, no sides should be found."""
    board = np.zeros((14, 14), dtype=np.int8)
    assert BlokusDuoBoard._no_sides(7, 7, 1, board)


def test_no_sides_with_adjacent_piece():
    """A friendly piece directly above should fail the no_sides check."""
    board = np.zeros((14, 14), dtype=np.int8)
    board[6, 7] = 1
    assert not BlokusDuoBoard._no_sides(7, 7, 1, board)


def test_at_least_one_corner_empty():
    """On an empty board, no corner should be found."""
    board = np.zeros((14, 14), dtype=np.int8)
    assert not BlokusDuoBoard._at_least_one_corner(7, 7, 1, board)


def test_at_least_one_corner_with_diagonal():
    """A friendly piece on the diagonal should satisfy the corner check."""
    board = np.zeros((14, 14), dtype=np.int8)
    board[6, 6] = 1
    assert BlokusDuoBoard._at_least_one_corner(7, 7, 1, board)


# -- Boundary bug fix tests ----------------------------------------------------


def test_no_sides_row0():
    """_no_sides should detect a friendly piece at row 0."""
    board = np.zeros((14, 14), dtype=np.int8)
    board[0, 5] = 1
    assert not BlokusDuoBoard._no_sides(1, 5, 1, board)


def test_no_sides_col0():
    """_no_sides should detect a friendly piece at col 0."""
    board = np.zeros((14, 14), dtype=np.int8)
    board[5, 0] = 1
    assert not BlokusDuoBoard._no_sides(5, 1, 1, board)


def test_corner_row0():
    """_at_least_one_corner should detect a diagonal piece at row 0."""
    board = np.zeros((14, 14), dtype=np.int8)
    board[0, 0] = 1
    assert BlokusDuoBoard._at_least_one_corner(1, 1, 1, board)


def test_corner_row13():
    """_at_least_one_corner should detect a diagonal piece at row 13."""
    board = np.zeros((14, 14), dtype=np.int8)
    board[13, 13] = 1
    assert BlokusDuoBoard._at_least_one_corner(12, 12, 1, board)


# -- IBoard protocol tests -----------------------------------------------------


def test_as_2d_returns_correct_shape(blokus_board: BlokusDuoBoard):
    """as_2d should return a (14, 14) array."""
    assert blokus_board.as_2d.shape == (14, 14)
    assert np.all(blokus_board.as_2d == 0)


def test_num_channels(blokus_board: BlokusDuoBoard):
    """num_channels should return 44."""
    assert blokus_board.num_channels == 44


def test_as_multi_channel_empty_board(blokus_board: BlokusDuoBoard):
    """All 44 channels should be zero on an empty board."""
    rep = blokus_board.as_multi_channel(1)
    assert rep.shape == (44, 14, 14)
    assert rep.dtype == np.float32
    assert np.all(rep == 0)


# -- Placement board & dtype tests ---------------------------------------------


def test_placement_board_dtype(blokus_board: BlokusDuoBoard):
    """Placement board should use int8 dtype."""
    assert blokus_board._piece_placement_board.dtype == np.int8


def test_placement_board_signed_ids(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """White pieces should have +id, black pieces should have -id."""
    white_action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(white_action, player_side=1)

    black_action = blokus_game.initial_actions[-1][0]
    board = board.with_piece(black_action, player_side=-1)

    ppb = board._piece_placement_board
    white_mask = ppb > 0
    assert np.any(white_mask)
    assert np.all(ppb[white_mask] == white_action.piece_id)

    black_mask = ppb < 0
    assert np.any(black_mask)
    assert np.all(ppb[black_mask] == -black_action.piece_id)


def test_as_2d_matches_sign_of_placement(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """as_2d should equal np.sign(_piece_placement_board)."""
    white_action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(white_action, player_side=1)

    expected = np.sign(board._piece_placement_board).astype(np.int8)
    np.testing.assert_array_equal(board.as_2d, expected)


# -- Immutability tests ---------------------------------------------------------


def test_with_piece_returns_new_board(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """with_piece() should return a different object."""
    action = blokus_game.initial_actions[1][0]
    new_board = blokus_board.with_piece(action, player_side=1)
    assert new_board is not blokus_board


def test_with_piece_is_independent(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """Placing on new board should not affect the original."""
    action = blokus_game.initial_actions[1][0]
    new_board = blokus_board.with_piece(action, player_side=1)

    assert np.all(blokus_board._piece_placement_board == 0)
    assert np.any(new_board._piece_placement_board != 0)


def test_with_piece_updates_remaining(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """Placed piece ID should be removed from remaining set."""
    action = blokus_game.initial_actions[1][0]
    new_board = blokus_board.with_piece(action, player_side=1)

    assert action.piece_id in blokus_board._white_piece_ids_remaining
    assert action.piece_id not in new_board._white_piece_ids_remaining
    assert new_board._black_piece_ids_remaining == blokus_board._black_piece_ids_remaining


# -- State key tests ------------------------------------------------------------


def test_state_key_deterministic(blokus_board: BlokusDuoBoard):
    """Same board state should produce same key."""
    assert blokus_board.state_key == blokus_board.state_key


def test_state_key_different_after_move(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """Different board states should produce different keys."""
    action = blokus_game.initial_actions[1][0]
    new_board = blokus_board.with_piece(action, player_side=1)
    assert blokus_board.state_key != new_board.state_key


def test_state_key_distinguishes_pieces(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """Two boards with different piece IDs at same position should have different keys."""
    action1 = blokus_game.initial_actions[1][0]
    action2 = blokus_game.initial_actions[1][1]

    if action1.piece_id != action2.piece_id:
        board1 = blokus_board.with_piece(action1, player_side=1)
        board2 = blokus_board.with_piece(action2, player_side=1)
        assert board1.state_key != board2.state_key


def test_state_key_matches_game_delegate(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """game.state_key(board) should match board.state_key."""
    assert blokus_game.state_key(blokus_board) == blokus_board.state_key


# -- Multi-channel encoding tests -----------------------------------------------


def test_as_multi_channel_after_placement(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """After placing a white piece, the correct per-piece channel should be non-zero."""
    action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(action, player_side=1)

    rep = board.as_multi_channel(1)

    piece_channel = rep[action.piece_id - 1]
    assert np.sum(piece_channel) > 0

    assert np.array_equal(rep[42], (rep[0:21].sum(axis=0) > 0).astype(np.float32))

    assert np.all(rep[21:42] == 0)
    assert np.all(rep[43] == 0)


def test_as_multi_channel_channels_mutually_exclusive(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """Each square should appear in at most one of channels 0-20."""
    action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(action, player_side=1)

    rep = board.as_multi_channel(1)

    current_sum = rep[0:21].sum(axis=0)
    assert np.all(current_sum <= 1)

    opponent_sum = rep[21:42].sum(axis=0)
    assert np.all(opponent_sum <= 1)


def test_as_multi_channel_aggregates_match_union(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """Aggregate channels should equal the union of per-piece planes."""
    white_action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(white_action, player_side=1)
    black_action = blokus_game.initial_actions[-1][0]
    board = board.with_piece(black_action, player_side=-1)

    rep = board.as_multi_channel(1)

    expected_current = (rep[0:21].sum(axis=0) > 0).astype(np.float32)
    np.testing.assert_array_equal(rep[42], expected_current)

    expected_opponent = (rep[21:42].sum(axis=0) > 0).astype(np.float32)
    np.testing.assert_array_equal(rep[43], expected_opponent)


def test_as_multi_channel_perspective_swap(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """as_multi_channel(1) and as_multi_channel(-1) should have swapped channel groups."""
    white_action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(white_action, player_side=1)
    black_action = blokus_game.initial_actions[-1][0]
    board = board.with_piece(black_action, player_side=-1)

    rep_white = board.as_multi_channel(1)
    rep_black = board.as_multi_channel(-1)

    np.testing.assert_array_equal(rep_white[0:21], rep_black[21:42])
    np.testing.assert_array_equal(rep_white[21:42], rep_black[0:21])
    np.testing.assert_array_equal(rep_white[42], rep_black[43])
    np.testing.assert_array_equal(rep_white[43], rep_black[42])


# -- Canonical form tests -------------------------------------------------------


def test_canonical_player1_is_self(blokus_board: BlokusDuoBoard):
    """canonical(1) should return self (identity)."""
    assert blokus_board.canonical(1) is blokus_board


def test_canonical_player_neg1_swaps_perspective(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
):
    """canonical(-1) should produce a board where as_multi_channel(1) shows black's perspective."""
    white_action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(white_action, player_side=1)

    canonical = board.canonical(-1)

    rep = canonical.as_multi_channel(1)
    assert np.all(rep[0:21] == 0), "Black has no pieces yet"
    assert np.sum(rep[21:42]) > 0, "White's piece should appear as opponent"

    assert np.all(canonical.as_2d == -board.as_2d)
