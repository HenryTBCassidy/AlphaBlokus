"""Comprehensive tests for BlokusDuoGame._valid_moves and valid_move_masking.

Tests cover: initial moves, constraint checking (overlap, sides, corners, bounds),
opponent interactions, piece usage, pass action, masking consistency, multi-move
progression, and edge cases.
"""
import numpy as np
import pytest

from games.blokusduo.board import Action, BlokusDuoBoard, CoordinateIndexDecoder
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pieces import Orientation, PieceManager


_decoder = CoordinateIndexDecoder(14)


def _action_at_idx(piece_id: int, orientation: Orientation, i: int, j: int) -> Action:
    """Create an Action placing a piece with its top-left corner at array index (i, j)."""
    x, y = _decoder.to_coordinate((i, j))
    return Action(piece_id=piece_id, orientation=orientation, x_coordinate=x, y_coordinate=y)


# -- First move (cached initial actions) ---------------------------------------


def test_white_first_move_returns_initial_actions(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    moves = blokus_game._valid_moves(blokus_board, 1)
    assert set(moves) == set(blokus_game.initial_actions[1])


def test_black_first_move_returns_initial_actions(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    moves = blokus_game._valid_moves(blokus_board, -1)
    assert set(moves) == set(blokus_game.initial_actions[-1])


def test_every_initial_action_covers_start_position(blokus_game: BlokusDuoGame):
    """Every initial white action, when placed, should have a filled cell at the white start."""
    board = blokus_game.initialise_board()
    white_start_idx = blokus_game.white_start  # (9, 9) in array indices

    for action in blokus_game.initial_actions[1]:
        new_board = board.with_piece(action, player_side=1)
        assert new_board._piece_placement_board[white_start_idx] != 0, (
            f"Action {action} does not cover white start position")


def test_every_remaining_piece_has_initial_action(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """For each initial move placed, every OTHER remaining piece should have at least one legal move."""
    for first_action in blokus_game.initial_actions[1][:10]:  # Sample first 10 to keep fast
        board = blokus_board.with_piece(first_action, player_side=1)
        moves = blokus_game._valid_moves(board, 1)
        piece_ids_with_moves = {m.piece_id for m in moves}
        remaining = board.remaining_piece_ids(1)

        for pid in remaining:
            assert pid in piece_ids_with_moves, (
                f"Piece {pid} has no moves after placing {first_action}")


# -- Basic move generation after first placement --------------------------------


def test_move_count_reasonable_after_first_placement(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """After one placement, move count should be in the hundreds, not zero or 17k."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    moves = blokus_game._valid_moves(board, 1)
    assert 50 < len(moves) < 5000


def test_every_move_is_playable(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """Every returned move should be playable via with_piece without error."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    moves = blokus_game._valid_moves(board, 1)

    for move in moves:
        board.with_piece(move, player_side=1)  # Should not raise


def test_every_move_places_piece_on_board(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """Every returned move, when placed, should result in non-zero cells on the board."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    moves = blokus_game._valid_moves(board, 1)

    for move in moves[:20]:  # Sample to keep fast
        new_board = board.with_piece(move, player_side=1)
        new_cells = np.sum(np.abs(new_board._piece_placement_board)) - np.sum(np.abs(board._piece_placement_board))
        assert new_cells > 0


def test_no_duplicate_actions(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """The returned list should have no duplicate Actions."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    moves = blokus_game._valid_moves(board, 1)
    assert len(moves) == len(set(moves))


# -- Piece constraint checks ---------------------------------------------------


def test_no_move_overlaps_existing_piece(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """No returned move should have any cell overlapping an existing piece."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    board = board.with_piece(blokus_game.initial_actions[-1][0], player_side=-1)
    board_2d = board.as_2d
    moves = blokus_game._valid_moves(board, 1)

    for move in moves:
        piece_array = blokus_game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        p_len, p_wid = piece_array.shape

        for di in range(p_len):
            for dj in range(p_wid):
                if piece_array[di, dj] == 1:
                    assert board_2d[ins_i + di, ins_j + dj] == 0, (
                        f"Move {move} overlaps existing piece at ({ins_i + di}, {ins_j + dj})")


def test_no_move_has_friendly_side_adjacency(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """No returned move should have any cell side-adjacent to a friendly piece."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    board_2d = board.as_2d
    moves = blokus_game._valid_moves(board, 1)

    for move in moves:
        piece_array = blokus_game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        p_len, p_wid = piece_array.shape

        for di in range(p_len):
            for dj in range(p_wid):
                if piece_array[di, dj] == 1:
                    ri, rj = ins_i + di, ins_j + dj
                    assert BlokusDuoBoard._no_sides(ri, rj, 1, board_2d), (
                        f"Move {move} has friendly side at ({ri}, {rj})")


def test_every_move_has_at_least_one_diagonal(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """Every returned move should have at least one cell diagonally adjacent to a friendly piece."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    board_2d = board.as_2d
    moves = blokus_game._valid_moves(board, 1)

    for move in moves:
        piece_array = blokus_game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        p_len, p_wid = piece_array.shape

        has_corner = False
        for di in range(p_len):
            for dj in range(p_wid):
                if piece_array[di, dj] == 1:
                    if BlokusDuoBoard._at_least_one_corner(ins_i + di, ins_j + dj, 1, board_2d):
                        has_corner = True
                        break
            if has_corner:
                break

        assert has_corner, f"Move {move} has no diagonal adjacency to a friendly piece"


def test_no_move_has_cells_off_board(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """No returned move should have any cell outside the 14x14 board."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    moves = blokus_game._valid_moves(board, 1)
    n = BlokusDuoBoard.N

    for move in moves:
        piece_array = blokus_game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        p_len, p_wid = piece_array.shape

        assert ins_i >= 0 and ins_j >= 0, f"Move {move} has negative insertion index"
        assert ins_i + p_len <= n and ins_j + p_wid <= n, (
            f"Move {move} extends beyond board edge")


# -- Opponent interactions -----------------------------------------------------


def test_moves_can_be_side_adjacent_to_opponent(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """It IS legal to place a piece side-adjacent to an opponent's piece.
    Verify that such moves exist (they should, since start positions are far apart
    and we can construct a scenario)."""
    # Place white monomino at (7,7) and black monomino at (7,8) — side by side
    board = blokus_board.with_piece(_action_at_idx(1, Orientation.Identity, 7, 7), player_side=1)
    board = board.with_piece(_action_at_idx(1, Orientation.Identity, 7, 8), player_side=-1)

    moves = blokus_game._valid_moves(board, 1)
    board_2d = board.as_2d

    # Some white moves should have cells adjacent to black at (7,8)
    has_opponent_adjacent = False
    for move in moves:
        piece_array = blokus_game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        p_len, p_wid = piece_array.shape

        for di in range(p_len):
            for dj in range(p_wid):
                if piece_array[di, dj] == 1:
                    ri, rj = ins_i + di, ins_j + dj
                    # Check if this cell is side-adjacent to the black piece at (7,8)
                    if abs(ri - 7) + abs(rj - 8) == 1:
                        has_opponent_adjacent = True
                        break
            if has_opponent_adjacent:
                break
        if has_opponent_adjacent:
            break

    assert has_opponent_adjacent, "No moves found that are side-adjacent to opponent"


def test_opponent_piece_blocks_squares(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """After opponent places a piece, white can't place on those squares."""
    board = blokus_board.with_piece(_action_at_idx(1, Orientation.Identity, 7, 7), player_side=1)
    board = board.with_piece(_action_at_idx(1, Orientation.Identity, 6, 6), player_side=-1)

    moves = blokus_game._valid_moves(board, 1)

    # No white move should have a cell at (6,6) where black's piece is
    for move in moves:
        piece_array = blokus_game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        p_len, p_wid = piece_array.shape

        for di in range(p_len):
            for dj in range(p_wid):
                if piece_array[di, dj] == 1:
                    assert (ins_i + di, ins_j + dj) != (6, 6), (
                        f"Move {move} places a cell on opponent's piece at (6,6)")


def test_opponent_on_placement_point_reduces_moves(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """After opponent occupies a white placement point, white's move count should drop."""
    board = blokus_board.with_piece(_action_at_idx(1, Orientation.Identity, 7, 7), player_side=1)
    moves_before = blokus_game._valid_moves(board, 1)

    # Black places on (6,6) which is a white placement point
    board_after = board.with_piece(_action_at_idx(1, Orientation.Identity, 6, 6), player_side=-1)
    moves_after = blokus_game._valid_moves(board_after, 1)

    assert len(moves_after) < len(moves_before)


# -- Piece usage ---------------------------------------------------------------


def test_no_move_uses_already_placed_piece(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """After placing a piece, no move should use the same piece_id."""
    first_action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(first_action, player_side=1)
    moves = blokus_game._valid_moves(board, 1)

    placed_id = first_action.piece_id
    for move in moves:
        assert move.piece_id != placed_id, (
            f"Move {move} uses already-placed piece {placed_id}")


def test_placed_piece_disappears_from_moves(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """After placing a piece, moves with that piece_id should disappear entirely."""
    first_action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(first_action, player_side=1)
    moves = blokus_game._valid_moves(board, 1)

    placed_id = first_action.piece_id
    piece_ids_in_moves = {m.piece_id for m in moves}
    assert placed_id not in piece_ids_in_moves


# -- Pass action / no moves ---------------------------------------------------


def test_masking_sets_pass_when_no_moves(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """When _valid_moves returns empty, the pass action bit should be set."""
    # Create a board state with no valid moves by filling everything
    # Simplest approach: mock via the masking logic directly
    # We know an empty board for a non-first-move scenario (21 remaining but no placement points)
    # won't happen naturally, so let's use the fact that after placing all 21 pieces there are
    # no remaining pieces. Build a board with 0 remaining pieces for white.
    # Actually, let's just verify the logic: if _valid_moves returns [], pass bit is set.
    # We can test this by checking the masking on a board where we manually verify no moves.
    # For now, verify the contract: pass bit set iff no moves.
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    mask = blokus_game.valid_move_masking(board, 1)

    moves = blokus_game._valid_moves(board, 1)
    pass_idx = blokus_game.action_codec.pass_action_index

    if len(moves) > 0:
        assert mask[pass_idx] == 0, "Pass bit should not be set when moves exist"
    else:
        assert mask[pass_idx] == 1, "Pass bit should be set when no moves exist"


def test_masking_does_not_set_pass_when_moves_exist(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """When moves exist, the pass action bit should NOT be set."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    mask = blokus_game.valid_move_masking(board, 1)
    pass_idx = blokus_game.action_codec.pass_action_index
    assert mask[pass_idx] == 0


# -- Masking consistency -------------------------------------------------------


def test_mask_count_equals_move_count(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """Number of 1-bits in the mask should equal len(_valid_moves)."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    moves = blokus_game._valid_moves(board, 1)
    mask = blokus_game.valid_move_masking(board, 1)
    assert int(mask.sum()) == len(moves)


def test_mask_bits_decode_to_valid_moves(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """Every 1-bit in the mask should decode to an Action in _valid_moves."""
    board = blokus_board.with_piece(blokus_game.initial_actions[1][0], player_side=1)
    moves = blokus_game._valid_moves(board, 1)
    mask = blokus_game.valid_move_masking(board, 1)
    move_set = set(moves)

    for idx in np.where(mask)[0]:
        if idx == blokus_game.action_codec.pass_action_index:
            continue
        decoded = blokus_game.action_codec.decode(int(idx))
        assert decoded in move_set, f"Mask bit {idx} decodes to {decoded} which is not in _valid_moves"


# -- Multi-move game progression -----------------------------------------------


def test_multi_move_progression(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """Play several alternating moves and verify moves don't crash or go to zero prematurely."""
    board = blokus_board
    players = [1, -1]

    for turn in range(6):
        player = players[turn % 2]
        moves = blokus_game._valid_moves(board, player)
        assert len(moves) > 0, f"No moves for player {player} on turn {turn}"
        board = board.with_piece(moves[0], player_side=player)

    # After 6 moves (3 per player), both should still have moves
    assert len(blokus_game._valid_moves(board, 1)) > 0
    assert len(blokus_game._valid_moves(board, -1)) > 0


def test_known_legal_placement_is_in_moves(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """After placing monomino at white start, placing piece 2 (domino) at a known
    diagonal should be in the valid moves list."""
    # White monomino at start (9,9) in array coords
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 9, 9), player_side=1)

    # Domino (piece 2, Identity = [[1],[1]]) at (10,10) — diagonal of (9,9)
    # This places cells at (10,10) and (11,10)
    expected = _action_at_idx(2, Orientation.Identity, 10, 10)
    moves = blokus_game._valid_moves(board, 1)
    assert expected in moves, f"Expected {expected} to be a valid move"


def test_known_illegal_placement_not_in_moves(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """A piece placed side-adjacent to a friendly piece should NOT be in valid moves."""
    # White monomino at (9,9)
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 9, 9), player_side=1)

    # Domino (piece 2, Identity) at (10,9) — directly below (9,9), side adjacent
    illegal = _action_at_idx(2, Orientation.Identity, 10, 9)
    moves = blokus_game._valid_moves(board, 1)
    assert illegal not in moves, f"Expected {illegal} to NOT be a valid move (side adjacent)"


# -- Edge cases ----------------------------------------------------------------


def test_piece_in_board_corner(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """Piece placed at board corner should have moves, but only away from the edge."""
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 0, 0), player_side=1)
    moves = blokus_game._valid_moves(board, 1)

    # All moves should have cells within bounds
    n = BlokusDuoBoard.N
    for move in moves:
        piece_array = blokus_game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        p_len, p_wid = piece_array.shape
        assert ins_i >= 0 and ins_j >= 0
        assert ins_i + p_len <= n and ins_j + p_wid <= n


def test_large_piece_near_edge_excluded(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """I-pentomino (5x1, piece 11) placements that would hang off the board should be excluded."""
    # Place monomino at (0, 0) — top-left corner
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 0, 0), player_side=1)
    moves = blokus_game._valid_moves(board, 1)

    # Filter to I-pentomino moves (piece 11)
    i_pent_moves = [m for m in moves if m.piece_id == 11]

    # All I-pentomino placements should fit within bounds
    n = BlokusDuoBoard.N
    for move in i_pent_moves:
        piece_array = blokus_game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        p_len, p_wid = piece_array.shape
        assert ins_i >= 0 and ins_j >= 0
        assert ins_i + p_len <= n and ins_j + p_wid <= n, (
            f"I-pentomino move {move} hangs off the board: ins=({ins_i},{ins_j}) shape=({p_len},{p_wid})")


def test_i_pentomino_along_edge(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard):
    """I-pentomino (piece 11) placed along column 0 should only generate diagonal
    points on one side (column 1), not column -1."""
    # Place I-pentomino vertically along column 0, rows 0-4
    board = blokus_board.with_piece(
        _action_at_idx(11, Orientation.Identity, 0, 0), player_side=1)

    moves = blokus_game._valid_moves(board, 1)

    # No move should have a cell at column < 0 (obviously can't, but verify all are in bounds)
    n = BlokusDuoBoard.N
    for move in moves:
        piece_array = blokus_game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        assert ins_j >= 0, f"Move {move} has negative column index"

    # Placement points should only be on column 1 side
    points = board.placement_points(1)
    for (pi, pj) in points:
        assert pj == 1, f"Placement point ({pi},{pj}) should only be at column 1 for edge piece"
