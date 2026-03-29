"""Tests for the placement point cache maintained by BlokusDuoBoard.

Placement points are board coordinates (array indices) where a player could
potentially place a piece — i.e. empty squares diagonally adjacent to a
friendly piece and not side-adjacent to a friendly piece.

All positions in these tests use array indices (i=row, j=col, top-left origin).
Board coordinates (x, y, bottom-left origin) are computed via the decoder.
"""
import numpy as np

from games.blokusduo.board import Action, BlokusDuoBoard, CoordinateIndexDecoder
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pieces import Orientation, PieceManager

# Helper: convert array index (i, j) to board coordinates (x, y)
_decoder = CoordinateIndexDecoder(14)


def _action_at_idx(piece_id: int, orientation: Orientation, i: int, j: int) -> Action:
    """Create an Action that places a piece with its top-left corner at array index (i, j)."""
    x, y = _decoder.to_coordinate((i, j))
    return Action(piece_id=piece_id, orientation=orientation, x_coordinate=x, y_coordinate=y)


# -- Basic placement point creation -------------------------------------------


def test_empty_board_has_no_placement_points(blokus_board: BlokusDuoBoard):
    """Fresh board should have empty placement caches for both players."""
    assert blokus_board.placement_points(1) == {}
    assert blokus_board.placement_points(-1) == {}


def test_monomino_creates_four_diagonal_points(blokus_board: BlokusDuoBoard):
    """Placing a 1x1 piece in the middle should create exactly 4 diagonal placement points."""
    action = _action_at_idx(piece_id=1, orientation=Orientation.Identity, i=7, j=7)
    board = blokus_board.with_piece(action, player_side=1)

    white_points = set(board.placement_points(1).keys())
    expected = {(6, 6), (6, 8), (8, 6), (8, 8)}
    assert white_points == expected


def test_domino_creates_correct_points(blokus_board: BlokusDuoBoard):
    """Placing a 2x1 vertical domino should create 6 diagonal placement points.

    Domino (piece 2, Identity) occupies (7,7) and (8,7):
        Diagonals: (6,6), (6,8), (7,6)*, (7,8)*, (9,6), (9,8)
        * (7,6) is side-adjacent to (8,7)? No — (7,6) is diagonal to (8,7) and side to... wait.
        Actually (7,6) is side-adjacent to (7,7) — left neighbor. So NOT valid.
        And (8,6) is side-adjacent to (8,7). So NOT valid.
        Similarly (7,8) is side-adjacent to (7,7). NOT valid.
        And (8,8) is side-adjacent to (8,7). NOT valid.

    Correct diagonals that pass both checks:
        (6,6): corner of (7,7), no sides → ✓
        (6,8): corner of (7,7), no sides → ✓
        (9,6): corner of (8,7), no sides → ✓
        (9,8): corner of (8,7), no sides → ✓
    """
    action = _action_at_idx(piece_id=2, orientation=Orientation.Identity, i=7, j=7)
    board = blokus_board.with_piece(action, player_side=1)

    white_points = set(board.placement_points(1).keys())
    expected = {(6, 6), (6, 8), (9, 6), (9, 8)}
    assert white_points == expected


def test_l_tromino_creates_correct_points(blokus_board: BlokusDuoBoard):
    """Placing the L-tromino (piece 3, Identity) which looks like:
        [[1, 0],
         [1, 1]]
    Placed at (7,7) occupies: (7,7), (8,7), (8,8).

    Diagonals to check:
        (6,6): corner of (7,7), not side of anything → ✓
        (6,8): corner of (7,7), side of (7,7)? No, (7,7) is at col 7 not 8.
               side of (8,8)? No — (6,8) is not adjacent to (8,8). → ✓ wait...
               Actually (6,8) is corner of (7,7) — j differs by 1, i differs by 1 ✓
               Side check: is any friendly piece at (5,8),(7,8),(6,7),(6,9)? (7,8) is not occupied. → ✓
        (7,6): side of (7,7) — j=6 is left of j=7. → ✗ (side adjacent)
        (7,9): corner of (8,8)? i=7, j=9: corner would need (8,8) — diff is (1,1) ✓
               Side check: (7,8) is side? (7,8) is not occupied → ✓
               But wait: is (7,9) side-adjacent to (7,7)? No, j differs by 2. → ✓
        (8,6): side of (8,7) → ✗
        (8,9): side of (8,8) → ✗ wait... (8,9) is to the right of (8,8), so side-adjacent → ✗
               Actually side means orthogonal neighbor: (8,9) is right of (8,8) → ✗
        (9,6): corner of (8,7), not side of anything occupied → ✓
        (9,8): side of (8,8) — (9,8) is below (8,8) → ✗
               Wait: (9,8) row=9, (8,8) row=8. |9-8|=1, |8-8|=0. That's a side. → ✗
        (9,9): corner of (8,8), side check: (8,9) not occupied, (9,8) not occupied → ✓
    """
    action = _action_at_idx(piece_id=3, orientation=Orientation.Identity, i=7, j=7)
    board = blokus_board.with_piece(action, player_side=1)

    white_points = set(board.placement_points(1).keys())
    expected = {(6, 6), (6, 8), (7, 9), (9, 6), (9, 9)}
    assert white_points == expected


# -- Edge and corner cases ----------------------------------------------------


def test_monomino_at_corner_creates_one_point(blokus_board: BlokusDuoBoard):
    """Placing a 1x1 piece at array index (0, 0) — only (1, 1) is a valid diagonal."""
    action = _action_at_idx(piece_id=1, orientation=Orientation.Identity, i=0, j=0)
    board = blokus_board.with_piece(action, player_side=1)

    white_points = set(board.placement_points(1).keys())
    assert white_points == {(1, 1)}


def test_monomino_at_edge_creates_two_points(blokus_board: BlokusDuoBoard):
    """Placing a 1x1 piece at (0, 7) — top edge, two diagonals below."""
    action = _action_at_idx(piece_id=1, orientation=Orientation.Identity, i=0, j=7)
    board = blokus_board.with_piece(action, player_side=1)

    white_points = set(board.placement_points(1).keys())
    assert white_points == {(1, 6), (1, 8)}


def test_monomino_at_board_corner_13_13(blokus_board: BlokusDuoBoard):
    """Placing at (13, 13) — bottom-right corner, only (12, 12) is valid."""
    action = _action_at_idx(piece_id=1, orientation=Orientation.Identity, i=13, j=13)
    board = blokus_board.with_piece(action, player_side=1)

    white_points = set(board.placement_points(1).keys())
    assert white_points == {(12, 12)}


# -- Invalidation by own piece -----------------------------------------------


def test_side_adjacent_piece_removes_placement_point(blokus_board: BlokusDuoBoard):
    """Place white monomino at (7,7) creating point at (6,6). Then place another
    white piece whose side touches (6,6) — the point should be removed."""
    # First piece: monomino at (7,7) → points at (6,6), (6,8), (8,6), (8,8)
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 7, 7), player_side=1)
    assert (6, 6) in board.placement_points(1)

    # Second piece: monomino at (6,5) — this is side-adjacent to (6,6)
    # (6,5) is left of (6,6), so (6,6) now has a friendly side
    board = board.with_piece(
        _action_at_idx(4, Orientation.Identity, 6, 5), player_side=1)
    assert (6, 6) not in board.placement_points(1)


def test_piece_placed_on_placement_point_removes_it(blokus_board: BlokusDuoBoard):
    """Place a piece directly on an existing placement point — it should be removed."""
    # Monomino at (7,7) → point at (6,6)
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 7, 7), player_side=1)
    assert (6, 6) in board.placement_points(1)

    # Place another white piece at (6,6) — occupies the placement point
    board = board.with_piece(
        _action_at_idx(4, Orientation.Identity, 6, 6), player_side=1)
    assert (6, 6) not in board.placement_points(1)


# -- Opponent interactions ----------------------------------------------------


def test_opponent_piece_removes_placement_point_by_occupation(blokus_board: BlokusDuoBoard):
    """If black places a piece on a white placement point, white loses that point."""
    # White monomino at (7,7) → white points at (6,6), (6,8), (8,6), (8,8)
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 7, 7), player_side=1)
    assert (6, 6) in board.placement_points(1)

    # Black places on (6,6)
    board = board.with_piece(
        _action_at_idx(1, Orientation.Identity, 6, 6), player_side=-1)
    assert (6, 6) not in board.placement_points(1)


def test_opponent_side_does_not_remove_placement_point(blokus_board: BlokusDuoBoard):
    """A black piece side-adjacent to a white placement point should NOT remove it.
    Only friendly sides invalidate placement points."""
    # White monomino at (7,7) → white points at (6,6), (6,8), (8,6), (8,8)
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 7, 7), player_side=1)
    assert (6, 6) in board.placement_points(1)

    # Black places at (6,5) — side-adjacent to white's placement point (6,6)
    # This should NOT invalidate (6,6) for white
    board = board.with_piece(
        _action_at_idx(4, Orientation.Identity, 6, 5), player_side=-1)
    assert (6, 6) in board.placement_points(1)


# -- Two players independent --------------------------------------------------


def test_players_have_independent_placement_points(blokus_board: BlokusDuoBoard):
    """Each player's placement points should only reflect their own pieces."""
    # White monomino at (7,7)
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 7, 7), player_side=1)

    # Black monomino at (3,3)
    board = board.with_piece(
        _action_at_idx(1, Orientation.Identity, 3, 3), player_side=-1)

    white_points = set(board.placement_points(1).keys())
    black_points = set(board.placement_points(-1).keys())

    # White's points should be around (7,7)
    assert white_points == {(6, 6), (6, 8), (8, 6), (8, 8)}

    # Black's points should be around (3,3)
    assert black_points == {(2, 2), (2, 4), (4, 2), (4, 4)}

    # No overlap
    assert white_points.isdisjoint(black_points)


# -- Placement point merging with multiple pieces -----------------------------


def test_second_piece_adds_new_points_and_preserves_existing(blokus_board: BlokusDuoBoard):
    """Placing a second piece far from the first should add new placement points
    while preserving all existing points from the first piece."""
    # White monomino at (2,2) → points at (1,1), (1,3), (3,1), (3,3)
    board = blokus_board.with_piece(
        _action_at_idx(1, Orientation.Identity, 2, 2), player_side=1)
    first_points = set(board.placement_points(1).keys())
    assert first_points == {(1, 1), (1, 3), (3, 1), (3, 3)}

    # White monomino at (10,10) — far away, no interference
    board = board.with_piece(
        _action_at_idx(2, Orientation.Identity, 10, 10), player_side=1)
    white_points = set(board.placement_points(1).keys())

    # All original points preserved
    assert first_points.issubset(white_points)
    # New points appeared around (10,10)
    assert (9, 9) in white_points
    assert (9, 11) in white_points
    # (10,10) is occupied, not a point
    assert (10, 10) not in white_points
