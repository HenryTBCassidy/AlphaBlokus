import numpy as np
import pytest

from games.blokusduo.game import (
    CoordinateIndexDecoder,
    at_least_one_corner,
    no_sides,
    valid_placement,
)


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
    # Bottom-left coordinate (0, 0) → array index (13, 0)
    assert decoder.to_idx((0, 0)) == (13, 0)
    # Top-right coordinate (13, 13) → array index (0, 13)
    assert decoder.to_idx((13, 13)) == (0, 13)
    # Top-left coordinate (0, 13) → array index (0, 0)
    assert decoder.to_idx((0, 13)) == (0, 0)
    # Bottom-right coordinate (13, 0) → array index (13, 13)
    assert decoder.to_idx((13, 0)) == (13, 13)


def test_valid_placement_empty_board():
    """Centre of an empty board has no corners touching, so should not be valid."""
    board = np.zeros((14, 14), dtype=int)
    # No friendly pieces adjacent, so at_least_one_corner will be False
    assert not valid_placement(7, 7, 1, board)


def test_no_sides_empty_board():
    """On an empty board, no sides should be found."""
    board = np.zeros((14, 14), dtype=int)
    assert no_sides(7, 7, 1, board)


def test_no_sides_with_adjacent_piece():
    """A friendly piece directly above should fail the no_sides check."""
    board = np.zeros((14, 14), dtype=int)
    board[6, 7] = 1  # Piece directly above (7, 7)
    assert not no_sides(7, 7, 1, board)


def test_at_least_one_corner_empty():
    """On an empty board, no corner should be found."""
    board = np.zeros((14, 14), dtype=int)
    assert not at_least_one_corner(7, 7, 1, board)


def test_at_least_one_corner_with_diagonal():
    """A friendly piece on the diagonal should satisfy the corner check."""
    board = np.zeros((14, 14), dtype=int)
    board[6, 6] = 1  # Piece at top-left diagonal of (7, 7)
    assert at_least_one_corner(7, 7, 1, board)
