"""Tests for PieceManager cached orientation arrays and filled cell indices."""
import numpy as np

from games.blokusduo.pieces import Orientation, PieceManager


def test_cached_arrays_match_piece_properties(piece_manager: PieceManager):
    """Cached arrays should match the values from Piece properties."""
    for piece in piece_manager.pieces.values():
        base = np.array(piece.fill_values)
        expected = {
            Orientation.Identity: base,
            Orientation.Rot90: np.rot90(base),
            Orientation.Rot180: np.rot90(base, 2),
            Orientation.Rot270: np.rot90(base, 3),
            Orientation.Flip: np.flip(base, axis=1),
            Orientation.Flip90: np.rot90(np.flip(base, axis=1)),
            Orientation.Flip180: np.rot90(np.flip(base, axis=1), 2),
            Orientation.Flip270: np.rot90(np.flip(base, axis=1), 3),
        }
        for orientation in piece.basis_orientations:
            cached = piece_manager.get_piece_orientation_array(piece.id, orientation)
            np.testing.assert_array_equal(cached, expected[orientation],
                err_msg=f"Piece {piece.id} {orientation} cache mismatch")


def test_cached_arrays_are_immutable(piece_manager: PieceManager):
    """Cached arrays should not be writable to prevent accidental mutation."""
    for piece in piece_manager.pieces.values():
        for orientation in piece.basis_orientations:
            arr = piece_manager.get_piece_orientation_array(piece.id, orientation)
            assert not arr.flags.writeable, (
                f"Piece {piece.id} {orientation} array should be immutable")


def test_cached_arrays_are_same_object_on_repeat_access(piece_manager: PieceManager):
    """Repeated calls should return the exact same array object (no allocation)."""
    piece = piece_manager.pieces[1]
    arr1 = piece_manager.get_piece_orientation_array(1, Orientation.Identity)
    arr2 = piece_manager.get_piece_orientation_array(1, Orientation.Identity)
    assert arr1 is arr2


def test_filled_cells_match_array(piece_manager: PieceManager):
    """Pre-computed filled cells should match np.argwhere on the cached array."""
    for piece in piece_manager.pieces.values():
        for orientation in piece.basis_orientations:
            arr = piece_manager.get_piece_orientation_array(piece.id, orientation)
            filled = piece_manager.get_filled_cells(piece.id, orientation)
            expected = np.argwhere(arr != 0)
            np.testing.assert_array_equal(filled, expected,
                err_msg=f"Piece {piece.id} {orientation} filled cells mismatch")


def test_filled_cells_count_matches_piece_size(piece_manager: PieceManager):
    """Number of filled cells should equal the number of 1s in the piece."""
    for piece in piece_manager.pieces.values():
        expected_count = int(np.array(piece.fill_values).sum())
        for orientation in piece.basis_orientations:
            filled = piece_manager.get_filled_cells(piece.id, orientation)
            assert len(filled) == expected_count, (
                f"Piece {piece.id} {orientation}: expected {expected_count} filled cells, got {len(filled)}")


def test_all_91_orientations_cached(piece_manager: PieceManager):
    """All 91 piece-orientation combinations should be cached."""
    count = 0
    for piece in piece_manager.pieces.values():
        for orientation in piece.basis_orientations:
            _ = piece_manager.get_piece_orientation_array(piece.id, orientation)
            _ = piece_manager.get_filled_cells(piece.id, orientation)
            count += 1
    assert count == 91
