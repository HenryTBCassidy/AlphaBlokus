import numpy as np

from games.blokusduo.pieces import Orientation, Piece, PieceManager, pieces_loader


def test_pieces_loader_count(pieces_path):
    pm = pieces_loader(pieces_path)
    assert len(pm.pieces) == 21


def test_pieces_loader_orientations(pieces_path):
    pm = pieces_loader(pieces_path)
    total = sum(len(p.basis_orientations) for p in pm.pieces.values())
    assert total == 91


def test_piece_manager_lookup_size(piece_manager: PieceManager):
    assert piece_manager.num_entries == 91


def test_piece_identity_shape(piece_manager: PieceManager):
    for piece in piece_manager.pieces.values():
        identity = piece.identity
        assert identity.ndim == 2
        assert identity.shape[0] > 0
        assert identity.shape[1] > 0
        # Only 0s and 1s
        assert set(np.unique(identity)).issubset({0, 1})


def test_piece_rotations_shape_consistent(piece_manager: PieceManager):
    """Rot90 swaps rows and cols relative to identity."""
    for piece in piece_manager.pieces.values():
        h, w = piece.identity.shape
        assert piece.rot90.shape == (w, h)
        assert piece.rot180.shape == (h, w)
        assert piece.rot270.shape == (w, h)


def test_piece_flip_preserves_cell_count(piece_manager: PieceManager):
    """All orientations should have the same number of filled cells."""
    for piece in piece_manager.pieces.values():
        count = np.sum(piece.identity)
        assert np.sum(piece.rot90) == count
        assert np.sum(piece.rot180) == count
        assert np.sum(piece.rot270) == count
        assert np.sum(piece.flip) == count
        assert np.sum(piece.flip90) == count
        assert np.sum(piece.flip180) == count
        assert np.sum(piece.flip270) == count


def test_all_orientations_produce_arrays(piece_manager: PieceManager):
    """Every orientation property on every piece returns a valid ndarray."""
    all_orientations = [
        Orientation.Identity, Orientation.Rot90, Orientation.Rot180, Orientation.Rot270,
        Orientation.Flip, Orientation.Flip90, Orientation.Flip180, Orientation.Flip270,
    ]
    for piece in piece_manager.pieces.values():
        for orient in all_orientations:
            arr = piece_manager.get_piece_orientation_array(piece.id, orient)
            assert isinstance(arr, np.ndarray)
            assert arr.ndim == 2
