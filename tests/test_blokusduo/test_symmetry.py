"""Tests for the Blokus Duo symmetry plumbing.

Covers the layered correctness checks described in
``docs/plans/blokus-symmetries.md`` — orientation lookup (S1), board
transpose (S2), action transpose (S3), and the higher-level equivariance /
invariance / round-trip properties (S5–S7).
"""
from __future__ import annotations

import numpy as np
import pytest

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
