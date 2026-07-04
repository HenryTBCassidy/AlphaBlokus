"""J2: the static kernel tables faithfully mirror ``MoveTables`` geometry.

Pure numpy — these tests run without the ``jax`` extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.blokusduo.board import Action
from alphablokus.games.blokusduo.jaxenv.tables import JaxTables, build_jax_tables
from alphablokus.games.blokusduo.movegen_tables import build_move_tables
from alphablokus.games.blokusduo.pieces import Orientation

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.game import BlokusDuoGame

# Pentobi's hard-coded placement count for the Duo variant.
EXPECTED_NUM_MOVES = 13_729

# Total squares across the 21 pieces: 1 + 2 + 2×3 + 5×4 + 12×5.
EXPECTED_TOTAL_SQUARES = 89


@pytest.fixture(scope="module")
def tables(blokus_game_module: BlokusDuoGame) -> JaxTables:
    return build_jax_tables(blokus_game_module)


def test_placeable_count(tables: JaxTables) -> None:
    assert int(tables.placeable.sum()) == EXPECTED_NUM_MOVES
    assert not tables.placeable[tables.pass_index]
    assert tables.pass_index == tables.action_size - 1


def test_row_sums_match_move_tables(tables: JaxTables, blokus_game_module: BlokusDuoGame) -> None:
    """Each placeable row holds exactly the cell counts MoveTables recorded."""
    move_tables = build_move_tables(blokus_game_module.piece_manager)
    action_ids = move_tables.action_id
    np.testing.assert_array_equal(tables.cover[action_ids].sum(axis=1), move_tables.n_cells)
    np.testing.assert_array_equal(tables.edge[action_ids].sum(axis=1), move_tables.n_adj)
    np.testing.assert_array_equal(tables.corner[action_ids].sum(axis=1), move_tables.n_attach)
    # Non-placeable rows (including pass) are all-zero.
    non_placeable = ~tables.placeable
    assert tables.cover[non_placeable].sum() == 0
    assert tables.edge[non_placeable].sum() == 0
    assert tables.corner[non_placeable].sum() == 0


def test_halos_disjoint_from_footprint(tables: JaxTables) -> None:
    """A footprint cell is never simultaneously in its own halo sets."""
    assert not np.any((tables.cover == 1) & (tables.edge == 1))
    assert not np.any((tables.cover == 1) & (tables.corner == 1))
    assert not np.any((tables.edge == 1) & (tables.corner == 1))


def test_piece_of_action(tables: JaxTables) -> None:
    assert tables.piece_of_action[tables.placeable].min() == 1
    assert tables.piece_of_action[tables.placeable].max() == 21
    assert np.all(tables.piece_of_action[~tables.placeable] == 0)


def test_piece_sizes(tables: JaxTables) -> None:
    assert tables.piece_sizes[0] == 0
    assert tables.piece_sizes[1] == 1  # monomino
    assert int(tables.piece_sizes.sum()) == EXPECTED_TOTAL_SQUARES


def test_start_cells(tables: JaxTables, blokus_game_module: BlokusDuoGame) -> None:
    """Start squares are array indices: White (4,4) -> 60, Black (9,9) -> 135.

    Pinned empirically: every legal first move for each player covers exactly
    this cell (the J3 parity test checks the full mask; here we check the
    engine's own initial-action cache).
    """
    n = blokus_game_module.board_size
    assert tables.start_cell[0] == 4 * n + 4
    assert tables.start_cell[1] == 9 * n + 9
    for slot, player in enumerate((1, -1)):
        for action in blokus_game_module.initial_actions[player]:
            action_id = blokus_game_module.action_codec.encode(action)
            assert tables.cover[action_id, tables.start_cell[slot]] == 1


def test_monomino_corner_placement_hand_check(tables: JaxTables, blokus_game_module: BlokusDuoGame) -> None:
    """Hand-computed geometry for the monomino at board (0, 0) = array (13, 0)."""
    action_id = blokus_game_module.action_codec.encode(
        Action(piece_id=1, orientation=Orientation.Identity, x_coordinate=0, y_coordinate=0)
    )
    n = blokus_game_module.board_size
    cell = 13 * n + 0
    assert set(np.flatnonzero(tables.cover[action_id])) == {cell}
    # Edge halo: (12,0) and (13,1). Corner halo: (12,1). Off-board cells dropped.
    assert set(np.flatnonzero(tables.edge[action_id])) == {12 * n + 0, 13 * n + 1}
    assert set(np.flatnonzero(tables.corner[action_id])) == {12 * n + 1}
