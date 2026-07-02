"""Static geometry tables for the JAX legality kernel (plan step J2).

The whole Blokus Duo rules engine reduces to three precomputed static matrices
over the full action space. For each geometrically-possible placement (already
enumerated by :func:`games.blokusduo.movegen_tables.build_move_tables`), we
scatter its cell lists into rows of three ``(action_size, 196)`` int8 matrices
indexed by **action id**:

- ``cover``  — the placement's footprint cells,
- ``edge``   — cells edge-adjacent to the footprint (own-colour contact here is
  illegal),
- ``corner`` — cells diagonally-adjacent but *not* edge-adjacent to the
  footprint (own-colour contact here is required, except on the first move).

Rows for the ~4,100 never-legal action ids (off-board placements) and the pass
action stay all-zero and are excluded via the static ``placeable`` mask.

Building *from* ``MoveTables`` rather than re-deriving geometry means the JAX
kernel inherits F2's enumeration exactly — any parity failure isolates to the
rule-condition translation, not the geometry.

Everything here is plain numpy (no jax import) so the tables can be built and
unit-tested without the ``jax`` extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from games.blokusduo.movegen_tables import NULL_CELL, NUM_CELLS, build_move_tables

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from games.blokusduo.game import BlokusDuoGame

#: Number of piece ids (1–21); index 0 of per-piece vectors is unused padding.
NUM_PIECE_IDS = 21


@dataclass(frozen=True)
class JaxTables:
    """Static arrays consumed by the JAX kernels. All plain numpy.

    Attributes:
        action_size: Full flat action space size (17,837 for Duo).
        num_cells: Board cells (196 for Duo).
        cover: int8 ``(action_size, num_cells)`` — footprint cells per action.
        edge: int8 ``(action_size, num_cells)`` — edge-adjacent halo per action.
        corner: int8 ``(action_size, num_cells)`` — diagonal (non-edge) attach
            halo per action.
        piece_of_action: int8 ``(action_size,)`` — piece id (1–21) per placeable
            action, 0 elsewhere (including pass).
        placeable: bool ``(action_size,)`` — True for the 13,729 on-board
            placements; False for off-board ids and pass.
        piece_sizes: int8 ``(NUM_PIECE_IDS + 1,)`` — squares per piece id,
            index 0 unused (0).
        start_cell: int32 ``(2,)`` — flat array-index start cell per player,
            index 0 = White (+1), index 1 = Black (-1).
        pass_index: Flat action id of the pass move.
    """

    action_size: int
    num_cells: int
    cover: NDArray
    edge: NDArray
    corner: NDArray
    piece_of_action: NDArray
    placeable: NDArray
    piece_sizes: NDArray
    start_cell: NDArray
    pass_index: int


def _scatter_cells(matrix: NDArray, action_ids: NDArray, cells: NDArray) -> None:
    """Set ``matrix[action_ids[m], cells[m, k]] = 1`` for every non-NULL cell."""
    rows = np.repeat(action_ids.astype(np.int64), cells.shape[1])
    flat_cells = cells.astype(np.int64).ravel()
    valid = flat_cells != NULL_CELL
    matrix[rows[valid], flat_cells[valid]] = 1


def build_jax_tables(game: BlokusDuoGame) -> JaxTables:
    """Build the static kernel tables from a game's piece definitions.

    Args:
        game: The rules engine to source geometry (via ``build_move_tables``),
            action encoding, and start squares from.

    Returns:
        Fully-populated :class:`JaxTables`.
    """
    move_tables = build_move_tables(game.piece_manager)
    action_size = game.get_action_size()

    cover = np.zeros((action_size, NUM_CELLS), dtype=np.int8)
    edge = np.zeros((action_size, NUM_CELLS), dtype=np.int8)
    corner = np.zeros((action_size, NUM_CELLS), dtype=np.int8)
    _scatter_cells(cover, move_tables.action_id, move_tables.cells)
    _scatter_cells(edge, move_tables.action_id, move_tables.adj_cells)
    _scatter_cells(corner, move_tables.action_id, move_tables.attach_cells)

    piece_of_action = np.zeros(action_size, dtype=np.int8)
    piece_of_action[move_tables.action_id] = move_tables.piece

    placeable = move_tables.action_to_move_id >= 0

    piece_sizes = np.zeros(NUM_PIECE_IDS + 1, dtype=np.int8)
    for piece_id, piece in game.piece_manager.pieces.items():
        piece_sizes[piece_id] = int(piece.identity.sum())

    # The engine's start squares are *array* indices, not board coordinates:
    # White's first move must cover array (4, 4) and Black's array (9, 9)
    # (see the ``get_symmetries`` docstring in games/blokusduo/game.py and the
    # J3 parity test, which pins this empirically against both generators).
    start_cell = np.empty(2, dtype=np.int32)
    for slot, (row, col) in enumerate((game.white_start, game.black_start)):
        start_cell[slot] = row * game.board_size + col

    return JaxTables(
        action_size=action_size,
        num_cells=NUM_CELLS,
        cover=cover,
        edge=edge,
        corner=corner,
        piece_of_action=piece_of_action,
        placeable=placeable,
        piece_sizes=piece_sizes,
        start_cell=start_cell,
        pass_index=game.action_codec.pass_action_index,
    )
