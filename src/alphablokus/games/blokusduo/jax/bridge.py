"""Converters between the Python engine's objects and JAX spike state.

Test/benchmark plumbing — the hot paths never touch this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphablokus.games.blokusduo.jax.tables import NUM_PIECE_IDS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphablokus.games.blokusduo.board import BlokusDuoBoard


def numpy_state_from_board(
    board: BlokusDuoBoard,
    current_player: int,
) -> tuple[NDArray, NDArray, NDArray, np.int8]:
    """Extract ``(ppb, remaining, last_piece, current_player)`` numpy arrays.

    Shapes/dtypes match :class:`games.blokusduo.jax.kernels.GameState` fields
    so rows can be stacked into batches and handed to jax wholesale.
    """
    ppb = board.to_compact().ravel().astype(np.int8)
    remaining = np.zeros((2, NUM_PIECE_IDS + 1), dtype=np.bool_)
    last_piece = np.zeros(2, dtype=np.int8)
    for slot, player_side in enumerate((1, -1)):
        for piece_id in board.remaining_piece_ids(player_side):
            remaining[slot, piece_id] = True
        last_piece[slot] = board.last_piece_played(player_side) or 0
    return ppb, remaining, last_piece, np.int8(current_player)
