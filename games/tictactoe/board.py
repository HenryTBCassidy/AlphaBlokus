from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.interfaces import IBoard


class Board(IBoard):
    """
    Board class for the game of TicTacToe.
    Implements the IBoard protocol.

    Default board size is 3x3.
    Board data:
      1=white(O), -1=black(X), 0=empty
      first dim is column , 2nd is row:
         pieces[0][0] is the top left square,
         pieces[2][0] is the bottom left square,
    Squares are stored and manipulated as (x,y) tuples.
    """
    N: int = 3  # Board size

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self) -> None:
        """Set up initial board configuration."""
        self.pieces: list[list[int]] = [[0] * self.N for _ in range(self.N)]

    # ── IBoard protocol ──────────────────────────────────────────────

    @property
    def as_2d(self) -> NDArray:
        """Flat H×W game state board (±1/0). Used by game logic."""
        return np.array(self.pieces)

    def as_multi_channel(self, current_player: int) -> NDArray:
        """Build 2-channel tensor for neural net input.

        Channel 0: current player's stones (binary)
        Channel 1: opponent's stones (binary)

        Args:
            current_player: Whose turn it is (1 or -1).

        Returns:
            NDArray of shape (2, n, n) with float32 dtype.
        """
        board = self.as_2d
        rep = np.zeros((2, self.N, self.N), dtype=np.float32)
        rep[0] = (board == current_player)
        rep[1] = (board == -current_player)
        return rep

    @property
    def num_channels(self) -> int:
        """Number of channels in as_multi_channel output."""
        return 2

    @property
    def state_key(self) -> bytes:
        """Hashable key that uniquely identifies this board state."""
        return self.as_2d.tobytes()

    def canonical(self, player: int) -> Board:
        """Return the board in canonical form (player 1 perspective)."""
        if player == 1:
            return self
        return Board._from_pieces((self.as_2d * player).tolist())

    # ── Indexer ───────────────────────────────────────────────────────

    def __getitem__(self, index: int) -> list[int]:
        return self.pieces[index]

    # ── Mutation ─────────────────────────────────────────────────────

    def with_move(self, move: tuple[int, int], color: int) -> Board:
        """Return a new board with the given move applied.

        Args:
            move: (x, y) position to place the piece.
            color: 1 for white, -1 for black.
        """
        new_pieces = [row[:] for row in self.pieces]
        x, y = move
        assert new_pieces[x][y] == 0
        new_pieces[x][y] = color
        return Board._from_pieces(new_pieces)

    # ── Private constructors ───────────────────────────────────────

    @classmethod
    def _from_pieces(cls, pieces: list[list[int]]) -> Board:
        """Construct a board from existing piece data (no mutation)."""
        board = object.__new__(cls)
        board.pieces = pieces
        return board
