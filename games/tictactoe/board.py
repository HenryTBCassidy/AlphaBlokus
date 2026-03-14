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
    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self, n: int = 3) -> None:
        """Set up initial board configuration."""
        self.n = n
        # Create the empty board array.
        self.pieces: list[list[int]] = []
        for i in range(self.n):
            self.pieces.append([0] * self.n)

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
        rep = np.zeros((2, self.n, self.n), dtype=np.float32)
        rep[0] = (board == current_player)
        rep[1] = (board == -current_player)
        return rep

    @property
    def num_channels(self) -> int:
        """Number of channels in as_multi_channel output."""
        return 2

    def copy(self) -> Board:
        """Return a deep copy of this board."""
        b = Board(self.n)
        b.pieces = [row[:] for row in self.pieces]
        return b

    def canonical(self, player: int) -> Board:
        """Return the board in canonical form (player 1 perspective)."""
        if player == 1:
            return self
        b = Board(self.n)
        b.pieces = (self.as_2d * player).tolist()
        return b

    # ── Indexer ───────────────────────────────────────────────────────

    def __getitem__(self, index: int) -> list[int]:
        return self.pieces[index]

    # ── Game logic ────────────────────────────────────────────────────

    def get_legal_moves(self, color: int) -> list[tuple[int, int]]:
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.
        """
        moves = set()  # stores the legal moves.

        # Get all the empty squares (color==0)
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    new_move = (x, y)
                    moves.add(new_move)
        return list(moves)

    def has_legal_moves(self) -> bool:
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def is_win(self, color: int) -> bool:
        """Check whether the given player has collected a triplet in any direction;
        @param color (1=white,-1=black)
        """
        win = self.n
        # check y-strips
        for y in range(self.n):
            count = 0
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
            if count == win:
                return True
        # check x-strips
        for x in range(self.n):
            count = 0
            for y in range(self.n):
                if self[x][y] == color:
                    count += 1
            if count == win:
                return True
        # check two diagonal strips
        count = 0
        for d in range(self.n):
            if self[d][d] == color:
                count += 1
        if count == win:
            return True
        count = 0
        for d in range(self.n):
            if self[d][self.n - d - 1] == color:
                count += 1
        if count == win:
            return True

        return False

    def execute_move(self, move: tuple[int, int], color: int) -> None:
        """Perform the given move on the board;
        color gives the color of the piece to play (1=white,-1=black)
        """
        (x, y) = move

        # Add the piece to the empty square.
        assert self[x][y] == 0
        self[x][y] = color
