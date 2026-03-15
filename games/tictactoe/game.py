import numpy as np
from numpy.typing import NDArray

from core.interfaces import IGame
from games.tictactoe.board import Board


class TicTacToeGame(IGame):

    def __init__(self) -> None:
        super().__init__()
        self.N: int = Board.N

    def initialise_board(self) -> Board:
        return Board()

    def get_board_size(self) -> tuple[int, int]:
        return self.N, self.N

    def get_action_size(self) -> int:
        return self.N * self.N + 1

    def get_next_state(self, board: Board, player: int, action: int) -> tuple[Board, int]:
        if action == self.N * self.N:
            return board, -player
        return board.with_move((action // self.N, action % self.N), player), -player

    def valid_move_masking(self, board: Board, player: int) -> NDArray:
        valids = [0] * self.get_action_size()
        legal_moves = self._get_legal_moves(board)
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legal_moves:
            valids[self.N * x + y] = 1
        return np.array(valids)

    def get_game_ended(self, board: Board, player: int) -> float:
        if self._is_win(board, player):
            return 1
        if self._is_win(board, -player):
            return -1
        if self._has_legal_moves(board):
            return 0
        return 1e-4

    @staticmethod
    def _get_legal_moves(board: Board) -> list[tuple[int, int]]:
        """Return all empty squares as (x, y) tuples."""
        n = Board.N
        moves = []
        for y in range(n):
            for x in range(n):
                if board[x][y] == 0:
                    moves.append((x, y))
        return moves

    @staticmethod
    def _has_legal_moves(board: Board) -> bool:
        """Check if any empty square exists."""
        n = Board.N
        for y in range(n):
            for x in range(n):
                if board[x][y] == 0:
                    return True
        return False

    @staticmethod
    def _is_win(board: Board, color: int) -> bool:
        """Check whether the given player has N in a row/column/diagonal."""
        n = Board.N
        # check y-strips
        for y in range(n):
            count = 0
            for x in range(n):
                if board[x][y] == color:
                    count += 1
            if count == n:
                return True
        # check x-strips
        for x in range(n):
            count = 0
            for y in range(n):
                if board[x][y] == color:
                    count += 1
            if count == n:
                return True
        # check two diagonal strips
        count = 0
        for d in range(n):
            if board[d][d] == color:
                count += 1
        if count == n:
            return True
        count = 0
        for d in range(n):
            if board[d][n - d - 1] == color:
                count += 1
        if count == n:
            return True
        return False

    def get_canonical_form(self, board: Board, player: int) -> Board:
        return board.canonical(player)

    def get_symmetries(self, board: Board, pi: NDArray) -> list[tuple[Board, NDArray]]:
        assert len(pi) == self.N ** 2 + 1  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.N, self.N))
        board_array = board.as_2d
        symmetries: list[tuple[Board, NDArray]] = []

        for i in range(1, 5):
            for is_flipped in [True, False]:
                new_array = np.rot90(board_array, i)
                new_pi = np.rot90(pi_board, i)
                if is_flipped:
                    new_array = np.fliplr(new_array)
                    new_pi = np.fliplr(new_pi)
                new_board = Board._from_pieces(new_array.tolist())
                new_pi_flat = np.append(new_pi.ravel(), pi[-1])
                symmetries.append((new_board, new_pi_flat))
        return symmetries

    def state_key(self, board: Board) -> bytes:
        return board.state_key

    @staticmethod
    def display(board: Board) -> None:
        n = board.N

        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")
            for x in range(n):
                piece = board[y][x]
                if piece == -1:
                    print("X ", end="")
                elif piece == 1:
                    print("O ", end="")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
