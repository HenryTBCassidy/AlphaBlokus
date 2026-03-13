import numpy as np
from numpy.typing import NDArray

from core.interfaces import IGame
from games.tictactoe.board import Board


class TicTacToeGame(IGame):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.n = n

    def initialise_board(self) -> NDArray:
        b = Board(self.n)
        return np.array(b.pieces)

    def get_board_size(self) -> tuple[int, int]:
        return self.n, self.n

    def get_action_size(self) -> int:
        return self.n * self.n + 1

    def get_next_state(self, board: NDArray, player: int, action: int) -> tuple[NDArray, int]:
        if action == self.n * self.n:
            return board, -player
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return b.pieces, -player

    def valid_move_masking(self, board: NDArray, player: int) -> NDArray:
        valids = [0] * self.get_action_size()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legal_moves = b.get_legal_moves(player)
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legal_moves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def get_game_ended(self, board: NDArray, player: int) -> float:
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # Draw has a very small value
        return 1e-4

    def get_canonical_form(self, board: NDArray, player: int) -> NDArray:
        return player * board

    def get_symmetries(self, board: NDArray, pi: NDArray) -> list[tuple[NDArray, NDArray]]:
        assert len(pi) == self.n ** 2 + 1  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        symmetries: list[tuple[NDArray, NDArray]] = []

        for i in range(1, 5):
            for is_flipped in [True, False]:
                new_board = np.rot90(board, i)
                new_pi = np.rot90(pi_board, i)
                if is_flipped:
                    new_board = np.fliplr(new_board)
                    new_pi = np.fliplr(new_pi)
                new_pi_flat = np.append(new_pi.ravel(), pi[-1])
                symmetries.append((new_board, new_pi_flat))
        return symmetries

    def state_key(self, board: NDArray) -> bytes:
        return board.tobytes()

    @staticmethod
    def display(board: NDArray) -> None:
        n = board.shape[0]

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
