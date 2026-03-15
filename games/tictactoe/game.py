import numpy as np
from numpy.typing import NDArray

from core.interfaces import IGame
from games.tictactoe.board import Board


class TicTacToeGame(IGame):
    N: int = 3  # Board size

    def __init__(self) -> None:
        super().__init__()

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
        legal_moves = board.get_legal_moves(player)
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legal_moves:
            valids[self.N * x + y] = 1
        return np.array(valids)

    def get_game_ended(self, board: Board, player: int) -> float:
        if board.is_win(player):
            return 1
        if board.is_win(-player):
            return -1
        if board.has_legal_moves():
            return 0
        # Draw has a very small value
        return 1e-4

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
