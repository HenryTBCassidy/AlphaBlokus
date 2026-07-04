"""Perfect-play oracle hooks for TicTacToe strength evaluation.

TicTacToe is small enough to solve exactly, so the framework can benchmark
against ground truth: an unbeatable arena opponent, and eval-set targets that
are game-theoretically optimal rather than gen-1 MCTS noise. Reached through
``registry.resolve_oracle`` — framework code never imports this module
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphablokus.games.tictactoe.board import Board
from alphablokus.games.tictactoe.minimax import MinimaxTicTacToePlayer
from alphablokus.interfaces import IOracle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphablokus.evaluation.players import Player
    from alphablokus.games.tictactoe.game import TicTacToeGame


class TicTacToeOracle(IOracle):
    """Minimax-backed perfect-play oracle for TicTacToe."""

    def __init__(self, game: TicTacToeGame) -> None:
        self._game = game

    def make_player(self) -> Player:
        """An unbeatable arena opponent (perfect-play negamax)."""
        return MinimaxTicTacToePlayer(self._game)

    def eval_targets(
        self,
        compact_boards: list[NDArray],
        action_size: int,
    ) -> tuple[NDArray, NDArray]:
        """Perfect-play eval-set targets for the given positions.

        Each row of the returned policies is a uniform distribution over
        all minimax-optimal actions; each value is the position's true
        game-theoretic value. Boards arrive as the compact 3×3 canonical grid
        (``Board.to_compact()``), so a :class:`Board` is rebuilt directly from
        it before querying the minimax solver.
        """
        minimax = MinimaxTicTacToePlayer(self._game)
        n = len(compact_boards)
        new_policies = np.zeros((n, action_size), dtype=np.float32)
        new_values = np.zeros(n, dtype=np.float32)

        for i, grid in enumerate(compact_boards):
            # Compact form is the canonical 3×3 grid (+1 side-to-move, -1 opponent).
            canonical_board = Board._from_pieces(np.asarray(grid).astype(int).tolist())

            new_values[i] = float(minimax.evaluate_position(canonical_board))
            optimal = minimax.optimal_actions(canonical_board)
            if optimal:
                weight = 1.0 / len(optimal)
                for action in optimal:
                    new_policies[i, action] = weight

        return new_policies, new_values
