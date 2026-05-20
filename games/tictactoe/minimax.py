"""Perfect-play minimax opponent for TicTacToe.

Used as a fixed external baseline for measuring whether a trained model has
internalised optimal play (R14 of the reporting overhaul). With perfect play
on both sides TicTacToe is a forced draw — so a "fully trained" model should
**never lose** against this opponent and should draw essentially every game.

Implementation: pure negamax with memoisation. **No alpha-beta** — caching
α-β results is only sound if you also store the value's bound type (exact /
upper / lower), which adds complexity for no real gain at TTT's ~5,400-state
size. The unpruned implementation handles the full game tree in <1s and the
memoisation makes repeated calls essentially free.
"""
from __future__ import annotations

import math

import numpy as np

from core.interfaces import IBoard
from games.tictactoe.game import TicTacToeGame


class MinimaxTicTacToePlayer:
    """Perfect-play TicTacToe opponent.

    Each ``__call__`` returns the action with the highest minimax value for
    the side-to-move (always +1 in the canonical-board convention). The
    underlying negamax search is memoised by board state key.
    """

    def __init__(self, game: TicTacToeGame) -> None:
        self._game = game
        self._cache: dict[bytes, float] = {}

    def __call__(self, board: IBoard) -> int:
        """Return the action with the highest minimax value from this state."""
        legal = np.flatnonzero(self._game.valid_move_masking(board, 1))
        best_action = int(legal[0])
        best_value = -math.inf
        for action in legal:
            next_board, next_player = self._game.get_next_state(board, 1, int(action))
            # Re-canonicalise for the opponent. Without this the recursion
            # would interpret the board with the wrong sign convention.
            next_canonical = self._game.get_canonical_form(next_board, next_player)
            # Opponent moves next; their best value, negated, is our value.
            value = -self._negamax(next_canonical)
            if value > best_value:
                best_value = value
                best_action = int(action)
        return best_action

    # -- internal --------------------------------------------------------------

    def _negamax(self, board: IBoard) -> float:
        """Negamax value of ``board`` from the side-to-move's perspective.

        ``board`` must be in canonical form (side to move = ``+1``). Pure
        unpruned minimax — values returned are always exact, so the cache
        is sound.
        """
        key = self._game.state_key(board)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        terminal = self._game.get_game_ended(board, 1)
        if terminal != 0:
            self._cache[key] = float(terminal)
            return float(terminal)

        best = -math.inf
        legal = np.flatnonzero(self._game.valid_move_masking(board, 1))
        for action in legal:
            next_board, next_player = self._game.get_next_state(board, 1, int(action))
            next_canonical = self._game.get_canonical_form(next_board, next_player)
            value = -self._negamax(next_canonical)
            if value > best:
                best = value

        self._cache[key] = best
        return best
