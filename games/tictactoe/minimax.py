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
        return self.optimal_actions(board)[0]

    def evaluate_position(self, board: IBoard) -> float:
        """Return the game-theoretic value of ``board`` from side-to-move's
        perspective: ``+1`` if the side-to-move can force a win, ``-1`` if they
        will lose against perfect play, ``0`` for a draw.

        Expects ``board`` in canonical form (side-to-move = ``+1``). For TTT
        every reachable non-terminal position has true value ``0`` (forced
        draw), so this is mostly useful on positions reached via random or
        early-net play that already contain a forced win/loss.
        """
        return self._negamax(board)

    def optimal_actions(self, board: IBoard) -> list[int]:
        """Return *all* actions whose minimax value matches the best value.

        Used by the TTT eval set so a network is credited for picking any
        of several equally-optimal moves, rather than being arbitrarily
        penalised when it picks a different one than the one we'd nominate.
        Falls back to ``[legal[0]]`` if the position is somehow terminal,
        which shouldn't happen in practice but keeps the API total.
        """
        legal = np.flatnonzero(self._game.valid_move_masking(board, 1))
        if len(legal) == 0:
            return [0]

        best_value = -math.inf
        action_values: list[tuple[int, float]] = []
        for action in legal:
            next_board, next_player = self._game.get_next_state(board, 1, int(action))
            next_canonical = self._game.get_canonical_form(next_board, next_player)
            value = -self._negamax(next_canonical)
            action_values.append((int(action), value))
            if value > best_value:
                best_value = value
        return [a for a, v in action_values if v >= best_value - 1e-9]

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
