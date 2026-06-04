"""Reusable Player implementations.

A ``Player`` is anything callable as ``board → action_index``. The existing
:class:`core.arena.Arena` already accepts arbitrary callables; this module
gives the common ones names so they're not re-defined ad-hoc in every test
or script.

Available players:

- :class:`RandomPlayer` — uniform over legal moves. Cheap baseline.
- :class:`NetworkPlayer` — neural-network-backed, plays via MCTS using the
  supplied checkpoint. The standard "trained model" player.
- :class:`MinimaxTicTacToePlayer` — perfect-play opponent for TTT.
- :class:`HumanPlayer` — reads moves from stdin.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from core.interfaces import IBoard, IGame, INeuralNetWrapper

if TYPE_CHECKING:
    from core.config import MCTSConfig


Player: TypeAlias = Callable[[IBoard], int]
"""Function signature for a player: takes a canonical board and returns an action index."""


class RandomPlayer:
    """Player that picks uniformly at random among legal moves.

    Useful as a cheap baseline opponent — every trained model should crush it.
    """

    def __init__(self, game: IGame) -> None:
        self._game = game
        self._rng = np.random.default_rng()

    def __call__(self, board: IBoard) -> int:
        valids = self._game.valid_move_masking(board, 1)
        legal_actions = np.flatnonzero(valids)
        return int(self._rng.choice(legal_actions))


class NetworkPlayer:
    """Player backed by a neural network + MCTS.

    Owns an :class:`core.mcts.MCTS` instance configured with the given
    network wrapper and search depth. The default ``temp=0`` gives
    deterministic best-move play; ``temp=1`` samples by visit count for
    self-play-style behaviour.

    Records the full policy from its most recent call for downstream
    analysis (top-K extraction for arena replays). Access via
    :meth:`get_last_policy`.
    """

    def __init__(
        self,
        game: IGame,
        nnet: INeuralNetWrapper,
        mcts_config: MCTSConfig,
        temp: float = 0.0,
    ) -> None:
        # Local import to avoid a cycle (mcts imports from core.interfaces).
        from core.mcts import MCTS

        self._game = game
        self._nnet = nnet
        self._mcts_config = mcts_config
        self._temp = temp
        self._mcts = MCTS(game, nnet, mcts_config)
        self._last_pi: np.ndarray | None = None

    def __call__(self, board: IBoard) -> int:
        # Run MCTS + get the action distribution at the configured temperature
        # for actual play (temp=0 → one-hot deterministic; temp=1 → sampled).
        pi_play = self._mcts.get_action_prob(board, temp=self._temp)

        # Separately, extract the *raw visit-count distribution* (i.e. what
        # the policy looks like before temperature is applied). This is the
        # informative record for replays — at temp=0 the play distribution is
        # one-hot and useless for "what was the model considering?" analysis.
        s = self._game.state_key(board)
        n_actions = self._game.get_action_size()
        counts = np.array(
            [self._mcts.visit_counts.get((s, a), 0) for a in range(n_actions)],
            dtype=float,
        )
        total = counts.sum()
        if total > 0:
            self._last_pi = counts / total
        else:
            self._last_pi = np.asarray(pi_play, dtype=float)

        if self._temp == 0:
            return int(np.argmax(pi_play))
        return int(np.random.choice(len(pi_play), p=pi_play))

    def get_last_policy(self) -> np.ndarray | None:
        """Return the policy vector from the most recent call, or None."""
        return self._last_pi

    def reset_search_tree(self) -> None:
        """Discard the MCTS tree between games for a clean evaluation slate.

        Called by :class:`core.arena.Arena` between games when present (via
        the existing ``startGame`` hook on the player).
        """
        from core.mcts import MCTS

        self._mcts = MCTS(self._game, self._nnet, self._mcts_config)

    # Arena's existing convention: if a player has ``startGame``, it's called
    # before each game starts. Use it to reset the MCTS tree so games don't
    # leak state.
    def startGame(self) -> None:  # noqa: N802 — Arena's pre-existing camelCase hook
        self.reset_search_tree()


# Note: ``load_network_player`` (a path-based factory that builds a wrapper
# from a checkpoint file on disk) lives in ``scripts/_player_loaders.py`` —
# it depends on per-game wrapper imports and config wiring that don't belong
# in this generic module.
