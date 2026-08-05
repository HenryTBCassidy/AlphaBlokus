"""Reusable Player implementations.

A ``Player`` is anything callable as ``board → action_index``. The existing
:class:`alphablokus.evaluation.arena.Arena` already accepts arbitrary callables; this module
gives the common ones names so they're not re-defined ad-hoc in every test
or script.

Available players:

- :class:`RandomPlayer` — uniform over legal moves. Cheap baseline.
- :class:`NetworkPlayer` — neural-network-backed, plays via MCTS using the
  supplied checkpoint. The standard "trained model" player.

Game-specific players (e.g. the TicTacToe minimax oracle) live with their
game under :mod:`alphablokus.games`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from alphablokus.interfaces import RESIGN_ACTION, IBoard, IGame, IPolicyValuePredictor

if TYPE_CHECKING:
    from alphablokus.config import MCTSConfig


Player: TypeAlias = Callable[[IBoard], int]
"""Function signature for a player: takes a canonical board and returns an action index."""


def sample_opening_prefix(game: IGame, sampler: Player, num_moves: int) -> tuple[int, ...]:
    """Sample a shared ``num_moves``-ply opening prefix for paired arena play.

    Plays the game forward from the initial position for up to ``num_moves``
    plies, drawing each ply from ``sampler`` (typically a :class:`NetworkPlayer`
    configured with a >0 play temperature, so it samples from the incumbent's
    MCTS visit distribution rather than playing deterministically). The captured
    action sequence is then replayed verbatim by both halves of a colour-swapped
    pair (see :meth:`alphablokus.evaluation.arena.Arena.play_games_paired`), so
    the first-mover advantage cancels exactly.

    The prefix is *game-level* — one action per global ply, applied to whichever
    side is to move — so it is robust to Blokus's non-strict alternation (a
    player may move twice in a row when the opponent has no legal move). Callers
    that need a deterministic prefix seed the global RNG before calling.

    Args:
        game: The game whose rules drive the forward simulation.
        sampler: The player asked to choose each opening ply.
        num_moves: Number of plies to sample (``<= 0`` returns an empty prefix).

    Returns:
        The sampled action sequence (may be shorter than ``num_moves`` if the
        game ends early or the sampler resigns).
    """
    if num_moves <= 0:
        return ()
    if hasattr(sampler, "startGame"):
        sampler.startGame()
    board = game.initialise_board()
    cur_player = 1
    actions: list[int] = []
    for _ in range(num_moves):
        if game.get_game_ended(board, cur_player) != 0:
            break
        canonical_board = game.get_canonical_form(board, cur_player)
        action = int(sampler(canonical_board))
        if action == RESIGN_ACTION:
            break
        actions.append(action)
        board, cur_player = game.get_next_state(board, cur_player, action)
    return tuple(actions)


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

    Owns an :class:`alphablokus.search.mcts.MCTS` instance configured with the given
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
        nnet: IPolicyValuePredictor,
        mcts_config: MCTSConfig,
        temp: float = 0.0,
        opening_temp: float = 0.0,
        opening_moves: int = 0,
        seed: int | None = None,
    ) -> None:
        """Create a network-backed player.

        Args:
            temp: Play temperature after the opening (0 → deterministic argmax).
            opening_temp: Temperature applied to the first ``opening_moves`` plies
                of each game. Set >0 to diversify openings so repeated deterministic
                games against a fixed opponent aren't near-identical (evaluation
                independence — see ``scripts/pentobi_benchmark``). The move counter
                resets each game via :meth:`startGame`.
            opening_moves: Number of the player's own opening plies to which
                ``opening_temp`` applies before reverting to ``temp``.
            seed: Seed for this player's own RNG, used for opening sampling.
                ``None`` keeps the legacy behaviour of drawing from numpy's global
                RNG, which nothing seeds — so evaluation runs were not reproducible
                while the *opponent* (``PentobiPlayer``) was carefully reseeded per
                game. Pass a seed for any run whose result gets compared to another.
        """
        # Local import to avoid a cycle (mcts imports from alphablokus.interfaces).
        from alphablokus.search.mcts import MCTS

        self._game = game
        self._nnet = nnet
        self._mcts_config = mcts_config
        self._temp = temp
        self._opening_temp = opening_temp
        self._opening_moves = opening_moves
        self._move_count = 0
        self._mcts = MCTS(game, nnet, mcts_config)
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self._last_pi: np.ndarray | None = None

    def __call__(self, board: IBoard) -> int:
        # Opening moves use ``opening_temp`` (sampling for position diversity),
        # then revert to the configured play temperature.
        temp = self._opening_temp if self._move_count < self._opening_moves else self._temp
        self._move_count += 1

        # Run MCTS + get the action distribution at the effective temperature
        # for actual play (temp=0 → one-hot deterministic; temp>0 → sampled).
        pi_play = self._mcts.get_action_prob(board, temp=temp)

        # Separately, extract the *raw visit-count distribution* (i.e. what
        # the policy looks like before temperature is applied). This is the
        # informative record for replays — at temp=0 the play distribution is
        # one-hot and useless for "what was the model considering?" analysis.
        counts = self._mcts.root_visit_counts(board).astype(float)
        total = counts.sum()
        if total > 0:
            self._last_pi = counts / total
        else:
            self._last_pi = np.asarray(pi_play, dtype=float)

        if temp == 0:
            return int(np.argmax(pi_play))
        if self._rng is not None:
            return int(self._rng.choice(len(pi_play), p=pi_play))
        return int(np.random.choice(len(pi_play), p=pi_play))

    def get_last_policy(self) -> np.ndarray | None:
        """Return the policy vector from the most recent call, or None."""
        return self._last_pi

    def reset_search_tree(self) -> None:
        """Discard the MCTS tree between games for a clean evaluation slate.

        Called by :class:`alphablokus.evaluation.arena.Arena` between games when present (via
        the existing ``startGame`` hook on the player).
        """
        from alphablokus.search.mcts import MCTS

        self._mcts = MCTS(self._game, self._nnet, self._mcts_config)

    # Arena's existing convention: if a player has ``startGame``, it's called
    # before each game starts. Use it to reset the MCTS tree so games don't
    # leak state.
    def startGame(self) -> None:  # noqa: N802 — Arena's pre-existing camelCase hook
        self.reset_search_tree()
        self._move_count = 0


# Note: ``load_network_player`` (a path-based factory that builds a wrapper
# from a checkpoint file on disk) lives in ``scripts/_player_loaders.py`` —
# it depends on per-game wrapper imports and config wiring that don't belong
# in this generic module.
