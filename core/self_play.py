"""Free-function self-play episode runner shared by Coach and parallel workers.

Extracted from ``Coach.execute_episode`` so the serial training loop and
the worker pool in ``core/parallel_self_play.py`` use the **same code
path**. That equivalence is the basis of the determinism test in
``tests/test_core/test_parallel_self_play.py`` — if both call sites
invoke this function with the same seed + same MCTS instance, they
produce identical training examples regardless of which process runs
the work.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.interfaces import IGame
    from core.mcts import MCTS

# (board_array, policy_array, value) — the shape Coach passes to
# ``nnet.train`` after a generation of self-play. Kept untyped at runtime
# because the inner board element is an NDArray returned by
# ``IBoard.as_multi_channel`` and the value is a float ∈ [-1, 1].
ProcessedExample = tuple[np.ndarray, np.ndarray, float]


def play_self_play_episode(
    game: IGame, mcts: MCTS, temp_threshold: int,
) -> list[ProcessedExample]:
    """Play one self-play game and return training examples for it.

    The single source of truth for the self-play episode loop. Called
    from:

    - ``Coach.execute_episode`` — when self-play runs sequentially in the
      training process.
    - ``core.parallel_self_play._worker_play_episode`` — when self-play
      runs in worker processes.

    Args:
        game: Game implementation providing rules and mechanics.
        mcts: A fresh MCTS instance bound to the same network the caller
            wants to use for action selection. The caller is responsible
            for constructing this (so the worker case can attach its own
            per-process network).
        temp_threshold: Move number after which exploration temperature
            collapses to 0 (deterministic argmax). Matches
            ``RunConfig.temp_threshold``.

    Returns:
        List of ``(board, policy, value)`` triples — one per position
        visited (including symmetry augmentations from
        ``game.get_symmetries``). The ``value`` field is filled in once
        the game ends, based on the perspective of the player at that
        position.
    """
    train_examples: list[tuple] = []
    board = game.initialise_board()
    current_player = 1
    move_count = 0

    while True:
        move_count += 1
        canonical_board = game.get_canonical_form(board, current_player)
        temperature = int(move_count < temp_threshold)

        # MCTS-improved policy. ``mcts`` accumulates per-move profiling
        # stats internally; the caller pulls them out via
        # ``mcts.get_episode_stats()`` after the episode ends.
        pi = mcts.get_action_prob(canonical_board, temp=temperature)

        # Symmetry augmentation: store every symmetric (board, policy)
        # pair the game exposes. Multiplies training-example count per
        # position by the size of the symmetry group.
        symmetries = game.get_symmetries(canonical_board, pi)
        for symmetric_board, symmetric_pi in symmetries:
            train_examples.append((symmetric_board, current_player, symmetric_pi, None))

        # Sample the move from the post-temperature distribution.
        action = np.random.choice(len(pi), p=pi)
        board, current_player = game.get_next_state(board, current_player, action)

        game_result = game.get_game_ended(board, current_player)
        if game_result != 0:
            # End of game: convert (board_obj, player, pi, _) into the
            # (NDArray, pi, value) shape the network trainer expects.
            # The value sign flips for each position where the player to
            # move differs from the player at game end.
            return [
                (x[0].as_multi_channel(1), x[2], game_result * ((-1) ** (x[1] != current_player)))
                for x in train_examples
            ]
