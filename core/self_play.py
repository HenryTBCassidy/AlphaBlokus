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

from core.sparse_policy import sparsify

if TYPE_CHECKING:
    from core.interfaces import IGame
    from core.mcts import MCTS

# (compact_board, sparse_policy, value) — the shape Coach passes to
# ``nnet.train`` after a generation of self-play. The board is stored
# **compact** (``IBoard.to_compact`` — e.g. the 196-byte int8 placement board
# for Blokus, vs the ~34.5 KB dense ``(44,14,14)`` encoding); the trainer
# re-encodes it lazily per mini-batch via ``IGame.encode_compact``. The policy
# is stored **sparse** as ``(indices, values)`` (see :mod:`core.sparse_policy`)
# because the dense 17,837-vector dominated replay-buffer RAM; the trainer
# densifies it. Kept untyped at runtime (compact board is an int8 NDArray,
# value is a float ∈ [-1, 1]).
ProcessedExample = tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]


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
        pi = mcts.get_action_prob(canonical_board, temp=temperature, add_root_noise=True)

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
            # (compact_board, pi, value) shape the network trainer expects.
            # The value sign flips for each position where the player to
            # move differs from the player at game end.
            #
            # The board is stored **compact** (``to_compact()`` — the minimal
            # canonical array, game-agnostic via the IBoard seam) rather than the
            # dense ``as_multi_channel(1)`` planes; the trainer re-encodes it
            # lazily per mini-batch. This keeps the replay buffer ~175× smaller.
            #
            # The policy is stored sparse (nonzero ``(indices, values)``): the
            # MCTS visit distribution is sparse, but a dense float32 vector is
            # ~71 KB and dominates replay-buffer RAM at scale. ``sparsify`` is
            # lossless — the trainer densifies it back. Symmetry augmentation
            # above operates on the dense ``pi``, so ``get_symmetries`` is
            # unaffected; we only sparsify the final stored result here.
            return [
                (
                    x[0].to_compact(),
                    sparsify(np.asarray(x[2], dtype=np.float32)),
                    game_result * ((-1) ** (x[1] != current_player)),
                )
                for x in train_examples
            ]
