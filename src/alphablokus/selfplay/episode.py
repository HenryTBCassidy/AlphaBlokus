"""Free-function self-play episode runner shared by Coach and parallel workers.

The serial loop in ``selfplay/generate.py`` and the worker pool in
``parallel/pool.py`` use this **same code path**. That equivalence is the basis of the determinism test in
``tests/parallel/test_pool.py`` — if both call sites
invoke this function with the same seed + same MCTS instance, they
produce identical training examples regardless of which process runs
the work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from alphablokus.storage.sparse_policy import sparsify

if TYPE_CHECKING:
    from alphablokus.interfaces import IGame
    from alphablokus.search.mcts import MCTS


class ProcessedExample(NamedTuple):
    """One stored self-play position — the shape ``nnet.train`` consumes.

    A ``NamedTuple`` rather than a bare tuple: the fields are read positionally in
    a dozen places (storage, the datasets, the holdout metrics), and a silent
    positional mix-up between ``value`` and ``player`` — two small numbers that
    both look plausible in either slot — is exactly the class of defect this
    project keeps paying for. It stays a tuple subclass so the existing
    ``example[0]``/``zip(*examples)`` call sites and the pickling the worker pool
    does are unaffected, and so a position still costs one tuple.

    Attributes:
        board: The position, stored **compact** (``IBoard.to_compact`` — e.g. the
            196-byte int8 placement board for Blokus, against the ~34.5 KB dense
            ``(44,14,14)`` encoding). The trainer re-encodes it lazily per
            mini-batch via ``IGame.encode_compact``.
        policy: The MCTS-improved policy, stored **sparse** as
            ``(indices, values)`` (see :mod:`alphablokus.storage.sparse_policy`)
            because the dense 17,837-vector dominated replay-buffer RAM. The
            trainer densifies it per mini-batch.
        value: The game outcome from the perspective of the side to move at this
            position; a float in ``[-1, 1]``.
        player: **Which side was actually to move** — ``+1`` White (the first
            mover), ``-1`` Black. The board is canonical (side-to-move
            perspective), so without this the absolute colour is gone: it can be
            *inferred* from piece-count parity only until the first pass, and
            passing happens in the endgame, precisely where the value signal
            matters most. Recorded explicitly for that reason
            (``docs/plans/selfplay-data-and-loop.md`` D1).
    """

    board: np.ndarray
    policy: tuple[np.ndarray, np.ndarray]
    value: float
    player: int


# One self-play game's positions, boundaries preserved so the games-sized
# replay buffer can evict whole games.
GameExamples = list[ProcessedExample]


def play_self_play_episode(
    game: IGame,
    mcts: MCTS,
    temp_threshold: int,
) -> list[ProcessedExample]:
    """Play one self-play game and return training examples for it.

    The single source of truth for the self-play episode loop. Called
    from:

    - ``alphablokus.selfplay.generate`` — when self-play runs sequentially
      in the training process.
    - ``alphablokus.parallel.pool._worker_play_self_play_episode`` — when
      self-play runs in worker processes.

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
        List of :class:`ProcessedExample` — one per position visited
        (including symmetry augmentations from ``game.get_symmetries``).
        The ``value`` field is filled in once the game ends, based on the
        perspective of the player at that position, and ``player`` records
        which side that was.
    """
    # (canonical board object, side to move, MCTS policy) per visited position;
    # the outcome is only known at the end, which is why this is not built as
    # ``ProcessedExample`` directly.
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
        symmetries = game.get_symmetries(canonical_board, np.asarray(pi))
        for symmetric_board, symmetric_pi in symmetries:
            train_examples.append((symmetric_board, current_player, symmetric_pi))

        # Sample the move from the post-temperature distribution.
        action = np.random.choice(len(pi), p=pi)
        board, current_player = game.get_next_state(board, current_player, action)

        game_result = game.get_game_ended(board, current_player)
        if game_result != 0:
            # End of game: convert (board_obj, player, pi) into the
            # ``ProcessedExample`` shape the network trainer expects. The
            # value sign flips for each position where the player to move
            # differs from the player at game end, and ``player`` carries the
            # real mover through to storage — the canonical board it is stored
            # alongside has already thrown the absolute colour away, and
            # piece-count parity cannot recover it once either side has passed
            # (``docs/plans/selfplay-data-and-loop.md`` D1).
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
                ProcessedExample(
                    board=position.to_compact(),
                    policy=sparsify(np.asarray(pi_at_position, dtype=np.float32)),
                    value=game_result * ((-1) ** (mover != current_player)),
                    player=mover,
                )
                for position, mover, pi_at_position in train_examples
            ]
