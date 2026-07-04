"""Frozen held-out eval set for per-epoch network diagnostics.

Built once from generation 1's self-play and persisted, so every epoch and
every resumed run measures the network against the same positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from alphablokus.storage.metrics import EvalSet
from alphablokus.storage.sparse_policy import as_dense

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.interfaces import IGame, IOracle
    from alphablokus.selfplay.episode import ProcessedExample


def build_or_load_eval_set(
    config: RunConfig,
    game: IGame,
    oracle: IOracle | None,
    train_examples: list[ProcessedExample],
    size: int,
) -> EvalSet | None:
    """Build (or load from disk) the frozen held-out eval set.

    When the game has a perfect-play ``oracle``, the self-play-derived targets
    are replaced with oracle targets: each position's ``target_policies`` row
    is uniform over all game-theoretically optimal actions, and
    ``target_values`` is the true minimax value of the position. This makes
    the per-generation top-K agreement plot answer the right question ("does
    the net pick a truly optimal move?") rather than chasing gen-1's noisy
    MCTS targets, which is what was making that curve dip over training.

    Without an oracle the eval set targets are MCTS visit distributions and
    final game outcomes recorded during gen 1's self-play.

    Sampled once from gen 1's training examples and saved to
    ``config.eval_set_directory`` as three numpy files: ``boards.npy``,
    ``target_policies.npy``, ``target_values.npy``. If any of the three is
    missing on disk we re-sample (which keeps things consistent between
    old runs and current ones).

    Returns:
        The eval set, or ``None`` when there is nothing on disk and no
        training examples to sample from yet.
    """
    eval_dir = config.eval_set_directory
    boards_path = eval_dir / "boards.npy"
    policies_path = eval_dir / "target_policies.npy"
    values_path = eval_dir / "target_values.npy"
    # Marker file: tells us *how* the targets were generated. We refuse to
    # reuse an on-disk eval set whose targets don't match the current
    # scheme — otherwise an old "selfplay-targets" file would silently
    # poison the metrics on a run that now expects oracle targets.
    marker_path = eval_dir / "targets_kind.txt"
    expected_kind = "minimax_v1" if oracle is not None else "selfplay_v1"
    if (
        boards_path.exists()
        and policies_path.exists()
        and values_path.exists()
        and marker_path.exists()
        and marker_path.read_text().strip() == expected_kind
    ):
        eval_set = EvalSet(
            boards=np.load(boards_path),
            target_policies=np.load(policies_path),
            target_values=np.load(values_path),
        )
        logger.info("Loaded eval set ({} positions, kind={}) from {}", len(eval_set), expected_kind, eval_dir)
        return eval_set

    if not train_examples:
        return None

    # Sample positions from the training examples (capped to actual size).
    # Boards are stored **compact** (``to_compact()``); the eval set feeds
    # boards straight to the network, so encode the sampled compact boards
    # to dense planes here. Policies are stored sparse (indices, values) —
    # densify to the full action-space vector the eval set holds.
    action_size = game.get_action_size()
    n = min(size, len(train_examples))
    rng = np.random.default_rng(seed=config.seed or 0)
    idx = rng.choice(len(train_examples), size=n, replace=False)
    sampled = [train_examples[i] for i in idx]
    sampled_compact = [ex[0] for ex in sampled]
    sampled_boards = np.array([game.encode_compact(b) for b in sampled_compact])
    target_policies = np.array([as_dense(ex[1], action_size) for ex in sampled])
    target_values = np.array([ex[2] for ex in sampled])

    if oracle is not None:
        target_policies, target_values = oracle.eval_targets(
            sampled_compact,
            action_size=action_size,
        )

    eval_set = EvalSet(
        boards=sampled_boards,
        target_policies=target_policies,
        target_values=target_values,
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    np.save(boards_path, eval_set.boards)
    np.save(policies_path, eval_set.target_policies)
    np.save(values_path, eval_set.target_values)
    marker_path.write_text(expected_kind)
    logger.info("Built eval set ({} positions, kind={}) → {}", n, expected_kind, eval_dir)
    return eval_set
