"""Tests for shared ``BaseNNetWrapper`` behaviour (LR scheduler)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
from alphablokus.storage.metrics import EvalSet, MetricsCollector
from alphablokus.storage.sparse_policy import sparsify

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.games.tictactoe.game import TicTacToeGame


def _lr_sequence(config: RunConfig, game: TicTacToeGame, steps: int) -> list[float]:
    """LR seen at each scheduler step (start value first, then after each step)."""
    wrapper = NNetWrapper(game, config)
    assert wrapper.scheduler is not None
    seq = [wrapper.optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        wrapper.scheduler.step()
        seq.append(wrapper.optimizer.param_groups[0]["lr"])
    return seq


def test_cosine_floor_never_drops_below_eta_min(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """With a non-zero ``lr_eta_min`` the cosine LR never anneals below it."""
    eta_min = 1e-4
    net_config = replace(test_config.net_config, lr_scheduler="cosine", lr_eta_min=eta_min)
    config = replace(test_config, num_generations=60, net_config=net_config)

    seq = _lr_sequence(config, ttt_game, steps=config.num_generations)

    assert min(seq) >= eta_min, f"LR dropped below floor {eta_min}: min={min(seq)}"
    # The final LR sits right at the floor (cosine reaches eta_min at T_max).
    assert seq[-1] == eta_min


def test_cosine_default_eta_min_is_unchanged(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """Default ``lr_eta_min=0.0`` reproduces the prior schedule exactly.

    The reference is a bare ``CosineAnnealingLR`` with the old hardcoded
    ``eta_min=0`` — the schedule must be identical value-for-value.
    """
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    net_config = replace(test_config.net_config, lr_scheduler="cosine")  # lr_eta_min defaults to 0.0
    config = replace(test_config, num_generations=60, net_config=net_config)
    assert config.net_config.lr_eta_min == 0.0

    seq = _lr_sequence(config, ttt_game, steps=config.num_generations)

    # Reference: the pre-change construction (eta_min=0, same T_max, same peak LR).
    ref_optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=config.net_config.learning_rate)
    ref_scheduler = CosineAnnealingLR(ref_optimizer, T_max=config.num_generations * config.net_config.epochs)
    ref = [ref_optimizer.param_groups[0]["lr"]]
    for _ in range(config.num_generations):
        ref_scheduler.step()
        ref.append(ref_optimizer.param_groups[0]["lr"])

    assert seq == ref, "Default lr_eta_min=0.0 changed the cosine schedule"


def _ttt_eval_positions(game: TicTacToeGame, count: int) -> list[np.ndarray]:
    """A few distinct canonical TTT compact boards from random short games."""
    rng = np.random.default_rng(0)
    positions: list[np.ndarray] = []
    for _ in range(count):
        board = game.initialise_board()
        player = 1
        for _ in range(rng.integers(1, 4)):
            legal = np.flatnonzero(game.valid_move_masking(board, player))
            board, player = game.get_next_state(board, player, int(rng.choice(legal)))
        positions.append(game.get_canonical_form(board, player).to_compact())
    return positions


def _uniform_over_legal(game: TicTacToeGame, compact: np.ndarray) -> np.ndarray:
    board = game.board_from_compact(compact)
    valids = game.valid_move_masking(board, 1)
    total = valids.sum()
    return (valids / total).astype(np.float32) if total > 0 else valids.astype(np.float32)


def _make_eval_set(game: TicTacToeGame, compacts: list[np.ndarray], *, with_compact: bool) -> EvalSet:
    boards = np.array([game.encode_compact(c) for c in compacts])
    policies = np.array([_uniform_over_legal(game, c) for c in compacts])
    values = np.zeros(len(compacts), dtype=np.float32)
    return EvalSet(
        boards=boards,
        target_policies=policies,
        target_values=values,
        compact_boards=np.array(compacts) if with_compact else None,
    )


def test_mcts_agreement_returns_none_without_compact_boards(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """No compact boards ⇒ nothing to re-search ⇒ the diagnostic is skipped."""
    wrapper = NNetWrapper(ttt_game, test_config)
    compacts = _ttt_eval_positions(ttt_game, 4)
    eval_set = _make_eval_set(ttt_game, compacts, with_compact=False)
    assert wrapper._compute_mcts_agreement(eval_set) is None


def test_mcts_agreement_is_computed_and_logged(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """With compact boards the net-vs-own-MCTS agreement is computed and logged."""
    wrapper = NNetWrapper(ttt_game, test_config)
    compacts = _ttt_eval_positions(ttt_game, 4)
    eval_set = _make_eval_set(ttt_game, compacts, with_compact=True)

    # Direct diagnostic: fractions in [0, 1].
    agreement = wrapper._compute_mcts_agreement(eval_set)
    assert agreement is not None
    top1, top5 = agreement
    assert 0.0 <= top1 <= top5 <= 1.0

    # End-to-end: train one generation and confirm the new series is persisted.
    examples = [(compact, sparsify(_uniform_over_legal(ttt_game, compact)), 0.0) for compact in compacts]
    metrics = MetricsCollector(config=test_config)
    wrapper.train(examples, generation=1, metrics=metrics, eval_set=eval_set)

    records = metrics._policy_accuracy_records
    assert records, "no policy-accuracy records logged"
    assert all("mcts_top1_accuracy" in r and "mcts_top5_accuracy" in r for r in records)
    assert all(0.0 <= r["mcts_top1_accuracy"] <= 1.0 for r in records)
