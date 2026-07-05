"""Tests for shared ``BaseNNetWrapper`` behaviour (LR scheduler)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper

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
