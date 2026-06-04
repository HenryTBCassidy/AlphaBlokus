"""Dirichlet root-noise tests.

Pins the exploration-noise contract: off by default (bit-identical search),
on => valid perturbation of the root priors only, and reproducible under a seed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.config import MCTSConfig, NetConfig, RunConfig
from core.mcts import MCTS
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper

if TYPE_CHECKING:
    from pathlib import Path


def _nnet(tmp_path: Path) -> tuple[TicTacToeGame, NNetWrapper]:
    import torch

    torch.manual_seed(0)
    game = TicTacToeGame()
    net_cfg = NetConfig(
        learning_rate=1e-3, dropout=0.3, epochs=1, batch_size=4, cuda=False,
        num_filters=32, num_residual_blocks=1,
    )
    run_cfg = RunConfig(
        game="tictactoe", run_name="t", num_generations=1, num_eps=2, temp_threshold=5,
        update_threshold=0.55, max_queue_length=10, num_arena_matches=2,
        max_generations_lookback=1, root_directory=tmp_path, load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=2, cpuct=1.0), net_config=net_cfg,
    )
    return game, NNetWrapper(game, run_cfg)


def test_epsilon_zero_is_a_noop(tmp_path: Path) -> None:
    """With ε=0, requesting root noise must change nothing — the search stays
    bit-identical to no-noise (so arena/Elo and existing behaviour are safe)."""
    game, nnet = _nnet(tmp_path)
    board = game.get_canonical_form(game.initialise_board(), 1)
    config = MCTSConfig(num_mcts_sims=25, cpuct=1.0, dirichlet_epsilon=0.0)

    def run(add_noise: bool) -> dict:
        np.random.seed(7)
        mcts = MCTS(game, nnet, config)
        mcts.get_action_prob(board, temp=1, add_root_noise=add_noise)
        return dict(mcts.visit_counts)

    assert run(add_noise=True) == run(add_noise=False)


def test_noise_perturbs_only_root_legal_priors(tmp_path: Path) -> None:
    """ε>0 changes the root priors, keeps them a valid distribution, and never
    puts mass on illegal moves."""
    game, nnet = _nnet(tmp_path)
    # A position with some illegal moves.
    board, player = game.get_next_state(game.initialise_board(), 1, 0)
    canonical = game.get_canonical_form(board, player)
    s = game.state_key(canonical)
    valids = game.valid_move_masking(canonical, 1)

    # Baseline priors (no noise).
    plain = MCTS(game, nnet, MCTSConfig(num_mcts_sims=4, cpuct=1.0, dirichlet_epsilon=0.0))
    plain.get_action_prob(canonical, temp=1, add_root_noise=False)
    base_priors = plain.policy_priors[s].copy()

    # Noised priors.
    np.random.seed(1)
    noised = MCTS(game, nnet, MCTSConfig(num_mcts_sims=4, cpuct=1.0,
                                         dirichlet_epsilon=0.25, dirichlet_alpha=0.3))
    noised.get_action_prob(canonical, temp=1, add_root_noise=True)
    noisy_priors = noised.policy_priors[s]

    assert not np.allclose(base_priors, noisy_priors), "noise did not change root priors"
    assert abs(float(noisy_priors.sum()) - 1.0) < 1e-6
    assert float(noisy_priors[np.asarray(valids) == 0].sum()) == 0.0, "mass on illegal moves"


def test_noise_is_reproducible_under_seed(tmp_path: Path) -> None:
    """Same net + same seed + ε>0 => identical visit counts (noise is seeded)."""
    game, nnet = _nnet(tmp_path)
    board = game.get_canonical_form(game.initialise_board(), 1)
    config = MCTSConfig(num_mcts_sims=25, cpuct=1.0, dirichlet_epsilon=0.25, dirichlet_alpha=0.3)

    def run() -> dict:
        np.random.seed(42)
        mcts = MCTS(game, nnet, config)
        mcts.get_action_prob(board, temp=1, add_root_noise=True)
        return dict(mcts.visit_counts)

    assert run() == run()
