"""Shared builders for the auxiliary-head tests.

``test_score_head.py`` and ``test_aux_heads.py`` both need a tiny CPU Blokus net at a
pinned seed, a run config to wrap it in, and a handful of real training examples. One
copy, so the two files cannot drift on what "the same net" means — which matters here
more than usual, since half of what these tests assert is that two nets built the same
way are byte-identical.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from alphablokus.config import MCTSConfig, NetConfig, RunConfig
from alphablokus.games.blokusduo.nn.net import AlphaBlokusDuo
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper as BlokusDuoNNetWrapper
from tests.conftest import RecordingMetrics

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.games.blokusduo.board import BlokusDuoBoard
    from alphablokus.games.blokusduo.game import BlokusDuoGame

#: One pinned seed everywhere, so "identical initialisation" is a real assertion.
SEED = 20260730


def net_config(**overrides: object) -> NetConfig:
    """A tiny CPU net config; ``overrides`` flips the auxiliary-head knobs."""
    base = NetConfig(
        learning_rate=5e-3,
        dropout=0.0,
        epochs=1,
        batch_size=4,
        cuda=False,
        num_filters=16,
        num_residual_blocks=1,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]  # kwargs are field values


def run_config(tmp_path: Path, config: NetConfig) -> RunConfig:
    """Minimal Blokus run config wrapping ``config``."""
    return RunConfig(
        game="blokusduo",
        run_name="aux_head_test",
        num_generations=1,
        num_eps=1,
        temp_threshold=5,
        update_threshold=0.55,
        num_arena_matches=2,
        root_directory=tmp_path,
        load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=2, cpuct=1.0),
        net_config=config,
    )


def build_net(game: BlokusDuoGame, board: BlokusDuoBoard, config: NetConfig) -> AlphaBlokusDuo:
    """Seeded construction, so two nets differ only where the architecture does."""
    rows, cols = game.get_board_size()
    torch.manual_seed(SEED)
    return AlphaBlokusDuo(
        board_rows=rows,
        board_cols=cols,
        action_size=game.get_action_size(),
        num_input_channels=board.num_channels,
        config=config,
    )


def examples(game: BlokusDuoGame, board: BlokusDuoBoard, count: int) -> list:
    """``count`` trivially-distinct training examples in the stored (sparse) shape."""
    built = []
    player = 1
    for i in range(count):
        legal = np.flatnonzero(game.valid_move_masking(board, player))
        action = int(legal[i % len(legal)])
        indices = np.array([action], dtype=np.int32)
        values = np.array([1.0], dtype=np.float32)
        compact = np.asarray(board.to_compact(), dtype=np.int8)
        built.append((compact, (indices, values), float((-1) ** i)))
        board, player = game.get_next_state(board, player, action)
        board = game.get_canonical_form(board, player)
        player = 1
    return built


def train_once(
    game: BlokusDuoGame,
    config: NetConfig,
    tmp_path: Path,
    training_examples: list,
    **targets: object,
) -> tuple[BlokusDuoNNetWrapper, RecordingMetrics]:
    """One seeded ``train()`` call, returning the wrapper and every logged loss row."""
    torch.manual_seed(SEED)
    wrapper = BlokusDuoNNetWrapper(game, run_config(tmp_path, config))
    metrics = RecordingMetrics()
    torch.manual_seed(SEED + 1)
    wrapper.train(training_examples, generation=1, metrics=metrics, **targets)  # type: ignore[arg-type]
    return wrapper, metrics
