"""Validate full-pass epoch training over the rolling buffer (R5).

``train`` runs ``epochs`` full, shuffled passes over the *whole* buffer — every
position is trained on exactly ``epochs`` times per generation (use all the
data). These tests lock that batch count and confirm it scales with the buffer
(not a fixed sample count).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.config import RunConfig
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper


class _CountingMetrics:
    """Minimal metrics stand-in that counts mini-batches via log_training."""

    def __init__(self) -> None:
        self.batches = 0

    def log_training(self, **_kwargs: object) -> None:
        self.batches += 1

    def log_training_throughput(self, **_kwargs: object) -> None:
        pass


def _buffer(action_size: int, n: int) -> list:
    """A buffer of ``n`` compact TTT positions with valid dense policies."""
    examples = []
    for i in range(n):
        board = np.zeros((3, 3), dtype=np.int8)
        board.flat[i % 9] = 1
        policy = np.full(action_size, 1.0 / action_size, dtype=np.float64)
        examples.append((board, policy, float((-1) ** i)))
    return examples


def _expected_batches(buffer_size: int, batch_size: int, epochs: int) -> int:
    # DataLoader without drop_last yields ceil(buffer/batch) batches per epoch.
    per_epoch = -(-buffer_size // batch_size)
    return per_epoch * epochs


def test_full_pass_batch_count(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """Batch count equals epochs × ceil(buffer_positions / batch_size)."""
    config = replace(  # batch_size 4 from the fixture; 3 full passes
        test_config, net_config=replace(test_config.net_config, epochs=3),
    )
    nnet = NNetWrapper(ttt_game, config)
    action_size = ttt_game.get_action_size()

    buffer = _buffer(action_size, 40)
    metrics = _CountingMetrics()
    nnet.train(buffer, generation=0, metrics=metrics)

    assert metrics.batches == _expected_batches(40, config.net_config.batch_size, 3)


def test_batch_count_scales_with_buffer(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """A bigger buffer means more batches — full passes use all the data."""
    config = replace(test_config, net_config=replace(test_config.net_config, epochs=1))
    action_size = ttt_game.get_action_size()

    counts = []
    for buffer_size in (20, 80):
        nnet = NNetWrapper(ttt_game, config)
        metrics = _CountingMetrics()
        nnet.train(_buffer(action_size, buffer_size), generation=0, metrics=metrics)
        counts.append(metrics.batches)

    bs = config.net_config.batch_size
    assert counts == [_expected_batches(20, bs, 1), _expected_batches(80, bs, 1)]
    assert counts[1] > counts[0]
