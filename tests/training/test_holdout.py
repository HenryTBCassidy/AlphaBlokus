"""Tests for game-level holdout splits + out-of-sample fit metrics (capacity probe)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
from alphablokus.storage.sparse_policy import sparsify
from alphablokus.training.holdout import evaluate_holdout, split_games_holdout

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.games.tictactoe.game import TicTacToeGame
    from alphablokus.selfplay.episode import GameExamples


def _fake_game(game_id: int, positions: int = 3) -> GameExamples:
    """A distinguishable fake game: the board array carries the game id."""
    return [
        (
            np.full((3, 3), game_id, dtype=np.int8),
            (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32)),
            1.0,
        )
        for _ in range(positions)
    ]


def _game_ids(games: list[GameExamples]) -> set[int]:
    return {int(game[0][0][0, 0]) for game in games}


def test_split_is_a_partition_at_game_granularity() -> None:
    games = [_fake_game(i) for i in range(20)]
    train, holdout = split_games_holdout(games, holdout_fraction=0.2, seed=7)

    assert len(train) + len(holdout) == 20
    assert len(holdout) == 4  # round(20 * 0.2)
    # No game straddles the split, and none is lost or duplicated.
    assert _game_ids(train).isdisjoint(_game_ids(holdout))
    assert _game_ids(train) | _game_ids(holdout) == set(range(20))
    # Positions within each held-out game stayed together.
    assert all(len(game) == 3 for game in holdout)


def test_split_is_deterministic_in_seed() -> None:
    games = [_fake_game(i) for i in range(30)]
    first = split_games_holdout(games, 0.1, seed=42)
    second = split_games_holdout(games, 0.1, seed=42)
    other_seed = split_games_holdout(games, 0.1, seed=43)

    assert _game_ids(first[1]) == _game_ids(second[1])
    assert _game_ids(first[1]) != _game_ids(other_seed[1])  # 3-of-30 collision is (30C3)⁻¹-unlikely


def test_split_nonzero_fraction_holds_out_at_least_one_game() -> None:
    games = [_fake_game(i) for i in range(3)]
    _train, holdout = split_games_holdout(games, holdout_fraction=0.01, seed=1)
    assert len(holdout) == 1


def test_split_zero_fraction_holds_out_nothing() -> None:
    games = [_fake_game(i) for i in range(3)]
    train, holdout = split_games_holdout(games, holdout_fraction=0.0, seed=1)
    assert holdout == [] and len(train) == 3


def test_split_validates_inputs() -> None:
    with pytest.raises(ValueError, match="holdout_fraction"):
        split_games_holdout([_fake_game(0)], holdout_fraction=1.0, seed=1)
    with pytest.raises(ValueError, match="at least one game"):
        split_games_holdout([], holdout_fraction=0.1, seed=1)


class _UniformPredictor:
    """Real (non-mock) minimal predictor: uniform policy, constant value."""

    def __init__(self, action_size: int, value: float) -> None:
        self._action_size = action_size
        self._value = value

    def predict_encoded(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = planes.shape[0]
        policies = np.full((n, self._action_size), 1.0 / self._action_size, dtype=np.float32)
        values = np.full((n,), self._value, dtype=np.float32)
        return policies, values


def test_evaluate_holdout_matches_closed_form() -> None:
    """One-hot targets vs a uniform predictor have CE = ln(A) and KL = ln(A)."""
    action_size = 10
    examples = [
        (
            np.zeros((3, 3), dtype=np.int8),
            (np.array([i % action_size], dtype=np.int32), np.array([1.0], dtype=np.float32)),
            1.0,
        )
        for i in range(5)
    ]
    predictor = _UniformPredictor(action_size, value=0.5)

    metrics = evaluate_holdout(
        predictor,
        examples,
        encode_fn=lambda board: board[np.newaxis].astype(np.float32),
        action_size=action_size,
        batch_size=2,  # exercises the batching path (5 examples → 3 batches)
    )

    assert metrics.policy_ce == pytest.approx(math.log(action_size))
    assert metrics.target_entropy == pytest.approx(0.0)  # one-hot targets
    assert metrics.policy_kl == pytest.approx(math.log(action_size))
    assert metrics.value_mse == pytest.approx(0.25)  # (1.0 − 0.5)²
    assert metrics.n_positions == 5


def test_evaluate_holdout_on_a_real_wrapper(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """A real TicTacToe wrapper produces finite metrics consistent with its own outputs."""
    wrapper = NNetWrapper(ttt_game, test_config)
    board = ttt_game.initialise_board()
    compact = ttt_game.get_canonical_form(board, 1).to_compact()
    action_size = ttt_game.get_action_size()
    uniform = np.full(action_size, 1.0 / action_size)
    examples = [(compact, sparsify(uniform), 0.0)]

    metrics = evaluate_holdout(
        wrapper,
        examples,
        encode_fn=ttt_game.encode_compact,
        action_size=action_size,
    )

    # Cross-check against the wrapper's own forward pass.
    policies, values = wrapper.predict_encoded(ttt_game.encode_compact(compact)[np.newaxis])
    expected_ce = float(-(uniform * np.log(policies[0])).sum())
    assert metrics.policy_ce == pytest.approx(expected_ce, rel=1e-5)
    assert metrics.value_mse == pytest.approx(float(values[0] ** 2), rel=1e-5)
    assert metrics.policy_kl == pytest.approx(metrics.policy_ce - metrics.target_entropy)
    assert np.isfinite(metrics.policy_ce) and np.isfinite(metrics.value_mse)


def test_evaluate_holdout_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        evaluate_holdout(
            _UniformPredictor(4, 0.0),
            [],
            encode_fn=lambda board: board,
            action_size=4,
        )
