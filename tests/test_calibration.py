"""Calibration cost model (alphablokus/calibration.py).

The £-arithmetic is pure and pinned here; the measurement helpers run the real
training loop (TTT for speed) and — when the jax extra is present — the real
jax self-play backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from alphablokus.calibration import (
    CostEstimate,
    NetSizeMeasurement,
    estimate_costs,
    format_markdown_table,
    measure_train_seconds_per_position,
    parse_net_sizes,
    recommend,
)
from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.games.tictactoe.game import TicTacToeGame


def _measurement(name: str, params: int, games_per_s: float | None, train_s: float) -> NetSizeMeasurement:
    return NetSizeMeasurement(
        name=name,
        num_filters=64,
        num_residual_blocks=4,
        parameters=params,
        selfplay_games_per_s=games_per_s,
        train_seconds_per_position=train_s,
        positions_per_game=29.0,
    )


def test_parse_net_sizes_mixes_presets_and_specs() -> None:
    assert parse_net_sizes("small, 192x12") == [("small", 64, 4), ("192x12", 192, 12)]


def test_parse_net_sizes_rejects_junk() -> None:
    with pytest.raises(ValueError, match="Bad net size"):
        parse_net_sizes("small,huge")


def test_estimate_costs_arithmetic() -> None:
    # 10 games/s → 1000s self-play for 10k games; 1e-4 s/pos × (1000 games × 29) × 1 epoch = 2.9s train.
    [estimate] = estimate_costs(
        [_measurement("m", 1_000_000, 10.0, 1e-4)],
        games_per_generation=10_000,
        replay_buffer_games=1_000,
        epochs=1,
        eval_overhead_fraction=0.0,
        rate_gbp_per_hour=3.6,  # 0.001 £/s
        budget_gbp=100.0,
    )
    assert estimate.selfplay_seconds_per_generation == pytest.approx(1000.0)
    assert estimate.train_seconds_per_generation == pytest.approx(2.9)
    assert estimate.seconds_per_generation == pytest.approx(1002.9)
    assert estimate.cost_gbp_per_generation == pytest.approx(1.0029)
    assert estimate.generations_in_budget == 99
    assert estimate.total_games_in_budget == 990_000


def test_eval_overhead_scales_total() -> None:
    [plain] = estimate_costs(
        [_measurement("m", 1, 10.0, 1e-4)],
        games_per_generation=1000,
        replay_buffer_games=1000,
        epochs=1,
        eval_overhead_fraction=0.0,
        rate_gbp_per_hour=1.0,
        budget_gbp=100.0,
    )
    [padded] = estimate_costs(
        [_measurement("m", 1, 10.0, 1e-4)],
        games_per_generation=1000,
        replay_buffer_games=1000,
        epochs=1,
        eval_overhead_fraction=0.5,
        rate_gbp_per_hour=1.0,
        budget_gbp=100.0,
    )
    assert padded.seconds_per_generation == pytest.approx(plain.seconds_per_generation * 1.5)


def test_skipped_selfplay_costs_training_only() -> None:
    [estimate] = estimate_costs(
        [_measurement("m", 1, None, 1e-4)],
        games_per_generation=10_000,
        replay_buffer_games=1_000,
        epochs=2,
        eval_overhead_fraction=0.0,
        rate_gbp_per_hour=1.0,
        budget_gbp=10.0,
    )
    assert estimate.selfplay_seconds_per_generation == 0.0
    assert estimate.train_seconds_per_generation == pytest.approx(5.8)


def _estimates_for_recommend() -> list[CostEstimate]:
    small = _measurement("small", 1_000_000, 20.0, 1e-5)
    large = _measurement("large", 10_000_000, 5.0, 4e-5)
    xl = _measurement("xl", 30_000_000, 1.0, 2e-4)
    return estimate_costs(
        [small, large, xl],
        games_per_generation=5_000,
        replay_buffer_games=20_000,
        epochs=1,
        eval_overhead_fraction=0.15,
        rate_gbp_per_hour=0.7,
        budget_gbp=100.0,
    )


def test_recommend_prefers_biggest_net_that_fits_the_floor() -> None:
    estimates = _estimates_for_recommend()
    choice = recommend(estimates, min_generations=30)
    assert choice is not None
    fitting = [e for e in estimates if e.generations_in_budget >= 30]
    assert choice.measurement.parameters == max(e.measurement.parameters for e in fitting)


def test_recommend_returns_none_when_nothing_fits() -> None:
    estimates = _estimates_for_recommend()
    assert recommend(estimates, min_generations=10_000_000) is None


def test_markdown_table_marks_recommendation() -> None:
    estimates = _estimates_for_recommend()
    choice = recommend(estimates, min_generations=30)
    table = format_markdown_table(estimates, choice)
    assert table.count("recommended") == 1
    assert "small" in table and "xl" in table


def test_measure_train_seconds_per_position_runs_real_loop(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    wrapper = NNetWrapper(ttt_game, test_config)
    seconds = measure_train_seconds_per_position(wrapper, ttt_game, num_positions=64)
    assert seconds > 0
    assert seconds < 1.0  # 64 TTT positions must train in well under a minute
