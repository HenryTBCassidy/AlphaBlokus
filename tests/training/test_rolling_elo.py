"""Rolling arena-derived Elo: chain, clamp, reject-no-advance, resume rebuild.

Covers the mechanism in ``docs/plans/archive/arena-derived-elo.md`` (S1–S3):
the candidate's Elo each generation is ``incumbent + compute_elo(arena)``, the
benchmark rolls forward only on acceptance, and on resume it is reconstructed
from the last *accepted* net's logged Elo. Real objects throughout (a real
``Coach`` and ``MetricsCollector``) — no mocks.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from alphablokus.evaluation.elo import compute_elo
from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
from alphablokus.storage.metrics import MetricsCollector
from alphablokus.training.coach import Coach, reconstruct_benchmark_elo

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.games.tictactoe.game import TicTacToeGame


def _coach(ttt_game: TicTacToeGame, config: RunConfig) -> Coach:
    """A real Coach on a tiny TTT config (checkpoints land under tmp_path)."""
    return Coach(ttt_game, NNetWrapper(ttt_game, config), config)


def test_chain_rolls_only_on_acceptance(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """Each gen's rolling Elo = incumbent + compute_elo; benchmark rolls on accept only."""
    coach = _coach(ttt_game, test_config)
    anchor = float(test_config.elo_baseline_rating)
    assert coach._benchmark_elo == anchor

    # Gen 1 accepted at 7-3-0 → +compute_elo above the anchor; benchmark rolls.
    delta1, _ = compute_elo(7, 3, 0)
    candidate1 = coach._record_rolling_elo(1, 7, 3, 0, accepted=True)
    assert candidate1 == pytest.approx(anchor + delta1)
    assert coach._benchmark_elo == pytest.approx(anchor + delta1)

    # Gen 2 accepted at 6-4-0 → measured against the *new* incumbent, rolls again.
    incumbent_after_1 = coach._benchmark_elo
    delta2, _ = compute_elo(6, 4, 0)
    candidate2 = coach._record_rolling_elo(2, 6, 4, 0, accepted=True)
    assert candidate2 == pytest.approx(incumbent_after_1 + delta2)
    assert coach._benchmark_elo == pytest.approx(incumbent_after_1 + delta2)


def test_clamp_on_sweep_is_finite(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """A 10-0-0 sweep yields the clamped ~+1200 delta, never inf."""
    coach = _coach(ttt_game, test_config)
    saturation, _ = compute_elo(10, 0, 0)
    candidate = coach._record_rolling_elo(1, 10, 0, 0, accepted=True)
    assert candidate == pytest.approx(test_config.elo_baseline_rating + saturation)
    assert saturation == pytest.approx(400 * math.log10(0.999 / 0.001), abs=1e-6)


def test_reject_does_not_advance_benchmark(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """A rejected gen logs a point but leaves the benchmark for the next gen."""
    coach = _coach(ttt_game, test_config)
    anchor = float(test_config.elo_baseline_rating)

    # Rejected gen 1: benchmark unchanged, but the provisional candidate Elo is
    # still returned/logged.
    delta_rej, _ = compute_elo(4, 6, 0)
    provisional = coach._record_rolling_elo(1, 4, 6, 0, accepted=False)
    assert provisional == pytest.approx(anchor + delta_rej)
    assert coach._benchmark_elo == anchor  # held

    # Gen 2 accepted: measured against the *unchanged* anchor, not the rejected
    # provisional value.
    delta_acc, _ = compute_elo(7, 3, 0)
    candidate2 = coach._record_rolling_elo(2, 7, 3, 0, accepted=True)
    assert candidate2 == pytest.approx(anchor + delta_acc)
    assert coach._benchmark_elo == pytest.approx(anchor + delta_acc)


def _write_rolling_history(config: RunConfig, points: list[tuple[int, float, bool]]) -> None:
    """Persist a rolling-Elo history via the real MetricsCollector + flush.

    ``points`` are ``(generation, rolling_elo, accepted)`` tuples; each is
    flushed to its own hive partition exactly as a real run would.
    """
    metrics = MetricsCollector(config=config)
    for generation, rolling_elo, accepted in points:
        metrics.log_rolling_elo(
            generation=generation,
            rolling_elo=rolling_elo,
            incumbent_elo=0.0,
            elo_delta=0.0,
            score_rate=0.5,
            wins=1,
            losses=1,
            draws=0,
            accepted=accepted,
        )
        metrics.flush(config, generation)


def test_resume_reconstruction_picks_last_accepted(test_config: RunConfig) -> None:
    """Reconstruction returns the last *accepted* net's Elo, not the last logged point."""
    _write_rolling_history(
        test_config,
        [
            (1, 500.0, True),
            (2, 480.0, False),  # rejected
            (3, 560.0, True),  # last accepted
            (4, 550.0, False),  # rejected — last logged, must NOT be chosen
        ],
    )
    assert reconstruct_benchmark_elo(test_config) == pytest.approx(560.0)


def test_resume_reconstruction_falls_back_to_anchor_when_empty(test_config: RunConfig) -> None:
    """No history → the configured anchor rating."""
    # Nothing written: directory absent.
    assert reconstruct_benchmark_elo(test_config) == pytest.approx(test_config.elo_baseline_rating)

    # History exists but nothing accepted yet → still the anchor.
    _write_rolling_history(test_config, [(1, 320.0, False), (2, 300.0, False)])
    assert reconstruct_benchmark_elo(test_config) == pytest.approx(test_config.elo_baseline_rating)


def test_resume_coach_reconstructs_benchmark(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """A resumed Coach adopts the reconstructed benchmark, a fresh one uses the anchor."""
    _write_rolling_history(test_config, [(1, 500.0, True), (2, 610.0, True)])

    fresh = _coach(ttt_game, test_config)
    assert fresh._benchmark_elo == pytest.approx(test_config.elo_baseline_rating)

    resumed = Coach(ttt_game, NNetWrapper(ttt_game, test_config), test_config, resume=True)
    assert resumed._benchmark_elo == pytest.approx(610.0)
