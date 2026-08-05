"""Draws score half a win, everywhere (`scripts/pentobi_benchmark`)."""

from __future__ import annotations

import pytest

from scripts.pentobi_benchmark import (
    PENTOBI_DUO_SIMS_BY_LEVEL,
    SCORING_WIN_DRAW_HALF,
    compute_headline_metrics,
    level_result,
)


def test_draws_count_half_in_the_score_and_not_at_all_in_win_rate() -> None:
    """The two definitions used to be mixed inside one report; both are now explicit."""
    row = level_result(9, net_wins=20, pentobi_wins=76, draws=4, records=[])
    assert row["score"] == pytest.approx(0.22)  # (20 + 2) / 100
    assert row["win_rate"] == pytest.approx(0.20)  # unchanged, wins only
    assert row["games"] == 100


def test_a_drawn_game_is_worth_half_a_won_game() -> None:
    all_draws = level_result(1, net_wins=0, pentobi_wins=0, draws=10, records=[])
    assert all_draws["score"] == pytest.approx(0.5)
    assert all_draws["win_rate"] == pytest.approx(0.0)


def test_interval_brackets_the_score_not_the_win_rate() -> None:
    row = level_result(9, net_wins=20, pentobi_wins=76, draws=4, records=[])
    low, high = row["ci"]
    assert low < row["score"] < high


def test_headline_metrics_use_the_score_definition() -> None:
    """A level won only on draws now counts as beaten; it used not to."""
    per_level = [
        level_result(1, net_wins=45, pentobi_wins=40, draws=15, records=[]),  # score 0.525
        level_result(2, net_wins=40, pentobi_wins=60, draws=0, records=[]),  # score 0.40
    ]
    metrics = compute_headline_metrics(per_level)
    assert metrics["pentobi_level"] == 1  # 0.525 > 0.5 on score; 0.45 on wins alone would not be
    assert metrics["scoring"] == SCORING_WIN_DRAW_HALF


def test_weighted_score_weights_by_level_and_counts_draws_half() -> None:
    per_level = [
        level_result(1, net_wins=50, pentobi_wins=50, draws=0, records=[]),
        level_result(9, net_wins=20, pentobi_wins=76, draws=4, records=[]),
    ]
    metrics = compute_headline_metrics(per_level)
    # (1*50 + 9*22) / (1*100 + 9*100)
    assert metrics["weighted_score"] == pytest.approx((50 + 198) / 1000)


def test_no_games_does_not_divide_by_zero() -> None:
    row = level_result(5, net_wins=0, pentobi_wins=0, draws=0, records=[])
    assert row["score"] == 0.0
    assert row["ci"] == (0.0, 0.0)
    assert compute_headline_metrics([row])["weighted_score"] == 0.0


def test_colour_split_is_recorded_when_supplied() -> None:
    """Needed to fit an unbiased Elo; aggregation used to discard it."""
    row = level_result(9, net_wins=20, pentobi_wins=76, draws=4, records=[], white_games=50, white_wins=18)
    assert row["white_games"] == 50
    assert row["white_wins"] == 18


def test_timing_summaries_report_median_alongside_mean() -> None:
    """Pentobi's per-move cost is skewed, so the mean alone misleads."""
    row = level_result(9, 1, 1, 0, [], pentobi_seconds=[0.5, 0.5, 26.0, 26.0], net_seconds=[1.3, 1.4])
    assert row["pentobi_seconds_per_move"]["median"] == pytest.approx(13.25)
    assert row["pentobi_seconds_per_move"]["mean"] == pytest.approx(13.25)
    assert row["pentobi_seconds_per_move"]["moves"] == 4
    assert row["net_seconds_per_move"]["moves"] == 2


def test_timing_keys_absent_when_not_measured() -> None:
    row = level_result(9, 1, 1, 0, [])
    assert "pentobi_seconds_per_move" not in row
    assert "net_seconds_per_move" not in row


def test_pentobi_level_budgets_match_the_engine_source() -> None:
    """From ``counts_duo`` in libpentobi_mcts/Player.cpp (Pentobi 31.0-dev).

    Recorded so a result says what it faced. The L6->L7 jump is 30x while every
    other step is 3-8x, which is why the ladder's levels are not an evenly spaced
    difficulty scale and reading them as one was misleading.
    """
    assert PENTOBI_DUO_SIMS_BY_LEVEL == {
        1: 3,
        2: 21,
        3: 77,
        4: 213,
        5: 861,
        6: 7280,
        7: 221867,
        8: 1109339,
        9: 5546695,
    }
    assert PENTOBI_DUO_SIMS_BY_LEVEL[7] / PENTOBI_DUO_SIMS_BY_LEVEL[6] > 30
