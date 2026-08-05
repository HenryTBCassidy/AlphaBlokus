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


# --------------------------------------------------------------------------- #
# Legacy payloads: a run laddered either side of 2026-08-05 holds both
# conventions, and keep-best compares the stored numbers directly.
# --------------------------------------------------------------------------- #


def _legacy_payload() -> dict:
    """A pre-2026-08-05 ladder payload: no ``scoring`` key, draws counted as losses."""
    levels = [
        {"level": 1, "games": 100, "net_wins": 79, "pentobi_wins": 20, "draws": 1},
        {"level": 2, "games": 100, "net_wins": 60, "pentobi_wins": 38, "draws": 2},
    ]
    weighted = sum(row["level"] * row["net_wins"] for row in levels) / sum(
        row["level"] * row["games"] for row in levels
    )
    return {
        "net": "accepted_10.pth.tar",
        "levels": levels,
        "metrics": {"weighted_score": weighted, "score": 139 / 200, "pentobi_level": 2},
    }


def test_a_legacy_payload_is_rescored_from_its_own_tallies() -> None:
    """Draws must not read as a strength change when the convention changed.

    Recomputing is exact — every legacy payload stores its per-level tallies — so the
    history stays intact instead of being excluded.
    """
    from alphablokus.evaluation.ladder_selection import normalised_scores

    payload = _legacy_payload()
    weighted, score = normalised_scores(payload)
    # (1*79.5 + 2*61) / (1*100 + 2*100)
    assert weighted == pytest.approx((79.5 + 122) / 300)
    assert score == pytest.approx((79.5 + 61) / 200)
    assert weighted > payload["metrics"]["weighted_score"]  # the uplift the raw compare would credit


def test_rescoring_a_legacy_payload_removes_a_bogus_promotion() -> None:
    """The concrete defect: a new checkpoint crowned by the convention, not by strength.

    On this project's real ladder files the uplift is 0.7-1.0 pp, so a *weaker* new
    checkpoint can out-score an older one on stored numbers alone.
    """
    from alphablokus.evaluation.ladder_selection import ladder_point_from_payload, select_best

    legacy = _legacy_payload()  # rescores to 0.6717
    newer = {
        "net": "accepted_20.pth.tar",
        "levels": [
            {"level": 1, "games": 100, "net_wins": 78, "pentobi_wins": 21, "draws": 1},
            {"level": 2, "games": 100, "net_wins": 60, "pentobi_wins": 38, "draws": 2},
        ],
        "metrics": {"weighted_score": (78.5 + 122) / 300, "score": 0.695, "scoring": SCORING_WIN_DRAW_HALF},
    }
    assert newer["metrics"]["weighted_score"] > legacy["metrics"]["weighted_score"]  # stored: newer "wins"

    points = [ladder_point_from_payload(legacy), ladder_point_from_payload(newer)]
    assert select_best(points).label == "accepted_10.pth.tar"  # rescored: the older net really is better


def test_a_current_payload_is_taken_as_written() -> None:
    """No recomputation when the payload already states the convention."""
    from alphablokus.evaluation.ladder_selection import normalised_scores

    payload = {
        "net": "accepted_30.pth.tar",
        "levels": [{"level": 9, "games": 100, "net_wins": 20, "pentobi_wins": 76, "draws": 4}],
        "metrics": {"weighted_score": 0.22, "score": 0.22, "scoring": SCORING_WIN_DRAW_HALF},
    }
    assert normalised_scores(payload) == (0.22, 0.22)


def test_a_legacy_payload_without_tallies_keeps_its_stored_score() -> None:
    """Nothing better is available; dropping it would shorten the drift history."""
    from alphablokus.evaluation.ladder_selection import normalised_scores

    payload = {"net": "donor.pth.tar", "metrics": {"weighted_score": 0.31}}
    assert normalised_scores(payload) == (0.31, None)


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
