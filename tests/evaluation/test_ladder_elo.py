"""Colour- and draw-aware ladder Elo (`evaluation/ladder_elo.py`)."""

from __future__ import annotations

import math

import pytest

from alphablokus.evaluation.ladder_elo import (
    colour_advantage_elo,
    opponent_elo,
    pooled_score,
    rung_from_level,
    slope_elo_per_doubling,
)


def test_even_score_means_even_strength() -> None:
    assert opponent_elo(0.5, colour_elo=0.0) == pytest.approx(0.0, abs=1e-3)
    assert opponent_elo(0.5, colour_elo=190.0) == pytest.approx(0.0, abs=1e-3)


def test_zero_colour_advantage_reproduces_the_naive_inversion() -> None:
    """With no first-mover edge the correct model collapses to -400*log10(1/s - 1)."""
    for score in (0.22, 0.405, 0.62, 0.77):
        naive = -400.0 * math.log10(1.0 / score - 1.0)
        assert opponent_elo(score, colour_elo=0.0) == pytest.approx(-naive, abs=0.05)


def test_colour_advantage_makes_the_true_gap_larger() -> None:
    """The headline correction: pooling halves of each colour flattens the curve.

    A naive inversion of gen-40's 0.22 at level 9 gives ~220 Elo. Because the
    ladder pools both colours and Blokus Duo's first mover takes ~75% of decisive
    games, the real gap is larger — around 280 Elo. Understating it is the
    flattering direction, which is why this is worth a test.
    """
    naive = opponent_elo(0.22, colour_elo=0.0)
    corrected = opponent_elo(0.22, colour_elo=colour_advantage_elo(0.75))
    assert naive == pytest.approx(220, abs=10)
    assert corrected > naive + 40
    assert corrected == pytest.approx(280, abs=25)


def test_correction_only_bites_away_from_an_even_score() -> None:
    """At 0.5 the flattening has nothing to flatten; it grows toward the tails."""
    colour = colour_advantage_elo(0.75)
    gaps = [abs(opponent_elo(s, colour) - opponent_elo(s, 0.0)) for s in (0.5, 0.55, 0.7, 0.85)]
    assert gaps[0] < 1.0
    assert gaps == sorted(gaps), f"correction should grow monotonically toward the tail, got {gaps}"


def test_pooled_score_is_flatter_than_a_plain_logistic() -> None:
    """The mechanism behind the correction, asserted directly."""
    colour = colour_advantage_elo(0.75)
    for gap in (100.0, 300.0):
        assert pooled_score(gap, colour) > pooled_score(gap, 0.0)


def test_pooled_score_is_symmetric_in_colour_advantage() -> None:
    """Which side gets the advantage cannot matter once games are split evenly."""
    assert pooled_score(150.0, 190.0) == pytest.approx(pooled_score(150.0, -190.0))


def test_round_trip_through_the_forward_model() -> None:
    colour = colour_advantage_elo(0.72)
    for gap in (-300.0, -50.0, 0.0, 120.0, 400.0):
        assert opponent_elo(pooled_score(gap, colour), colour) == pytest.approx(gap, abs=0.01)


def test_colour_advantage_from_an_even_split_is_zero() -> None:
    assert colour_advantage_elo(0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_degenerate_scores_are_rejected_rather_than_returning_infinity(bad: float) -> None:
    with pytest.raises(ValueError, match="must be in"):
        opponent_elo(bad, colour_elo=0.0)
    with pytest.raises(ValueError, match="must be in"):
        colour_advantage_elo(bad)


def test_rung_interval_is_as_wide_as_the_noise_floor_implies() -> None:
    """Documents why per-level Elo comparisons are unusable at 100 games.

    Near a score of 0.2 one percentage point is ~11 Elo and the binomial SE is
    ~4pp, so a single rung's 95% interval spans roughly 170 Elo. The saturation
    reading of L7/L8/L9 (+275/+230/+220) sat entirely inside this.
    """
    rung = rung_from_level(9, score=0.22, games=100, colour_elo=colour_advantage_elo(0.75))
    assert rung.ci_low < rung.opponent_elo < rung.ci_high
    assert 120 < rung.ci_width < 260, f"expected a ~170 Elo interval, got {rung.ci_width:.0f}"


def test_more_games_narrow_the_interval() -> None:
    colour = colour_advantage_elo(0.75)
    narrow = rung_from_level(9, score=0.22, games=400, colour_elo=colour)
    wide = rung_from_level(9, score=0.22, games=100, colour_elo=colour)
    assert narrow.ci_width < wide.ci_width
    # Quadrupling the games should roughly halve the interval.
    assert narrow.ci_width == pytest.approx(wide.ci_width / 2, rel=0.2)


def test_rung_rejects_zero_games() -> None:
    with pytest.raises(ValueError, match="games must be positive"):
        rung_from_level(9, score=0.22, games=0, colour_elo=0.0)


def test_slope_recovers_a_planted_elo_per_doubling() -> None:
    """A synthetic ladder with a known slope must be recovered."""
    sims = {5: 861, 6: 7280, 7: 221867, 9: 5546695}
    planted = 30.0
    base = math.log2(sims[5])
    rungs = [
        rung_from_level(level, score=pooled_score(planted * (math.log2(n) - base), 0.0), games=100, colour_elo=0.0)
        for level, n in sims.items()
    ]
    slope = slope_elo_per_doubling(rungs, sims)
    assert slope is not None
    assert slope == pytest.approx(planted, rel=0.05)


def test_slope_needs_two_known_budgets() -> None:
    rung = rung_from_level(9, score=0.22, games=100, colour_elo=0.0)
    assert slope_elo_per_doubling([rung], {9: 5546695}) is None
    assert slope_elo_per_doubling([rung], {}) is None
