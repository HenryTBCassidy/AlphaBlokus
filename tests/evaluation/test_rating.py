"""Tests for the pool BayesElo fit (``evaluation/rating.py``).

Real numbers, no mocks. The recovery test synthesises game counts from known
ground-truth ratings via the exact Bradley–Terry probabilities, so the fit has a
deterministic target to hit.
"""

from __future__ import annotations

import math

from alphablokus.evaluation.rating import RatingResult, fit_bayeselo


def _expected_win(rating_i: float, rating_j: float) -> float:
    """Bradley–Terry probability that i beats j given Elo ratings."""
    gamma_i = 10.0 ** (rating_i / 400.0)
    gamma_j = 10.0 ** (rating_j / 400.0)
    return gamma_i / (gamma_i + gamma_j)


def _round_robin_from_truth(
    truth: dict[str, float],
    games_per_pair: int,
) -> tuple[list[str], dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Build (players, wins, draws) matching expected counts from true ratings."""
    players = list(truth)
    wins: dict[tuple[str, str], int] = {}
    draws: dict[tuple[str, str], int] = {}
    for a_idx, a in enumerate(players):
        for b in players[a_idx + 1 :]:
            p = _expected_win(truth[a], truth[b])
            a_wins = round(p * games_per_pair)
            wins[(a, b)] = a_wins
            wins[(b, a)] = games_per_pair - a_wins
            draws[(a, b)] = 0
    return players, wins, draws


def test_recovers_ground_truth_ratings() -> None:
    """Fit recovers ordering and pairwise gaps of known ratings within ~30 Elo."""
    truth = {"p0": 0.0, "p1": 100.0, "p2": 200.0, "p3": 300.0}
    players, wins, draws = _round_robin_from_truth(truth, games_per_pair=2000)

    result = fit_bayeselo(players, wins, draws, anchor="p0", anchor_rating=0.0)

    # Ordering preserved.
    ordered = sorted(players, key=lambda p: result.ratings[p])
    assert ordered == ["p0", "p1", "p2", "p3"]
    # Gaps vs the anchor recovered to within ~30 Elo.
    for p, true_rating in truth.items():
        assert abs(result.ratings[p] - true_rating) < 30.0
    assert result.converged


def test_undefeated_and_winless_players_get_finite_ratings() -> None:
    """The prior keeps a 100%-win and a 0%-win player finite — no ±1200 clamp.

    This is the entire point vs ``compute_elo``, which would saturate both.
    """
    players = ["top", "mid", "bot"]
    wins = {
        ("top", "mid"): 20,
        ("mid", "top"): 0,
        ("top", "bot"): 20,
        ("bot", "top"): 0,
        ("mid", "bot"): 20,
        ("bot", "mid"): 0,
    }
    draws: dict[tuple[str, str], int] = {}

    result = fit_bayeselo(players, wins, draws, prior_games=2.0, anchor="mid")

    for rating in result.ratings.values():
        assert math.isfinite(rating)
    assert result.ratings["top"] > result.ratings["mid"] > result.ratings["bot"]


def test_anchor_pins_chosen_player_exactly() -> None:
    """``anchor`` places that player's rating exactly at ``anchor_rating``."""
    truth = {"gen0": 0.0, "gen1": 150.0, "gen2": 300.0}
    players, wins, draws = _round_robin_from_truth(truth, games_per_pair=500)

    result = fit_bayeselo(players, wins, draws, anchor="gen0", anchor_rating=400.0)
    assert result.ratings["gen0"] == 400.0
    # Others shift by the same offset, preserving gaps.
    assert result.ratings["gen2"] > result.ratings["gen1"] > 400.0


def test_deterministic() -> None:
    """Same input yields identical output."""
    truth = {"a": 0.0, "b": 120.0, "c": 240.0}
    players, wins, draws = _round_robin_from_truth(truth, games_per_pair=300)
    r1 = fit_bayeselo(players, wins, draws, anchor="a")
    r2 = fit_bayeselo(players, wins, draws, anchor="a")
    assert r1 == r2
    assert isinstance(r1, RatingResult)


def test_single_player_sits_at_anchor() -> None:
    """A one-player pool is degenerate but must not blow up — it sits at R=0."""
    result = fit_bayeselo(["only"], {}, {})
    assert result.ratings["only"] == 0.0
    assert result.converged
