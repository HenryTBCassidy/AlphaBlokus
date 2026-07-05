"""Behaviour-locking tests for the pairwise ``compute_elo`` arithmetic.

These pin the existing single-anchor Elo function so nothing regresses when the
pool tournament (``evaluation/rating.py``) takes over as the canonical strength
curve. The headline fact these tests document is the **±1200 saturation**: once
the live net wins ~100% of games vs the frozen gen-0 baseline, ``compute_elo``
clamps and can no longer tell a strong net from a much stronger one. That
saturation is exactly why the vs-gen-0 curve flatlines, and why the pool
BayesElo fit (``fit_bayeselo``) exists.
"""

from __future__ import annotations

import math

import pytest

from alphablokus.evaluation.elo import compute_elo

# The clamp is score_rate ∈ [0.001, 0.999]; the extreme Elo magnitude is
# 400·log10(0.999/0.001) ≈ 1199.83. Any perfect (or near-perfect) score lands
# here regardless of how lopsided it really was — the saturation ceiling.
SATURATION_ELO = 400 * math.log10(0.999 / 0.001)


def test_perfect_score_saturates_at_the_clamp() -> None:
    """A 1-0-0 and a 20-0-0 sweep both return the *same* ~+1200 Elo.

    This is the whole motivation for the pool tournament: a net that wins
    every game against gen-0 is indistinguishable — under this metric — from
    one that wins every game by a far larger margin. The curve flatlines.
    """
    elo_one, rate_one = compute_elo(1, 0, 0)
    elo_many, rate_many = compute_elo(20, 0, 0)

    assert rate_one == 1.0
    assert rate_many == 1.0
    assert elo_one == pytest.approx(SATURATION_ELO, abs=1e-6)
    assert elo_one == pytest.approx(elo_many, abs=1e-6)  # 1 win == 20 wins: saturated


def test_no_games_returns_zero() -> None:
    """The ``total == 0`` guard returns a neutral ``(0.0, 0.0)``."""
    assert compute_elo(0, 0, 0) == (0.0, 0.0)


def test_symmetry_under_swapping_wins_and_losses() -> None:
    """Swapping wins and losses negates the Elo difference (draws fixed)."""
    elo, _ = compute_elo(7, 3, 2)
    elo_swapped, _ = compute_elo(3, 7, 2)
    assert elo == pytest.approx(-elo_swapped, abs=1e-9)


def test_mid_range_value_is_exact_chess_formula() -> None:
    """A resolvable 75% score maps to the textbook 400·log10(0.75/0.25) ≈ +190.8."""
    elo, rate = compute_elo(3, 1, 0)
    assert rate == pytest.approx(0.75)
    assert elo == pytest.approx(400 * math.log10(0.75 / 0.25), abs=1e-6)
    assert elo == pytest.approx(190.85, abs=0.01)


def test_draws_count_as_half() -> None:
    """Draws contribute 0.5 each to the score rate."""
    _, rate = compute_elo(2, 0, 2)
    assert rate == pytest.approx((2 + 0.5 * 2) / 4)  # 0.75
