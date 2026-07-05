"""Pool BayesElo — a maximum-a-posteriori Bradley–Terry rating fit.

The frozen-baseline ``compute_elo`` (``evaluation/elo.py``) saturates at ±1200
once a net wins ~100% of its games against a single fixed opponent, so it can no
longer separate a strong net from a much stronger one. DeepMind never rated
against a single anchor: they played games *among a pool* of checkpoints and fit
one consistent rating per player with **BayesElo** (Rémi Coulom's Bradley–Terry
fit). This module implements that fit.

Model. Each player *i* has an Elo rating ``R_i``; define ``γ_i = 10^(R_i / 400)``.
The probability *i* beats *j* is ``γ_i / (γ_i + γ_j)`` (the Bradley–Terry model,
equivalent to the logistic Elo expectation with ``c_elo = 1/400``). Draws count
as half a win to each side — standard BayesElo handling, not modelled explicitly.

Fit. Maximum likelihood by Minorization–Maximization (Hunter 2004), the
algorithm behind BayesElo. The "Bayes" is a weak prior: every player plays
``prior_games`` virtual draws against a fixed γ=1 (R=0) pseudo-player, which
keeps an undefeated or winless player's rating finite (an MLE would send it to
±∞). This is the MAP estimate under that prior.

See ``docs/research/pool-elo-methodology.md`` for the full derivation and the
DeepMind lineage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RatingResult:
    """Result of a BayesElo fit.

    Attributes:
        ratings: Player id → Elo rating (post-anchoring).
        iterations: MM iterations run before convergence (or the cap).
        converged: Whether the fit met ``tol`` before ``max_iters``.
    """

    ratings: dict[str, float]
    iterations: int
    converged: bool


def fit_bayeselo(
    players: list[str],
    wins: dict[tuple[str, str], int],
    draws: dict[tuple[str, str], int],
    *,
    prior_games: float = 2.0,
    anchor: str | None = None,
    anchor_rating: float = 0.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> RatingResult:
    """Fit pool Elo ratings by Bradley–Terry Minorization–Maximization.

    Args:
        players: Ordered list of unique player ids. Ratings are returned for
            exactly these players.
        wins: ``(i, j) -> count`` of games player *i* won against *j*. Store the
            two directions separately (``(i, j)`` and ``(j, i)``).
        draws: ``(i, j) -> count`` of drawn games between *i* and *j*. Store once
            per unordered pair (the fit symmetrises internally).
        prior_games: Virtual draws each player plays against a fixed R=0 anchor
            pseudo-player. The regularisation strength — larger pulls ratings
            gently toward the anchor and guarantees finite ratings even for an
            undefeated / winless player. Must be non-negative.
        anchor: If given, all ratings are shifted so this player sits exactly at
            ``anchor_rating``. Must be one of ``players``.
        anchor_rating: The Elo assigned to ``anchor`` after fitting.
        max_iters: Iteration cap for the MM loop.
        tol: Convergence threshold on ``max |Δ ln γ|`` across an iteration.

    Returns:
        A :class:`RatingResult` with one rating per player.

    Raises:
        ValueError: If ``players`` is empty or has duplicates, if a win/draw key
            references an unknown player, or if ``anchor`` is not a player, or if
            ``prior_games`` is negative.
    """
    if not players:
        raise ValueError("players must be non-empty")
    if len(set(players)) != len(players):
        raise ValueError("players must be unique")
    if prior_games < 0:
        raise ValueError(f"prior_games must be >= 0, got {prior_games}")
    if anchor is not None and anchor not in players:
        raise ValueError(f"anchor {anchor!r} is not one of players")

    idx = {p: i for i, p in enumerate(players)}
    n = len(players)

    # win_matrix[i, j] = games i beat j; draw_matrix symmetric = drawn games.
    win_matrix = np.zeros((n, n), dtype=np.float64)
    draw_matrix = np.zeros((n, n), dtype=np.float64)

    for (i, j), count in wins.items():
        if i not in idx or j not in idx:
            raise ValueError(f"wins references unknown player in pair {(i, j)!r}")
        win_matrix[idx[i], idx[j]] += count
    for (i, j), count in draws.items():
        if i not in idx or j not in idx:
            raise ValueError(f"draws references unknown player in pair {(i, j)!r}")
        # Symmetrise: a drawn pair contributes to both orientations.
        draw_matrix[idx[i], idx[j]] += count
        draw_matrix[idx[j], idx[i]] += count

    # Total games between i and j (symmetric), zero on the diagonal.
    n_matrix = win_matrix + win_matrix.T + draw_matrix
    np.fill_diagonal(n_matrix, 0.0)

    # Each player's total score (wins + half-draws) plus half the virtual draws.
    score = win_matrix.sum(axis=1) + 0.5 * draw_matrix.sum(axis=1) + 0.5 * prior_games

    # MM iterations (Jacobi): compute every denominator from the previous γ,
    # then update all γ together and renormalise to the geometric-mean gauge.
    gamma = np.ones(n, dtype=np.float64)
    converged = False
    iterations = 0
    while iterations < max_iters:
        iterations += 1
        pair_sums = gamma[:, None] + gamma[None, :]  # γ_i + γ_j
        denom = (n_matrix / pair_sums).sum(axis=1) + prior_games / (gamma + 1.0)
        new_gamma = score / denom
        # Fix the scale (gauge) freedom: divide by the geometric mean.
        new_gamma /= math.exp(np.log(new_gamma).mean())
        delta = float(np.abs(np.log(new_gamma) - np.log(gamma)).max())
        gamma = new_gamma
        if delta < tol:
            converged = True
            break

    elo = 400.0 * np.log10(gamma)
    if anchor is not None:
        elo = elo - elo[idx[anchor]] + anchor_rating

    ratings = {p: float(elo[idx[p]]) for p in players}
    return RatingResult(ratings=ratings, iterations=iterations, converged=converged)
