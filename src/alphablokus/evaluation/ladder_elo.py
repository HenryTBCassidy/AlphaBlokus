"""Colour- and draw-aware Elo for the Pentobi ladder.

Converting a ladder score to Elo with ``-400*log10(1/s - 1)`` is wrong here, and
wrong in a direction that flatters us.

The ladder plays half its games as each colour and pools the result. With a
first-mover advantage ``c``, the pooled score against an opponent ``d`` Elo away is

    s(d) = [sigma(d + c) + sigma(d - c)] / 2

which is **flatter** in ``d`` than ``sigma(d)`` — averaging a logistic over
+/-c pulls the curve toward 0.5. Inverting a pooled score with a plain logistic
therefore *understates* the gap everywhere off 0.5, and Blokus Duo has a large
``c``: roughly 75% of decisive games go to the first mover. At c ~ 190 Elo an
observed 0.22 implies a true gap near -280, not the -220 a naive inversion gives.

This module inverts the correct expression instead, and fits ``c`` from the
recorded colour split rather than assuming it. Draws count as half a game
throughout, matching ``scripts/pentobi_benchmark.level_result``.

Sign convention: a returned Elo is **the opponent's rating relative to our net**,
so positive means the opponent is stronger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Elo's logistic scale factor: a 400-point gap is a 10:1 expected score.
_SCALE = 400.0 / math.log(10.0)


def _sigma(elo: float) -> float:
    """Expected score for a player ``elo`` points ahead."""
    return 1.0 / (1.0 + math.exp(-elo / _SCALE))


def colour_advantage_elo(white_score: float) -> float:
    """Elo value of moving first, from the first mover's score in balanced games.

    Args:
        white_score: Score achieved by the first mover, draws counted as half,
            across games where both sides are otherwise equally matched.

    Returns:
        The advantage in Elo. 0.5 gives 0; higher scores give positive values.

    Raises:
        ValueError: If ``white_score`` is not strictly inside (0, 1) — a 0 or 1
            score implies infinite advantage and cannot be converted.
    """
    if not 0.0 < white_score < 1.0:
        raise ValueError(f"white_score must be in (0, 1) to convert to Elo, got {white_score!r}")
    return _SCALE * math.log(white_score / (1.0 - white_score))


def pooled_score(gap_elo: float, colour_elo: float) -> float:
    """Expected pooled score against an opponent ``gap_elo`` ahead, halves of each colour.

    This is the forward model the ladder actually samples from, so it is what has
    to be inverted. Note it is symmetric in ``colour_elo``: playing half the games
    with the advantage and half against it does not cancel to ``sigma(-gap)``, it
    flattens the curve.
    """
    return 0.5 * (_sigma(-gap_elo + colour_elo) + _sigma(-gap_elo - colour_elo))


def opponent_elo(score: float, colour_elo: float, *, tolerance: float = 1e-6) -> float:
    """Invert :func:`pooled_score`: the opponent's Elo from our pooled score.

    Args:
        score: Our pooled score (wins + draws/2) / games, strictly inside (0, 1).
        colour_elo: First-mover advantage, from :func:`colour_advantage_elo`.
            Pass 0.0 to recover the naive logistic inversion.
        tolerance: Bisection tolerance in Elo.

    Returns:
        The opponent's rating relative to our net; positive means stronger.

    Raises:
        ValueError: If ``score`` is not strictly inside (0, 1).
    """
    if not 0.0 < score < 1.0:
        raise ValueError(f"score must be in (0, 1) to convert to Elo, got {score!r}")
    # pooled_score is strictly decreasing in the gap, so bisect on a bracket wide
    # enough for anything the ladder can produce at 100+ games.
    low, high = -4000.0, 4000.0
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if pooled_score(mid, colour_elo) > score:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


@dataclass(frozen=True)
class LadderRung:
    """One level's result, converted to Elo.

    Attributes:
        level: Pentobi level.
        score: Our pooled score, draws as half.
        games: Games played.
        opponent_elo: Pentobi's rating relative to our net; positive = stronger.
        ci_low: Lower bound of the 95% interval on ``opponent_elo``.
        ci_high: Upper bound of the 95% interval on ``opponent_elo``.
    """

    level: int
    score: float
    games: int
    opponent_elo: float
    ci_low: float
    ci_high: float

    @property
    def ci_width(self) -> float:
        """Full width of the interval, in Elo."""
        return self.ci_high - self.ci_low


def rung_from_level(level: int, score: float, games: int, colour_elo: float, *, z: float = 1.96) -> LadderRung:
    """Convert one level's score into Elo with an interval.

    The interval comes from propagating the binomial SE on the score through the
    inverse, which is what makes explicit how coarse these numbers are: near a
    score of 0.2 at 100 games, one percentage point is worth about 11 Elo and the
    SE is ~4pp, so a single rung carries roughly +/-87 Elo. Adjacent rungs differ
    by less than their own intervals, which is why per-level "Elo per doubling"
    figures are not usable and only the aggregate slope is.
    """
    if games <= 0:
        raise ValueError(f"games must be positive, got {games}")
    se = math.sqrt(max(score * (1.0 - score), 1e-9) / games)
    lo_score = min(max(score - z * se, 1e-6), 1.0 - 1e-6)
    hi_score = min(max(score + z * se, 1e-6), 1.0 - 1e-6)
    return LadderRung(
        level=level,
        score=score,
        games=games,
        opponent_elo=opponent_elo(score, colour_elo),
        # A higher score for us means a weaker opponent, so the score bounds map to
        # the Elo bounds the other way round.
        ci_low=opponent_elo(hi_score, colour_elo),
        ci_high=opponent_elo(lo_score, colour_elo),
    )


def slope_elo_per_doubling(rungs: list[LadderRung], sims_by_level: dict[int, int]) -> float | None:
    """Least-squares slope of opponent Elo against log2 of its simulation budget.

    Aggregated over all supplied rungs on purpose. Adjacent-level slopes are
    dominated by per-rung noise (see :func:`rung_from_level`); only a fit across
    many doublings has usable signal.

    Returns:
        Elo gained per doubling of the opponent's search, or None if fewer than two
        rungs have a known budget.
    """
    points = [(math.log2(sims_by_level[r.level]), r.opponent_elo) for r in rungs if r.level in sims_by_level]
    if len(points) < 2:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
