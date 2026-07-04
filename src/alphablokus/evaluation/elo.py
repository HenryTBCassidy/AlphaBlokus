"""Chess-style Elo arithmetic for the frozen-baseline strength curve."""

from __future__ import annotations

import math


def compute_elo(wins: int, losses: int, draws: int) -> tuple[float, float]:
    """Chess-style Elo difference vs an anchor opponent.

    Score rate = (wins + 0.5·draws) / total_games, clamped to [0.001, 0.999]
    to avoid log(0). Elo difference = 400 · log₁₀(score_rate / (1−score_rate)).
    Returns ``(elo_diff, score_rate)``.
    """
    total = wins + losses + draws
    if total == 0:
        return 0.0, 0.0
    raw = (wins + 0.5 * draws) / total
    score_rate = max(0.001, min(0.999, raw))
    elo = 400 * math.log10(score_rate / (1 - score_rate))
    return elo, raw
