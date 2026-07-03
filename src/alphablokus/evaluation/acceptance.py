"""Single source of truth for the arena-acceptance decision.

The training loop (``Coach._should_accept_new_network``) and the reporting
layer (``reporting/training._accepted_mask``) historically each computed
acceptance independently, with predictable divergence: a previous bug had
the report using the old draws-excluded rule while the training code had
already switched to score-based. The chart then misreported which gens
were accepted.

This module exists so neither side ever recomputes the rule. ``Coach``
calls :func:`is_accepted_score_rule`; the reporting layer prefers the
explicitly persisted ``accepted`` column on the arena parquet and only
falls back to :func:`is_accepted_score_rule` when reading older runs.
That way the single decision lives in one place and is logged with the
result, so visualisations always agree with the training-time decision.
"""
from __future__ import annotations


def is_accepted_score_rule(
    new_wins: int, prev_wins: int, draws: int, threshold: float,
) -> bool:
    """Chess-style score-based acceptance.

    Draws count as 0.5 each in both numerator and denominator. The new
    network is accepted iff its **score** meets or exceeds ``threshold``::

        score = (new_wins + 0.5 * draws) / (new_wins + prev_wins + draws)

    Score is undefined (``threshold`` cannot be met) when no arena games
    were played; that case returns ``False``.

    This is the AlphaGo-Zero-style threshold criterion, modified to count
    draws — the draws-excluded form silently rejects every gen in
    forced-draw games (TTT under perfect play, late-stage Blokus where
    both nets have converged to similar policies). Score-based handles
    both cases cleanly.
    """
    total_games = new_wins + prev_wins + draws
    if total_games == 0:
        return False
    score = (new_wins + 0.5 * draws) / total_games
    return score >= threshold


def acceptance_score(new_wins: int, prev_wins: int, draws: int) -> float:
    """Return the raw score the acceptance rule compares to the threshold.

    Useful for plotting and KPI cards: shows how close-to-threshold a
    rejected gen was. Returns ``0.0`` when no arena games happened.
    """
    total_games = new_wins + prev_wins + draws
    if total_games == 0:
        return 0.0
    return (new_wins + 0.5 * draws) / total_games
