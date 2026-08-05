"""Keep-best-by-external-ladder selection and the drift circuit-breaker.

Weight-flow decisions are never made by candidate-vs-incumbent arena again:
between near-equal nets the arena score is structurally pinned to ~0.50 by
first-mover colour (a ~+100-Elo-class real gap reads as 0.525 paired), so both
a strict gate (froze ``blokus_search_harder`` at 0/17) and a loose one (let
``blokus_paired_gate_rerun`` regress L4 → L3) fail near equality. The Pentobi
ladder is the one instrument that repeatedly resolved differences the arena
called a tie, so the run's *product* is chosen by ladder — see
docs/research/regression-and-next-steps.md §1.2/§4 and
docs/plans/post-regression-recovery.md P3.

This module is the pure, tested logic; ``scripts/mini_ladder.py`` produces the
:class:`LadderPoint` history it consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# ``accepted_12.pth.tar`` / ``rejected_3.pth.tar`` — the Coach's per-generation
# checkpoint naming (training/coach.py).
_CHECKPOINT_GENERATION_RE = re.compile(r"(?:accepted|rejected)_(\d+)")

# Circuit-breaker defaults (post-regression-recovery P3): trip when
# ``DEFAULT_CONSECUTIVE_DROPS`` consecutive evaluations sit at least
# ``DEFAULT_DROP`` weighted-score below the best seen so far. Replayed against
# ``blokus_paired_gate_rerun`` these values trip by ~gen 8–10, before the
# run's terminal slide.
DEFAULT_DROP: float = 0.05
DEFAULT_CONSECUTIVE_DROPS: int = 2


@dataclass(frozen=True)
class LadderPoint:
    """One checkpoint's mini-ladder result.

    Attributes:
        label: Checkpoint filename the ladder was run on (e.g.
            ``accepted_12.pth.tar``).
        weighted_score: The ladder's weighted score (``metrics.weighted_score``
            from ``scripts/pentobi_benchmark.py`` — level-weighted win rate),
            the selection quantity.
        generation: Generation parsed from ``label`` when it follows the
            Coach's ``accepted_<n>``/``rejected_<n>`` naming, else ``None``.
        pentobi_level: Headline "beats level" metric, informational only.
        score: Unweighted win rate across the laddered levels, informational.
    """

    label: str
    weighted_score: float
    generation: int | None = None
    pentobi_level: int | None = None
    score: float | None = None


@dataclass(frozen=True)
class DriftAlarm:
    """A tripped drift circuit-breaker.

    Attributes:
        tripped_at: The evaluation that completed the consecutive-drop streak.
        best_before: The best point seen up to (and excluding) the streak —
            the checkpoint a stopped run should resume from.
        consecutive_drops: Length of the streak that tripped the alarm.
    """

    tripped_at: LadderPoint
    best_before: LadderPoint
    consecutive_drops: int


def checkpoint_generation(label: str) -> int | None:
    """Parse the generation from a Coach checkpoint filename, or ``None``.

    Recognises ``accepted_<n>`` / ``rejected_<n>`` anywhere in ``label`` (so
    full paths work); anything else (``best.pth.tar``, a donor net) has no
    generation.
    """
    match = _CHECKPOINT_GENERATION_RE.search(label)
    return int(match.group(1)) if match else None


# The only condition Coach may promote or stop a run on. Everything else — an
# equal-time comparison, a book-on run, a search-scaling arm — is a one-off
# measurement on a different scale, and folding it into this series would compare
# scores that are not comparable.
LADDER_CONDITION = "ladder"


def is_longitudinal(payload: Mapping[str, Any]) -> bool:
    """Whether this payload belongs to the longitudinal ladder series.

    Payloads written before 2026-08-05 have no ``condition`` key and are all
    longitudinal ladder results, so a missing key means yes.

    This filter is load-bearing rather than tidy-minded. ``Coach._check_ladder_and_drift``
    reads *every* ``ladder_*.json`` in one directory as a single series and feeds it
    to keep-best-by-ladder and the drift circuit-breaker. A fair-fight result (book
    on, 300 games, level 9 only) has a weighted score that means something entirely
    different from a 100-game book-free L1-9 sweep, so absorbing one would corrupt
    promotion and could trip the catastrophe stop on nothing. Conditions are kept in
    separate directories as the first line of defence; this is the second, so that
    pointing them at one directory is still safe.
    """
    return str(payload.get("condition", LADDER_CONDITION)) == LADDER_CONDITION


def ladder_point_from_payload(payload: Mapping[str, Any]) -> LadderPoint:
    """Build a :class:`LadderPoint` from a ladder JSON payload.

    Accepts the schema written by ``reporting/pentobi_ladder.write_ladder_result``
    (``scripts/pentobi_benchmark.py``'s output): ``net`` + a ``metrics`` dict
    with ``weighted_score`` / ``pentobi_level`` / ``score``.
    """
    label = str(payload["net"])
    metrics = payload["metrics"]
    return LadderPoint(
        label=label,
        weighted_score=float(metrics["weighted_score"]),
        generation=checkpoint_generation(label),
        pentobi_level=int(metrics["pentobi_level"]) if "pentobi_level" in metrics else None,
        score=float(metrics["score"]) if "score" in metrics else None,
    )


def select_best(points: Sequence[LadderPoint]) -> LadderPoint:
    """Return the keep-best checkpoint: highest weighted ladder score.

    Ties break toward the *lowest* generation (least exposure to training
    drift), with generation-less points last; a residual tie keeps the earliest
    evaluation. Raises ``ValueError`` on an empty sequence.
    """
    if not points:
        raise ValueError("select_best needs at least one ladder point")

    def key(indexed: tuple[int, LadderPoint]) -> tuple[float, int, int]:
        index, point = indexed
        generation = point.generation if point.generation is not None else 10**9
        # max() keeps the first of equal keys, so higher weighted first, then
        # lower generation, then earlier evaluation.
        return (point.weighted_score, -generation, -index)

    return max(enumerate(points), key=key)[1]


def detect_drift(
    points: Sequence[LadderPoint],
    *,
    drop: float = DEFAULT_DROP,
    consecutive: int = DEFAULT_CONSECUTIVE_DROPS,
) -> DriftAlarm | None:
    """Detect a sustained ladder regression in an evaluation-ordered history.

    Walks ``points`` in order, tracking the best weighted score seen so far. An
    evaluation is a *drop* when it sits at least ``drop`` below that best;
    ``consecutive`` drops in a row trip the alarm (a single drop is within
    ladder noise and resets nothing but its own streak). Returns the first
    alarm, or ``None``.

    ``points`` must be in evaluation order (the order the checkpoints were
    produced), else "consecutive" is meaningless.
    """
    if consecutive < 1:
        raise ValueError(f"consecutive must be >= 1, got {consecutive}")

    best: LadderPoint | None = None
    streak = 0
    for point in points:
        if best is None:
            best = point
            continue
        if point.weighted_score <= best.weighted_score - drop:
            streak += 1
            if streak >= consecutive:
                return DriftAlarm(tripped_at=point, best_before=best, consecutive_drops=streak)
        else:
            streak = 0
            if point.weighted_score > best.weighted_score:
                best = point
    return None
