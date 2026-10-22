"""What Pentobi conditions exist, where each writes, and which one drives promotion.

A *condition* names the instrument a Pentobi result belongs to. It exists because
``Coach._check_ladder_and_drift`` reads a directory of ``ladder_*.json`` as a single
series and feeds it to keep-best-by-ladder and the drift circuit-breaker: a book-on
300-game level-9-only score and a book-free 100-game L1-9 weighted score are not the
same measurement, and pooling them would corrupt promotion and could trip the
catastrophe stop on nothing.

Only :attr:`LadderCondition.LADDER` is longitudinal. Everything else is a one-off
comparison on its own scale, quarantined in its own directory
(:func:`condition_directory`) and filtered out of the promotion series
(``ladder_selection.is_longitudinal``).

This registry lives in the package rather than in ``scripts/pentobi_benchmark.py`` so
that the producer (the benchmark script), the consumer (``Coach`` via
``ladder_selection``) and ``scripts/mini_ladder.py`` all name the same set, and adding
a condition is one edit instead of three.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class LadderCondition(StrEnum):
    """The instrument a Pentobi benchmark result belongs to.

    ``StrEnum`` so a member is its own wire format: it compares equal to the string
    stored in a payload's ``condition`` key and serialises straight back out through
    ``json.dumps``, which keeps payloads written before this enum existed readable.

    Attributes:
        LADDER: The longitudinal yardstick — 400 simulations, Pentobi's book off,
            levels 1-9. The only condition ``Coach`` promotes or stops a run on, and
            the only one whose settings are pinned (``condition_conflicts`` in
            ``scripts/pentobi_benchmark.py``).
        FAIR_FIGHT: Equal-thinking-time comparison: book on, top levels only, our side
            given the simulations that match Pentobi's wall-clock.
        SEARCH_SCALING: One arm of a sims sweep (400 / 1,600 / 6,400) asking how much
            strength *our* search buys per doubling.
        COLOUR_CHECK: A colour-split diagnostic — is a lopsided first-mover/second-mover
            record real, or a colour-handling bug?
        BOOK_DELTA: Book-on versus book-off at a fixed level, measuring what Pentobi's
            opening book is worth. Converts every pre-2026-08-05 number, all of which
            faced a book-free engine, onto the as-shipped scale.
    """

    LADDER = "ladder"
    FAIR_FIGHT = "fair-fight"
    SEARCH_SCALING = "search-scaling"
    COLOUR_CHECK = "colour-check"
    BOOK_DELTA = "book-delta"


# One directory per condition, and they MUST stay distinct: the directory is the first
# line of defence against a foreign result being read as longitudinal. The mapping is
# total (every member has an entry) so that adding a condition and forgetting its
# directory is a test failure now rather than a ``KeyError`` at the end of a run that
# has already spent its box hours. ``LADDER``'s entry records the conventional name of
# the longitudinal home; the directory itself comes from
# ``RunConfig.pentobi_ladder_directory``, which is what a run's existing history sits in.
CONDITION_DIRNAMES: Final[Mapping[LadderCondition, str]] = {
    LadderCondition.LADDER: "PentobiLadder",
    LadderCondition.FAIR_FIGHT: "PentobiFairFight",
    LadderCondition.SEARCH_SCALING: "PentobiSearchScaling",
    LadderCondition.COLOUR_CHECK: "PentobiColourCheck",
    LadderCondition.BOOK_DELTA: "PentobiBookDelta",
}


def is_longitudinal_condition(condition: str) -> bool:
    """Whether ``condition`` is the series Coach may promote or stop a run on.

    Exactly one condition is, by construction. An unrecognised string is *not*
    longitudinal, so a payload written by a future or misspelled condition stays out
    of the promotion series rather than silently joining it.
    """
    return condition == LadderCondition.LADDER


def condition_directory(ladder_directory: Path, condition: str) -> Path:
    """Where a result for ``condition`` is written.

    The longitudinal series keeps ``ladder_directory`` itself (that is where a run's
    history already sits); every other condition gets a sibling directory named by
    :data:`CONDITION_DIRNAMES`.

    Args:
        ladder_directory: The run's ``PentobiLadder/`` directory
            (``RunConfig.pentobi_ladder_directory``) — the longitudinal home.
        condition: The condition being written.

    Returns:
        The directory to write into.

    Raises:
        ValueError: If ``condition`` is not a known condition, or if a one-off
            condition would resolve to the longitudinal directory. The second case
            cannot happen with the shipped table and is checked anyway: a duplicated
            dirname would drop one-off results straight into the directory Coach
            reads, which is the failure the whole mechanism exists to prevent.
    """
    try:
        known = LadderCondition(condition)
    except ValueError:
        raise ValueError(
            f"Unknown Pentobi condition {condition!r}; expected one of "
            f"{', '.join(sorted(c.value for c in LadderCondition))}.",
        ) from None
    if known is LadderCondition.LADDER:
        return ladder_directory
    directory = ladder_directory.parent / CONDITION_DIRNAMES[known]
    if directory == ladder_directory:
        raise ValueError(
            f"Condition {known.value!r} resolves to the longitudinal directory {ladder_directory}; "
            f"its results would be read as ladder results by Coach's keep-best and drift checks.",
        )
    return directory
