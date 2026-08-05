"""Condition separation: a one-off comparison must never enter the promotion series."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from alphablokus.evaluation.ladder_selection import (
    LADDER_CONDITION,
    is_longitudinal,
    ladder_point_from_payload,
)
from alphablokus.reporting.pentobi_ladder import load_ladder_results, write_ladder_result

if TYPE_CHECKING:
    from pathlib import Path


def _payload(net: str, weighted: float, condition: str | None) -> dict:
    payload: dict = {"net": net, "metrics": {"weighted_score": weighted, "pentobi_level": 4, "score": 0.5}}
    if condition is not None:
        payload["condition"] = condition
    return payload


def test_missing_condition_is_treated_as_longitudinal() -> None:
    """Every payload written before 2026-08-05 lacks the key and is a ladder result."""
    assert is_longitudinal(_payload("accepted_1.pth.tar", 0.34, None))


def test_explicit_ladder_condition_is_longitudinal() -> None:
    assert is_longitudinal(_payload("accepted_1.pth.tar", 0.34, LADDER_CONDITION))


@pytest.mark.parametrize("condition", ["fair-fight", "search-scaling", "book-delta"])
def test_other_conditions_are_excluded(condition: str) -> None:
    assert not is_longitudinal(_payload("accepted_1.pth.tar", 0.9, condition))


def test_a_fair_fight_result_cannot_reach_keep_best_or_the_drift_breaker() -> None:
    """The defect this guards: Coach reads a directory as one series.

    A fair-fight run (book on, level 9 only, 300 games) produces a weighted score
    that means something entirely different from a book-free 100-game L1-9 sweep.
    Before the condition filter, dropping one into ``PentobiLadder/`` would have fed
    it to keep-best-by-ladder and the drift circuit-breaker — corrupting promotion
    and potentially tripping the catastrophe stop on nothing.
    """
    results = [
        _payload("accepted_10.pth.tar", 0.34, None),
        _payload("accepted_20.pth.tar", 0.36, LADDER_CONDITION),
        _payload("accepted_20.pth.tar", 0.95, "fair-fight"),  # different scale entirely
    ]
    kept = [ladder_point_from_payload(r) for r in results if is_longitudinal(r)]
    assert [p.weighted_score for p in kept] == [0.34, 0.36]
    assert max(p.weighted_score for p in kept) < 0.9


def test_write_and_read_round_trips_condition_and_context(tmp_path: Path) -> None:
    context = {"pentobi": {"book": True, "threads": 1}, "net": {"sims": 4800}}
    write_ladder_result(
        tmp_path,
        net="accepted_40.pth.tar",
        sims=4800,
        games_per_level=300,
        per_level=[{"level": 9, "games": 300, "net_wins": 120, "pentobi_wins": 170, "draws": 10, "score": 0.4167}],
        metrics={"weighted_score": 0.4167, "pentobi_level": 0, "score": 0.4167},
        condition="fair-fight",
        context=context,
    )
    (loaded,) = load_ladder_results(tmp_path)
    assert loaded["condition"] == "fair-fight"
    assert loaded["context"] == context
    assert not is_longitudinal(loaded)


def test_default_written_condition_is_the_ladder(tmp_path: Path) -> None:
    """Existing callers that pass no condition keep writing longitudinal results."""
    write_ladder_result(
        tmp_path,
        net="accepted_1.pth.tar",
        sims=400,
        games_per_level=100,
        per_level=[{"level": 1, "games": 100, "net_wins": 77, "pentobi_wins": 23, "draws": 0, "score": 0.77}],
        metrics={"weighted_score": 0.77, "pentobi_level": 1, "score": 0.77},
    )
    (loaded,) = load_ladder_results(tmp_path)
    assert loaded["condition"] == LADDER_CONDITION
    assert is_longitudinal(loaded)


def test_pentobi_player_requires_an_explicit_book_decision() -> None:
    """``nobook`` is keyword-only with no default, on purpose.

    The book was inactive for the project's entire history because the binary's
    directory held no book files, while ``param_base`` still reported ``use_book 1``.
    A default here — either way — would let that recur silently on a rebuild.
    """
    from alphablokus.games.blokusduo.pentobi.player import PentobiPlayer

    param = inspect.signature(PentobiPlayer.__init__).parameters["nobook"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty
