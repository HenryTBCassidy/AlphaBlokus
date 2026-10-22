"""Condition separation: a one-off comparison must never enter the promotion series."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from alphablokus.evaluation.ladder_conditions import (
    CONDITION_DIRNAMES,
    LadderCondition,
    condition_directory,
    is_longitudinal_condition,
)
from alphablokus.evaluation.ladder_selection import (
    LADDER_CONDITION,
    LadderPoint,
    detect_drift,
    is_longitudinal,
    ladder_point_from_payload,
    select_best,
)
from alphablokus.reporting.pentobi_ladder import load_ladder_results, write_ladder_result


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


@pytest.mark.parametrize("condition", ["fair-fight", "search-scaling", "book-delta", "typo-condition"])
def test_other_conditions_are_excluded(condition: str) -> None:
    """Hand-written payloads too: the filter is on the string, not on the enum member."""
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


# --- The registry: every condition the queue needs, each with its own home ------------

ONE_OFF_CONDITIONS = [c for c in LadderCondition if c is not LadderCondition.LADDER]


def test_the_conditions_the_queue_needs_all_exist() -> None:
    """I1: only ``ladder`` and ``fair-fight`` existed, so three queued jobs had nowhere to write.

    A colour-check diagnostic was run mislabelled as ``fair-fight`` for exactly this
    reason, which put a two-level 20-game payload in the directory holding the
    equal-time 300-game level-9 result.
    """
    assert {"ladder", "fair-fight", "search-scaling", "colour-check", "book-delta"} <= {
        c.value for c in LadderCondition
    }


def test_exactly_one_condition_is_longitudinal() -> None:
    """Promotion reads one series. A second longitudinal condition must be a deliberate act.

    ``is_longitudinal`` is the filter that keeps a one-off comparison out of keep-best
    and the drift circuit-breaker, so widening it is not a tidy-up — it is a change to
    what the Coach promotes on, and it should break a test.
    """
    longitudinal = [c for c in LadderCondition if is_longitudinal_condition(c)]
    assert longitudinal == [LadderCondition.LADDER]


def test_every_condition_has_a_directory_and_they_are_all_distinct() -> None:
    """Total and unique. A missing entry would ``KeyError`` only after the games are played."""
    assert set(CONDITION_DIRNAMES) == set(LadderCondition)
    dirnames = list(CONDITION_DIRNAMES.values())
    assert len(set(dirnames)) == len(dirnames)


@pytest.mark.parametrize("condition", ONE_OFF_CONDITIONS)
def test_a_one_off_condition_writes_outside_the_promotion_directory(condition: LadderCondition) -> None:
    """The first line of defence: Coach only ever reads ``pentobi_ladder_directory``."""
    ladder_dir = Path("temp/runs/blokus/some_run/PentobiLadder")
    out_dir = condition_directory(ladder_dir, condition)
    assert out_dir != ladder_dir
    assert out_dir.parent == ladder_dir.parent


def test_the_ladder_condition_writes_to_the_runs_historical_home() -> None:
    ladder_dir = Path("temp/runs/blokus/some_run/PentobiLadder")
    assert condition_directory(ladder_dir, LadderCondition.LADDER) == ladder_dir


def test_condition_directories_are_all_different() -> None:
    ladder_dir = Path("temp/runs/blokus/some_run/PentobiLadder")
    resolved = [condition_directory(ladder_dir, c) for c in LadderCondition]
    assert len(set(resolved)) == len(resolved)


def test_an_unknown_condition_is_refused_a_directory() -> None:
    """A misspelled label must fail loudly rather than default into the ladder's directory."""
    with pytest.raises(ValueError, match="Unknown Pentobi condition"):
        condition_directory(Path("temp/runs/blokus/some_run/PentobiLadder"), "fair_fight")


# --- The leak test: a new condition cannot reach promotion or the drift breaker --------


def _promotion_series(payloads: list[dict]) -> list[LadderPoint]:
    """The filter Coach applies (``coach.py`` ``_check_ladder_and_drift``), verbatim."""
    return [
        ladder_point_from_payload(r)
        for r in payloads
        if "weighted_score" in r.get("metrics", {}) and is_longitudinal(r)
    ]


@pytest.mark.parametrize("condition", ONE_OFF_CONDITIONS)
def test_no_one_off_condition_can_reach_keep_best_or_the_drift_breaker(condition: LadderCondition) -> None:
    """Parametrised over the enum, so a condition added later is covered without being added here.

    The intruders are built to do maximum damage if they leak: one scores far above
    every ladder point (it would be crowned by keep-best and become the run's product),
    and two score far below (two consecutive drops of >= 0.05 is exactly what trips the
    drift circuit-breaker and stops the run). Neither is a regression or an improvement
    — they are different instruments, and their scores are not on this scale.
    """
    ladder_only = [
        _payload("accepted_10.pth.tar", 0.34, None),  # pre-2026-08-05: no condition key
        _payload("accepted_20.pth.tar", 0.36, LADDER_CONDITION),
        _payload("accepted_30.pth.tar", 0.35, LADDER_CONDITION),
    ]
    intruders = [
        _payload("accepted_30.pth.tar", 0.95, condition),
        _payload("accepted_30.pth.tar", 0.05, condition),
        _payload("accepted_30.pth.tar", 0.04, condition),
    ]

    clean = _promotion_series(ladder_only)
    polluted = _promotion_series([*ladder_only, *intruders])

    assert [p.weighted_score for p in polluted] == [p.weighted_score for p in clean]
    assert select_best(polluted).weighted_score == 0.36
    assert detect_drift(polluted) is None


@pytest.mark.parametrize("condition", ONE_OFF_CONDITIONS)
def test_a_one_off_payload_written_to_disk_is_filtered_on_read(
    condition: LadderCondition,
    tmp_path: Path,
) -> None:
    """Second line of defence, end to end: even pointing both at one directory is safe."""
    per_level = [{"level": 9, "games": 100, "net_wins": 90, "pentobi_wins": 10, "draws": 0, "score": 0.9}]
    write_ladder_result(
        tmp_path,
        net="accepted_40.pth.tar",
        sims=6400,
        games_per_level=100,
        per_level=per_level,
        metrics={"weighted_score": 0.9, "pentobi_level": 9, "score": 0.9, "scoring": "win_draw_half"},
        condition=condition,
    )
    write_ladder_result(
        tmp_path,
        net="accepted_41.pth.tar",
        sims=400,
        games_per_level=100,
        per_level=[{"level": 1, "games": 100, "net_wins": 34, "pentobi_wins": 66, "draws": 0, "score": 0.34}],
        metrics={"weighted_score": 0.34, "pentobi_level": 0, "score": 0.34, "scoring": "win_draw_half"},
    )
    kept = [r for r in load_ladder_results(tmp_path) if is_longitudinal(r)]
    assert [r["metrics"]["weighted_score"] for r in kept] == [0.34]


def test_every_reader_of_the_ladder_directory_applies_the_condition_filter() -> None:
    """Two modules read the results directory as a series; a third must not forget to filter.

    ``Coach._check_ladder_and_drift`` and ``reporting.data`` both call
    ``load_ladder_results``, and both must exclude one-off conditions — the report would
    otherwise draw a strength curve through two different instruments. This binds the
    real call sites instead of the copy in :func:`_promotion_series`.
    """
    src = Path(__file__).resolve().parents[2] / "src" / "alphablokus"
    sources = {path: path.read_text() for path in src.rglob("*.py")}
    readers = {
        path: text
        for path, text in sources.items()
        # The module that defines the loader is not a consumer of it.
        if "load_ladder_results(" in text and "def load_ladder_results(" not in text
    }
    assert readers, "no module calls load_ladder_results — has it been renamed?"
    unfiltered = [str(path.relative_to(src)) for path, text in readers.items() if "is_longitudinal" not in text]
    assert not unfiltered, f"these read ladder results without the condition filter: {unfiltered}"


# --- The CLI surface ------------------------------------------------------------------


@pytest.mark.parametrize("condition", list(LadderCondition))
def test_the_cli_accepts_every_condition(condition: LadderCondition) -> None:
    """Binds the real ``--condition`` call site: the flag offered only two values before I1."""
    import argparse

    from scripts.pentobi_benchmark import add_condition_argument

    parser = argparse.ArgumentParser()
    add_condition_argument(parser)
    assert parser.parse_args(["--condition", condition.value]).condition is condition


def test_the_cli_still_defaults_to_the_ladder() -> None:
    import argparse

    from scripts.pentobi_benchmark import add_condition_argument

    parser = argparse.ArgumentParser()
    add_condition_argument(parser)
    assert parser.parse_args([]).condition is LadderCondition.LADDER


def test_the_cli_rejects_an_unknown_condition() -> None:
    """Parse time, not after the games: the output directory is resolved at the very end."""
    import argparse

    from scripts.pentobi_benchmark import add_condition_argument

    parser = argparse.ArgumentParser()
    add_condition_argument(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--condition", "search_scaling"])


@pytest.mark.parametrize("condition", ONE_OFF_CONDITIONS)
def test_the_yardstick_guard_constrains_only_the_ladder(condition: LadderCondition) -> None:
    """A new condition must not inherit the ladder's fixed 400-sims/book-off requirement."""
    from scripts.pentobi_benchmark import condition_conflicts

    assert condition_conflicts(condition, nobook=False, sims=6400) is None


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


def test_the_ladder_condition_rejects_a_book_on_run() -> None:
    """``--condition`` defaults to ``ladder``, so ``--book`` alone would poison the series.

    The output directory and the ``condition`` key defend against a *foreign* payload
    being read as longitudinal. Neither defends against a run that claims to be the
    ladder while changing the yardstick — and since the default condition is the
    ladder, passing ``--book`` and nothing else did exactly that.
    """
    from scripts.pentobi_benchmark import EVAL_SIMS_DEFAULT, condition_conflicts

    message = condition_conflicts(LadderCondition.LADDER, nobook=False, sims=EVAL_SIMS_DEFAULT)
    assert message is not None
    assert "--book" in message
    assert "fair-fight" in message  # the message names the escape hatch


def test_the_ladder_condition_rejects_a_changed_simulation_budget() -> None:
    """The fixed 400 sims are what make a per-level score convertible to one Elo scale."""
    from scripts.pentobi_benchmark import condition_conflicts

    message = condition_conflicts(LadderCondition.LADDER, nobook=True, sims=6400)
    assert message is not None
    assert "6400" in message


def test_the_ladder_condition_accepts_its_own_settings() -> None:
    from scripts.pentobi_benchmark import EVAL_SIMS_DEFAULT, condition_conflicts

    assert condition_conflicts(LadderCondition.LADDER, nobook=True, sims=EVAL_SIMS_DEFAULT) is None


@pytest.mark.parametrize(("nobook", "sims"), [(False, 4800), (True, 400), (False, 400)])
def test_a_one_off_condition_may_use_any_settings(nobook: bool, sims: int) -> None:
    """Only the longitudinal series is constrained; a one-off comparison is free."""
    from scripts.pentobi_benchmark import condition_conflicts

    assert condition_conflicts(LadderCondition.FAIR_FIGHT, nobook=nobook, sims=sims) is None


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
