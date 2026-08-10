"""``scripts/mini_ladder.py`` must actually be able to call the benchmark.

The mini ladder is the run's *selection* instrument: ``Coach`` reads its results for
keep-best-by-ladder and the drift circuit-breaker. When ``PentobiPlayer`` gained a
required ``nobook`` argument, this script's call into
``pentobi_benchmark.benchmark_levels_parallel`` was not updated, so every mini-ladder
run would have died with ``TypeError`` — silently disabling external checkpoint
selection and drift detection on a multi-day box run. No Pentobi binary is needed to
catch that, only a check that the call site and the callee still agree.
"""

from __future__ import annotations

import argparse
import inspect
from typing import TYPE_CHECKING, Any

import pytest

from scripts import mini_ladder, pentobi_benchmark

if TYPE_CHECKING:
    from alphablokus.config import RunConfig


def _args(**overrides: Any) -> argparse.Namespace:
    """A Namespace with the defaults ``mini_ladder``'s parser would produce."""
    defaults = {
        "levels": mini_ladder.DEFAULT_LEVELS,
        "games": 4,
        "sims": pentobi_benchmark.EVAL_SIMS_DEFAULT,
        "batch": 16,
        "seed": 1,
        "opening_temp": 1.0,
        "opening_moves": 4,
        "workers": 2,
        "cpu_net": False,
        "mps": False,
    }
    return argparse.Namespace(**{**defaults, **overrides})


def test_run_one_net_call_is_compatible_with_the_benchmark_signature(
    test_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind ``_run_one_net``'s real call against the real callee signature.

    The stub validates the keyword set with ``Signature.bind`` before returning, so a
    missing required argument (the ``nobook`` regression) or a stale keyword fails here
    exactly as it would on the box — without needing ``pentobi-gtp``.
    """
    real_signature = inspect.signature(pentobi_benchmark.benchmark_levels_parallel)
    seen: dict[str, Any] = {}

    def fake_benchmark(**kwargs: Any) -> list[dict[str, Any]]:
        real_signature.bind(**kwargs)  # raises TypeError on a mismatched call site
        seen.update(kwargs)
        return [
            pentobi_benchmark.level_result(level, net_wins=3, pentobi_wins=1, draws=0, records=[])
            for level in kwargs["levels"]
        ]

    monkeypatch.setattr(pentobi_benchmark, "benchmark_levels_parallel", fake_benchmark)

    point, duration_s = mini_ladder._run_one_net("run.json", test_config, "accepted_5.pth.tar", _args())

    assert point.label == "accepted_5.pth.tar"
    assert point.generation == 5
    assert duration_s >= 0.0
    assert seen["levels"] == [3, 4, 5, 6]


def test_mini_ladder_keeps_pentobis_book_off(test_config: RunConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """The longitudinal instrument is book-free, and has no option to be otherwise.

    Every ladder number the project has published faced a book-free Pentobi. A book-on
    result in this series would be a different scale under the same name, and Coach
    would promote on it.
    """
    seen: dict[str, Any] = {}

    def fake_benchmark(**kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        return [pentobi_benchmark.level_result(3, net_wins=2, pentobi_wins=2, draws=0, records=[])]

    monkeypatch.setattr(pentobi_benchmark, "benchmark_levels_parallel", fake_benchmark)
    mini_ladder._run_one_net("run.json", test_config, "accepted_5.pth.tar", _args(levels="3"))

    assert seen["nobook"] is True
    assert mini_ladder.LONGITUDINAL_NOBOOK is True


def test_mini_ladder_result_is_longitudinal_and_records_its_context(
    test_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its payload must be self-describing and admissible to the promotion series."""
    from alphablokus.evaluation.ladder_selection import is_longitudinal
    from alphablokus.reporting.pentobi_ladder import load_ladder_results

    def fake_benchmark(**kwargs: Any) -> list[dict[str, Any]]:
        return [pentobi_benchmark.level_result(3, net_wins=2, pentobi_wins=2, draws=0, records=[])]

    monkeypatch.setattr(pentobi_benchmark, "benchmark_levels_parallel", fake_benchmark)
    mini_ladder._run_one_net("run.json", test_config, "accepted_5.pth.tar", _args(levels="3"))

    (payload,) = load_ladder_results(test_config.pentobi_ladder_directory)
    assert is_longitudinal(payload)
    assert payload["context"]["pentobi"]["book"] is False
    assert payload["context"]["net"]["sims"] == pentobi_benchmark.EVAL_SIMS_DEFAULT


def test_mini_ladder_refuses_a_non_ladder_simulation_budget(test_config: RunConfig) -> None:
    """It writes only the longitudinal series, so it must hold that series' yardstick.

    A 6,400-sim mini ladder appended to ``PentobiLadder/`` would sit on the promotion
    curve next to 400-sim results, and Coach compares those numbers directly.
    """
    with pytest.raises(SystemExit, match="ladder"):
        mini_ladder._run_one_net("run.json", test_config, "accepted_5.pth.tar", _args(sims=6400))
