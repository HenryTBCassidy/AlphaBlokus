"""CLI entry-point behaviour (``alphablokus.cli.main``).

Covers the crash-safe reporting contract (H2): a crash *inside* ``learn()`` must
still leave a rendered ``report.html`` behind, built from the per-generation
parquets that completed before the crash. Before this fix the render sat after
``learn()`` returned, so the crashed ``blokus_cloud_60`` run produced no report.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import alphablokus.cli as cli
from alphablokus.games.base_wrapper import BaseNNetWrapper

if TYPE_CHECKING:
    from pathlib import Path


def _write_tictactoe_config(tmp_path: Path) -> Path:
    """Write a tiny two-generation TicTacToe run config to a JSON file."""
    config = {
        "game": "tictactoe",
        "run_name": "cli_crash_test",
        "num_generations": 2,
        "num_eps": 2,
        "temp_threshold": 5,
        "update_threshold": 0.55,
        "num_arena_matches": 2,
        "replay_buffer_games": 20,
        "root_directory": str(tmp_path),
        "load_model": False,
        "minimax_games_per_gen": 0,
        "symmetry_diagnostic_positions": 0,
        "mcts_config": {"num_mcts_sims": 2, "cpuct": 1.0},
        "net_config": {
            "learning_rate": 0.001,
            "dropout": 0.3,
            "epochs": 1,
            "batch_size": 4,
            "cuda": False,
            "num_filters": 16,
            "num_residual_blocks": 1,
        },
    }
    config_path = tmp_path / "cli_crash_test.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


@pytest.mark.slow
def test_report_renders_when_learn_crashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash during generation 2's train step still yields a report from gen-1 data."""
    config_path = _write_tictactoe_config(tmp_path)

    original_train = BaseNNetWrapper.train
    calls = {"count": 0}

    def train_then_crash(self: BaseNNetWrapper, *args: object, **kwargs: object) -> None:
        calls["count"] += 1
        if calls["count"] >= 2:  # generation 1 completed and flushed; crash generation 2
            raise RuntimeError("injected training crash")
        original_train(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(BaseNNetWrapper, "train", train_then_crash)
    # The config lives in a tmp dir with no git repository, so the provenance
    # guard cannot check it against a commit and refuses by default (A5). This
    # test is about crash-safe reporting, so it takes the documented override.
    monkeypatch.setattr(
        "sys.argv",
        ["alphablokus", "--config", str(config_path), "--allow-uncommitted-config"],
    )

    with pytest.raises(RuntimeError, match="injected training crash"):
        cli.main()

    from alphablokus.config import load_args

    config = load_args(config_path)
    report_path = config.report_directory / "report.html"
    assert report_path.exists(), "report.html must be rendered from gen-1 data even when learn() crashes"
    assert report_path.stat().st_size > 0
