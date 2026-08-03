"""Behavioural test for the report's shipped JavaScript (assets/report.js).

The replay browser once shipped a crash that only fired when a placed move was
painted: pressing play blanked the panel with a ``ReferenceError``, while the
initial render — which paints no moves — looked perfect. Rendering was checked;
behaviour was not.

So this module renders a real ``report.html`` and then *drives* it under node
(``report_js_harness.js``): play a game to the end, step, scrub every position,
select every alternative, switch games and generations, toggle the theme. Any
error raised along the way fails the test.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from alphablokus.reporting.report import create_html_report

if TYPE_CHECKING:
    from alphablokus.config import RunConfig

_HARNESS = Path(__file__).parent / "report_js_harness.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node to execute report.js")


def _replay_rows(generation: int, game_idx: int, moves: list[tuple[int, int, list[int], list[float]]]) -> list[dict]:
    return [
        {
            "generation": generation,
            "game_idx": game_idx,
            "move_idx": move_idx,
            "player": player,
            "action": action,
            "top_k_actions": top_k_actions,
            "top_k_probs": top_k_probs,
            "played_prob": top_k_probs[0],
            "outcome": 1.0 if game_idx % 2 == 0 else -1.0,
            "player1_was_white": game_idx % 2 == 0,
        }
        for move_idx, (player, action, top_k_actions, top_k_probs) in enumerate(moves)
    ]


def _replay_frame() -> pd.DataFrame:
    """Two generations × two TicTacToe games, with alternatives on every move.

    Deliberately more than one generation and one game: switching between them
    is part of what has to keep working.
    """
    game_a: list[tuple[int, int, list[int], list[float]]] = [
        (1, 4, [4, 0, 8], [0.6, 0.3, 0.1]),
        (-1, 0, [0, 2, 6], [0.5, 0.3, 0.2]),
        (1, 8, [8, 2, 6], [0.7, 0.2, 0.1]),
        (-1, 2, [2, 6], [0.8, 0.2]),
        (1, 6, [6], [1.0]),
    ]
    game_b: list[tuple[int, int, list[int], list[float]]] = [
        (1, 0, [0, 4], [0.55, 0.45]),
        (-1, 4, [4, 8], [0.6, 0.4]),
        (1, 8, [8, 2], [0.5, 0.5]),
    ]
    rows: list[dict] = []
    for generation in (1, 2):
        rows += _replay_rows(generation, 0, game_a)
        rows += _replay_rows(generation, 1, game_b)
    return pd.DataFrame(rows)


@pytest.fixture
def rendered_report(test_config: RunConfig) -> Path:
    """A report.html built from a run directory with arena data and replays."""
    arena_dir = test_config.arena_data_directory / "generation=1"
    arena_dir.mkdir(parents=True)
    pd.DataFrame(
        {"wins": [2], "losses": [2], "draws": [0], "accepted": [False], "white_wins": [4], "black_wins": [0]}
    ).to_parquet(arena_dir / "data.parquet")

    frame = _replay_frame()
    for generation, group in frame.groupby("generation"):
        replay_dir = test_config.arena_replays_directory / f"generation={generation}"
        replay_dir.mkdir(parents=True)
        group.drop(columns=["generation"]).to_parquet(replay_dir / "games.parquet")

    create_html_report(test_config)
    return test_config.report_directory / "report.html"


def _drive(report: Path) -> dict:
    result = subprocess.run(
        ["node", str(_HARNESS), str(report)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"driving report.js raised:\n{result.stderr}"
    return json.loads(result.stdout)


def test_the_replay_browser_survives_being_used(rendered_report: Path) -> None:
    """Play to the end, step, scrub, pick alternatives, switch games, retheme.

    This is the regression test for the crash on play: painting any placed move
    used to throw, so every assertion past "press play" failed at once.
    """
    outcome = _drive(rendered_report)
    assert outcome["moves"] == 5, "expected the five-move game to load first"
    for action in ("press play", "play to the end", "step back", "scrub every position", "select alternatives"):
        assert action in outcome["actions"], f"{action} did not complete"
    assert outcome["theme"] == "light", "two theme toggles should return to light"


def test_the_page_renders_its_signals_charts_and_key(rendered_report: Path) -> None:
    outcome = _drive(rendered_report)
    assert outcome["signals"] == 6, "one tile per signal"
    assert outcome["charts"] > 0, "charts must render"
    assert outcome["keyRows"] > 0, "the event key must render a row per event"


def test_a_crash_while_painting_a_move_is_caught(rendered_report: Path, tmp_path: Path) -> None:
    """The harness must actually fail when the board painter throws.

    Re-injects the original defect — an undeclared variable read while painting
    placed moves, fatal under ``"use strict"`` — and requires a non-zero exit.
    Without this, a harness that silently swallowed errors would look green.
    """
    broken = tmp_path / "broken_report.html"
    html = rendered_report.read_text(encoding="utf-8")
    injected = html.replace("var isLast = !altActive && i === k - 1;", "var isLast = i === paintUpTo - 1;")
    assert injected != html, "the board painter's isLast line moved; update this test"
    broken.write_text(injected, encoding="utf-8")

    result = subprocess.run(["node", str(_HARNESS), str(broken)], capture_output=True, text=True, check=False)
    assert result.returncode != 0, "the harness passed a report whose replay browser crashes"
    assert "paintUpTo is not defined" in result.stderr
