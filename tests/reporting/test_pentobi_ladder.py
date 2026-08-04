"""Pentobi ladder persistence (reporting/pentobi_ladder.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from alphablokus.reporting.pentobi_ladder import (
    load_ladder_results,
    parse_levels,
    write_ladder_result,
)

if TYPE_CHECKING:
    from pathlib import Path


def _per_level(level: int, net_wins: int, games: int = 20) -> dict:
    pentobi_wins = games - net_wins
    return {
        "level": level,
        "games": games,
        "net_wins": net_wins,
        "pentobi_wins": pentobi_wins,
        "draws": 0,
        "win_rate": net_wins / games,
        "ci": (0.1, 0.9),
        "records": ["<GameRecord>"],  # must be dropped on write
    }


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1-5", [1, 2, 3, 4, 5]),
        ("1,3,9", [1, 3, 9]),
        ("1-2, 9", [1, 2, 9]),
        ("7", [7]),
    ],
)
def test_parse_levels(spec: str, expected: list[int]) -> None:
    assert parse_levels(spec) == expected


@pytest.mark.parametrize("spec", ["", "0-3", "8-10", "abc"])
def test_parse_levels_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_levels(spec)


def test_write_then_load_roundtrip_drops_records(tmp_path: Path) -> None:
    path = write_ladder_result(
        tmp_path,
        net="best.pth.tar",
        sims=400,
        games_per_level=20,
        per_level=[_per_level(1, 15), _per_level(2, 4)],
        metrics={"pentobi_level": 1, "score": 0.475, "weighted_score": 0.38},
    )
    assert path.parent == tmp_path
    [result] = load_ladder_results(tmp_path)
    assert result["net"] == "best.pth.tar"
    assert result["metrics"]["pentobi_level"] == 1
    assert [row["level"] for row in result["levels"]] == [1, 2]
    assert all("records" not in row for row in result["levels"])


def test_load_from_missing_or_empty_directory(tmp_path: Path) -> None:
    assert load_ladder_results(tmp_path / "nowhere") == []
    assert load_ladder_results(tmp_path) == []  # exists but empty
