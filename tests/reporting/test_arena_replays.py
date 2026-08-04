"""Compact replay payloads (reporting/arena_replays.py) + the end-to-end
report render they feed into."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from alphablokus.reporting.arena_replays import build_replay_payload
from alphablokus.reporting.report import create_html_report

if TYPE_CHECKING:
    from alphablokus.config import RunConfig

# A three-move TicTacToe fragment: X (P1, White) takes the centre, O replies
# corner, X passes (action 9 = N*N). Outcome +1 = Player 1 (previous net) won.
_MOVES = [
    # (move_idx, player, action, top_k_actions, top_k_probs, played_prob)
    (0, 1, 4, [4, 0, 8], [0.6, 0.3, 0.1], 0.6),
    (1, -1, 0, [0, 2, 0], [0.5, 0.4, 0.0], 0.5),
    (2, 1, 9, [9], [1.0], 1.0),
]


def _replay_frame() -> pd.DataFrame:
    rows = [
        {
            "generation": 1,
            "game_idx": 0,
            "move_idx": move_idx,
            "player": player,
            "action": action,
            "top_k_actions": top_k_actions,
            "top_k_probs": top_k_probs,
            "played_prob": played_prob,
            "outcome": 1.0,
            "player1_was_white": True,
        }
        for move_idx, player, action, top_k_actions, top_k_probs, played_prob in _MOVES
    ]
    return pd.DataFrame(rows)


def test_replay_payload_diffs_cells_and_decodes_moves(test_config: RunConfig) -> None:
    payload = build_replay_payload(_replay_frame(), test_config)
    assert payload is not None
    assert payload["game"] == "tictactoe"
    assert payload["rows"] == 3 and payload["cols"] == 3

    [game] = payload["gens"]["1"]
    assert game["winner"] == "prev"
    assert game["label"] == "White wins — previous net"

    first, second, third = game["moves"]
    assert first["p"] == 1
    assert [cell[:2] for cell in first["cells"]] == [[1, 1]]  # action 4 = centre
    assert first["prob"] == 0.6
    assert second["cells"][0][:2] == [0, 0]
    assert third["cells"] == [] and third["cap"] == "Pass"  # pass leaves the board unchanged


def test_alternatives_exclude_played_action_and_zero_probs(test_config: RunConfig) -> None:
    payload = build_replay_payload(_replay_frame(), test_config)
    assert payload is not None
    [game] = payload["gens"]["1"]

    first_alts = game["moves"][0]["alts"]
    assert [alt["cells"] for alt in first_alts] == [[[0, 0]], [[2, 2]]]  # actions 0 and 8, played 4 dropped
    assert [alt["prob"] for alt in first_alts] == [0.3, 0.1]

    # Move 2's top-k padded a zero-probability duplicate — it must be dropped.
    second_alts = game["moves"][1]["alts"]
    assert [alt["cells"] for alt in second_alts] == [[[0, 2]]]


def test_new_net_win_maps_to_the_right_colour(test_config: RunConfig) -> None:
    df = _replay_frame()
    df["outcome"] = -1.0  # Player 2 (the new candidate) won
    df["player1_was_white"] = False  # so the candidate played White
    [game] = build_replay_payload(df, test_config)["gens"]["1"]
    assert game["winner"] == "new"
    assert game["label"] == "White wins — new net"


def test_create_html_report_end_to_end(test_config: RunConfig) -> None:
    """A run directory holding one arena table + replays renders to a single
    self-contained HTML file with the payload and both assets inlined."""
    arena_dir = test_config.arena_data_directory / "generation=1"
    arena_dir.mkdir(parents=True)
    pd.DataFrame({"wins": [3], "losses": [1], "draws": [0], "accepted": [True]}).to_parquet(arena_dir / "data.parquet")
    replays_dir = test_config.arena_replays_directory / "generation=1"
    replays_dir.mkdir(parents=True)
    _replay_frame().drop(columns=["generation"]).to_parquet(replays_dir / "games.parquet")

    create_html_report(test_config)

    html = (test_config.report_directory / "report.html").read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert 'id="report-data"' in html
    assert "ReplayBrowser" in html  # report.js inlined
    assert ".signals" in html  # report.css inlined
    assert "cdn.plot.ly" not in html and "https://" not in html.split("</style>")[0]  # offline: no CDN
