"""Analyse how legal move counts evolve across random Blokus Duo games.

Plays N random games, recording per-turn stats and action sequences, then
produces an HTML report with charts, summary statistics, and interactive
game replays with board visualisation.

Usage:
    uv run python -m scripts.move_count_analysis [--num-games 100] [--replay-games 3]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from games.blokusduo.game import BlokusDuoGame
from reporting.display_blokusduo import BOARD_CSS, build_game_replay_html

OUTPUT_DIR = Path("temp/move_count_analysis")
PIECES_PATH = Path("games/blokusduo/pieces.json")


# -- Data collection -----------------------------------------------------------


def play_random_game(game: BlokusDuoGame) -> tuple[list[dict], list[dict]]:
    """Play a single random game, recording stats and actions per turn.

    Returns:
        Tuple of (turn_records, action_sequence).
    """
    board = game.initialise_board()
    players = [1, -1]
    turn = 0
    records = []
    actions = []
    consecutive_passes = 0

    while consecutive_passes < 2:
        player = players[turn % 2]
        t0 = time.perf_counter()
        moves = game._valid_moves(board, player)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        records.append({
            "turn": turn,
            "player": player,
            "num_moves": len(moves),
            "remaining_pieces": len(board.remaining_piece_ids(player)),
            "placement_points": len(board.placement_points(player)),
            "valid_moves_time_ms": elapsed_ms,
        })

        if not moves:
            actions.append({"turn": turn, "player": player, "pass": True})
            consecutive_passes += 1
            turn += 1
            continue

        consecutive_passes = 0
        action = random.choice(moves)
        actions.append({
            "turn": turn,
            "player": player,
            "pass": False,
            "piece_id": int(action.piece_id),
            "orientation": str(action.orientation.value),
            "x": int(action.x_coordinate),
            "y": int(action.y_coordinate),
        })
        board = board.with_piece(action, player_side=player)
        turn += 1

    return records, actions


def run_analysis(num_games: int) -> tuple[pd.DataFrame, list[list[dict]]]:
    """Play num_games random games and collect all data.

    Returns:
        Tuple of (turn_stats_df, list_of_action_sequences).
    """
    game = BlokusDuoGame(pieces_config_path=PIECES_PATH)
    all_records = []
    all_actions = []

    for game_idx in range(num_games):
        if (game_idx + 1) % 10 == 0:
            print(f"  Game {game_idx + 1}/{num_games}")
        records, actions = play_random_game(game)
        for r in records:
            r["game_id"] = game_idx
        all_records.extend(records)
        all_actions.append(actions)

    return pd.DataFrame(all_records), all_actions


# -- Board rendering -----------------------------------------------------------


# -- Report generation ---------------------------------------------------------


def build_report(
    df: pd.DataFrame,
    all_actions: list[list[dict]],
    output_dir: Path,
) -> None:
    """Generate HTML report with charts, summary stats, and game replays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    game = BlokusDuoGame(pieces_config_path=PIECES_PATH)

    num_games = df["game_id"].nunique()

    # Game length distribution
    game_lengths = df.groupby("game_id")["turn"].max() + 1
    avg_length = game_lengths.mean()
    min_length = game_lengths.min()
    max_length = game_lengths.max()

    # Per-turn aggregates
    turn_stats = df.groupby("turn").agg(
        mean_moves=("num_moves", "mean"),
        std_moves=("num_moves", "std"),
        median_moves=("num_moves", "median"),
        mean_placement_pts=("placement_points", "mean"),
        mean_remaining=("remaining_pieces", "mean"),
        mean_time_ms=("valid_moves_time_ms", "mean"),
    ).reset_index()
    turn_stats["std_moves"] = turn_stats["std_moves"].fillna(0)

    # Per-turn by player
    player_turn_stats = df.groupby(["turn", "player"]).agg(
        mean_moves=("num_moves", "mean"),
        std_moves=("num_moves", "std"),
    ).reset_index()
    player_turn_stats["std_moves"] = player_turn_stats["std_moves"].fillna(0)
    player_turn_stats["player_label"] = player_turn_stats["player"].map({1: "White", -1: "Black"})

    peak_moves = turn_stats["mean_moves"].max()
    peak_turn = turn_stats.loc[turn_stats["mean_moves"].idxmax(), "turn"]
    total_time = df.groupby("game_id")["valid_moves_time_ms"].sum()
    avg_game_time_ms = total_time.mean()

    # -- Charts --

    fig_moves = go.Figure()
    fig_moves.add_trace(go.Scatter(
        x=turn_stats["turn"], y=turn_stats["mean_moves"] + turn_stats["std_moves"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig_moves.add_trace(go.Scatter(
        x=turn_stats["turn"], y=turn_stats["mean_moves"] - turn_stats["std_moves"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(99, 110, 250, 0.15)", showlegend=False,
    ))
    fig_moves.add_trace(go.Scatter(
        x=turn_stats["turn"], y=turn_stats["mean_moves"],
        mode="lines", line=dict(color="#636efa", width=2), name="Mean",
    ))
    fig_moves.update_layout(
        title="Legal Moves Per Turn (Mean +/- Std Dev)",
        xaxis_title="Turn", yaxis_title="Number of Legal Moves",
        height=450, template="plotly_white",
    )

    fig_by_player = px.line(
        player_turn_stats, x="turn", y="mean_moves", color="player_label",
        title="Legal Moves Per Turn By Player",
        labels={"mean_moves": "Mean Legal Moves", "turn": "Turn", "player_label": "Player"},
        color_discrete_map={"White": "#636efa", "Black": "#ef553b"},
        height=450, template="plotly_white",
    )

    fig_points = go.Figure()
    fig_points.add_trace(go.Scatter(
        x=turn_stats["turn"], y=turn_stats["mean_placement_pts"],
        mode="lines", line=dict(color="#00cc96", width=2),
    ))
    fig_points.update_layout(
        title="Placement Points Per Turn (Mean)",
        xaxis_title="Turn", yaxis_title="Number of Placement Points",
        height=450, template="plotly_white",
    )

    fig_hist = px.histogram(
        game_lengths, nbins=20,
        title="Game Length Distribution",
        labels={"value": "Total Turns", "count": "Games"},
        height=400, template="plotly_white",
    )
    fig_hist.update_layout(xaxis_title="Total Turns", yaxis_title="Number of Games")

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=turn_stats["turn"], y=turn_stats["mean_time_ms"],
        mode="lines", line=dict(color="#ab63fa", width=2),
    ))
    fig_time.update_layout(
        title="valid_moves() Call Time Per Turn (Mean)",
        xaxis_title="Turn", yaxis_title="Time (ms)",
        height=450, template="plotly_white",
    )

    fig_remaining = go.Figure()
    fig_remaining.add_trace(go.Scatter(
        x=turn_stats["turn"], y=turn_stats["mean_remaining"],
        mode="lines", line=dict(color="#ffa15a", width=2),
    ))
    fig_remaining.update_layout(
        title="Remaining Pieces Per Turn (Mean)",
        xaxis_title="Turn", yaxis_title="Remaining Pieces",
        height=450, template="plotly_white",
    )

    # -- Game replays (all games, shown/hidden via dropdown) --
    print(f"Building replays for all {num_games} games...")
    replays_html = ""
    game_options_html = ""
    for gid in range(num_games):
        game_len = len(all_actions[gid])
        replays_html += f'<div class="game-container" id="game-{gid}" style="display:none;">\n'
        replays_html += build_game_replay_html(game, all_actions[gid], gid)
        replays_html += '</div>\n'
        game_options_html += f'<option value="{gid}">Game {gid} ({game_len} turns)</option>\n'

    # -- HTML Report --
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Move Count Analysis — Blokus Duo</title>
    <script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
    <style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1280px; margin: 0 auto; padding: 24px 32px; color: #2a3f5f;
    background: #ffffff;
}}
h1 {{ border-bottom: 2px solid #636efa; padding-bottom: 8px; margin-bottom: 4px; }}
.subtitle {{ color: #6b7280; font-size: 14px; margin-bottom: 24px; }}
h2 {{ margin-top: 40px; color: #636efa; font-size: 20px; }}
.kpi-grid {{ display: flex; gap: 14px; margin: 20px 0 32px 0; }}
.kpi-card {{
    flex: 1; padding: 14px 18px; border-radius: 8px;
    background: #f8f9fb; border: 1px solid #e5e7eb;
}}
.kpi-value {{ font-size: 26px; font-weight: 700; color: #1a1a2e; }}
.kpi-label {{ font-size: 12px; color: #6b7280; margin-top: 2px; text-transform: uppercase;
             letter-spacing: 0.5px; }}
.chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
details {{ margin: 12px 0; }}
details > summary {{
    cursor: pointer; font-weight: 600; color: #636efa; font-size: 14px;
    padding: 8px 0; list-style: none;
}}
details > summary::before {{ content: "\\25B8  "; }}
details[open] > summary::before {{ content: "\\25BE  "; }}
{BOARD_CSS}
.game-selector {{ margin: 16px 0; }}
.game-selector select {{
    font-size: 14px; padding: 6px 12px; border: 1px solid #e5e7eb;
    border-radius: 6px; background: #f8f9fb; color: #2a3f5f;
    min-width: 200px;
}}
.game-selector label {{ font-weight: 600; margin-right: 8px; }}
    </style>
</head>
<body>
<h1>Move Count Analysis</h1>
<p class="subtitle">Blokus Duo — {num_games} random games</p>

<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-value">{avg_length:.1f}</div>
        <div class="kpi-label">Avg Game Length (turns)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{peak_moves:.0f}</div>
        <div class="kpi-label">Peak Mean Legal Moves (turn {int(peak_turn)})</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{avg_game_time_ms:.0f} ms</div>
        <div class="kpi-label">Avg Total valid_moves Time / Game</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{min_length} — {max_length}</div>
        <div class="kpi-label">Game Length Range (turns)</div>
    </div>
</div>

<h2>Legal Moves</h2>
<div class="chart-grid">
    {fig_moves.to_html(full_html=False, include_plotlyjs=False)}
    {fig_by_player.to_html(full_html=False, include_plotlyjs=False)}
</div>

<h2>Board State</h2>
<div class="chart-grid">
    {fig_points.to_html(full_html=False, include_plotlyjs=False)}
    {fig_remaining.to_html(full_html=False, include_plotlyjs=False)}
</div>

<h2>Performance</h2>
<div class="chart-grid">
    {fig_time.to_html(full_html=False, include_plotlyjs=False)}
    {fig_hist.to_html(full_html=False, include_plotlyjs=False)}
</div>

<h2>Game Replays</h2>
<p class="section-desc">
    Board state at each turn. Piece IDs shown in cells.
    Placement points: <span style="background:#c5cae9;padding:1px 4px;">\u25c7</span> White
    <span style="background:#ffcdd2;padding:1px 4px;">\u25cb</span> Black
    <span style="background:#ffd700;padding:1px 4px;">*</span> Both.
</p>
<div class="game-selector">
    <label for="game-select">Select game:</label>
    <select id="game-select" onchange="showGame(this.value)">
        <option value="">-- Choose a game --</option>
        {game_options_html}
    </select>
</div>
{replays_html}
<script>
function showGame(gameId) {{
    document.querySelectorAll('.game-container').forEach(el => el.style.display = 'none');
    if (gameId !== '') {{
        const el = document.getElementById('game-' + gameId);
        if (el) el.style.display = 'block';
    }}
}}
</script>

</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html)
    print(f"\nReport saved to {report_path}")

    # Save raw data
    csv_path = output_dir / "turn_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to {csv_path}")

    # Save action sequences
    actions_path = output_dir / "game_actions.json"
    actions_path.write_text(json.dumps(all_actions, indent=2))
    print(f"Action sequences saved to {actions_path}")


def main():
    parser = argparse.ArgumentParser(description="Blokus Duo move count analysis")
    parser.add_argument("--num-games", type=int, default=100, help="Number of random games to play")
    args = parser.parse_args()

    print(f"Playing {args.num_games} random Blokus Duo games...")
    t0 = time.perf_counter()
    df, all_actions = run_analysis(args.num_games)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")

    build_report(df, all_actions, OUTPUT_DIR)


if __name__ == "__main__":
    main()
