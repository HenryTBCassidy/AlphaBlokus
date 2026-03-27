"""Profile MCTS performance on Blokus Duo (or TicTacToe).

Plays N games using MCTS with detailed profiling, then produces an HTML report
showing where time is spent: neural net inference, move generation, game-ended
checks, and UCB selection.

Usage:
    uv run python -m scripts.mcts_profiling [--num-games 5] [--num-sims 50]
    uv run python -m scripts.mcts_profiling --checkpoint temp/some_run/Nets/best.pth.tar
    uv run python -m scripts.mcts_profiling --game tictactoe --num-sims 25
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.config import MCTSConfig, NetConfig, RunConfig
from core.mcts import MCTS, MCTSEpisodeStats, MCTSMoveStats
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.neuralnets.wrapper import NNetWrapper as BlokusDuoNNetWrapper
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper as TicTacToeNNetWrapper

OUTPUT_DIR = Path("temp/mcts_profiling")
PIECES_PATH = Path("games/blokusduo/pieces.json")


def _create_game_and_net(
    game_name: str,
    checkpoint: str | None,
    num_sims: int,
) -> tuple:
    """Create game, net, and MCTS config."""
    # Minimal net config for profiling (small net = fast inference, isolates move gen cost)
    net_config = NetConfig(
        learning_rate=0.001,
        dropout=0.3,
        epochs=1,
        batch_size=4,
        cuda=False,
        num_filters=32,
        num_residual_blocks=1,
    )
    mcts_config = MCTSConfig(
        num_mcts_sims=num_sims,
        cpuct=1.0,
        profiling_level="detailed",
    )
    # RunConfig needed by the wrapper (uses net_config + directory paths)
    run_config = RunConfig(
        game=game_name,
        run_name="mcts_profiling",
        num_generations=1,
        num_eps=1,
        temp_threshold=10,
        update_threshold=0.55,
        max_queue_length=100,
        num_arena_matches=2,
        max_generations_lookback=1,
        root_directory=OUTPUT_DIR,
        load_model=False,
        mcts_config=mcts_config,
        net_config=net_config,
    )

    match game_name:
        case "blokusduo":
            game = BlokusDuoGame(pieces_config_path=PIECES_PATH)
            nnet = BlokusDuoNNetWrapper(game, run_config)
        case "tictactoe":
            game = TicTacToeGame()
            nnet = TicTacToeNNetWrapper(game, run_config)
        case _:
            raise ValueError(f"Unknown game: {game_name}")

    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        nnet.load_checkpoint(checkpoint)

    return game, nnet, mcts_config


def play_mcts_game(game, nnet, mcts_config: MCTSConfig) -> MCTSEpisodeStats:
    """Play one game using MCTS and return detailed profiling stats."""
    mcts = MCTS(game, nnet, mcts_config)
    board = game.initialise_board()
    current_player = 1
    move_count = 0

    while True:
        move_count += 1
        canonical = game.get_canonical_form(board, current_player)
        temp = 1 if move_count < 10 else 0

        pi = mcts.get_action_prob(canonical, temp=temp)
        action = np.random.choice(len(pi), p=pi)
        board, current_player = game.get_next_state(board, current_player, action)

        result = game.get_game_ended(board, current_player)
        if result != 0:
            return mcts.get_episode_stats()


def run_profiling(game_name: str, num_games: int, num_sims: int, checkpoint: str | None) -> list[MCTSEpisodeStats]:
    """Play num_games MCTS games and collect stats."""
    game, nnet, mcts_config = _create_game_and_net(game_name, checkpoint, num_sims)
    all_stats = []

    for i in range(num_games):
        t0 = time.perf_counter()
        stats = play_mcts_game(game, nnet, mcts_config)
        elapsed = time.perf_counter() - t0
        all_stats.append(stats)
        print(f"  Game {i + 1}/{num_games}: {stats.num_moves} moves, "
              f"{stats.total_search_time_s:.1f}s search, "
              f"{elapsed:.1f}s total")

    return all_stats


def _move_stats_to_df(all_stats: list[MCTSEpisodeStats]) -> pd.DataFrame:
    """Flatten per-move stats from all games into a DataFrame."""
    records = []
    for game_id, episode in enumerate(all_stats):
        for ms in episode.move_stats:
            records.append({
                "game_id": game_id,
                "move_number": ms.move_number,
                "num_sims": ms.num_sims,
                "search_time_s": ms.search_time_s,
                "inference_time_s": ms.inference_time_s,
                "valid_moves_time_s": ms.valid_moves_time_s,
                "game_ended_time_s": ms.game_ended_time_s,
                "num_leaf_expansions": ms.num_leaf_expansions,
                "num_valid_moves": ms.num_valid_moves,
            })
    return pd.DataFrame(records)


def _episode_stats_to_df(all_stats: list[MCTSEpisodeStats]) -> pd.DataFrame:
    """Flatten episode-level stats into a DataFrame."""
    records = []
    for game_id, s in enumerate(all_stats):
        records.append({
            "game_id": game_id,
            "num_moves": s.num_moves,
            "total_sims": s.total_sims,
            "total_search_time_s": s.total_search_time_s,
            "total_inference_time_s": s.total_inference_time_s,
            "total_valid_moves_time_s": s.total_valid_moves_time_s,
            "total_game_ended_time_s": s.total_game_ended_time_s,
            "num_leaf_expansions": s.num_leaf_expansions,
            "num_valid_moves_calls": s.num_valid_moves_calls,
            "num_game_ended_calls": s.num_game_ended_calls,
            "tree_size": s.tree_size,
            "tree_memory_bytes": s.tree_memory_bytes,
        })
    return pd.DataFrame(records)


def build_report(all_stats: list[MCTSEpisodeStats], game_name: str, num_sims: int, output_dir: Path) -> None:
    """Generate HTML report from profiling data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    move_df = _move_stats_to_df(all_stats)
    episode_df = _episode_stats_to_df(all_stats)
    num_games = len(all_stats)

    # -- KPI values --
    avg_game_time = episode_df["total_search_time_s"].mean()
    avg_moves = episode_df["num_moves"].mean()
    avg_time_per_move = move_df["search_time_s"].mean()
    avg_sims_per_sec = episode_df["total_sims"].sum() / episode_df["total_search_time_s"].sum()
    avg_tree_memory_mb = episode_df["tree_memory_bytes"].mean() / (1024 * 1024)

    # -- Time breakdown (aggregate across all games) --
    total_inference = episode_df["total_inference_time_s"].sum()
    total_valid_moves = episode_df["total_valid_moves_time_s"].sum()
    total_game_ended = episode_df["total_game_ended_time_s"].sum()
    total_search = episode_df["total_search_time_s"].sum()
    total_other = max(0, total_search - total_inference - total_valid_moves - total_game_ended)

    fig_pie = go.Figure(data=[go.Pie(
        labels=["Neural Net Inference", "Move Generation", "Game-Ended Checks", "Other (UCB, tree ops)"],
        values=[total_inference, total_valid_moves, total_game_ended, total_other],
        marker_colors=["#636efa", "#00cc96", "#ef553b", "#ffa15a"],
        textinfo="label+percent",
        hole=0.35,
    )])
    fig_pie.update_layout(
        title="Time Breakdown (All Games)",
        height=450, template="plotly_white",
    )

    # -- Per-turn timing curves --
    turn_stats = move_df.groupby("move_number").agg(
        mean_search=("search_time_s", "mean"),
        mean_inference=("inference_time_s", "mean"),
        mean_valid_moves=("valid_moves_time_s", "mean"),
        mean_game_ended=("game_ended_time_s", "mean"),
        mean_num_valid=("num_valid_moves", "mean"),
        mean_leaf=("num_leaf_expansions", "mean"),
    ).reset_index()

    fig_timing = go.Figure()
    fig_timing.add_trace(go.Scatter(
        x=turn_stats["move_number"], y=turn_stats["mean_inference"] * 1000,
        mode="lines", name="Inference", line=dict(color="#636efa", width=2),
    ))
    fig_timing.add_trace(go.Scatter(
        x=turn_stats["move_number"], y=turn_stats["mean_valid_moves"] * 1000,
        mode="lines", name="Move Generation", line=dict(color="#00cc96", width=2),
    ))
    fig_timing.add_trace(go.Scatter(
        x=turn_stats["move_number"], y=turn_stats["mean_game_ended"] * 1000,
        mode="lines", name="Game-Ended", line=dict(color="#ef553b", width=2),
    ))
    fig_timing.update_layout(
        title="Mean Time Per Move By Component",
        xaxis_title="Move Number", yaxis_title="Time (ms)",
        height=450, template="plotly_white",
    )

    # -- Per-turn search time --
    fig_search = go.Figure()
    fig_search.add_trace(go.Scatter(
        x=turn_stats["move_number"], y=turn_stats["mean_search"] * 1000,
        mode="lines", name="Total Search", line=dict(color="#ab63fa", width=2),
    ))
    fig_search.update_layout(
        title="Mean Total Search Time Per Move",
        xaxis_title="Move Number", yaxis_title="Time (ms)",
        height=450, template="plotly_white",
    )

    # -- Legal moves per turn --
    fig_valid = go.Figure()
    fig_valid.add_trace(go.Scatter(
        x=turn_stats["move_number"], y=turn_stats["mean_num_valid"],
        mode="lines", line=dict(color="#636efa", width=2),
    ))
    fig_valid.update_layout(
        title="Mean Legal Moves Per Turn (MCTS Perspective)",
        xaxis_title="Move Number", yaxis_title="Number of Legal Moves",
        height=450, template="plotly_white",
    )

    # -- Leaf expansions per turn --
    fig_leaf = go.Figure()
    fig_leaf.add_trace(go.Scatter(
        x=turn_stats["move_number"], y=turn_stats["mean_leaf"],
        mode="lines", line=dict(color="#ffa15a", width=2),
    ))
    fig_leaf.update_layout(
        title="Mean Leaf Expansions Per Move",
        xaxis_title="Move Number", yaxis_title="Leaf Expansions",
        height=450, template="plotly_white",
    )

    # -- Move time histogram --
    fig_hist = px.histogram(
        move_df, x="search_time_s", nbins=50,
        title="Distribution of Search Time Per Move",
        labels={"search_time_s": "Search Time (s)"},
        height=400, template="plotly_white",
    )
    fig_hist.update_layout(xaxis_title="Search Time (s)", yaxis_title="Count")

    # -- HTML --
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>MCTS Profiling — {game_name}</title>
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
    </style>
</head>
<body>
<h1>MCTS Profiling</h1>
<p class="subtitle">{game_name} — {num_games} games, {num_sims} sims/move</p>

<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-value">{avg_game_time:.1f}s</div>
        <div class="kpi-label">Avg Search Time / Game</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{avg_time_per_move * 1000:.0f}ms</div>
        <div class="kpi-label">Avg Search Time / Move</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{avg_sims_per_sec:.0f}</div>
        <div class="kpi-label">Simulations / Second</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{avg_moves:.0f}</div>
        <div class="kpi-label">Avg Moves / Game</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{avg_tree_memory_mb:.1f} MB</div>
        <div class="kpi-label">Avg Tree Memory / Game</div>
    </div>
</div>

<h2>Time Breakdown</h2>
<div class="chart-grid">
    {fig_pie.to_html(full_html=False, include_plotlyjs=False)}
    {fig_hist.to_html(full_html=False, include_plotlyjs=False)}
</div>

<h2>Per-Move Analysis</h2>
<div class="chart-grid">
    {fig_timing.to_html(full_html=False, include_plotlyjs=False)}
    {fig_search.to_html(full_html=False, include_plotlyjs=False)}
</div>

<h2>Search Characteristics</h2>
<div class="chart-grid">
    {fig_valid.to_html(full_html=False, include_plotlyjs=False)}
    {fig_leaf.to_html(full_html=False, include_plotlyjs=False)}
</div>

</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html)
    print(f"\nReport saved to {report_path}")

    # Save raw data
    move_df.to_csv(output_dir / "move_stats.csv", index=False)
    episode_df.to_csv(output_dir / "episode_stats.csv", index=False)
    print(f"Raw data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MCTS profiling for AlphaBlokus")
    parser.add_argument("--num-games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--num-sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--game", type=str, default="blokusduo", choices=["blokusduo", "tictactoe"])
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: temp/mcts_profiling/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    print(f"MCTS Profiling: {args.game}, {args.num_games} games, {args.num_sims} sims/move")
    t0 = time.perf_counter()
    all_stats = run_profiling(args.game, args.num_games, args.num_sims, args.checkpoint)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")

    build_report(all_stats, args.game, args.num_sims, output_dir)


if __name__ == "__main__":
    main()
