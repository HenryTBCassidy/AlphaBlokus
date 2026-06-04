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

from core.config import MCTSConfig, NetConfig, RunConfig
from core.mcts import MCTS, MCTSEpisodeStats
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.neuralnets.wrapper import NNetWrapper as BlokusDuoNNetWrapper
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper as TicTacToeNNetWrapper
from reporting.mcts_profiling import build_single_phase_report

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

    report_path = build_single_phase_report(
        all_stats,
        title="MCTS Profiling",
        subtitle=f"{args.game} — {args.num_games} games, {args.num_sims} sims/move",
        output_dir=output_dir,
        wall_clock_s=elapsed,
    )
    print(f"\nReport saved to {report_path}")
    print(f"Raw data saved to {output_dir}")


if __name__ == "__main__":
    main()
