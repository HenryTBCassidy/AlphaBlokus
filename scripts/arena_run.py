"""Generic head-to-head player evaluation.

Pit any two players (network checkpoints, random, perfect-play minimax) against
each other and print the win/loss/draw breakdown. Used internally by the
:meth:`core.coach.Coach._evaluate_strength_vs_baselines` flow and exposed here
as a CLI for ad-hoc evaluation: "is gen 11 actually stronger than gen 5?",
"how does our trained model fare against random?", etc.

Examples::

    # Trained model vs random baseline.
    uv run python -m scripts.arena_run \
        --config run_configurations/smoke_test.json \
        --player1 temp/smoke_test/Nets/best.pth.tar \
        --player2 random \
        --num-games 50

    # Trained model vs perfect minimax (TTT only).
    uv run python -m scripts.arena_run \
        --config run_configurations/smoke_test.json \
        --player1 temp/smoke_test/Nets/accepted_3.pth.tar \
        --player2 minimax \
        --num-games 20

    # Two checkpoints from the same run head-to-head.
    uv run python -m scripts.arena_run \
        --config run_configurations/ttt_full.json \
        --player1 temp/ttt_full/Nets/accepted_2.pth.tar \
        --player2 temp/ttt_full/Nets/accepted_11.pth.tar \
        --num-games 30

The ``--config`` flag points at the JSON whose net architecture matches the
checkpoints; you can't load a checkpoint into a wrapper with a different
``num_filters`` / ``num_residual_blocks`` setup.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.arena import Arena
from core.config import RunConfig, load_args
from core.interfaces import IGame, INeuralNetWrapper
from core.players import NetworkPlayer, Player, RandomPlayer


def _get_game(game_name: str) -> IGame:
    if game_name == "tictactoe":
        from games.tictactoe.game import TicTacToeGame
        return TicTacToeGame()
    if game_name == "blokusduo":
        from games.blokusduo.game import BlokusDuoGame
        return BlokusDuoGame()
    raise ValueError(f"Unknown game: {game_name!r}")


def _get_nnet_class(game_name: str) -> type[INeuralNetWrapper]:
    if game_name == "tictactoe":
        from games.tictactoe.neuralnets.wrapper import NNetWrapper
        return NNetWrapper
    if game_name == "blokusduo":
        from games.blokusduo.neuralnets.wrapper import NNetWrapper
        return NNetWrapper
    raise ValueError(f"Unknown game: {game_name!r}")


def _load_checkpoint_into_wrapper(
    checkpoint_path: Path,
    wrapper: INeuralNetWrapper,
    config: RunConfig,
) -> None:
    """Load a state_dict from an arbitrary path (bypassing config.net_directory).

    The base wrapper's ``load_checkpoint`` resolves via the run's net directory,
    which doesn't help when loading checkpoints from other runs.
    """
    map_location = None if config.net_config.cuda else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    wrapper.nnet.load_state_dict(ckpt["state_dict"])


def build_player(spec: str, game: IGame, config: RunConfig) -> Player:
    """Materialise a Player from a CLI spec string.

    Supported specs:

    - ``"random"`` — uniform random over legal moves.
    - ``"minimax"`` — perfect-play minimax (TTT only).
    - any other string — treated as a filesystem path to a ``.pth.tar``
      checkpoint, loaded into a :class:`NetworkPlayer`.
    """
    if spec == "random":
        return RandomPlayer(game)
    if spec == "minimax":
        if config.game != "tictactoe":
            raise ValueError(
                "Minimax player is only available for TicTacToe; "
                f"current config game is {config.game!r}"
            )
        from games.tictactoe.minimax import MinimaxTicTacToePlayer
        return MinimaxTicTacToePlayer(game)

    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    nnet_class = _get_nnet_class(config.game)
    wrapper = nnet_class(game, config)
    _load_checkpoint_into_wrapper(path, wrapper, config)
    return NetworkPlayer(game=game, nnet=wrapper, mcts_config=config.mcts_config, temp=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pit two players in head-to-head games.")
    parser.add_argument("--config", required=True, help="Path to a RunConfig JSON.")
    parser.add_argument(
        "--player1", required=True,
        help="Checkpoint path, or one of: random, minimax (TTT only).",
    )
    parser.add_argument(
        "--player2", required=True,
        help="Checkpoint path, or one of: random, minimax (TTT only).",
    )
    parser.add_argument(
        "--num-games", type=int, default=20,
        help="Number of games (rounded down to nearest even number; "
             "players alternate starting positions). Default: 20.",
    )
    args = parser.parse_args()

    config = load_args(args.config)
    game = _get_game(config.game)
    p1 = build_player(args.player1, game, config)
    p2 = build_player(args.player2, game, config)

    print(f"Game: {config.game}")
    print(f"Player 1: {args.player1}")
    print(f"Player 2: {args.player2}")
    print(f"Playing {args.num_games} games (MCTS sims: {config.mcts_config.num_mcts_sims}) ...")

    arena = Arena(p1, p2, game)
    p1_wins, p2_wins, draws = arena.play_games(args.num_games)
    total = p1_wins + p2_wins + draws

    print()
    print(f"Player 1 wins: {p1_wins:>4} ({100 * p1_wins / total:.1f}%)")
    print(f"Player 2 wins: {p2_wins:>4} ({100 * p2_wins / total:.1f}%)")
    print(f"Draws:         {draws:>4} ({100 * draws / total:.1f}%)")
    print(f"Total games:   {total}")


if __name__ == "__main__":
    main()
