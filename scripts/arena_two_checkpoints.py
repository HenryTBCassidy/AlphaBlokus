"""Head-to-head arena between two checkpoints (G8 A/B final-net comparison).

Both nets play through the existing python ``NetworkPlayer``/``Arena``
machinery (noise-free, temp 0) with the Pentobi-harness opening randomisation
(``opening_temp``/``opening_moves``) so games decorrelate. Colours alternate
via ``Arena.play_games``'s half-swap. Reports W/L/D for net A with a 95%
Wilson interval.

Usage (box)::

    uv run python -m scripts.arena_two_checkpoints \
        --net-a temp/runs/blokus/<run>/Nets/best.pth.tar --label-a jax \
        --net-b temp/runs/blokus/<other-run>/Nets/best.pth.tar --label-b python \
        --games 100 --sims 400
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from loguru import logger

from alphablokus.games.blokusduo.pieces import default_pieces_path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIECES_PATH = default_pieces_path()


def _wilson(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games == 0:
        return 0.0, 1.0
    p = wins / games
    denom = 1 + z**2 / games
    centre = (p + z**2 / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z**2 / (4 * games**2)) / denom
    return centre - half, centre + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-a", type=Path, required=True)
    parser.add_argument("--net-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--opening-temp", type=float, default=1.0)
    parser.add_argument("--opening-moves", type=int, default=4)
    parser.add_argument(
        "--paired",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Colour-swapped paired play (shared opening prefix per pair) — cancels first-mover advantage.",
    )
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from alphablokus.config import MCTSConfig, NetConfig, RunConfig
    from alphablokus.evaluation.arena import Arena
    from alphablokus.evaluation.players import NetworkPlayer
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper

    game = BlokusDuoGame(pieces_config_path=PIECES_PATH)
    game.enable_optimised_movegen()
    config = RunConfig(
        game="blokusduo",
        run_name="arena_two_checkpoints",
        num_generations=1,
        num_eps=1,
        temp_threshold=12,
        update_threshold=0.55,
        num_arena_matches=args.games,
        root_directory=REPO_ROOT / "temp",
        load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=args.sims, cpuct=2.5, mcts_batch_size=16),
        net_config=NetConfig(
            learning_rate=1e-3,
            dropout=0.0,
            epochs=1,
            batch_size=8,
            cuda=args.cuda,
            num_filters=args.filters,
            num_residual_blocks=args.blocks,
            fp16_inference=args.cuda,
        ),
    )

    nnets = []
    for checkpoint in (args.net_a, args.net_b):
        nnet = NNetWrapper(game, config)
        nnet.load_checkpoint(filename=str(checkpoint.resolve()))
        nnets.append(nnet)

    # In paired mode the shared opening prefix is the ONLY diversification, so
    # the two competitors play deterministically (temp 0, no per-player opening
    # schedule) after it; the prefix is sampled from a separate temp>0 sampler.
    # In unpaired mode we keep the legacy per-player opening schedule.
    player_opening_temp = 0.0 if args.paired else args.opening_temp
    player_opening_moves = 0 if args.paired else args.opening_moves
    players = [
        NetworkPlayer(
            game,
            nnet,
            config.mcts_config,
            temp=0.0,
            opening_temp=player_opening_temp,
            opening_moves=player_opening_moves,
        )
        for nnet in nnets
    ]

    arena = Arena(players[0], players[1], game)
    if args.paired:
        # Sample each pair's opening prefix from net A's visit distribution at
        # ``opening_temp`` (a distinct temp>0 player — the competitors are temp 0).
        prefix_sampler = NetworkPlayer(game, nnets[0], config.mcts_config, temp=args.opening_temp)
        a_wins, b_wins, draws, records = arena.play_games_paired(
            args.games, prefix_sampler=prefix_sampler, opening_moves=args.opening_moves, record=True
        )
    else:
        a_wins, b_wins, draws, records = arena.play_games(args.games, record=True)

    # Colour split — the diagnostic the old gate was blind to. ``outcome`` is
    # from net A's perspective; ``player1_was_white`` is net A's colour that game.
    a_white_wins = sum(1 for r in records if r.outcome > 0 and r.player1_was_white)
    a_black_wins = sum(1 for r in records if r.outcome > 0 and not r.player1_was_white)
    b_white_wins = sum(1 for r in records if r.outcome < 0 and not r.player1_was_white)
    b_black_wins = sum(1 for r in records if r.outcome < 0 and r.player1_was_white)
    white_wins = a_white_wins + b_white_wins
    black_wins = a_black_wins + b_black_wins
    decisive = white_wins + black_wins
    white_win_rate = white_wins / decisive if decisive else 0.0

    low, high = _wilson(a_wins + 0.5 * draws, args.games)
    logger.info(
        "{} vs {} [{}]: {}-{}-{} → {} score {:.1%} (95% CI {:.1%}–{:.1%})",
        args.label_a,
        args.label_b,
        "paired" if args.paired else "unpaired",
        a_wins,
        b_wins,
        draws,
        args.label_a,
        (a_wins + 0.5 * draws) / args.games,
        low,
        high,
    )
    logger.info(
        "colour split: White won {:.1%} of {} decisive games | {} wins-as-Black {} vs {} wins-as-Black {}",
        white_win_rate,
        decisive,
        args.label_a,
        a_black_wins,
        args.label_b,
        b_black_wins,
    )
    out = args.out or REPO_ROOT / "temp" / "benchmarks" / f"arena_{args.label_a}_vs_{args.label_b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "net_a": str(args.net_a),
                "net_b": str(args.net_b),
                "label_a": args.label_a,
                "label_b": args.label_b,
                "paired": args.paired,
                "games": args.games,
                "sims": args.sims,
                "opening_temp": args.opening_temp,
                "opening_moves": args.opening_moves,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "draws": draws,
                "a_score_rate": (a_wins + 0.5 * draws) / args.games,
                "wilson_95": [low, high],
                "white_win_rate": white_win_rate,
                "a_white_wins": a_white_wins,
                "a_black_wins": a_black_wins,
                "b_white_wins": b_white_wins,
                "b_black_wins": b_black_wins,
            },
            indent=2,
        )
    )
    logger.info("written to {}", out)


if __name__ == "__main__":
    main()
