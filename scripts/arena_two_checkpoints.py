"""Head-to-head arena between two checkpoints (G8 A/B final-net comparison).

Both nets play through the existing python ``NetworkPlayer``/``Arena``
machinery (noise-free, temp 0) with the Pentobi-harness opening randomisation
(``opening_temp``/``opening_moves``) so games decorrelate. Colours alternate
via ``Arena.play_games``'s half-swap. Reports W/L/D for net A with a 95%
Wilson interval.

Usage (box)::

    PYTHONPATH=$PWD uv run python -m scripts.arena_two_checkpoints \
        --net-a temp/runs/blokus/ab_jax_10/Nets/best.pth.tar --label-a jax \
        --net-b temp/runs/blokus/ab_python_10/Nets/best.pth.tar --label-b python \
        --games 100 --sims 400
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
PIECES_PATH = REPO_ROOT / "games" / "blokusduo" / "pieces.json"


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
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from core.arena import Arena
    from core.config import MCTSConfig, NetConfig, RunConfig
    from core.players import NetworkPlayer
    from games.blokusduo.game import BlokusDuoGame
    from games.blokusduo.neuralnets.wrapper import NNetWrapper

    game = BlokusDuoGame(pieces_config_path=PIECES_PATH)
    game.enable_optimised_movegen()
    config = RunConfig(
        game="blokusduo", run_name="arena_two_checkpoints", num_generations=1, num_eps=1,
        temp_threshold=12, update_threshold=0.55, num_arena_matches=args.games,
        root_directory=REPO_ROOT / "temp", load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=args.sims, cpuct=2.5, mcts_batch_size=16),
        net_config=NetConfig(
            learning_rate=1e-3, dropout=0.0, epochs=1, batch_size=8, cuda=args.cuda,
            num_filters=args.filters, num_residual_blocks=args.blocks, fp16_inference=args.cuda,
        ),
    )

    players = []
    for checkpoint in (args.net_a, args.net_b):
        nnet = NNetWrapper(game, config)
        nnet.load_checkpoint(filename=str(checkpoint.resolve()))
        players.append(NetworkPlayer(
            game, nnet, config.mcts_config, temp=0.0,
            opening_temp=args.opening_temp, opening_moves=args.opening_moves,
        ))

    arena = Arena(players[0], players[1], game)
    a_wins, b_wins, draws, _records = arena.play_games(args.games)
    low, high = _wilson(a_wins + 0.5 * draws, args.games)
    logger.info(
        "{} vs {}: {}-{}-{} → {} score rate {:.1%} (95% CI {:.1%}–{:.1%})",
        args.label_a, args.label_b, a_wins, b_wins, draws,
        args.label_a, (a_wins + 0.5 * draws) / args.games, low, high,
    )
    out = args.out or REPO_ROOT / "temp" / "benchmarks" / f"arena_{args.label_a}_vs_{args.label_b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "net_a": str(args.net_a), "net_b": str(args.net_b),
        "label_a": args.label_a, "label_b": args.label_b,
        "games": args.games, "sims": args.sims,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_score_rate": (a_wins + 0.5 * draws) / args.games,
        "wilson_95": [low, high],
    }, indent=2))
    logger.info("written to {}", out)


if __name__ == "__main__":
    main()
