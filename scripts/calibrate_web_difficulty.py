"""Calibrate the web-play difficulty ladder: sims → relative strength.

Plays round-robin arena matches between play-time strength levels (raw-policy
argmax and PUCT at increasing sim budgets) with the SAME net — exactly the
knob the web difficulty selector moves. Strength is determined by the search
budget, not the host, so these Python numbers transfer to the browser tier
given the parity suites (docs/plans/web-play.md W4/W6/W11).

Openings are diversified with opening_temp sampling so repeated deterministic
games don't collapse to one line. Results print as win rates + relative Elo
(logistic), recorded in docs/research/web-play-calibration.md.

Usage:
    uv run python scripts/calibrate_web_difficulty.py \
        --config run_configurations/blokus_run3_overnight.json \
        --checkpoint temp/box_nets/accepted_82.pth.tar \
        --pairs 0:32 32:128 128:400 --games 16
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

from alphablokus.config import MCTSConfig, load_args
from alphablokus.evaluation.arena import Arena
from alphablokus.evaluation.players import NetworkPlayer
from alphablokus.registry import instantiate_game_and_network

if TYPE_CHECKING:
    from alphablokus.interfaces import IBoard, IGame, INeuralNetWrapper

#: Player's own opening plies played at temperature 1 for game diversity.
OPENING_MOVES = 4


class PolicyArgmaxPlayer:
    """Raw-policy player (the web tier's Level 1): argmax priors over legal moves.

    Samples its first ``OPENING_MOVES`` plies from the masked policy so
    repeated games diversify, mirroring ``NetworkPlayer``'s opening_temp.
    """

    def __init__(self, game: IGame, nnet: INeuralNetWrapper, seed: int) -> None:
        self._game = game
        self._nnet = nnet
        self._rng = np.random.default_rng(seed)
        self._move_count = 0

    def __call__(self, board: IBoard) -> int:
        priors, _ = self._nnet.predict(board)
        mask = self._game.valid_move_masking(board, 1)
        masked = priors * mask
        total = masked.sum()
        self._move_count += 1
        if total <= 0:
            return int(self._rng.choice(np.flatnonzero(mask)))
        if self._move_count <= OPENING_MOVES:
            return int(self._rng.choice(len(masked), p=masked / total))
        return int(np.argmax(masked))

    def startGame(self) -> None:  # noqa: N802 — Arena's pre-existing camelCase hook
        self._move_count = 0


def build_player(game: IGame, nnet: INeuralNetWrapper, sims: int, seed: int) -> NetworkPlayer | PolicyArgmaxPlayer:
    """A player for one strength level: sims=0 → raw policy, else PUCT MCTS."""
    if sims == 0:
        return PolicyArgmaxPlayer(game, nnet, seed)
    mcts_config = MCTSConfig(
        num_mcts_sims=sims,
        cpuct=2.5,
        profiling_level="none",
        mcts_batch_size=16,
    )
    return NetworkPlayer(game, nnet, mcts_config, temp=0.0, opening_temp=1.0, opening_moves=OPENING_MOVES)


def relative_elo(wins: int, losses: int, draws: int) -> float:
    """Logistic Elo difference implied by a match score (clamped score)."""
    total_games = wins + losses + draws
    score = (wins + 0.5 * draws) / total_games
    score = min(max(score, 1 / (2 * total_games)), 1 - 1 / (2 * total_games))
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=16, help="Games per pair (halved per colour).")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["0:32", "32:128", "128:400"],
        help="sims pairs to match, e.g. 0:32 32:128 (0 = raw policy).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON results path.")
    args = parser.parse_args()

    config = load_args(args.config)
    if config.net_config.cuda and not torch.cuda.is_available():
        config = replace(config, net_config=replace(config.net_config, cuda=False))
    game, nnet = instantiate_game_and_network(config)
    nnet.load_checkpoint(str(args.checkpoint.resolve()))
    if config.game == "blokusduo":
        game.enable_optimised_movegen()  # type: ignore[attr-defined]  # Blokus-only speedup

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = []
    for pair in args.pairs:
        low_str, high_str = pair.split(":")
        low_sims, high_sims = int(low_str), int(high_str)
        start = time.perf_counter()
        arena = Arena(
            build_player(game, nnet, high_sims, args.seed),
            build_player(game, nnet, low_sims, args.seed + 1),
            game,
        )
        high_wins, low_wins, draws, _ = arena.play_games(args.games)
        elapsed = time.perf_counter() - start
        elo = relative_elo(high_wins, low_wins, draws)
        results.append(
            {
                "pair": f"{low_sims} vs {high_sims} sims",
                "highSims": high_sims,
                "lowSims": low_sims,
                "highWins": high_wins,
                "lowWins": low_wins,
                "draws": draws,
                "eloHighMinusLow": round(elo, 1),
                "elapsedS": round(elapsed, 1),
            }
        )
        logger.info(
            "{} sims vs {} sims: {}-{}-{} → Δelo ≈ {:+.0f} for the higher budget ({:.0f}s)",
            high_sims,
            low_sims,
            high_wins,
            low_wins,
            draws,
            elo,
            elapsed,
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"games_per_pair": args.games, "results": results}, indent=1))
        logger.info("Wrote {}", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
