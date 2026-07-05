"""Verify a browser-engine game record against the Python reference stack.

Counterpart of ``web/scripts/agreement_game.ts`` (plan step W11): the TS
engine + exported ONNX net played full games with raw-policy argmax; this
script replays each record through the reference ``BlokusDuoGame`` + torch
and asserts, ply by ply:

- the recorded legal-move set matches ``valid_move_masking`` exactly,
- the recorded action is what torch's policy argmax over legal moves picks
  (near-ties within float tolerance are accepted and reported),
- the recorded net value matches torch within tolerance,
- the final scores and result match the reference scoring.

Usage:
    (cd web && npm run agreement)
    uv run python scripts/verify_web_agreement.py \
        --config run_configurations/blokus_run3_overnight.json \
        --checkpoint temp/box_nets/accepted_82.pth.tar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from alphablokus.config import load_args
from alphablokus.registry import instantiate_game_and_network

VALUE_TOLERANCE = 1e-4
PROB_TIE_TOLERANCE = 1e-6


def main() -> int:
    """CLI entry point. Returns non-zero on any disagreement."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--record",
        type=Path,
        default=repo_root / "web" / "tests" / "artifacts" / "agreement_games.json",
        help="Game record written by `npm run agreement`.",
    )
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    config = load_args(args.config)
    game, wrapper = instantiate_game_and_network(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    wrapper.nnet.load_state_dict(checkpoint["state_dict"])

    near_ties = 0
    total_plies = 0
    for game_index, game_record in enumerate(record["games"]):
        board = game.initialise_board()
        player = 1
        for ply_index, ply in enumerate(game_record["plies"]):
            at = f"game {game_index} ply {ply_index}"
            total_plies += 1

            mask = game.valid_move_masking(board, player)
            legal = [int(action) for action in np.flatnonzero(mask)]
            if legal != ply["legal"]:
                logger.error("{}: legal-move sets differ (py {} vs ts {})", at, len(legal), len(ply["legal"]))
                return 1

            canonical = game.get_canonical_form(board, player)
            priors, value = wrapper.predict(canonical)
            if abs(float(value) - ply["value"]) > VALUE_TOLERANCE:
                logger.error("{}: value differs (py {:.6f} vs ts {:.6f})", at, float(value), ply["value"])
                return 1

            legal_priors = np.asarray([priors[action] for action in legal])
            python_action = legal[int(np.argmax(legal_priors))]
            if python_action != ply["action"]:
                # fp32 graph-vs-graph rounding can flip a near-tie; accept it
                # only when the two candidates are indistinguishable in torch.
                ts_prob = float(priors[ply["action"]])
                py_prob = float(legal_priors.max())
                if abs(ts_prob - py_prob) > PROB_TIE_TOLERANCE:
                    logger.error(
                        "{}: chosen actions differ (py {} p={:.6g} vs ts {} p={:.6g})",
                        at,
                        python_action,
                        py_prob,
                        ply["action"],
                        ts_prob,
                    )
                    return 1
                near_ties += 1

            board, player = game.get_next_state(board, player, ply["action"])

        ended = game.get_game_ended(board, 1)
        if ended == 0:
            logger.error("game {}: reference says the game is not over at the recorded terminal", game_index)
            return 1
        white_score = game._calculate_score(board, 1)
        black_score = game._calculate_score(board, -1)
        if [white_score, black_score] != game_record["finalScores"]:
            logger.error(
                "game {}: final scores differ (py {} vs ts {})",
                game_index,
                [white_score, black_score],
                game_record["finalScores"],
            )
            return 1

    logger.info(
        "Agreement verified: {} games, {} plies, {} near-tie argmax flips accepted.",
        len(record["games"]),
        total_plies,
        near_ties,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
