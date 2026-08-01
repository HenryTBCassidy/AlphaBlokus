"""Generate parity fixtures pinning the TS web engine to the Python reference.

Plays seeded random games with the **reference** ``BlokusDuoGame`` (the
authoritative rules engine) and records, for every ply: the signed placement
board, inventories, last-piece markers, the exact legal action-id set, a
sha256 of the 44-channel float32 encoding, the game-ended value, and the
action taken. The vitest suite (``web/tests/rules_parity.test.ts``) replays
each game through the TS engine and asserts everything matches exactly.

With ``--config``/``--checkpoint`` it additionally dumps torch net outputs
(top-K policy entries + value) for a subset of positions, used by the ONNX
output-parity test (``web/tests/net_parity.test.ts``).

Usage:
    uv run python scripts/generate_web_parity_fixtures.py
    uv run python scripts/generate_web_parity_fixtures.py \
        --config run_configurations/blokus_run3_overnight.json \
        --checkpoint temp/runs/blokus/blokus_run3_overnight/Nets/accepted_82.pth.tar
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pieces import default_pieces_path

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.board import BlokusDuoBoard

#: Policy entries kept per position in the net fixture (full 17,837-vector
#: dumps would bloat the committed JSON for no extra signal).
NET_TOP_K = 50


def _ply_record(game: BlokusDuoGame, board: BlokusDuoBoard, player: int) -> dict[str, Any]:
    """Snapshot one position exactly as the TS engine will reconstruct it."""
    mask = game.valid_move_masking(board, player)
    legal = np.flatnonzero(mask).astype(int).tolist()
    planes = board.as_multi_channel(player).astype(np.float32)
    return {
        "ppbB64": base64.b64encode(board.placement_grid.tobytes()).decode("ascii"),
        "currentPlayer": player,
        "remaining": [
            sorted(board.remaining_piece_ids(1)),
            sorted(board.remaining_piece_ids(-1)),
        ],
        "lastPiece": [
            board.last_piece_played(1) or 0,
            board.last_piece_played(-1) or 0,
        ],
        "legal": legal,
        "encodingSha256": hashlib.sha256(planes.tobytes()).hexdigest(),
        "gameEnded": float(game.get_game_ended(board, player)),
    }


def _play_random_game(game: BlokusDuoGame, seed: int) -> list[dict[str, Any]]:
    """One full random-playout game; returns per-ply records (terminal included).

    The terminal record carries ``action: None`` — the TS replay stops there.
    """
    rng = np.random.default_rng(seed)
    board = game.initialise_board()
    player = 1
    plies: list[dict[str, Any]] = []
    while True:
        record = _ply_record(game, board, player)
        if record["gameEnded"] != 0.0:
            record["action"] = None
            plies.append(record)
            return plies
        action = int(rng.choice(record["legal"]))
        record["action"] = action
        plies.append(record)
        board, player = game.get_next_state(board, player, action)


def _net_records(
    config_path: Path,
    checkpoint_path: Path,
    games: list[list[dict[str, Any]]],
    positions_per_game: int,
) -> list[dict[str, Any]]:
    """Torch net outputs (top-K policy + value) for a spread of fixture positions."""
    import torch

    from alphablokus.config import load_args
    from alphablokus.registry import instantiate_game_and_network
    from alphablokus.training.checkpoint_compat import load_state_dict_compat

    config = load_args(config_path)
    game, wrapper = instantiate_game_and_network(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_state_dict_compat(wrapper.nnet, checkpoint["state_dict"])

    records: list[dict[str, Any]] = []
    for plies in games:
        step = max(1, len(plies) // positions_per_game)
        for ply in plies[::step][:positions_per_game]:
            ppb = np.frombuffer(base64.b64decode(ply["ppbB64"]), dtype=np.int8).reshape(14, 14)
            canonical = (ppb * ply["currentPlayer"]).astype(np.int8)
            planes = game.encode_compact(canonical)
            policies, values = wrapper.predict_encoded(planes[np.newaxis, ...])
            policy = policies[0]
            top_k = np.argsort(policy)[::-1][:NET_TOP_K]
            records.append(
                {
                    "ppbB64": ply["ppbB64"],
                    "currentPlayer": ply["currentPlayer"],
                    "value": float(values[0]),
                    "policySum": float(policy.sum()),
                    "topActions": top_k.astype(int).tolist(),
                    "topProbs": policy[top_k].astype(float).tolist(),
                }
            )
    return records


def main() -> int:
    """CLI entry point."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8, help="Number of seeded random games to record.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (game i uses seed + i).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "web" / "tests" / "fixtures",
        help="Directory for the fixture JSON files.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Run-config JSON (net fixture only).")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Torch checkpoint (net fixture only).")
    parser.add_argument("--net-positions-per-game", type=int, default=4, help="Positions per game in the net fixture.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())

    games = [_play_random_game(game, args.seed + index) for index in range(args.games)]
    total_plies = sum(len(plies) for plies in games)
    rules_path = args.out_dir / "rules_parity.json"
    rules_path.write_text(json.dumps({"seed": args.seed, "games": games}, indent=None))
    logger.info("Wrote {} games / {} plies to {}", len(games), total_plies, rules_path)

    if args.checkpoint is not None:
        if args.config is None:
            parser.error("--config is required with --checkpoint.")
        records = _net_records(args.config, args.checkpoint, games, args.net_positions_per_game)
        net_path = args.out_dir / "net_parity.json"
        net_path.write_text(json.dumps({"checkpoint": str(args.checkpoint), "positions": records}, indent=None))
        logger.info("Wrote {} net positions to {}", len(records), net_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
