"""Ad-hoc diagnostic: replay net-vs-Pentobi games and dump the anatomy of each result.

Not a permanent tool — probes *how* the net loses (score margin, board coverage,
pieces placed, when it gets walled off, opening move) rather than just win/loss.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

os.environ.setdefault("ALPHABLOKUS_MPS", "1")  # eval-only MPS opt-in (see base_wrapper)

import numpy as np

from core.arena import Arena
from core.config import load_args
from core.game_factory import instantiate_game_and_network
from core.players import NetworkPlayer
from games.blokusduo.pentobi_player import PentobiPlayer


def eval_cfg(base, sims):
    return replace(base, num_mcts_sims=sims, mcts_batch_size=16, dirichlet_epsilon=0.0, sim_schedule="flat")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--sims", type=int, default=800)
    args = ap.parse_args()

    cfg = load_args(args.config)
    import torch
    if cfg.net_config.cuda and not torch.cuda.is_available():
        cfg = replace(cfg, net_config=replace(cfg.net_config, cuda=False))
    game, nnet = instantiate_game_and_network(cfg)
    nnet.load_checkpoint(filename=args.net)
    net_player = NetworkPlayer(game, nnet, eval_cfg(cfg.mcts_config, args.sims), temp=0.0,
                               opening_temp=1.0, opening_moves=4)

    pentobi = PentobiPlayer(game, args.level, seed=1)
    try:
        nw, pw, d, records = Arena(net_player, pentobi, game).play_games(args.games, record=True)
    finally:
        pentobi.close()

    codec = game.action_codec
    piece_sq = {pid: int(game.piece_manager.pieces[pid].identity.sum())
                for pid in game.piece_manager.pieces}
    total_sq = sum(piece_sq.values())

    print(f"\n=== net {nw}-{pw}-{d} at L{args.level}, {args.sims} sims, {args.games} games ===")
    print(f"(total piece squares per side = {total_sq})\n")
    rows = []
    for gi, rec in enumerate(records):
        # net is player1; player1_was_white → net's PlayerSide (+1 white / -1 black)
        net_side = 1 if rec.player1_was_white else -1
        board = game.initialise_board()
        net_moves = pen_moves = 0
        net_first_pass = pen_first_pass = None
        net_open = None
        for m in rec.moves:
            is_net = (m.player == net_side)
            is_pass = codec.is_pass(m.action)
            if is_net:
                net_moves += 1
                if net_open is None and not is_pass:
                    a = codec.decode(m.action)
                    net_open = (a.piece_id, a.x_coordinate, a.y_coordinate)
                if is_pass and net_first_pass is None:
                    net_first_pass = net_moves
            else:
                pen_moves += 1
                if is_pass and pen_first_pass is None:
                    pen_first_pass = pen_moves
            board, _ = game.get_next_state(board, m.player, m.action)

        net_rem = board.remaining_piece_ids(net_side)
        pen_rem = board.remaining_piece_ids(-net_side)
        net_placed_sq = total_sq - sum(piece_sq[p] for p in net_rem)
        pen_placed_sq = total_sq - sum(piece_sq[p] for p in pen_rem)
        net_score = game._calculate_score(board, net_side)
        pen_score = game._calculate_score(board, -net_side)
        margin = net_score - pen_score
        won = margin > 0
        rows.append((net_placed_sq, pen_placed_sq, 21 - len(net_rem), 21 - len(pen_rem), margin))
        print(
            f"g{gi:2d} {'W' if net_side==1 else 'B'} | "
            f"{'WIN ' if won else 'loss'} margin {margin:+3d} | "
            f"squares net {net_placed_sq:2d} vs pen {pen_placed_sq:2d} | "
            f"pieces net {21-len(net_rem):2d} vs pen {21-len(pen_rem):2d} | "
            f"net 1st pass @ move {net_first_pass} (of {net_moves}) | open {net_open}"
        )

    arr = np.array([r for r in rows], dtype=float)
    print("\n--- averages ---")
    print(f"squares placed:  net {arr[:,0].mean():.1f}  vs  pentobi {arr[:,1].mean():.1f}")
    print(f"pieces placed:   net {arr[:,2].mean():.1f}  vs  pentobi {arr[:,3].mean():.1f}")
    print(f"score margin:    {arr[:,4].mean():+.1f}  (min {arr[:,4].min():+.0f}, max {arr[:,4].max():+.0f})")


if __name__ == "__main__":
    main()
