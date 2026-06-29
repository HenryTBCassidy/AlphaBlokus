"""Throughput benchmark for the real 16-worker self-play path.

Times ``run_self_play_episodes_parallel`` (the exact path Coach uses, pool/spawn
overhead included) and reports games/s. Seeds torch before building the net and
reuses one saved checkpoint, so every checkpoint/code version plays the *same*
games — the games/s ratio is pure code speed, not net-init noise.

Usage:
    PYTHONPATH=. python scripts/bench_parallel.py --config <cfg> --workers 16 --eps 80
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from core.config import load_args
from core.game_factory import instantiate_game_and_network
from core.parallel_self_play import run_self_play_episodes_parallel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--eps", type=int, default=80)
    args = ap.parse_args()

    config = load_args(args.config)
    config = replace(config, num_eps=args.eps)

    # Fixed weights across runs → identical games → clean games/s comparison.
    torch.manual_seed(0)
    _game, nnet = instantiate_game_and_network(config)
    ckpt = "bench_parallel_ckpt.pth"
    nnet.save_checkpoint(ckpt)

    t0 = time.perf_counter()
    examples, stats = run_self_play_episodes_parallel(
        config, generation=0, checkpoint_path=ckpt, num_workers=args.workers,
    )
    wall = time.perf_counter() - t0

    n_games = len(examples)
    total_sims = sum(s.total_sims for s in stats)
    total_moves = sum(s.num_moves for s in stats)
    print("================ PARALLEL SELF-PLAY THROUGHPUT ================")
    print(f"workers={args.workers}  eps={n_games}  wall={wall:.1f}s  "
          f"(pool/spawn overhead included)")
    print(f"games/s        = {n_games / wall:.3f}")
    print(f"total moves    = {total_moves}   total sims = {total_sims}")
    print(f"sims/s         = {total_sims / wall:.0f}")
    print("===============================================================")


if __name__ == "__main__":
    main()
