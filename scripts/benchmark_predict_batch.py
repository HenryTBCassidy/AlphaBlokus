"""Microbenchmark: ``predict_batch(K)`` vs K serial ``predict()`` calls.

Isolates the *pure GPU-side* batching speedup from all MCTS / worker-contention
overhead — the clean counterpart to the end-to-end phase benchmark
(`scripts/benchmark_phases.py`), which under 8-worker contention is noisy.

For each batch size K it times evaluating K leaf positions two ways:
  - one ``predict_batch(K boards)`` call, vs
  - K separate ``predict(board)`` calls,
both producing the same K (policy, value) results. Reports per-leaf latency and
the batched-vs-serial speedup. Run on the *idle* GPU for a clean number.

Usage:
    uv run python -m scripts.benchmark_predict_batch                 # default sweep
    uv run python -m scripts.benchmark_predict_batch --cuda --iters 50
    uv run python -m scripts.benchmark_predict_batch --ks 1,8,16,32
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from core.config import MCTSConfig, NetConfig, RunConfig
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.neuralnets.wrapper import NNetWrapper

_PIECES = Path(__file__).resolve().parent.parent / "games" / "blokusduo" / "pieces.json"


def _make_boards(game: BlokusDuoGame, n: int, seed: int) -> list:
    """Build ``n`` distinct canonical boards by playing random legal moves.

    Board *content* barely affects forward-pass time (fixed 44x14x14 input), but
    using distinct real positions keeps the measurement honest.
    """
    rng = np.random.default_rng(seed)
    boards = []
    board = game.initialise_board()
    player = 1
    while len(boards) < n:
        valids = np.where(np.asarray(game.valid_move_masking(board, player)) > 0)[0]
        if len(valids) == 0:
            board, player = game.initialise_board(), 1
            continue
        boards.append(game.get_canonical_form(board, player))
        board, player = game.get_next_state(board, player, int(rng.choice(valids)))
    return boards


def _sync(cuda: bool) -> None:
    if cuda:
        torch.cuda.synchronize()


def _time_batched(nnet: NNetWrapper, boards: list, iters: int, cuda: bool) -> float:
    """Mean wall-clock (s) of one ``predict_batch(boards)`` call."""
    _sync(cuda)
    start = time.perf_counter()
    for _ in range(iters):
        nnet.predict_batch(boards)
    _sync(cuda)
    return (time.perf_counter() - start) / iters


def _time_serial(nnet: NNetWrapper, boards: list, iters: int, cuda: bool) -> float:
    """Mean wall-clock (s) of evaluating the same boards via K serial predict()."""
    _sync(cuda)
    start = time.perf_counter()
    for _ in range(iters):
        for board in boards:
            nnet.predict(board)
    _sync(cuda)
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark predict_batch vs serial predict")
    parser.add_argument("--cuda", action="store_true", help="Run on GPU (default: CPU)")
    parser.add_argument("--ks", type=str, default="1,2,4,8,16,32",
                        help="Comma-separated batch sizes to sweep")
    parser.add_argument("--iters", type=int, default=30, help="Timed iterations per K (after warmup)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (discarded)")
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    game = BlokusDuoGame(pieces_config_path=_PIECES)
    net_cfg = NetConfig(
        learning_rate=1e-3, dropout=0.3, epochs=1, batch_size=64, cuda=args.cuda,
        num_filters=args.filters, num_residual_blocks=args.blocks,
    )
    run_cfg = RunConfig(
        game="blokusduo", run_name="microbench", num_generations=1, num_eps=2,
        temp_threshold=5, update_threshold=0.55, max_queue_length=10, num_arena_matches=2,
        max_generations_lookback=1, root_directory=Path("/tmp/microbench"), load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=1, cpuct=1.0), net_config=net_cfg,
    )
    nnet = NNetWrapper(game, run_cfg)
    device = "CUDA" if args.cuda else "CPU"
    print(f"Device: {device} | net {args.filters}f x {args.blocks}b | "
          f"iters={args.iters} (warmup {args.warmup})")
    print(f"{'K':>4} {'serial/leaf':>13} {'batched/leaf':>14} {'speedup':>9} {'batched leaves/s':>18}")

    max_boards = _make_boards(game, max(ks), seed=42)
    for k in ks:
        boards = max_boards[:k]
        # Warmup both paths (cuDNN autotune, allocator).
        for _ in range(args.warmup):
            nnet.predict_batch(boards)
            nnet.predict(boards[0])
        _sync(args.cuda)

        batched = _time_batched(nnet, boards, args.iters, args.cuda)
        serial = _time_serial(nnet, boards, args.iters, args.cuda)
        serial_per_leaf = serial / k
        batched_per_leaf = batched / k
        speedup = serial / batched if batched > 0 else float("nan")
        leaves_per_s = k / batched if batched > 0 else float("nan")
        print(f"{k:>4} {serial_per_leaf * 1e3:>10.3f}ms {batched_per_leaf * 1e3:>11.3f}ms "
              f"{speedup:>8.2f}x {leaves_per_s:>16.0f}")


if __name__ == "__main__":
    main()
