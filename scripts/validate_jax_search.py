"""G4 box-scale validation: jax/mctx search vs python MCTS on a trained net.

For N dev-cache positions and one trained checkpoint, computes noise-free
visit distributions from:

- python MCTS, K=1 (the exact reference search),
- python MCTS, K=16 (production's virtual-loss approximation — the yardstick),
- jax search at each requested ``--top-k``.

Reports top-1 agreement and visit-distribution overlap (Σ min(p,q)) of
everything against the K=1 reference, plus timing. The acceptance gate from
``docs/plans/archive/jax-selfplay-pipeline.md`` G4: some K must reach the K=16
yardstick's agreement with the reference.

Usage (box)::

    PYTHONPATH=$PWD uv run python -m scripts.validate_jax_search \
        --checkpoint temp/runs/blokus/blokus_run3_overnight/Nets/accepted_82.pth.tar \
        --filters 128 --blocks 8 --sims 400 --positions 200 --top-k 64 128 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from alphablokus.games.blokusduo.pieces import default_pieces_path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIECES_PATH = default_pieces_path()
DEV_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "blokus_duo_positions" / "dev_5000.npz"


def _top1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a.argmax(axis=1) == b.argmax(axis=1)))


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum(axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--top-k", type=int, nargs="*", default=[64, 128, 256])
    parser.add_argument("--cpuct", type=float, default=2.5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import jax

    from alphablokus.config import MCTSConfig, NetConfig, RunConfig
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board
    from alphablokus.games.blokusduo.jax.checkpoint import convert_torch_checkpoint, params_to_device
    from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels
    from alphablokus.games.blokusduo.jax.search import SearchConfig, dense_policy, make_search
    from alphablokus.games.blokusduo.jax.tables import build_jax_tables
    from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper
    from alphablokus.search.mcts import MCTS
    from tests.fixtures.blokus_positions import iter_cached_positions

    game = BlokusDuoGame(pieces_config_path=PIECES_PATH)
    game.enable_optimised_movegen()
    run_config = RunConfig(
        game="blokusduo", run_name="validate_jax_search", num_generations=1, num_eps=1,
        temp_threshold=12, update_threshold=0.55, num_arena_matches=2,
        root_directory=REPO_ROOT / "temp", load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=args.sims, cpuct=args.cpuct),
        net_config=NetConfig(
            learning_rate=1e-3, dropout=0.0, epochs=1, batch_size=8,
            cuda=False,  # python reference runs CPU; jax uses the GPU
            num_filters=args.filters, num_residual_blocks=args.blocks,
        ),
    )
    nnet = NNetWrapper(game, run_config)
    nnet.load_checkpoint(filename=str(args.checkpoint.resolve()))

    boards, players = [], []
    for _, (board, player, sequence) in enumerate(iter_cached_positions(game, DEV_CACHE_PATH)):
        # Exclude first-move strata and terminal positions (the play loop never
        # searches a terminal state; python get_action_prob has no visits there).
        if 4 <= len(sequence) <= 26 and game.get_game_ended(board, player) == 0:
            boards.append(board)
            players.append(player)
        if len(boards) >= args.positions:
            break
    logger.info("validating on {} positions, {} sims", len(boards), args.sims)

    def python_distribution(batch_size: int) -> tuple[np.ndarray, float]:
        config = MCTSConfig(num_mcts_sims=args.sims, cpuct=args.cpuct, mcts_batch_size=batch_size)
        start = time.perf_counter()
        rows = []
        for board, player in zip(boards, players, strict=True):
            mcts = MCTS(game, nnet, config)
            canonical = game.get_canonical_form(board, player)
            rows.append(np.asarray(mcts.get_action_prob(canonical, temp=1)))
        return np.stack(rows), time.perf_counter() - start

    k1, k1_seconds = python_distribution(1)
    logger.info("python K=1 done in {:.1f}s", k1_seconds)
    # Yardstick K: at tiny sim budgets K=16's first virtual-loss batch is fully
    # consumed expanding the root (zero root-edge visits) — keep S >> K.
    yardstick_k = min(16, max(2, args.sims // 4))
    k16, k16_seconds = python_distribution(yardstick_k)
    logger.info("python K={} done in {:.1f}s", yardstick_k, k16_seconds)

    kernels = make_kernels(build_jax_tables(game))
    states = GameState(*(
        np.stack(rows) for rows in zip(
            *(numpy_state_from_board(b, p) for b, p in zip(boards, players, strict=True)), strict=True
        )
    ))
    params = params_to_device(
        convert_torch_checkpoint(args.checkpoint.resolve(), args.blocks), dtype=args.dtype,
    )

    report = {
        "sims": args.sims, "positions": len(boards), "cpuct": args.cpuct, "dtype": args.dtype,
        "checkpoint": str(args.checkpoint),
        "yardstick_k": yardstick_k,
        "yardstick_k16_vs_k1": {"top1": _top1(k16, k1), "overlap": _overlap(k16, k1)},
        "python_seconds": {"k1": k1_seconds, "k16": k16_seconds},
        "jax": {},
    }
    logger.info(
        "yardstick python K=16 vs K=1: top1 {:.3f}, overlap {:.3f}",
        report["yardstick_k16_vs_k1"]["top1"], report["yardstick_k16_vs_k1"]["overlap"],
    )

    for top_k in args.top_k:
        search = make_search(kernels, SearchConfig(
            num_simulations=args.sims, top_k=top_k, cpuct=args.cpuct, dtype=args.dtype,
        ))
        result = search(params, jax.random.PRNGKey(0), states)  # compile
        jax.block_until_ready(result.action_weights)
        start = time.perf_counter()
        result = search(params, jax.random.PRNGKey(0), states)
        jax.block_until_ready(result.action_weights)
        seconds = time.perf_counter() - start
        dense = np.asarray(dense_policy(result.action_weights, result.topk_ids, kernels.action_size))
        entry = {
            "top1_vs_k1": _top1(dense, k1), "overlap_vs_k1": _overlap(dense, k1),
            "top1_vs_k16": _top1(dense, k16), "overlap_vs_k16": _overlap(dense, k16),
            "seconds": seconds,
            "sims_per_second": len(boards) * args.sims / seconds,
        }
        report["jax"][top_k] = entry
        logger.info(
            "jax K={:>3}: vs-K1 top1 {:.3f} overlap {:.3f} | vs-K16 top1 {:.3f} | "
            "{:.1f}s ({:,.0f} sims/s incl. batch-of-{} search)",
            top_k, entry["top1_vs_k1"], entry["overlap_vs_k1"], entry["top1_vs_k16"],
            seconds, entry["sims_per_second"], len(boards),
        )

    out = args.out or REPO_ROOT / "temp" / "benchmarks" / "validate_jax_search.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("report written to {}", out)


if __name__ == "__main__":
    main()
