"""G7: apples-to-apples throughput of the python vs jax self-play backends.

Same trained checkpoint, same flat sim budget, same temperature/noise
hyperparameters; the only difference is the backend. Reports games/s, sims/s
(true sim counts from per-game stats), positions/s (with the 2× augmentation),
and VRAM where available. The python baseline uses the production
configuration (N workers, all-GPU, K=16 leaf batching); the jax backend sweeps
``(batch_size, top_k)``.

Usage (box)::

    PYTHONPATH=$PWD uv run python -m scripts.benchmark_selfplay_backends \
        --checkpoint temp/runs/blokus/blokus_run3_overnight/Nets/accepted_82.pth.tar \
        --filters 128 --blocks 8 --sims 400 \
        --python-episodes 64 --python-workers 16 \
        --jax-batch-sizes 128 256 512 --jax-top-k 128

Writes JSON next to the other benchmark artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from loguru import logger

from alphablokus.games.blokusduo.pieces import default_pieces_path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIECES_PATH = default_pieces_path()


def _gpu_memory_mib() -> int | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        return int(output)
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return None


def _base_config(args, *, num_eps: int, workers: int, backend: str, jax_selfplay=None):
    from alphablokus.core.config import JaxSelfPlayConfig, MCTSConfig, NetConfig, RunConfig

    return RunConfig(
        game="blokusduo", run_name="bench_backends", num_generations=1, num_eps=num_eps,
        temp_threshold=12, update_threshold=0.55, num_arena_matches=2,
        root_directory=REPO_ROOT / "temp" / "bench_backends", load_model=False,
        mcts_config=MCTSConfig(
            num_mcts_sims=args.sims, cpuct=2.5, dirichlet_epsilon=0.25, dirichlet_alpha=0.03,
            mcts_batch_size=16, sim_schedule="flat",
        ),
        net_config=NetConfig(
            learning_rate=1e-3, dropout=0.0, epochs=1, batch_size=256,
            cuda=backend == "python" and args.python_cuda,
            num_filters=args.filters, num_residual_blocks=args.blocks, fp16_inference=True,
        ),
        selfplay_backend=backend,  # informational; we call the backends directly
        jax_selfplay=jax_selfplay or JaxSelfPlayConfig(),
        num_parallel_workers=workers, worker_cuda=args.python_cuda,
        use_optimised_movegen=True, seed=42,
    )


def _summarise(label: str, stats, seconds: float, num_games: int) -> dict:
    total_sims = sum(s.total_sims for s in stats)
    total_moves = sum(s.num_moves for s in stats)
    entry = {
        "label": label, "seconds": seconds, "games": num_games,
        "games_per_second": num_games / seconds,
        "sims_per_second": total_sims / seconds,
        "moves_per_second": total_moves / seconds,
        "gpu_memory_mib": _gpu_memory_mib(),
    }
    logger.info(
        "{:<28} {:>6.1f}s | {:>6.2f} games/s | {:>10,.0f} sims/s | VRAM {} MiB",
        label, seconds, entry["games_per_second"], entry["sims_per_second"],
        entry["gpu_memory_mib"],
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--python-episodes", type=int, default=64)
    parser.add_argument("--python-workers", type=int, default=16)
    parser.add_argument("--python-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--jax-batch-sizes", type=int, nargs="*", default=[128, 256, 512])
    parser.add_argument("--jax-top-k", type=int, default=128)
    parser.add_argument("--jax-dtype", default="bfloat16")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from alphablokus.core.config import JaxSelfPlayConfig
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.jax.backend import generate_self_play_games
    from alphablokus.games.blokusduo.neuralnets.wrapper import NNetWrapper

    checkpoint = args.checkpoint.resolve()
    results = []

    if not args.skip_python:
        from alphablokus.parallel.pool import run_self_play_episodes_parallel

        config = _base_config(args, num_eps=args.python_episodes, workers=args.python_workers,
                              backend="python")
        game = BlokusDuoGame(pieces_config_path=PIECES_PATH)
        nnet = NNetWrapper(game, config)
        nnet.load_checkpoint(filename=str(checkpoint))
        nnet.save_checkpoint(filename="bench_init.pth.tar")  # workers load from net_directory
        start = time.perf_counter()
        _examples, stats = run_self_play_episodes_parallel(
            config=config, generation=1, checkpoint_path="bench_init.pth.tar",
            num_workers=args.python_workers,
        )
        results.append(_summarise(
            f"python x{args.python_workers} K=16", stats, time.perf_counter() - start,
            args.python_episodes,
        ))

    for batch_size in args.jax_batch_sizes:
        jax_config = JaxSelfPlayConfig(
            batch_size=batch_size, top_k=args.jax_top_k, dtype=args.jax_dtype, wave_plies=32,
        )
        config = _base_config(args, num_eps=2 * batch_size, workers=1, backend="jax",
                              jax_selfplay=jax_config)
        game = BlokusDuoGame(pieces_config_path=PIECES_PATH)
        nnet = NNetWrapper(game, config)
        nnet.load_checkpoint(filename=str(checkpoint))
        nnet.save_checkpoint(filename="bench_init.pth.tar")
        # Warm-up generation compiles the actor (artefact cache keys on shapes,
        # not num_eps, so a small warm-up covers the real run); measure the second.
        import dataclasses

        warmup = dataclasses.replace(config, num_eps=max(8, batch_size // 8))
        generate_self_play_games(warmup, generation=1, checkpoint_path="bench_init.pth.tar")
        start = time.perf_counter()
        _examples, stats = generate_self_play_games(
            config, generation=2, checkpoint_path="bench_init.pth.tar",
        )
        results.append(_summarise(
            f"jax B={batch_size} K={args.jax_top_k} {args.jax_dtype}", stats,
            time.perf_counter() - start, config.num_eps,
        ))

    out = args.out or REPO_ROOT / "temp" / "benchmarks" / "backend_throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args) | {"checkpoint": str(checkpoint), "out": str(out)},
                               "results": results}, indent=2, default=str))
    logger.info("report written to {}", out)


if __name__ == "__main__":
    main()
