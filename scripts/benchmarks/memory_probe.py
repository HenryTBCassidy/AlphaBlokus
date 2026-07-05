"""Full-buffer training memory probe — know the peak RAM before renting a GPU.

The training memory peak lands at the *buffer-fill generation* (when the rolling
replay buffer first reaches capacity), which is deep into a paid run — so short
validation runs never reach it and the OOM only shows up hours in
(docs/plans/fix-training-oom.md). This script reproduces that peak cheaply: it
builds a full-size synthetic replay buffer and drives the training DataLoader at
the config's worker count, then prints the **measured peak process-tree RSS**
next to the pre-flight guard's **estimate** and the machine's **available RAM**,
so the memory cost is known before committing budget.

Run it at *full scale on the target box* (or a cheap big-RAM pod) before a paid
run — that is the check short runs cannot give you. On a small dev box, point it
at a reduced buffer with ``--games`` (the per-worker multiplier is what matters,
and it is visible at any buffer size).

Usage::

    # Full production buffer at the config's worker count (run on the box):
    uv run python -m scripts.benchmarks.memory_probe --config run_configurations/blokus_cloud_v2.json

    # Reduced buffer for a laptop, sweeping worker counts:
    uv run python -m scripts.benchmarks.memory_probe \\
        --config run_configurations/blokus_cloud_v2.json --games 3000 --workers 0,2,4,8
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset

from alphablokus.config import RunConfig, load_args
from alphablokus.games.base_wrapper import _LazyPolicyDataset, resolve_dataloader_context
from alphablokus.registry import instantiate_game
from alphablokus.storage.sparse_policy import sparsify
from alphablokus.training.diagnostics import available_ram_bytes, estimate_peak_ram_bytes
from alphablokus.training.memmap_dataset import MemmapPolicyDataset

_GB = 1024**3


class PeakTreeRSS:
    """Sample peak RSS of this process **and all its children** in a thread.

    ``resource.getrusage`` only sees the main process, so it misses the whole
    point here — the DataLoader *workers* are where the memory goes. psutil walks
    the process tree; the peak is what would get the run OOM-killed.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self._process = psutil.Process()
        self._interval_s = interval_s
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _tree_rss_bytes(self) -> int:
        total = self._process.memory_info().rss
        for child in self._process.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                total += child.memory_info().rss
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak_bytes = max(self._peak_bytes, self._tree_rss_bytes())
            time.sleep(self._interval_s)

    def __enter__(self) -> PeakTreeRSS:
        self._peak_bytes = self._tree_rss_bytes()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes


def build_synthetic_buffer(action_size: int, num_games: int, positions_per_game: int, nnz: int, seed: int) -> list:
    """A full-size replay buffer of synthetic positions (compact board, sparse policy, value).

    Shaped like Blokus self-play output: 14×14 int8 compact boards and sparse
    ``nnz``-nonzero policies. The point is the *volume and layout*, not the
    contents — the memory footprint is what we measure.
    """
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(num_games * positions_per_game):
        board = rng.integers(-21, 22, (14, 14)).astype(np.int8)
        dense = np.zeros(action_size, dtype=np.float32)
        idx = rng.choice(action_size, size=nnz, replace=False)
        weights = rng.random(nnz).astype(np.float32)
        dense[idx] = weights / weights.sum()
        examples.append((board, sparsify(dense), float(rng.uniform(-1.0, 1.0))))
    return examples


def _build_loader(config: RunConfig, dataset: Dataset, workers: int) -> DataLoader:
    """Build the training DataLoader exactly as ``BaseNNetWrapper.train`` does."""
    perf = config.net_config.perf
    kwargs: dict[str, object] = {}
    if workers > 0:
        kwargs = {
            "num_workers": workers,
            "persistent_workers": perf.persistent_workers,
            "prefetch_factor": perf.prefetch_factor,
        }
        if perf.dataloader_context != "fork":
            kwargs["multiprocessing_context"] = resolve_dataloader_context(perf.dataloader_context)
    return DataLoader(
        dataset,
        batch_size=config.net_config.batch_size,
        shuffle=True,
        pin_memory=perf.pin_memory and torch.cuda.is_available(),
        **kwargs,
    )


def probe_workers(config: RunConfig, examples: list, workers: int, scratch_root: Path) -> int:
    """Build the dataset + loader at ``workers`` workers, iterate once, return peak tree RSS bytes."""
    game = instantiate_game(config)
    action_size = game.get_action_size()
    boards_np, raw_pis, vs_np = zip(*examples, strict=True)

    memmap_dir: Path | None = None
    dataset: Dataset
    if workers > 0:
        memmap_dir = scratch_root / f"memmap_w{workers}"
        dataset = MemmapPolicyDataset.build(examples, action_size, game.encode_compact, memmap_dir)
    else:
        dataset = _LazyPolicyDataset(list(boards_np), list(raw_pis), list(vs_np), action_size, game.encode_compact)

    loader = _build_loader(config, dataset, workers)
    try:
        with PeakTreeRSS() as peak:
            for _batch in loader:
                pass
        return peak.peak_bytes
    finally:
        del loader
        if memmap_dir is not None:
            shutil.rmtree(memmap_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Run config JSON to probe.")
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="Buffer size in games to build (0 = the config's replay_buffer_games). Reduce on a small dev box.",
    )
    parser.add_argument(
        "--workers",
        default="",
        help="Comma-separated worker counts to sweep (default: the config's dataloader_workers).",
    )
    parser.add_argument(
        "--positions-per-game", type=int, default=65, help="Synthetic positions per game (~Blokus avg)."
    )
    parser.add_argument("--nnz", type=int, default=150, help="Nonzeros per synthetic sparse policy.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_args(args.config)
    num_games = args.games or config.replay_buffer_games
    worker_counts = (
        [int(w) for w in args.workers.split(",")] if args.workers else [config.net_config.perf.dataloader_workers]
    )

    game = instantiate_game(config)
    action_size = game.get_action_size()
    print(
        f"config={args.config} game={config.game} action_size={action_size}\n"
        f"buffer={num_games} games × {args.positions_per_game} pos/game "
        f"= {num_games * args.positions_per_game} positions  (nnz={args.nnz}, batch={config.net_config.batch_size})"
    )
    print(f"building synthetic buffer ({num_games} games)...")
    examples = build_synthetic_buffer(action_size, num_games, args.positions_per_game, args.nnz, args.seed)

    available = available_ram_bytes()
    print(f"\navailable RAM (physical or tighter cgroup cap): {available / _GB:.1f} GB\n")
    print(f"  {'workers':>7}  {'measured peak RSS':>18}  {'guard estimate':>15}  {'verdict':>8}")
    print("  " + "-" * 58)
    with tempfile.TemporaryDirectory(prefix="memory_probe_") as scratch:
        scratch_root = Path(scratch)
        for workers in worker_counts:
            measured = probe_workers(config, examples, workers, scratch_root)
            # Estimate for exactly what was probed: the built buffer at this worker count.
            probe_config = replace(
                config,
                replay_buffer_games=num_games,
                num_eps=0,
                net_config=replace(config.net_config, perf=replace(config.net_config.perf, dataloader_workers=workers)),
            )
            estimate = estimate_peak_ram_bytes(probe_config)
            fits = "FITS" if measured < available else "OVER"
            print(
                f"  {workers:>7}  {measured / _GB:>15.2f} GB  {estimate / _GB:>12.2f} GB  {fits:>8}"
                + ("  <- estimate < measured!" if estimate < measured else "")
            )
    print(
        "\nRun this at the FULL buffer on the target box before a paid run; if 'measured' approaches "
        "'available RAM', lower replay_buffer_games / dataloader_workers before launching."
    )


if __name__ == "__main__":
    main()
