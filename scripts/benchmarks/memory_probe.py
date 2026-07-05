"""Full-buffer training memory probe — know the peak RAM before renting a GPU.

The training memory peak lands at the *buffer-fill generation* (when the rolling
replay buffer first reaches capacity), which is deep into a paid run — so short
validation runs never reach it and the OOM only shows up hours in
(docs/plans/fix-training-oom.md). This script reproduces that peak cheaply: it
builds a full-size synthetic replay buffer and drives the training DataLoader at
the config's worker count, then prints the **measured peak physical memory**
next to the pre-flight guard's **estimate** and the machine's **available RAM**,
so the memory cost is known before committing budget.

**Physical, not summed RSS.** The memmap-backed dataset (``training/memmap_dataset``)
lets N DataLoader workers ``mmap`` and *share* one buffer file. Summed process-tree
RSS counts those shared pages once *per worker*, so it credits the sharing fix as
an ~N× cost (it reported ~44 GB at 8 workers on a 40k-game buffer that actually
used ~14 GB physical and ran to completion on a 31 GB box). This probe measures
the real physical figure instead — the number the OOM-killer acts on:

1. **cgroup accounting** (Linux/container): ``memory.current`` — the kernel's own
   tally of the cgroup's physical footprint, compared against ``memory.max``.
2. **summed PSS** (Linux, no cgroup): proportional set size splits each shared
   page across its sharers, so the tree sum ≈ physical.
3. **summed RSS** (last resort, e.g. macOS): over-counts shared memory; printed
   with a loud warning. Run the probe on the Linux box/pod for a real number.

The printed peak reconciles against ``estimate_peak_ram_bytes``: that guard models
the memmapped buffer **once** (not once per worker) and stays a *conservative
upper bound* on physical — so ``estimate >= measured-physical`` is the healthy
case, and ``estimate < measured`` would mean the guard is under-counting.

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
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Callable

_GB = 1024**3

# cgroup "current usage" files, newest scheme first — the kernel's live tally of
# the cgroup's physical memory, which is what the OOM-killer compares against the
# limit (``memory.max`` / ``memory.limit_in_bytes``, read by the guard's
# ``available_ram_bytes``). ``memory.current`` sampled over the section is the
# true physical peak on a pod. A module-level tuple so tests can point it at a
# temp file. (cgroup v2 also exposes a kernel high-water ``memory.peak``; we
# sample ``current`` instead so the figure is scoped to *this* probe section
# rather than the cgroup's whole lifetime.)
_CGROUP_CURRENT_PATHS = (
    Path("/sys/fs/cgroup/memory.current"),  # cgroup v2
    Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),  # cgroup v1
)


class MemorySource(StrEnum):
    """Which accounting the probe uses for *physical* memory, best first."""

    CGROUP = "cgroup"
    PSS = "PSS"
    RSS = "RSS"


class PeakPhysicalMemory:
    """Sample peak *physical* memory over a section, in a background thread.

    Unlike summed process-tree RSS — which counts pages shared between the
    DataLoader workers once *per worker*, inflating the memmapped buffer ~N× —
    this samples the resolved physical source (cgroup accounting, else summed
    PSS, else summed RSS with a warning). The peak is the high-water figure that
    would get the run OOM-killed.

    A pre-resolved ``source`` can be injected (so a sweep resolves once and
    tests can drive a specific branch); by default it resolves the best source
    available on this machine.
    """

    def __init__(
        self,
        interval_s: float = 0.02,
        source: tuple[MemorySource, Callable[[], int]] | None = None,
    ) -> None:
        self._interval_s = interval_s
        self._source, self._read_bytes = source if source is not None else resolve_physical_source()
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak_bytes = max(self._peak_bytes, self._read_bytes())
            time.sleep(self._interval_s)

    def __enter__(self) -> PeakPhysicalMemory:
        self._peak_bytes = self._read_bytes()
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

    @property
    def source(self) -> MemorySource:
        return self._source


def select_memory_source(*, has_cgroup: bool, has_pss: bool) -> MemorySource:
    """Pick the physical-memory source: cgroup > PSS > RSS.

    cgroup accounting is the true container figure the OOM-killer acts on; summed
    PSS approximates physical outside a container; summed RSS is the last resort
    and over-counts shared memory.

    Args:
        has_cgroup: A cgroup ``memory.current`` file is readable.
        has_pss: ``psutil.memory_full_info().pss`` is available (Linux).

    Returns:
        The best available :class:`MemorySource`.
    """
    if has_cgroup:
        return MemorySource.CGROUP
    if has_pss:
        return MemorySource.PSS
    return MemorySource.RSS


def resolve_physical_source(
    cgroup_paths: tuple[Path, ...] = _CGROUP_CURRENT_PATHS,
    process: psutil.Process | None = None,
) -> tuple[MemorySource, Callable[[], int]]:
    """Resolve the best physical-memory source and a callable that samples it.

    Returns the chosen :class:`MemorySource` and a function reading current
    physical bytes from it. Prints a warning when only summed RSS is available
    (macOS / no cgroup / no PSS), since RSS over-counts the shared memmapped
    buffer — the probe belongs on the Linux box/pod where cgroup/PSS is present.
    """
    process = process if process is not None else psutil.Process()
    source = select_memory_source(
        has_cgroup=_read_cgroup_current(cgroup_paths) is not None,
        has_pss=_read_tree_pss(process) is not None,
    )
    if source is MemorySource.CGROUP:
        return source, lambda: _read_cgroup_current(cgroup_paths) or 0
    if source is MemorySource.PSS:
        return source, lambda: _read_tree_pss(process) or 0
    print(
        "  WARNING: no cgroup accounting or PSS available (bare macOS?); falling back to summed\n"
        "  process-tree RSS, which OVER-COUNTS shared memory — the memmapped buffer is counted\n"
        "  once per DataLoader worker, so this figure is inflated ~N×. Run on the Linux box/pod\n"
        "  for a trustworthy physical number."
    )
    return source, lambda: _read_tree_rss(process)


def _read_cgroup_current(paths: tuple[Path, ...] = _CGROUP_CURRENT_PATHS) -> int | None:
    """Current physical memory charged to this cgroup in bytes, or None.

    Reads cgroup v2 ``memory.current`` then v1 ``memory.usage_in_bytes`` — the
    kernel's own tally of the container's physical footprint, exactly what the
    OOM-killer weighs against the limit. None when no cgroup file is present
    (e.g. a bare macOS dev box).
    """
    for path in paths:
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def _read_tree_pss(process: psutil.Process) -> int | None:
    """Summed PSS of ``process`` and all children in bytes, or None if unsupported.

    PSS (proportional set size) splits each shared page evenly across the
    processes mapping it, so summing over the tree ≈ physical memory even when N
    DataLoader workers share one memmapped buffer — unlike summed RSS, which
    counts the shared buffer once per worker. Linux-only (reads ``smaps``);
    returns None elsewhere (macOS ``memory_full_info`` exposes no ``pss``).
    """
    try:
        total = process.memory_full_info().pss
    except (psutil.Error, AttributeError):
        return None
    for child in process.children(recursive=True):
        with contextlib.suppress(psutil.Error, AttributeError):
            total += child.memory_full_info().pss
    return int(total)


def _read_tree_rss(process: psutil.Process) -> int:
    """Summed RSS of ``process`` and all children in bytes (over-counts sharing).

    Each process that maps a shared page counts the whole page, so with the
    memmapped buffer this inflates by ~workers×. Last-resort fallback only.
    """
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        with contextlib.suppress(psutil.Error):
            total += child.memory_info().rss
    return int(total)


def _verdict(measured_bytes: int, available_bytes: int) -> str:
    """FITS when the measured physical peak is under available RAM, else OVER."""
    return "FITS" if measured_bytes < available_bytes else "OVER"


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


def probe_workers(
    config: RunConfig,
    examples: list,
    workers: int,
    scratch_root: Path,
    source: tuple[MemorySource, Callable[[], int]],
) -> int:
    """Build the dataset + loader at ``workers`` workers, iterate once, return peak physical bytes."""
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
        with PeakPhysicalMemory(source=source) as peak:
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

    # Resolve the physical-memory source once (prints the RSS warning at most
    # once), then reuse it across the sweep.
    source = resolve_physical_source()
    available = available_ram_bytes()
    print(f"\nphysical-memory source: {source[0]}")
    print(f"available RAM (physical or tighter cgroup cap): {available / _GB:.1f} GB\n")
    peak_column = f"peak physical ({source[0]})"
    print(f"  {'workers':>7}  {peak_column:>22}  {'guard estimate':>15}  {'verdict':>8}")
    print("  " + "-" * 62)
    with tempfile.TemporaryDirectory(prefix="memory_probe_") as scratch:
        scratch_root = Path(scratch)
        for workers in worker_counts:
            measured = probe_workers(config, examples, workers, scratch_root, source)
            # Estimate for exactly what was probed: the built buffer at this worker count.
            probe_config = replace(
                config,
                replay_buffer_games=num_games,
                num_eps=0,
                net_config=replace(config.net_config, perf=replace(config.net_config.perf, dataloader_workers=workers)),
            )
            estimate = estimate_peak_ram_bytes(probe_config)
            fits = _verdict(measured, available)
            print(
                f"  {workers:>7}  {measured / _GB:>19.2f} GB  {estimate / _GB:>12.2f} GB  {fits:>8}"
                + ("  <- estimate < measured!" if estimate < measured else "")
            )
    print(
        "\nRun this at the FULL buffer on the target box before a paid run; if 'peak physical' approaches "
        "'available RAM', lower replay_buffer_games / dataloader_workers before launching."
    )


if __name__ == "__main__":
    main()
