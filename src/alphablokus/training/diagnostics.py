"""Process-level resource diagnostics and pre-flight guards.

Two halves of the same OOM story (``docs/plans/archive/oom-hardening.md`` O8):
:func:`get_memory_snapshot` makes memory visible *during* a run (RSS + peak
RSS at phase transitions), and :func:`check_ram_budget` refuses configs whose
estimated peak cannot fit the machine *before* a run starts — turning a 3 a.m.
OOM kill into an instant config error.
"""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import torch
from loguru import logger

if TYPE_CHECKING:
    from alphablokus.config import RunConfig

# Rough RAM footprint of one buffered self-play game (its ``ProcessedExample``
# list, symmetry augmentation included): compact board + sparse policy arrays
# + tuple/object overhead, ~2 KB × ~100 positions for Blokus. Deliberately
# generous — this feeds a pre-flight budget check, not an allocator. Unknown
# games fall back to the largest estimate.
_EST_BYTES_PER_BUFFERED_GAME = {"blokusduo": 256_000, "tictactoe": 8_000}

# Rough RAM footprint of ONE dense position in flight through the DataLoader:
# the encoded (C, H, W) board planes + the densified full-action-space policy,
# float32. Blokus = 44×14×14×4 + 17,837×4 ≈ 106 KB. Used to size the workers'
# prefetched-batch working set. Unknown games fall back to the largest estimate.
_EST_DENSE_POSITION_BYTES = {"blokusduo": 106_000, "tictactoe": 500}

# Fixed overhead outside the replay buffer: torch runtime + net + optimizer,
# CUDA context, metrics buffers. Generous round number.
_EST_FIXED_OVERHEAD_BYTES = 6 * 1024**3

# Per-DataLoader-worker fixed cost: a forkserver/spawn worker re-imports torch
# and carries its own small heap. Measured ~0.4–0.6 GB in the M1 profile
# (docs/plans/fix-training-oom.md); rounded up. The workers' *variable* cost —
# prefetched dense batches — is modelled separately from config below.
_EST_WORKER_BASE_BYTES = 700 * 1024**2

# Fraction of the buffer each DataLoader worker additionally *copies*. Before the
# memmap-backed dataset (M2) a forkserver worker pickled a full copy of the
# in-RAM buffer, so this was ~1.0 — the OOM: peak ≈ buffer × (1 + workers). The
# memmap dataset shares the buffer through the OS page cache, so workers no
# longer copy it and this is ~0. Kept explicit so the model documents (and the
# guard stays honest about) the exact term the fix closed.
_EST_WORKER_BUFFER_COPY_FRACTION = 0.0

# Fraction of physical/cgroup RAM the estimated peak may claim before we refuse
# to start. Leaves headroom for the OS, transients, and estimate error.
_RAM_BUDGET_FRACTION = 0.8

# cgroup memory-limit files, newest scheme first. A container is frequently
# capped far below the host's physical RAM, and ``psutil.virtual_memory().total``
# reports the *host* — so a config that fits the host but not the container would
# OOM despite a "passing" check. Read the cgroup limit and take the tighter of
# the two. A module-level tuple so tests can point it at a temp file.
_CGROUP_MEMORY_LIMIT_PATHS = (
    Path("/sys/fs/cgroup/memory.max"),  # cgroup v2
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),  # cgroup v1
)


@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time memory usage of the current process.

    All values are in bytes. ``gpu_bytes`` is ``None`` when no GPU is
    available or the backend doesn't expose an allocation counter.
    ``process_peak_rss_bytes`` is the high-water RSS since process start —
    the number that actually gets a process OOM-killed, which a point-in-time
    RSS read can miss entirely.
    """

    process_rss_bytes: int
    process_peak_rss_bytes: int
    gpu_bytes: float | None


def get_memory_snapshot() -> MemorySnapshot:
    """Take a cross-platform snapshot of the current process's memory usage.

    Uses ``psutil`` for process RSS (works identically on macOS, Linux,
    and Windows — always returns bytes) and ``getrusage`` for the peak RSS
    (``ru_maxrss`` is bytes on macOS, kilobytes on Linux — normalised here).

    For GPU memory, checks CUDA first, then MPS (Apple Silicon).
    Returns ``None`` for ``gpu_bytes`` if no GPU is available.
    """
    rss = psutil.Process().memory_info().rss

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    # ``ru_maxrss`` and psutil's RSS come from different subsystems sampled a
    # moment apart, and ``ru_maxrss`` is KB-quantised on Linux — so the recorded
    # peak can read fractionally below the current RSS. Clamp: the peak-so-far
    # is by definition at least the current RSS.
    peak_bytes = max(peak_bytes, rss)

    gpu_mem: float | None = None
    if torch.cuda.is_available():
        gpu_mem = float(torch.cuda.memory_allocated())
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_mem = float(torch.mps.current_allocated_memory())

    return MemorySnapshot(process_rss_bytes=rss, process_peak_rss_bytes=peak_bytes, gpu_bytes=gpu_mem)


def estimate_peak_ram_bytes(config: RunConfig) -> int:
    """Estimate a training run's peak RAM from config alone (no allocation).

    Models the three terms that actually drive the peak — deliberately coarse
    and generous; its job is to catch order-of-magnitude mistakes, not to be an
    allocator:

    - **Resident buffer:** ``(replay_buffer_games + num_eps) × bytes/game``. The
      ``+ num_eps`` covers the fresh-game list still referencing games the deque
      has begun evicting during the save.
    - **DataLoader workers:** ``workers × (base + prefetch_factor × batch_size ×
      dense-position bytes + copy_fraction × buffer)``. The prefetch term is the
      dense batches in flight; ``copy_fraction`` is the per-worker buffer copy
      that OOM-killed the run pre-M2 (now ~0 via the memmap-backed dataset). Zero
      when ``dataloader_workers == 0`` (the in-process path).
    - **Framework:** torch runtime + net + optimizer + CUDA context.

    This is the number :func:`check_ram_budget` compares against the machine, and
    the number ``scripts/benchmarks/memory_probe.py`` prints alongside a measured
    peak so the two can be reconciled before a paid run.
    """
    bytes_per_game = _EST_BYTES_PER_BUFFERED_GAME.get(config.game, max(_EST_BYTES_PER_BUFFERED_GAME.values()))
    buffer_bytes = (config.replay_buffer_games + config.num_eps) * bytes_per_game

    perf = config.net_config.perf
    worker_bytes = 0
    if perf.dataloader_workers > 0:
        dense_position_bytes = _EST_DENSE_POSITION_BYTES.get(config.game, max(_EST_DENSE_POSITION_BYTES.values()))
        prefetch_bytes = perf.prefetch_factor * config.net_config.batch_size * dense_position_bytes
        per_worker = _EST_WORKER_BASE_BYTES + prefetch_bytes + int(_EST_WORKER_BUFFER_COPY_FRACTION * buffer_bytes)
        worker_bytes = perf.dataloader_workers * per_worker

    return buffer_bytes + worker_bytes + _EST_FIXED_OVERHEAD_BYTES


def available_ram_bytes() -> int:
    """RAM the run may actually use — the tighter of physical and the cgroup cap.

    ``psutil.virtual_memory().total`` reports the *host*; a container is often
    capped well below that, so a config that fits the host can still OOM. Take
    the minimum of physical RAM and any cgroup ``memory.max`` limit.
    """
    total = psutil.virtual_memory().total
    cgroup = _cgroup_memory_limit_bytes()
    if cgroup is not None:
        return min(total, cgroup)
    return total


def _cgroup_memory_limit_bytes(paths: tuple[Path, ...] = _CGROUP_MEMORY_LIMIT_PATHS) -> int | None:
    """Read the container's cgroup memory limit in bytes, or None if unset.

    Returns None when no cgroup limit file is present or the limit is unbounded
    (cgroup v2 writes the literal ``max``; v1 writes a huge sentinel, which the
    ``min`` in :func:`available_ram_bytes` clamps to physical RAM anyway).
    """
    for path in paths:
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def check_ram_budget(config: RunConfig) -> None:
    """Refuse a run whose estimated peak RAM exceeds the machine's budget.

    Compares :func:`estimate_peak_ram_bytes` against ``_RAM_BUDGET_FRACTION`` of
    :func:`available_ram_bytes` (physical RAM or the tighter cgroup cap). Always
    logs the numbers so every run records what it expected to use, then aborts
    *before* training — turning a 3 a.m. OOM kill into an instant config error.

    Raises:
        ValueError: When the estimated peak exceeds the budget, naming the knobs
            to lower (``replay_buffer_games``, ``num_eps``, ``dataloader_workers``,
            ``pin_memory``) or to move to a bigger box.
    """
    estimated_bytes = estimate_peak_ram_bytes(config)
    available_bytes = available_ram_bytes()
    physical_bytes = psutil.virtual_memory().total
    budget_bytes = int(available_bytes * _RAM_BUDGET_FRACTION)
    limited_by_cgroup = available_bytes < physical_bytes

    logger.info(
        "RAM budget check: estimated peak ≈{:.1f} GB "
        "(buffer {} games + {} DataLoader workers) vs budget {:.1f} GB "
        "({:.0%} of {:.1f} GB available{})",
        estimated_bytes / 1024**3,
        config.replay_buffer_games + config.num_eps,
        config.net_config.perf.dataloader_workers,
        budget_bytes / 1024**3,
        _RAM_BUDGET_FRACTION,
        available_bytes / 1024**3,
        f", cgroup-capped from {physical_bytes / 1024**3:.1f} GB physical" if limited_by_cgroup else " physical",
    )
    if estimated_bytes > budget_bytes:
        raise ValueError(
            f"Estimated peak RAM ≈{estimated_bytes / 1024**3:.1f} GB exceeds the "
            f"{_RAM_BUDGET_FRACTION:.0%} budget of this machine's "
            f"{available_bytes / 1024**3:.1f} GB available "
            f"({budget_bytes / 1024**3:.1f} GB"
            f"{'; cgroup-capped' if limited_by_cgroup else ''}). Lower replay_buffer_games "
            f"(={config.replay_buffer_games}), num_eps (={config.num_eps}), or "
            f"net_config.perf.dataloader_workers (={config.net_config.perf.dataloader_workers}) / "
            f"pin_memory (={config.net_config.perf.pin_memory}), or run on a bigger box. "
            "See docs/plans/fix-training-oom.md M3 and docs/research/training-memory-model.md."
        )
