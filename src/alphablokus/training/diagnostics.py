"""Process-level resource diagnostics and pre-flight guards.

Two halves of the same OOM story (``docs/plans/oom-hardening.md`` O8):
:func:`get_memory_snapshot` makes memory visible *during* a run (RSS + peak
RSS at phase transitions), and :func:`check_ram_budget` refuses configs whose
estimated peak cannot fit the machine *before* a run starts — turning a 3 a.m.
OOM kill into an instant config error.
"""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
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

# Fixed overhead outside the replay buffer: torch runtime + net + optimizer,
# CUDA context, worker processes, metrics buffers. Generous round number.
_EST_FIXED_OVERHEAD_BYTES = 6 * 1024**3

# Fraction of physical RAM the estimated peak may claim before we refuse to
# start. Leaves headroom for the OS, transients, and estimate error.
_RAM_BUDGET_FRACTION = 0.8


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

    gpu_mem: float | None = None
    if torch.cuda.is_available():
        gpu_mem = float(torch.cuda.memory_allocated())
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_mem = float(torch.mps.current_allocated_memory())

    return MemorySnapshot(process_rss_bytes=rss, process_peak_rss_bytes=peak_bytes, gpu_bytes=gpu_mem)


def check_ram_budget(config: RunConfig) -> None:
    """Refuse a run whose estimated peak RAM exceeds the machine's budget.

    Estimates the replay buffer's steady-state footprint from config (buffer
    capacity plus one generation of fresh games, at a per-game byte estimate)
    plus a fixed process-overhead allowance, and compares it against
    ``_RAM_BUDGET_FRACTION`` of physical RAM. Always logs the numbers so every
    run records what it expected to use.

    The estimate is deliberately coarse and generous — its job is to catch
    configs that are off by an order of magnitude (the ones that OOM-kill the
    box hours in), not to be an allocator.

    Raises:
        ValueError: When the estimated peak exceeds the budget. Lower
            ``replay_buffer_games`` / ``num_eps`` (or use a bigger box).
    """
    bytes_per_game = _EST_BYTES_PER_BUFFERED_GAME.get(config.game, max(_EST_BYTES_PER_BUFFERED_GAME.values()))
    # Buffer capacity + one generation of fresh games: fresh games stream into
    # the buffer, but during save the fresh list still references games the
    # deque may have evicted, so budget for both.
    buffered_games = config.replay_buffer_games + config.num_eps
    estimated_bytes = buffered_games * bytes_per_game + _EST_FIXED_OVERHEAD_BYTES
    total_bytes = psutil.virtual_memory().total
    budget_bytes = int(total_bytes * _RAM_BUDGET_FRACTION)

    logger.info(
        "RAM budget check: estimated peak ≈{:.1f} GB ({} buffered games ≈{:.1f} GB + ≈{:.1f} GB overhead) "
        "vs budget {:.1f} GB ({:.0%} of {:.1f} GB physical)",
        estimated_bytes / 1024**3,
        buffered_games,
        buffered_games * bytes_per_game / 1024**3,
        _EST_FIXED_OVERHEAD_BYTES / 1024**3,
        budget_bytes / 1024**3,
        _RAM_BUDGET_FRACTION,
        total_bytes / 1024**3,
    )
    if estimated_bytes > budget_bytes:
        raise ValueError(
            f"Estimated peak RAM ≈{estimated_bytes / 1024**3:.1f} GB exceeds the "
            f"{_RAM_BUDGET_FRACTION:.0%} budget of this machine's {total_bytes / 1024**3:.1f} GB "
            f"({budget_bytes / 1024**3:.1f} GB). Lower replay_buffer_games "
            f"(={config.replay_buffer_games}) and/or num_eps (={config.num_eps}), or run on a "
            "bigger box. See docs/plans/oom-hardening.md O8."
        )
