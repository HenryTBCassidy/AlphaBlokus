"""Tests for the memory guardrails (``training/diagnostics.py``).

Covers the O8 pre-flight budget check and its M3 extension
(docs/plans/fix-training-oom.md): the DataLoader-worker term and the cgroup
memory-limit read.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from alphablokus.config import TrainingPerfConfig
from alphablokus.training import diagnostics
from alphablokus.training.diagnostics import (
    _cgroup_memory_limit_bytes,
    available_ram_bytes,
    check_ram_budget,
    estimate_peak_ram_bytes,
    get_memory_snapshot,
)

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.config import RunConfig


def _blokus(config: RunConfig, *, replay_buffer_games: int, num_eps: int, workers: int) -> RunConfig:
    """A blokusduo config with the given buffer size and DataLoader worker count."""
    return replace(
        config,
        game="blokusduo",
        replay_buffer_games=replay_buffer_games,
        num_eps=num_eps,
        net_config=replace(config.net_config, perf=TrainingPerfConfig(dataloader_workers=workers)),
    )


def test_memory_snapshot_reports_peak_rss() -> None:
    """Peak RSS is populated and can't be below the current RSS."""
    snapshot = get_memory_snapshot()
    assert snapshot.process_rss_bytes > 0
    assert snapshot.process_peak_rss_bytes >= snapshot.process_rss_bytes


def test_ram_budget_accepts_small_config(test_config: RunConfig) -> None:
    """A tiny test config is nowhere near the budget."""
    check_ram_budget(test_config)  # must not raise


def test_ram_budget_refuses_oversized_config(test_config: RunConfig) -> None:
    """A buffer that can't possibly fit is refused at startup, not at 3 a.m."""
    oversized = replace(test_config, game="blokusduo", replay_buffer_games=10**9, num_eps=10**6)
    with pytest.raises(ValueError, match="peak RAM"):
        check_ram_budget(oversized)


def test_ram_budget_unknown_game_uses_largest_estimate(test_config: RunConfig) -> None:
    """Unknown games fall back to the most conservative per-game estimate."""
    unknown = replace(test_config, game="somefuturegame", replay_buffer_games=10**9, num_eps=10**6)
    with pytest.raises(ValueError, match="peak RAM"):
        check_ram_budget(unknown)


# ── M3: worker term + cgroup limit (docs/plans/fix-training-oom.md) ──


def test_dataloader_workers_raise_the_estimate(test_config: RunConfig) -> None:
    """More DataLoader workers must raise the estimated peak (the term M1 missed)."""
    base = _blokus(test_config, replay_buffer_games=5_000, num_eps=1_000, workers=0)
    with_workers = _blokus(test_config, replay_buffer_games=5_000, num_eps=1_000, workers=8)
    assert estimate_peak_ram_bytes(with_workers) > estimate_peak_ram_bytes(base)


def test_ram_budget_fires_for_production_buffer_with_workers(
    test_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 60k-buffer + 8-worker config is refused on a modest box (the OOM config)."""
    oom_config = _blokus(test_config, replay_buffer_games=60_000, num_eps=10_000, workers=8)
    monkeypatch.setattr(diagnostics, "available_ram_bytes", lambda: 16 * 1024**3)
    with pytest.raises(ValueError, match="dataloader_workers"):
        check_ram_budget(oom_config)


def test_ram_budget_passes_for_safe_config_on_ample_box(
    test_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A modest buffer with a few workers fits a large box — must not raise."""
    safe_config = _blokus(test_config, replay_buffer_games=5_000, num_eps=1_000, workers=4)
    monkeypatch.setattr(diagnostics, "available_ram_bytes", lambda: 128 * 1024**3)
    check_ram_budget(safe_config)  # must not raise


def test_available_ram_uses_tighter_cgroup_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cgroup limit below physical RAM is what the budget is measured against."""
    huge_physical = 512 * 1024**3
    monkeypatch.setattr(diagnostics.psutil, "virtual_memory", lambda: type("M", (), {"total": huge_physical})())
    monkeypatch.setattr(diagnostics, "_cgroup_memory_limit_bytes", lambda: 8 * 1024**3)
    assert available_ram_bytes() == 8 * 1024**3


def test_cgroup_limit_reads_a_numeric_file(tmp_path: Path) -> None:
    """A cgroup memory.max holding a byte count is read back exactly."""
    limit_file = tmp_path / "memory.max"
    limit_file.write_text("17179869184\n")  # 16 GiB
    assert _cgroup_memory_limit_bytes((limit_file,)) == 16 * 1024**3


def test_cgroup_limit_treats_max_as_unbounded(tmp_path: Path) -> None:
    """cgroup v2's literal ``max`` means no limit (falls back to physical RAM)."""
    limit_file = tmp_path / "memory.max"
    limit_file.write_text("max\n")
    assert _cgroup_memory_limit_bytes((limit_file,)) is None


def test_cgroup_limit_absent_returns_none(tmp_path: Path) -> None:
    """No cgroup file present (e.g. bare macOS) → None, physical RAM is used."""
    assert _cgroup_memory_limit_bytes((tmp_path / "nope.max",)) is None
