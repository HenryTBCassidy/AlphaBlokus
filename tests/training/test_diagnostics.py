"""Tests for the O8 memory guardrails (``training/diagnostics.py``)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from alphablokus.training.diagnostics import check_ram_budget, get_memory_snapshot

if TYPE_CHECKING:
    from alphablokus.config import RunConfig


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
