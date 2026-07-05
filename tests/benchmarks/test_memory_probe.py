"""Tests for the training memory probe (``scripts/benchmarks/memory_probe.py``).

The probe measures *physical* memory rather than summed process-tree RSS, which
over-counts the pages the DataLoader workers share through the memmapped buffer
(docs/plans/fix-training-oom.md). These tests pin the source-selection ladder
(cgroup → PSS → RSS), the cgroup reader, and that the resolved physical figure
is the one the verdict acts on. The DataLoader end-to-end run is left to the
manual box/pod invocation — CI must not depend on building a full buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.benchmarks.memory_probe import (
    MemorySource,
    PeakPhysicalMemory,
    _read_cgroup_current,
    _read_tree_rss,
    _verdict,
    resolve_physical_source,
    select_memory_source,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _NoPssProcess:
    """A macOS-like process stand-in: RSS available, PSS not.

    ``memory_full_info().pss`` is what ``_read_tree_pss`` probes for; macOS
    exposes no ``pss``, so raising ``AttributeError`` there drives the RSS
    fallback branch deterministically on any OS.
    """

    def __init__(self, rss_bytes: int) -> None:
        self._rss_bytes = rss_bytes

    def memory_full_info(self) -> object:
        raise AttributeError("no pss on this platform")

    def memory_info(self) -> object:
        return type("_MInfo", (), {"rss": self._rss_bytes})()

    def children(self, recursive: bool = True) -> list[object]:
        return []


# ── source selection ladder: cgroup > PSS > RSS ──


def test_select_source_prefers_cgroup() -> None:
    """cgroup accounting wins whenever it is available, even if PSS is too."""
    assert select_memory_source(has_cgroup=True, has_pss=True) is MemorySource.CGROUP


def test_select_source_falls_back_to_pss() -> None:
    """Without cgroup, summed PSS is chosen over summed RSS."""
    assert select_memory_source(has_cgroup=False, has_pss=True) is MemorySource.PSS


def test_select_source_last_resort_rss() -> None:
    """With neither cgroup nor PSS (bare macOS), summed RSS is the last resort."""
    assert select_memory_source(has_cgroup=False, has_pss=False) is MemorySource.RSS


# ── cgroup current-usage reader ──


def test_cgroup_current_reads_a_numeric_file(tmp_path: Path) -> None:
    """A cgroup ``memory.current`` holding a byte count is read back exactly."""
    current = tmp_path / "memory.current"
    current.write_text("17179869184\n")  # 16 GiB
    assert _read_cgroup_current((current,)) == 16 * 1024**3


def test_cgroup_current_absent_returns_none(tmp_path: Path) -> None:
    """No cgroup file present (e.g. bare macOS) → None, so PSS/RSS is used."""
    assert _read_cgroup_current((tmp_path / "nope.current",)) is None


# ── resolve_physical_source: wires the reader to the chosen source ──


def test_resolve_uses_cgroup_current_when_present(tmp_path: Path) -> None:
    """A readable cgroup file selects CGROUP and its reader returns that usage."""
    current = tmp_path / "memory.current"
    current.write_text("456\n")
    source, read_bytes = resolve_physical_source(cgroup_paths=(current,))
    assert source is MemorySource.CGROUP
    assert read_bytes() == 456


def test_resolve_falls_back_to_rss_and_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """No cgroup + no PSS → summed RSS, with a warning that it over-counts sharing."""
    source, read_bytes = resolve_physical_source(
        cgroup_paths=(tmp_path / "absent.current",),
        process=_NoPssProcess(rss_bytes=123),  # type: ignore[arg-type]
    )
    assert source is MemorySource.RSS
    assert read_bytes() == 123
    warning = capsys.readouterr().out
    assert "OVER-COUNTS" in warning


def test_read_tree_rss_matches_stub() -> None:
    """The RSS reader sums the (childless) stub process's RSS."""
    assert _read_tree_rss(_NoPssProcess(rss_bytes=999)) == 999  # type: ignore[arg-type]


# ── the resolved physical figure is what the verdict acts on ──


def test_peak_physical_uses_injected_source() -> None:
    """The sampler reports the injected source's reading, not summed RSS."""
    with PeakPhysicalMemory(interval_s=0.001, source=(MemorySource.CGROUP, lambda: 789)) as peak:
        pass
    assert peak.source is MemorySource.CGROUP
    assert peak.peak_bytes == 789


def test_verdict_uses_the_physical_peak() -> None:
    """A physical peak under available RAM FITS; one over it is OVER."""
    with PeakPhysicalMemory(interval_s=0.001, source=(MemorySource.CGROUP, lambda: 10 * 1024**3)) as peak:
        pass
    assert _verdict(peak.peak_bytes, available_bytes=31 * 1024**3) == "FITS"
    assert _verdict(peak.peak_bytes, available_bytes=8 * 1024**3) == "OVER"
