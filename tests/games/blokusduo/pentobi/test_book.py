"""Book engagement is verified, not assumed (fair-pentobi-benchmark F1).

Omitting ``--nobook`` only *requests* the opening book. Pentobi looks for its
``.blksgf`` files beside the binary and plays book-free without them, which is how the
project's whole measurement history came to face a book-free engine while the setting
reported ``use_book 1``. Recording ``book: true`` on the strength of a missing flag
would repeat that with a payload that now looks authoritative.

The classification is tested here directly; driving a real engine needs the binary,
which only exists on the box, so :func:`probe_book` must degrade to "unknown" rather
than fail anywhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphablokus.games.blokusduo.pentobi.book import classify_probe, probe_book

if TYPE_CHECKING:
    import pytest


def test_a_book_hit_has_no_search_tree() -> None:
    """The measured signature: instant return, empty ``move_values`` (0.47 s at L9)."""
    probe = classify_probe(seconds=0.47, move="f8,d9,e9,f9,e10", root_children=0)
    assert probe.engaged is True
    assert "book hit" in probe.detail


def test_a_searched_move_reports_root_children() -> None:
    """The measured contrast: ~26 s at L9 with a populated ``move_values``."""
    probe = classify_probe(seconds=25.82, move="f8,e9,f9,g9,e10", root_children=137)
    assert probe.engaged is False
    assert "no book" in probe.detail


def test_emptiness_decides_regardless_of_wall_clock() -> None:
    """Timing is corroboration only: a low level searches fast enough to look instant.

    Level 1 searches 3 simulations, so a sub-second answer says nothing about the book;
    only the absence of a search tree does.
    """
    assert classify_probe(seconds=0.01, move="e10", root_children=12).engaged is False
    assert classify_probe(seconds=30.0, move="e10", root_children=0).engaged is True


def test_probe_degrades_to_unknown_without_a_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """No engine to probe with is "could not determine", never "fine"."""
    monkeypatch.setenv("PENTOBI_GTP_PATH", "/nonexistent/pentobi-gtp")
    probe = probe_book(9)
    assert probe.engaged is None
    assert "unavailable" in probe.detail
    assert probe.as_dict()["engaged"] is None


def test_the_benchmark_refuses_to_record_an_unloaded_book() -> None:
    """A book-on run that demonstrably had no book must abort, not record ``book: true``."""
    from scripts.pentobi_benchmark import book_probe_conflict

    message = book_probe_conflict(classify_probe(seconds=25.8, move="f8,e9", root_children=137))
    assert message is not None
    assert ".blksgf" in message  # tells the operator how to fix it


def test_an_engaged_or_unverifiable_book_does_not_abort() -> None:
    """Only a *proven* absence is fatal; an unrunnable probe is recorded as unverified."""
    from alphablokus.games.blokusduo.pentobi.book import BookProbe
    from scripts.pentobi_benchmark import book_probe_conflict

    assert book_probe_conflict(classify_probe(seconds=0.5, move="f8", root_children=0)) is None
    unknown = BookProbe(engaged=None, seconds=0.0, move="", root_children=0, detail="probe unavailable: no binary")
    assert book_probe_conflict(unknown) is None
