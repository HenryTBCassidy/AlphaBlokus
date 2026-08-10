"""Is Pentobi's opening book actually loaded? (fair-pentobi-benchmark F1)

Asking for the book is not the same as getting it. ``pentobi_gtp/Main.cpp`` sets
``books_dir`` to the directory holding the binary, so a build tree without
``.blksgf`` files runs book-free while ``param_base`` still reports ``use_book 1``
— which is why the project's entire measurement history faced a book-free engine
without anyone choosing that. Dropping ``--nobook`` therefore *requests* the book;
it does not establish that one loaded. Recording ``book: true`` on that basis would
repeat the original defect with a payload that now looks authoritative.

**The observable.** A book hit returns a move with no search tree behind it, so
``move_values`` comes back empty; a searched move always has root children (and at
level 9 takes ~26 s against the book's ~0.5 s). Emptiness is the discriminator —
the timing is only corroborating evidence, because a low level searches fast enough
that its wall-clock says nothing.

The probe needs the real binary, which exists only on the box, so it degrades to
``engaged=None`` ("could not determine") whenever the engine cannot be started or
driven. Callers must treat ``None`` as "unverified", never as "fine".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from alphablokus.games.blokusduo.pentobi.gtp import GtpError, PentobiGtp

# GTP colour token for the first mover (our White=+1 ↔ 'b', pinned in H2).
_FIRST_MOVER = "b"


@dataclass(frozen=True)
class BookProbe:
    """What one book-engagement probe observed.

    Attributes:
        engaged: True if the engine answered from its book, False if it searched,
            ``None`` if the probe could not run at all (no binary, engine error).
        seconds: Wall-clock the opening move took.
        move: The move the engine returned, for the record.
        root_children: ``move_values`` entry count — 0 means no search tree.
        detail: Human-readable summary, suitable for a log line or a payload.
    """

    engaged: bool | None
    seconds: float
    move: str
    root_children: int
    detail: str

    def as_dict(self) -> dict[str, object]:
        """The probe as a JSON-serialisable record for a result payload."""
        return {
            "engaged": self.engaged,
            "seconds": round(self.seconds, 3),
            "move": self.move,
            "root_children": self.root_children,
            "detail": self.detail,
        }


def classify_probe(seconds: float, move: str, root_children: int) -> BookProbe:
    """Decide book vs search from one opening move's observables.

    Split out from :func:`probe_book` so the decision is testable without the
    binary: only the driving needs a live engine.
    """
    engaged = root_children == 0
    detail = (
        f"opening move {move!r} returned in {seconds:.2f}s with no search tree — book hit"
        if engaged
        else f"opening move {move!r} returned in {seconds:.2f}s with {root_children} root children — searched, no book"
    )
    return BookProbe(engaged=engaged, seconds=seconds, move=move, root_children=root_children, detail=detail)


def probe_book(level: int, *, binary: str | None = None, threads: int = 1) -> BookProbe:
    """Start a book-enabled engine and see whether its first move came from the book.

    Args:
        level: Level to probe at — the same one the benchmark will use, since the
            book is consulted per position and the timing evidence is level-dependent.
        binary: Explicit ``pentobi-gtp`` path (defaults to the usual resolution).
        threads: Engine thread count, mirroring the benchmark's.

    Returns:
        The probe result. ``engaged=None`` means the probe could not run, which is
        the expected outcome anywhere the binary is absent (a Mac, CI).
    """
    try:
        engine = PentobiGtp(level, binary=binary, threads=threads, seed=1, nobook=False)
    except (FileNotFoundError, GtpError, OSError) as exc:
        logger.warning("Could not start a Pentobi engine to verify the opening book: {}", exc)
        return BookProbe(engaged=None, seconds=0.0, move="", root_children=0, detail=f"probe unavailable: {exc}")
    try:
        start = time.perf_counter()
        move = engine.genmove(_FIRST_MOVER)
        seconds = time.perf_counter() - start
        values = engine.move_values()
    except GtpError as exc:
        logger.warning("Pentobi book probe failed mid-conversation: {}", exc)
        return BookProbe(engaged=None, seconds=0.0, move="", root_children=0, detail=f"probe failed: {exc}")
    finally:
        engine.close()
    probe = classify_probe(seconds, move, len(values))
    logger.info("Pentobi book probe at level {}: {}", level, probe.detail)
    return probe
