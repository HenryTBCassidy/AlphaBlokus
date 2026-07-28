"""Parser for Pentobi's GTP ``move_values`` response — the v2 corpus's soft target (V1).

``pentobi_gtp/GtpEngine.cpp::cmd_move_values`` dumps the **root children of the last
search**, strongest-first, one per line::

    846665 846668.0 0.747 [F]f8,f9,g9,e10,f10
    170044 170047.0 0.687 [F]f8,e9,f9,g9,e10
     ...
    <visits> <value_count> <value> [PIECE]<cells>

Pentobi builds this tree on every search anyway, so harvesting it is free: the corpus
drive pattern is ``reg_genmove <c>`` → ``move_values`` → ``play <c> <our choice>``, which
labels a position with the expert's *whole* preference distribution and then lets us play
something else.

Parsing rules, every one of them measured against real L9 output
(``local/probes/mv_deep.txt``, checked into ``tests/.../data/move_values_l9.txt``):

- Three numeric fields then the move; the move is **everything after the first ``]``**.
  The ``[F]`` / ``[T5]`` / ``[1]`` piece-name prefix is Pentobi's own annotation and
  ``PentobiMoveTranslator.cells_to_action`` will not parse it. A line with no ``[`` is
  still accepted (the whole remainder is the move).
- **Values may be negative.** They are win-probability-like for the side to move but not
  bounded to [0, 1] — deep in a game they go below zero.
- Values are only meaningful for **visited** children: an unvisited child reports a tiny
  ``value_count`` (~3.0, the prior's weight) and the prior's value, so ``value_count`` is
  kept as the visited/unvisited discriminator rather than discarded.
- **An empty response is legal** and yields an empty :class:`MoveValues`: a forced pass,
  a terminal position, or (if the opening book were ever enabled) a book hit returns a
  move with no search tree behind it. Callers handle emptiness; the parser never raises
  on it.

A position's backed-up value is :attr:`MoveValues.search_value` — the **top child's**
value. GTP ``get_value`` is useless for this: Pentobi never updates the root node's own
value, so it returns a constant 0.
"""

from __future__ import annotations

from dataclasses import dataclass


class MoveValuesParseError(ValueError):
    """Raised when a ``move_values`` line does not have the measured 4-field shape."""


@dataclass(frozen=True)
class MoveValueEntry:
    """One root child of a Pentobi search: how hard it was searched and how it scored."""

    visits: int
    value_count: float  # ~3.0 for an unvisited child (prior weight only) — see is_visited
    value: float  # side-to-move perspective; NOT bounded to [0, 1]
    cells: str  # piece-name prefix already stripped, e.g. "f8,f9,g9,e10,f10"

    @property
    def is_visited(self) -> bool:
        """Whether the search actually visited this child (else ``value`` is the prior's)."""
        return self.visits > 0


@dataclass(frozen=True)
class MoveValues:
    """A parsed ``move_values`` response: the expert's preferences at one position."""

    entries: tuple[MoveValueEntry, ...]  # visit-descending, as the engine emits them
    total_visits: int  # sum over entries — the denominator for visit shares

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        """False for an empty response (no search tree) — the caller's emptiness check."""
        return bool(self.entries)

    @property
    def top(self) -> MoveValueEntry | None:
        """The strongest-first entry the engine emitted, or ``None`` if empty."""
        return self.entries[0] if self.entries else None

    @property
    def search_value(self) -> float | None:
        """The position's backed-up value for the side to move = the top child's value.

        ``None`` when the response is empty. Never read GTP ``get_value`` instead: it is
        a constant 0 because Pentobi does not update the root node's own value.
        """
        top = self.top
        return top.value if top is not None else None

    @property
    def visit_shares(self) -> tuple[float, ...]:
        """Each entry's share of :attr:`total_visits` (all zeros if nothing was visited)."""
        if self.total_visits <= 0:
            return tuple(0.0 for _ in self.entries)
        return tuple(entry.visits / self.total_visits for entry in self.entries)


def strip_piece_prefix(move: str) -> str:
    """Drop Pentobi's ``[PIECE]`` annotation from a move string.

    ``"[T5]f9,d10,e10,f10,f11"`` → ``"f9,d10,e10,f10,f11"``. A move with no annotation is
    returned unchanged, so this is safe to apply to any Pentobi move string.
    """
    _, sep, rest = move.partition("]")
    return (rest if sep else move).strip()


def parse_move_values(response: str) -> MoveValues:
    """Parse a ``move_values`` GTP response payload into :class:`MoveValues`.

    Args:
        response: The response body as :meth:`PentobiGtp.send` returns it — the ``=``
            status marker already stripped. An empty (or whitespace-only) body is legal
            and yields an empty result.

    Raises:
        MoveValuesParseError: If a non-blank line is not ``<visits> <value_count>
            <value> <move>`` — a shape change in the engine we must not silently ignore.
    """
    entries: list[MoveValueEntry] = []
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=3)
        if len(fields) != 4:
            raise MoveValuesParseError(f"expected '<visits> <value_count> <value> <move>', got {raw_line!r}")
        visits_text, value_count_text, value_text, move = fields
        try:
            entry = MoveValueEntry(
                visits=int(visits_text),
                value_count=float(value_count_text),
                value=float(value_text),
                cells=strip_piece_prefix(move),
            )
        except ValueError as exc:
            raise MoveValuesParseError(f"unparseable move_values line {raw_line!r}: {exc}") from exc
        entries.append(entry)
    return MoveValues(entries=tuple(entries), total_visits=sum(entry.visits for entry in entries))
