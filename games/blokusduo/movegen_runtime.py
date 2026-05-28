"""F2 runtime move generator backed by precomputed lookup tables.

This module is the hot path the F2 optimisation rewrites. Instead of the
existing array-based ``BlokusDuoGame._generate_valid_moves`` (which
loops over every piece × every orientation × every placement point and
runs ``_all_cells_valid`` per candidate), F2 uses two precomputed tables
built in :mod:`games.blokusduo.movegen_tables`:

- ``MoveTables``: every legal ``(piece, orientation, anchor)`` triple
  numbered 0..13,728 with cell lists and adj/attach metadata.
- ``LookupTable``: indexed by ``(anchor, adj_status, piece)``, returns
  the list of move IDs that match those constraints. ``adj_status`` is
  a 6-bit packing of "is each of these 6 neighbour cells forbidden?"
  — see ``docs/research/pentobi/move-generation.md`` §4f.

At runtime the work per anchor is:

1. Compute the 6-bit ``adj_status`` from 6 byte reads.
2. For each piece in inventory, look up a short candidate list.
3. For each candidate, verify its (up to 5) cells aren't forbidden via
   an unrolled OR over byte reads.

How "forbidden" maps from the existing board
--------------------------------------------

``forbidden[cell]`` for colour C is true iff *placing a same-colour piece
cell here would be illegal*. That happens for two reasons:

- The cell is **occupied** by any piece (own or opponent). Captured by
  ``board._piece_placement_board != 0``.
- The cell is **edge-adjacent to a same-colour piece** (Blokus's
  no-edge-touching-own-colour rule). Captured by
  ``board.side_danger_zone(player)``.

The union of these two is F2's ``forbidden`` byte array.

The first-move case
-------------------

Until a player has placed a piece, ``placement_points`` is empty (the
attach-point set requires at least one own piece on the board). The
existing code handles this by falling back to a precomputed
``initial_actions`` list. F2 does the same — for the special case of
no pieces played, defer to the existing fast path. The optimisation
matters for the *thousands* of subsequent moves, not the one-off first
move.

Module-level table caching
--------------------------

Building both tables is ~200 ms total. Doing it per ``valid_move_mask``
call would dominate. Instead a module-level singleton (lazily built on
first access via :func:`get_default_generator`) holds them across
calls. For multi-process self-play, each worker rebuilds at import
time — also ~200 ms once per process, then free for thousands of
subsequent calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from games.blokusduo.movegen_tables import (
    BOARD_SIZE,
    NUM_CELLS,
    LookupTable,
    MoveTables,
    build_lookup_table,
    build_move_tables,
)
from games.blokusduo.pieces import pieces_loader

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from games.blokusduo.board import BlokusDuoBoard
    from games.blokusduo.game import BlokusDuoGame


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class F2MoveGenerator:
    """Stateless front-end to the precomputed move tables.

    The generator owns the static tables but holds no per-board state.
    Each ``valid_move_mask`` call derives its dynamic ``forbidden`` array
    fresh from the board (cheap — single numpy expression).
    """

    def __init__(self, tables: MoveTables, lookup: LookupTable) -> None:
        self._tables = tables
        self._lookup = lookup
        # Hoist commonly-accessed arrays to instance attributes so the
        # hot loop doesn't pay attribute-access cost per iteration.
        self._move_cells = tables.cells
        self._move_action_id = tables.action_id
        self._lookup_begin = lookup.begin
        self._lookup_size = lookup.size
        self._lookup_move_ids = lookup.move_ids
        self._adj_status_cells = lookup.adj_status_cells

    @classmethod
    def from_pieces_json(cls, pieces_path: Path) -> F2MoveGenerator:
        """Build a generator by loading pieces.json and constructing both tables."""
        pm = pieces_loader(pieces_path)
        tables = build_move_tables(pm, BOARD_SIZE)
        lookup = build_lookup_table(tables, BOARD_SIZE)
        return cls(tables, lookup)

    def valid_move_mask(
        self, game: BlokusDuoGame, board: BlokusDuoBoard, player: int,
    ) -> NDArray:
        """Return the legal-actions bool mask for ``(board, player)``.

        Output shape: ``(game.action_codec.action_size,)`` = ``(17837,)``.
        Bit ``i`` is True iff action ``i`` is a legal move. If no legal
        placements exist, the pass action (last index) is set to True —
        matching the existing :meth:`BlokusDuoGame.valid_move_masking`.

        Args:
            game: The :class:`BlokusDuoGame` instance (used for the
                first-move precomputed-action fallback and the action
                codec's pass index).
            board: Current board state.
            player: 1 for White, -1 for Black.
        """
        action_size = game.action_codec.action_size
        out = np.zeros(action_size, dtype=bool)

        # First-move special case — defer to the existing fast path.
        # When a player hasn't placed any pieces, attach_points is empty;
        # the legal moves are determined by Blokus's starting-cell rule
        # which the existing code already encodes in ``_initial_actions``.
        if len(board.remaining_piece_ids(player)) == 21:
            self._fill_first_move_mask(game, board, player, out)
            if not out.any():
                out[game.action_codec.pass_action_index] = True
            return out

        # Hot path: build the forbidden mask, then iterate attach points.
        forbidden = self._build_forbidden_array(board, player)
        remaining = board.remaining_piece_ids(player)

        # Dedup set — a single move can be emitted by multiple attach
        # points (one for each of its footprint cells that's an attach
        # point). Pentobi calls this MoveMarker; we use a plain Python
        # set since action_size is only ~17k.
        seen: set[int] = set()

        for (anchor_row, anchor_col) in board.placement_points(player):
            anchor = anchor_row * BOARD_SIZE + anchor_col
            if forbidden[anchor]:
                # Attach point itself has become forbidden since it was
                # added (a later edge-adjacent piece covered it). Skip.
                continue
            adj_status = self._compute_adj_status(forbidden, anchor)
            self._fill_legal_actions_for_anchor(
                anchor, adj_status, remaining, forbidden, out, seen,
            )

        if not out.any():
            out[game.action_codec.pass_action_index] = True
        return out

    # -- Hot-path helpers ------------------------------------------------------

    def _build_forbidden_array(
        self, board: BlokusDuoBoard, player: int,
    ) -> NDArray:
        """Build the (NUM_CELLS + 1,) uint8 forbidden array.

        ``forbidden[cell]`` is 1 iff a same-colour piece cell cannot
        legally be placed at ``cell``. The last slot
        (``forbidden[NULL_CELL]``) is always 0 so the unrolled per-cell
        check can read padded slots safely.

        forbidden[cell] = (any piece occupies cell) OR (cell is
        edge-adjacent to one of the player's pieces).
        """
        forbidden_2d = (board._piece_placement_board != 0) | board.side_danger_zone(player)
        forbidden = np.zeros(NUM_CELLS + 1, dtype=np.uint8)
        forbidden[:NUM_CELLS] = forbidden_2d.ravel().astype(np.uint8)
        return forbidden

    def _compute_adj_status(self, forbidden: NDArray, anchor: int) -> int:
        """Read 6 forbidden bytes and pack into the 6-bit adj_status."""
        adj_cells = self._adj_status_cells[anchor]
        return (
            int(forbidden[adj_cells[0]])
            | (int(forbidden[adj_cells[1]]) << 1)
            | (int(forbidden[adj_cells[2]]) << 2)
            | (int(forbidden[adj_cells[3]]) << 3)
            | (int(forbidden[adj_cells[4]]) << 4)
            | (int(forbidden[adj_cells[5]]) << 5)
        )

    def _fill_legal_actions_for_anchor(
        self,
        anchor: int,
        adj_status: int,
        remaining_pieces: frozenset[int],
        forbidden: NDArray,
        out: NDArray,
        seen: set[int],
    ) -> None:
        """For one anchor + adj_status, iterate each remaining piece's
        candidate list and mark legal moves in ``out``. Uses ``seen``
        as a dedup set across anchor calls."""
        for piece_id in remaining_pieces:
            piece_idx = piece_id - 1  # piece IDs are 1..21
            begin = int(self._lookup_begin[anchor, adj_status, piece_idx])
            size = int(self._lookup_size[anchor, adj_status, piece_idx])
            if size == 0:
                continue
            candidates = self._lookup_move_ids[begin : begin + size]
            for move_id_np in candidates:
                move_id = int(move_id_np)
                if move_id in seen:
                    continue
                cells = self._move_cells[move_id]
                if (
                    forbidden[cells[0]]
                    | forbidden[cells[1]]
                    | forbidden[cells[2]]
                    | forbidden[cells[3]]
                    | forbidden[cells[4]]
                ):
                    continue
                seen.add(move_id)
                out[self._move_action_id[move_id]] = True

    def _fill_first_move_mask(
        self, game: BlokusDuoGame, board: BlokusDuoBoard, player: int,
        out: NDArray,
    ) -> None:
        """Populate ``out`` with the legal first moves for a player.

        Delegates to the existing ``_generate_valid_moves`` for the
        first-move case; F2's hot path doesn't need to be optimised for
        the one-off opening move per colour per game.
        """
        for action in game._generate_valid_moves(board, player):  # noqa: SLF001
            action_id = game.action_codec.encode(action)
            out[action_id] = True


# ---------------------------------------------------------------------------
# Module-level singleton — build tables once per process
# ---------------------------------------------------------------------------

_default_generator: F2MoveGenerator | None = None
_DEFAULT_PIECES_PATH = Path(__file__).resolve().parent / "pieces.json"


def get_default_generator() -> F2MoveGenerator:
    """Return a process-singleton :class:`F2MoveGenerator`.

    Lazily builds the tables on first call (~200 ms) and caches the
    result for the lifetime of the process. Thread-safe for the typical
    "construct-once-then-read" usage pattern; not safe if multiple
    threads race the first call (negligible risk in practice).
    """
    global _default_generator
    if _default_generator is None:
        _default_generator = F2MoveGenerator.from_pieces_json(_DEFAULT_PIECES_PATH)
    return _default_generator
