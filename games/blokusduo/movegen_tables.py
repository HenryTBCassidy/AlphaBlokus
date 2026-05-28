"""Precomputed move tables for the F2 move-gen rewrite.

This module enumerates every legal placement on a 14×14 Blokus Duo board
and assigns each one a unique move ID 0..13,728. The static tables
produced here are the foundation the (post-F2) runtime move generator
will use.

See ``docs/plans/move-gen-optimisation.md`` §P4 for the rationale and
``docs/research/pentobi/move-generation.md`` for the original Pentobi
design this implementation is based on.

What gets enumerated
--------------------

A "placement" is a triple ``(piece, orientation, anchor_cell)``:

- *piece*: one of the 21 Blokus pieces (id 1..21).
- *orientation*: one of the piece's unique basis orientations (1..8 per
  piece; 91 distinct piece-orientations across all 21 pieces).
- *anchor_cell*: the top-left corner of the piece's bounding box when
  placed on the board, expressed as a cell index 0..195
  (``cell_index = row * 14 + col`` with top-left origin).

Many ``(piece, orientation, anchor)`` triples in the 17,836-slot action
space would extend off the board — e.g. a piece with a 5-row footprint
anchored at row 11 would have cells on rows 11..15, but rows 14..15
don't exist. Those triples are dropped. The remaining count is **13,729
for Blokus Duo**, verified by ``scripts/count_onboard_placements.py``
against Pentobi's hard-coded constant.

The bidirectional ``action_id ↔ move_id`` mapping
-------------------------------------------------

The rest of the codebase (MCTS, training, replay, etc.) thinks in
**action IDs** — a number 0..17,836 from the action codec's
14×14×91+1 cartesian product. F2 internals think in **move IDs** —
a number 0..13,728 indexing the dense list of legal placements.

Both are needed:

- After move generation, F2 produces move IDs but the rest of the
  system wants a 17,837-length boolean mask of legal actions.
  ``move_to_action_id`` does that conversion.
- When the rest of the system applies an action (via
  ``Coach.execute_episode`` → ``Board.with_piece(action)``), F2's
  incremental state update needs to know which move ID that action
  corresponds to. ``action_to_move_id[action_id]`` does that (returning
  ``-1`` for the ~4,107 actions that are not legal placements).

Cell index convention
---------------------

We use **row-major, top-left origin**: ``cell_index = row * 14 + col``.
This matches numpy's natural indexing of a (14, 14) array and is the
convention F2's runtime forbidden/attach arrays use.

The :class:`ActionCodec` uses (x, y) coordinates with bottom-left
origin. Conversion happens at the boundary inside
:func:`build_move_tables`; downstream code never sees board coordinates.

``NULL_CELL`` sentinel
----------------------

A ``MoveInfo``-style array stores up to 5 cells per piece, padded with
:data:`NULL_CELL` for shorter pieces. ``NULL_CELL = 196`` = "one past
the last valid cell index", which is the natural "out of band" value
and matches Pentobi's ``Point::null()`` convention.

For the runtime check ``forbidden[cell]`` to be safe when ``cell ==
NULL_CELL``, the ``forbidden`` array must be allocated with shape
``(num_cells + 1,)`` and the ``NULL_CELL`` slot must always be 0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from games.blokusduo.board import ActionCodec, CoordinateIndexDecoder
from games.blokusduo.pieces import pieces_loader

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from games.blokusduo.pieces import PieceManager


# Board geometry constants. Hardcoded for Blokus Duo; if we ever support
# other variants this becomes parametric.
BOARD_SIZE = 14
NUM_CELLS = BOARD_SIZE * BOARD_SIZE  # 196

#: Sentinel value for "no cell" in fixed-size cell arrays. Indexing
#: ``forbidden[NULL_CELL]`` must yield 0 — see module docstring.
NULL_CELL: int = NUM_CELLS  # 196

#: Maximum number of cells any Blokus piece occupies (the largest pieces
#: are pentominoes with 5 cells).
MAX_CELLS_PER_PIECE: int = 5

#: Maximum number of cells edge-adjacent to a piece. A 5-cell straight
#: line has up to 12 edge-adjacent cells (2 along the spine + 5+5 along
#: the long sides), and more L-shapes can have up to 14. We use 16 for
#: comfortable headroom and to match Pentobi's MAX_ADJ_ATTACH for Duo.
MAX_ADJ_CELLS: int = 16

#: Maximum number of cells diagonally-adjacent to a piece. Same bound
#: as MAX_ADJ_CELLS for the same reason.
MAX_ATTACH_CELLS: int = 16

#: Number of bits in the ``adj_status`` value — 6 cells around each
#: anchor (4 edge-neighbours + 2 diagonals). See
#: ``docs/research/pentobi/move-generation.md`` §4f for the exact 6
#: cells and their bit ordering.
ADJ_STATUS_BITS: int = 6
ADJ_STATUS_COUNT: int = 1 << ADJ_STATUS_BITS  # 64

#: The 6 adj_status cells expressed as ``(row_offset, col_offset)`` from
#: the anchor, in bit order. Matches Pentobi's
#: ``init_adj_status_points`` algorithm: first the 4 edge-neighbours
#: (N, W, E, S) in ``get_adj_coord`` order, then 2 of the 4 diagonals
#: (NW, SE) in ``get_diag_coord`` order.
ADJ_STATUS_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),   # bit 0: N
    (0, -1),   # bit 1: W
    (0, 1),    # bit 2: E
    (1, 0),    # bit 3: S
    (-1, -1),  # bit 4: NW
    (1, 1),    # bit 5: SE
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveTables:
    """Static geometry tables for all 13,729 legal Blokus Duo placements.

    Every array is indexed by ``move_id`` ∈ ``[0, num_moves)`` unless
    otherwise stated. Arrays are read-only after construction.
    """

    #: Total count of legal placements. Should equal 13,729 for Duo —
    #: verified at construction time against
    #: ``scripts/count_onboard_placements.py``'s independent count.
    num_moves: int

    #: ``piece[move_id]`` = piece id (1..21).
    piece: NDArray  # uint8[num_moves]

    #: ``anchor[move_id]`` = anchor cell index (0..195).
    anchor: NDArray  # uint8[num_moves]

    #: ``action_id[move_id]`` = the corresponding integer in the
    #: 17,837-length action space the rest of the codebase uses.
    action_id: NDArray  # uint32[num_moves]

    #: ``cells[move_id, :]`` = the piece's cell indices, padded with
    #: NULL_CELL. ``n_cells[move_id]`` says how many are real.
    cells: NDArray  # uint8[num_moves, MAX_CELLS_PER_PIECE]
    n_cells: NDArray  # uint8[num_moves]

    #: ``adj_cells[move_id, :]`` = cells edge-adjacent to the piece's
    #: footprint (would become forbidden for the playing colour when
    #: this move is played). NULL_CELL-padded. ``n_adj`` says how many.
    adj_cells: NDArray  # uint8[num_moves, MAX_ADJ_CELLS]
    n_adj: NDArray  # uint8[num_moves]

    #: ``attach_cells[move_id, :]`` = cells diagonally-adjacent to the
    #: piece's footprint (become new attach points for the playing
    #: colour). NULL_CELL-padded. ``n_attach`` says how many.
    attach_cells: NDArray  # uint8[num_moves, MAX_ATTACH_CELLS]
    n_attach: NDArray  # uint8[num_moves]

    #: ``action_to_move_id[action_id]`` = the move_id corresponding to
    #: that action, or ``-1`` if the action is not a legal placement
    #: (off-board or the pass action). Used for incremental state
    #: updates when the rest of the codebase plays an action.
    action_to_move_id: NDArray  # int32[action_size]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_move_tables(
    piece_manager: PieceManager,
    board_size: int = BOARD_SIZE,
    *,
    verbose: bool = False,
) -> MoveTables:
    """Enumerate every legal ``(piece, orientation, anchor)`` triple,
    drop off-board placements, and return the static tables.

    Args:
        piece_manager: The piece definitions to enumerate over.
        board_size: 14 for Blokus Duo. Parameterised for future variants.
        verbose: If True, print per-piece counts and timing as we go.

    Returns:
        :class:`MoveTables` with all arrays populated. ``num_moves`` is
        guaranteed to equal Pentobi's hard-coded 13,729 for the Duo
        variant — an assertion fires otherwise.
    """
    t0 = time.perf_counter()

    action_codec = ActionCodec(board_size=board_size, piece_manager=piece_manager)
    decoder = CoordinateIndexDecoder(board_size=board_size)

    # First pass: enumerate all legal placements (drop off-board ones).
    # We don't know the final count until the end, so collect into lists.
    records: list[dict] = []

    # The orientation codec gives us the canonical 0..90 ordering. We
    # walk by orientation_id so move_ids land in a stable, predictable
    # order (deterministic across runs).
    num_orientations = piece_manager.num_entries

    for orientation_id in range(num_orientations):
        piece_id, orientation = piece_manager.get_piece_orientation(orientation_id)
        piece_grid = piece_manager.get_piece_orientation_array(piece_id, orientation)

        for y in range(board_size):
            for x in range(board_size):
                # Convert board (x, y) → array (length_idx, width_idx) =
                # (row, col). This matches the action codec / board.with_piece.
                row, col = decoder.to_idx((x, y))
                placement_cells = _piece_footprint(piece_grid, row, col, board_size)
                if placement_cells is None:
                    continue  # off-board

                anchor_cell_idx = row * board_size + col
                action_id = action_codec.encode_from_components(
                    piece_id=piece_id,
                    orientation_id=orientation_id,
                    x=x,
                    y=y,
                )
                adj_set, attach_set = _adj_and_attach(
                    placement_cells, board_size,
                )
                records.append({
                    "piece_id": piece_id,
                    "orientation_id": orientation_id,
                    "anchor": anchor_cell_idx,
                    "action_id": action_id,
                    "cells": placement_cells,
                    "adj": adj_set,
                    "attach": attach_set,
                })

    num_moves = len(records)

    # Allocate fixed-size arrays now we know the count.
    piece_arr = np.zeros(num_moves, dtype=np.uint8)
    anchor_arr = np.zeros(num_moves, dtype=np.uint8)
    action_id_arr = np.zeros(num_moves, dtype=np.uint32)

    cells_arr = np.full((num_moves, MAX_CELLS_PER_PIECE), NULL_CELL, dtype=np.uint8)
    n_cells_arr = np.zeros(num_moves, dtype=np.uint8)
    adj_cells_arr = np.full((num_moves, MAX_ADJ_CELLS), NULL_CELL, dtype=np.uint8)
    n_adj_arr = np.zeros(num_moves, dtype=np.uint8)
    attach_cells_arr = np.full((num_moves, MAX_ATTACH_CELLS), NULL_CELL, dtype=np.uint8)
    n_attach_arr = np.zeros(num_moves, dtype=np.uint8)

    action_to_move_id_arr = np.full(action_codec.action_size, -1, dtype=np.int32)

    # Second pass: fill the arrays.
    for move_id, rec in enumerate(records):
        piece_arr[move_id] = rec["piece_id"]
        anchor_arr[move_id] = rec["anchor"]
        action_id_arr[move_id] = rec["action_id"]
        cells = rec["cells"]
        if len(cells) > MAX_CELLS_PER_PIECE:
            raise AssertionError(
                f"Piece {rec['piece_id']} has {len(cells)} cells, exceeds "
                f"MAX_CELLS_PER_PIECE={MAX_CELLS_PER_PIECE}.",
            )
        n_cells_arr[move_id] = len(cells)
        cells_arr[move_id, : len(cells)] = cells

        adj = sorted(rec["adj"])
        if len(adj) > MAX_ADJ_CELLS:
            raise AssertionError(
                f"Move {move_id} (piece {rec['piece_id']}) has {len(adj)} adj "
                f"cells, exceeds MAX_ADJ_CELLS={MAX_ADJ_CELLS}.",
            )
        n_adj_arr[move_id] = len(adj)
        adj_cells_arr[move_id, : len(adj)] = adj

        attach = sorted(rec["attach"])
        if len(attach) > MAX_ATTACH_CELLS:
            raise AssertionError(
                f"Move {move_id} (piece {rec['piece_id']}) has {len(attach)} "
                f"attach cells, exceeds MAX_ATTACH_CELLS={MAX_ATTACH_CELLS}.",
            )
        n_attach_arr[move_id] = len(attach)
        attach_cells_arr[move_id, : len(attach)] = attach

        action_to_move_id_arr[rec["action_id"]] = move_id

    tables = MoveTables(
        num_moves=num_moves,
        piece=piece_arr,
        anchor=anchor_arr,
        action_id=action_id_arr,
        cells=cells_arr,
        n_cells=n_cells_arr,
        adj_cells=adj_cells_arr,
        n_adj=n_adj_arr,
        attach_cells=attach_cells_arr,
        n_attach=n_attach_arr,
        action_to_move_id=action_to_move_id_arr,
    )

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"Built MoveTables in {elapsed * 1000:.0f}ms: {num_moves:,} moves")
        _print_per_piece_counts(tables)

    return tables


# ---------------------------------------------------------------------------
# P5: the (anchor, adj_status, piece) → candidate move IDs lookup table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LookupTable:
    """The (anchor, adj_status, piece) → candidate move IDs precomputed table.

    For each ``(anchor, adj_status, piece)`` triple this stores the list
    of move IDs that:

    1. Use the given ``piece``.
    2. Are anchored at the given ``anchor`` cell.
    3. **Do not** occupy any of the 6 cells around the anchor that
       ``adj_status`` flags as forbidden.

    At runtime the move generator computes ``adj_status`` once per
    anchor (6 byte reads + bit-pack), then looks up a short candidate
    list per piece and verifies each candidate with a separate
    cells-not-forbidden check. The lookup pre-filters out the moves
    that can't possibly be legal because of the local 6-cell forbidden
    pattern — a cheap early-exit before the (slightly more expensive)
    per-cell check.

    Storage layout:

    - ``begin[anchor, adj_status, piece]`` and
      ``size[anchor, adj_status, piece]`` describe a slice into
      ``move_ids``. ``begin`` is ``uint32`` (the flat buffer can grow
      past 2^24); ``size`` is ``uint16`` (per-list size always small).
    - ``move_ids`` is a flat ``uint16`` buffer of all candidate lists
      concatenated. Move IDs fit in 16 bits since 13,729 < 2^16.

    Memory: ``begin`` and ``size`` are ~1.5 MB combined (196 × 64 × 21
    cells). ``move_ids`` is ~5 MB if we hit Pentobi's reported
    2.6M-entry total.
    """

    #: Total number of move IDs across all (anchor, adj_status, piece)
    #: candidate lists combined.
    total_entries: int

    #: ``begin[anchor, adj_status, piece]`` = start index into ``move_ids``.
    begin: NDArray  # uint32[NUM_CELLS, ADJ_STATUS_COUNT, 21]

    #: ``size[anchor, adj_status, piece]`` = length of the list at this slot.
    size: NDArray  # uint16[NUM_CELLS, ADJ_STATUS_COUNT, 21]

    #: Flat backing store of move IDs. The candidate list for
    #: ``(anchor, adj_status, piece)`` is the slice
    #: ``move_ids[begin[a,s,p] : begin[a,s,p] + size[a,s,p]]``.
    move_ids: NDArray  # uint16[total_entries]

    #: Per-anchor lookup of the 6 adj_status cells, with NULL_CELL for
    #: off-board neighbours. Used at runtime to compute adj_status by
    #: reading ``forbidden[c]`` for each of these cells and packing the
    #: bits. Shape ``(196, 6)`` of ``uint8``.
    adj_status_cells: NDArray  # uint8[NUM_CELLS, ADJ_STATUS_BITS]


def build_lookup_table(
    tables: MoveTables, board_size: int = BOARD_SIZE, *, verbose: bool = False,
) -> LookupTable:
    """Build the (anchor, adj_status, piece) lookup table from the
    geometry tables.

    Algorithm:

    1. For each anchor cell, compute the 6 adj_status cells (returning
       NULL_CELL for any that fall off the board).
    2. For each move ID in ``tables``, compute its ``incompat_mask``
       — which of the 6 adj_status cells (for its anchor) the move's
       footprint cells occupy. If a move's footprint includes
       adj_status cell i, then the move is incompatible with any
       ``adj_status`` value where bit i is set.
    3. For each ``(anchor, piece)`` pair, group the move IDs that
       match. For each of the 64 possible ``adj_status`` values, the
       candidate list is the subset of move IDs whose
       ``incompat_mask & adj_status == 0``.
    4. Pack the candidate lists into a single flat buffer with
       begin/size offsets per ``(anchor, adj_status, piece)``.

    Args:
        tables: The :class:`MoveTables` from :func:`build_move_tables`.
        board_size: Board dimension (14 for Duo).
        verbose: If True, print timing + size info.

    Returns:
        :class:`LookupTable` with the begin/size/move_ids arrays
        populated and ``total_entries`` set.
    """
    t0 = time.perf_counter()

    n_anchors = board_size * board_size
    n_pieces = 21

    # Step 1: adj_status cells per anchor.
    adj_status_cells = np.full((n_anchors, ADJ_STATUS_BITS), NULL_CELL, dtype=np.uint8)
    for anchor in range(n_anchors):
        ar, ac = divmod(anchor, board_size)
        for bit, (dr, dc) in enumerate(ADJ_STATUS_OFFSETS):
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                adj_status_cells[anchor, bit] = nr * board_size + nc
            # else stays NULL_CELL — off-board

    # Pre-compute each move's footprint as a Python set for the lookup
    # below. (We don't allocate this as a numpy array because we only
    # need it during table construction.)
    footprints: list[set[int]] = []
    for move_id in range(tables.num_moves):
        n = int(tables.n_cells[move_id])
        footprints.append(set(int(c) for c in tables.cells[move_id, :n]))

    def incompat_mask_for_anchor(move_id: int, anchor: int) -> int:
        """The 6-bit mask of adj_status cells (for ``anchor``) that the
        move's footprint occupies. A move is compatible with a given
        adj_status iff ``(mask & adj_status) == 0``.

        Note this is anchor-dependent — the same move has different
        incompat_masks at different attach points because the 6
        adj_status cells around each anchor are different.
        """
        footprint = footprints[move_id]
        mask = 0
        for bit in range(ADJ_STATUS_BITS):
            asc = int(adj_status_cells[anchor, bit])
            if asc != NULL_CELL and asc in footprint:
                mask |= (1 << bit)
        return mask

    # Step 3: group move IDs by (footprint_cell, piece) for fast adj_status filtering.
    #
    # IMPORTANT: a move appears in the list for EACH of its footprint
    # cells, not just the bounding-box top-left. This matches Pentobi's
    # design: ``placement_points`` (a.k.a. attach points) can be any
    # cell the player could anchor a piece on, and a Blokus piece's
    # cells are interchangeable as "the cell that touches the attach
    # point." A single move whose 5-cell footprint includes 2 attach
    # points appears in 2 different (anchor, piece) lists — runtime
    # dedup via a "seen" set keeps it from being counted twice.
    moves_by_anchor_piece: list[list[list[int]]] = [
        [[] for _ in range(n_pieces)] for _ in range(n_anchors)
    ]
    for move_id in range(tables.num_moves):
        p = int(tables.piece[move_id]) - 1  # piece IDs are 1..21
        n = int(tables.n_cells[move_id])
        for cell_idx_in_piece in range(n):
            footprint_cell = int(tables.cells[move_id, cell_idx_in_piece])
            moves_by_anchor_piece[footprint_cell][p].append(move_id)

    # Step 4: build the flat buffer + begin/size arrays.
    # For each (anchor, piece), precompute each base-move's
    # incompat_mask vs this anchor (it depends on the anchor — see
    # ``incompat_mask_for_anchor`` docstring). Then for each adj_status
    # value, filter the precomputed list by mask.
    begin = np.zeros((n_anchors, ADJ_STATUS_COUNT, n_pieces), dtype=np.uint32)
    size = np.zeros((n_anchors, ADJ_STATUS_COUNT, n_pieces), dtype=np.uint16)
    move_ids_buffer: list[int] = []
    cursor = 0
    for anchor in range(n_anchors):
        for piece_idx in range(n_pieces):
            base = moves_by_anchor_piece[anchor][piece_idx]
            if not base:
                # Empty base list — every (anchor, status, piece_idx)
                # slot stays at size 0.
                for status in range(ADJ_STATUS_COUNT):
                    begin[anchor, status, piece_idx] = cursor
                continue
            # Precompute the incompat_mask of each base move vs this anchor.
            base_masks = [incompat_mask_for_anchor(m, anchor) for m in base]
            for status in range(ADJ_STATUS_COUNT):
                candidates = [
                    base[i] for i, mask in enumerate(base_masks)
                    if (mask & status) == 0
                ]
                begin[anchor, status, piece_idx] = cursor
                size[anchor, status, piece_idx] = len(candidates)
                if len(candidates) > 0xFFFF:
                    raise AssertionError(
                        f"List at ({anchor}, {status}, {piece_idx}) has "
                        f"{len(candidates)} > 65535 entries — uint16 size won't fit.",
                    )
                move_ids_buffer.extend(candidates)
                cursor += len(candidates)

    move_ids = np.array(move_ids_buffer, dtype=np.uint16)
    total_entries = len(move_ids)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"Built LookupTable in {elapsed:.2f}s: {total_entries:,} entries")
        mb = (begin.nbytes + size.nbytes + move_ids.nbytes + adj_status_cells.nbytes) / 1024 / 1024
        print(f"  begin:            {begin.nbytes / 1024:.1f} KB ({begin.shape} {begin.dtype})")
        print(f"  size:             {size.nbytes / 1024:.1f} KB")
        print(f"  move_ids:         {move_ids.nbytes / 1024:.1f} KB")
        print(f"  adj_status_cells: {adj_status_cells.nbytes / 1024:.1f} KB")
        print(f"  total:            {mb:.2f} MB")
        nonempty = int((size > 0).sum())
        total_slots = n_anchors * ADJ_STATUS_COUNT * n_pieces
        print(f"  non-empty slots:  {nonempty:,} / {total_slots:,} ({100*nonempty/total_slots:.1f}%)")

    return LookupTable(
        total_entries=total_entries,
        begin=begin,
        size=size,
        move_ids=move_ids,
        adj_status_cells=adj_status_cells,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _piece_footprint(
    piece_grid: NDArray, anchor_row: int, anchor_col: int, board_size: int,
) -> list[int] | None:
    """Return the cell indices the piece occupies when anchored at
    ``(anchor_row, anchor_col)``, or ``None`` if any cell is off-board.

    ``piece_grid`` is the 2D 0/1 array from
    ``PieceManager.get_piece_orientation_array``.
    """
    p_rows, p_cols = piece_grid.shape
    cells: list[int] = []
    for i in range(p_rows):
        for j in range(p_cols):
            if piece_grid[i, j] == 0:
                continue
            r, c = anchor_row + i, anchor_col + j
            if not (0 <= r < board_size and 0 <= c < board_size):
                return None  # placement extends off the board
            cells.append(r * board_size + c)
    return cells


def _adj_and_attach(
    footprint: list[int], board_size: int,
) -> tuple[set[int], set[int]]:
    """Compute the edge-adjacent and diagonally-adjacent cell sets.

    Definitions follow Blokus rules:
    - **adj**: cells sharing an edge with at least one footprint cell.
      These become forbidden for the playing colour when the move is
      played (no same-colour piece may edge-touch).
    - **attach**: cells sharing only a corner with at least one
      footprint cell, and NOT edge-touching the footprint. These
      become new attach points for the playing colour. A cell that's
      both edge- and diagonally-adjacent counts only as ``adj``.

    Footprint cells themselves are excluded from both sets.
    Off-board candidate cells are excluded from both sets.

    Two-pass implementation matters: the ``adj`` set must be fully
    populated before we check diagonals, because a single cell can be
    edge-adjacent to one footprint cell and diagonally-adjacent to
    another (e.g. an L-shaped piece). Single-pass processing would
    leak such cells into the ``attach`` set whenever the diagonal-
    contributing footprint cell happened to be processed first.
    """
    footprint_set = set(footprint)
    edge_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diag_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Pass 1: collect all edge-adjacent cells.
    adj: set[int] = set()
    for cell_idx in footprint:
        r, c = divmod(cell_idx, board_size)
        for dr, dc in edge_offsets:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < board_size and 0 <= nc < board_size):
                continue
            neighbour = nr * board_size + nc
            if neighbour in footprint_set:
                continue
            adj.add(neighbour)

    # Pass 2: collect diagonals that aren't already in adj.
    attach: set[int] = set()
    for cell_idx in footprint:
        r, c = divmod(cell_idx, board_size)
        for dr, dc in diag_offsets:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < board_size and 0 <= nc < board_size):
                continue
            neighbour = nr * board_size + nc
            if neighbour in footprint_set or neighbour in adj:
                continue
            attach.add(neighbour)

    return adj, attach


def _print_per_piece_counts(tables: MoveTables) -> None:
    """Print number of legal placements per piece id for sanity checking."""
    unique, counts = np.unique(tables.piece, return_counts=True)
    print("  Per-piece legal placements (piece_id → count):")
    for piece_id, count in zip(unique, counts, strict=True):
        print(f"    piece {int(piece_id):2d}: {int(count):>5,} placements")


# ---------------------------------------------------------------------------
# Patch the ActionCodec with the helper we need
# ---------------------------------------------------------------------------
# The existing ActionCodec.encode takes an ``Action`` dataclass. Building the
# Action requires an Orientation enum, which we already decoded. To avoid
# the round-trip, we add a low-level encode helper.

def _encode_from_components(
    self: ActionCodec, *, piece_id: int, orientation_id: int, x: int, y: int,
) -> int:
    """Encode from primitive components without constructing an Action.

    Equivalent to ``self.encode(Action(piece_id, orientation, x, y))`` but
    skips the ``Action`` construction and the piece→orientation_id lookup
    (we already have the id).
    """
    del piece_id  # used only for symmetry of the call site
    return (
        y * (self._board_size * self._num_orientations)
        + x * self._num_orientations
        + orientation_id
    )


# Monkey-patch on import — this is a hot helper for table building.
ActionCodec.encode_from_components = _encode_from_components  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CLI entry point for inspection / one-off building
# ---------------------------------------------------------------------------

def main() -> int:
    """Build the tables and print summary stats. Useful for sanity-checking."""
    pieces_path = Path(__file__).resolve().parent / "pieces.json"
    piece_manager = pieces_loader(pieces_path)

    t0 = time.perf_counter()
    tables = build_move_tables(piece_manager, BOARD_SIZE, verbose=True)
    elapsed = time.perf_counter() - t0

    print()
    print(f"Total build time: {elapsed:.2f}s")
    print(f"num_moves: {tables.num_moves:,}")
    print("Memory (rough):")
    print(f"  cells:        {tables.cells.nbytes / 1024:.1f} KB "
          f"({tables.cells.shape} {tables.cells.dtype})")
    print(f"  adj_cells:    {tables.adj_cells.nbytes / 1024:.1f} KB")
    print(f"  attach_cells: {tables.attach_cells.nbytes / 1024:.1f} KB")
    print(f"  action_to_move_id: {tables.action_to_move_id.nbytes / 1024:.1f} KB")
    total_kb = sum(
        getattr(tables, f).nbytes
        for f in ["piece", "anchor", "action_id", "cells", "n_cells",
                  "adj_cells", "n_adj", "attach_cells", "n_attach",
                  "action_to_move_id"]
    ) / 1024
    print(f"  total:        {total_kb:.1f} KB")

    expected = 13_729
    if tables.num_moves != expected:
        print(f"\nFAIL: expected {expected:,} moves, got {tables.num_moves:,}")
        return 1

    print()
    print("=" * 60)
    print("Building lookup table (P5) on top of geometry tables...")
    print("=" * 60)
    lookup = build_lookup_table(tables, BOARD_SIZE, verbose=True)

    print()
    print(f"OK: built MoveTables ({tables.num_moves:,} moves) + "
          f"LookupTable ({lookup.total_entries:,} entries).")
    print(f"Matches Pentobi's Move.h:32 count of {expected:,}.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
