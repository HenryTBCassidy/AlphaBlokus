"""Unit tests for the F2 P4 geometry tables.

The :class:`MoveTables` produced by :func:`build_move_tables` is the
foundation of the F2 runtime move generator. Any bug here propagates to
every downstream piece of F2, so this test pins the invariants tightly.

Tests run in ~100ms (table build is fast).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from games.blokusduo.movegen_tables import (
    ADJ_STATUS_BITS,
    ADJ_STATUS_COUNT,
    ADJ_STATUS_OFFSETS,
    BOARD_SIZE,
    MAX_ADJ_CELLS,
    MAX_ATTACH_CELLS,
    MAX_CELLS_PER_PIECE,
    NULL_CELL,
    NUM_CELLS,
    build_lookup_table,
    build_move_tables,
)
from games.blokusduo.pieces import pieces_loader

PIECES_PATH = Path(__file__).resolve().parent.parent.parent / "games" / "blokusduo" / "pieces.json"
PENTOBI_DUO_MOVE_COUNT = 13_729


@pytest.fixture(scope="module")
def tables():
    """Build the tables once, reuse across this module's tests."""
    pm = pieces_loader(PIECES_PATH)
    return build_move_tables(pm, BOARD_SIZE)


# ---------------------------------------------------------------------------
# Basic shape and count invariants
# ---------------------------------------------------------------------------

def test_move_count_matches_pentobi(tables) -> None:
    """Headline check: same count as scripts/count_onboard_placements.py."""
    assert tables.num_moves == PENTOBI_DUO_MOVE_COUNT, (
        f"Expected {PENTOBI_DUO_MOVE_COUNT:,} moves, got {tables.num_moves:,}. "
        "Did pieces.json change?"
    )


def test_array_shapes(tables) -> None:
    n = tables.num_moves
    assert tables.piece.shape == (n,)
    assert tables.anchor.shape == (n,)
    assert tables.action_id.shape == (n,)
    assert tables.cells.shape == (n, MAX_CELLS_PER_PIECE)
    assert tables.n_cells.shape == (n,)
    assert tables.adj_cells.shape == (n, MAX_ADJ_CELLS)
    assert tables.n_adj.shape == (n,)
    assert tables.attach_cells.shape == (n, MAX_ATTACH_CELLS)
    assert tables.n_attach.shape == (n,)
    # action_to_move_id is sized to the full action space (17,837).
    assert tables.action_to_move_id.shape == (17_837,)


def test_array_dtypes(tables) -> None:
    assert tables.piece.dtype == np.uint8
    assert tables.anchor.dtype == np.uint8
    assert tables.action_id.dtype == np.uint32
    assert tables.cells.dtype == np.uint8
    assert tables.n_cells.dtype == np.uint8
    assert tables.adj_cells.dtype == np.uint8
    assert tables.n_adj.dtype == np.uint8
    assert tables.attach_cells.dtype == np.uint8
    assert tables.n_attach.dtype == np.uint8
    assert tables.action_to_move_id.dtype == np.int32


# ---------------------------------------------------------------------------
# Per-move structural invariants
# ---------------------------------------------------------------------------

def test_n_cells_in_range(tables) -> None:
    """Every piece has between 1 and 5 cells (Duo has only mono- through penta-ominoes)."""
    assert (tables.n_cells >= 1).all()
    assert (tables.n_cells <= MAX_CELLS_PER_PIECE).all()


def test_anchors_on_board(tables) -> None:
    """Every anchor cell is a valid board cell (0..195)."""
    assert (tables.anchor < NUM_CELLS).all()


def test_pieces_in_range(tables) -> None:
    """Piece ids are 1..21 (Blokus has 21 pieces, ids 1-indexed)."""
    assert (tables.piece >= 1).all()
    assert (tables.piece <= 21).all()
    # All 21 pieces are represented at least once.
    assert set(np.unique(tables.piece).tolist()) == set(range(1, 22))


def test_action_ids_unique_and_in_range(tables) -> None:
    """Each move maps to a unique action id, and all action ids are < 17,837."""
    assert (tables.action_id < 17_837).all()
    assert len(np.unique(tables.action_id)) == tables.num_moves, (
        "Duplicate action_id in the tables — encoding bug."
    )


def test_cells_padded_with_null(tables) -> None:
    """Cell slots beyond ``n_cells`` are filled with ``NULL_CELL``."""
    for i in range(tables.num_moves):
        n = int(tables.n_cells[i])
        real_cells = tables.cells[i, :n]
        padded = tables.cells[i, n:]
        assert (real_cells < NUM_CELLS).all(), (
            f"Move {i}: cell index >= NUM_CELLS in real-cell region"
        )
        assert (padded == NULL_CELL).all(), (
            f"Move {i}: cells beyond n_cells={n} not padded with NULL_CELL"
        )


def test_no_duplicate_cells_per_move(tables) -> None:
    """A piece's footprint cells must all be distinct."""
    for i in range(tables.num_moves):
        n = int(tables.n_cells[i])
        cells = tables.cells[i, :n]
        assert len(set(cells.tolist())) == n, (
            f"Move {i} has duplicate footprint cells: {cells.tolist()}"
        )


def test_adj_and_attach_disjoint_from_footprint(tables) -> None:
    """``adj`` and ``attach`` sets must not include any footprint cells."""
    for i in range(tables.num_moves):
        footprint = set(tables.cells[i, : tables.n_cells[i]].tolist())
        adj = set(tables.adj_cells[i, : tables.n_adj[i]].tolist())
        attach = set(tables.attach_cells[i, : tables.n_attach[i]].tolist())
        assert footprint.isdisjoint(adj), (
            f"Move {i}: footprint overlaps with adj cells"
        )
        assert footprint.isdisjoint(attach), (
            f"Move {i}: footprint overlaps with attach cells"
        )


def test_adj_and_attach_disjoint_from_each_other(tables) -> None:
    """Edge-adjacency wins: a cell that's both edge- and diagonally-adjacent
    must be in ``adj`` only, never in ``attach``."""
    for i in range(tables.num_moves):
        adj = set(tables.adj_cells[i, : tables.n_adj[i]].tolist())
        attach = set(tables.attach_cells[i, : tables.n_attach[i]].tolist())
        overlap = adj & attach
        assert not overlap, (
            f"Move {i}: cell(s) {overlap} in both adj and attach sets"
        )


# ---------------------------------------------------------------------------
# action_id ↔ move_id round-trip
# ---------------------------------------------------------------------------

def test_action_to_move_round_trip(tables) -> None:
    """For every legal move, action_to_move_id ∘ action_id is identity."""
    for move_id in range(tables.num_moves):
        action_id = int(tables.action_id[move_id])
        round_tripped = int(tables.action_to_move_id[action_id])
        assert round_tripped == move_id, (
            f"Round-trip broken for move_id {move_id}: action {action_id} → "
            f"move {round_tripped} (expected {move_id})"
        )


def test_offboard_actions_marked(tables) -> None:
    """Action IDs that don't correspond to a legal placement get -1."""
    # 17,837 total slots = 13,729 legal + 1 pass + 4,107 off-board.
    illegal_count = int((tables.action_to_move_id == -1).sum())
    legal_count = int((tables.action_to_move_id >= 0).sum())
    assert legal_count == tables.num_moves, (
        f"Expected {tables.num_moves} legal action slots, got {legal_count}"
    )
    # 17,837 - 13,729 = 4,108 (4,107 off-board + 1 pass).
    assert illegal_count == 17_837 - tables.num_moves


# ===========================================================================
# P5: LookupTable tests
# ===========================================================================

@pytest.fixture(scope="module")
def lookup(tables):
    """Build the lookup table once, reuse across this module's tests."""
    return build_lookup_table(tables, BOARD_SIZE)


# ---------------------------------------------------------------------------
# Shape and structural invariants
# ---------------------------------------------------------------------------

def test_lookup_array_shapes(lookup) -> None:
    assert lookup.begin.shape == (NUM_CELLS, ADJ_STATUS_COUNT, 21)
    assert lookup.size.shape == (NUM_CELLS, ADJ_STATUS_COUNT, 21)
    assert lookup.move_ids.shape == (lookup.total_entries,)
    assert lookup.adj_status_cells.shape == (NUM_CELLS, ADJ_STATUS_BITS)


def test_lookup_array_dtypes(lookup) -> None:
    assert lookup.begin.dtype == np.uint32
    assert lookup.size.dtype == np.uint16
    assert lookup.move_ids.dtype == np.uint16
    assert lookup.adj_status_cells.dtype == np.uint8


def test_lookup_slices_dont_overflow(lookup) -> None:
    """For every (anchor, status, piece) slot, ``begin + size`` must
    stay within the move_ids buffer."""
    end = lookup.begin.astype(np.int64) + lookup.size.astype(np.int64)
    assert (end <= lookup.total_entries).all(), (
        "Some slot's begin+size exceeds move_ids buffer length"
    )


def test_adj_status_cells_correct_for_anchor_middle(lookup) -> None:
    """For an interior anchor (7, 7), all 6 adj_status cells are
    on-board and match the documented offsets."""
    anchor = 7 * BOARD_SIZE + 7
    for bit, (dr, dc) in enumerate(ADJ_STATUS_OFFSETS):
        expected_cell = (7 + dr) * BOARD_SIZE + (7 + dc)
        actual = int(lookup.adj_status_cells[anchor, bit])
        assert actual == expected_cell, (
            f"adj_status_cells[(7,7), bit {bit}] = {actual}, expected {expected_cell}"
        )


def test_adj_status_cells_offboard_marked_at_corner(lookup) -> None:
    """At the top-left corner (0, 0), N, W, NW are all off-board and
    must be marked with NULL_CELL."""
    anchor = 0  # (0, 0)
    # bit 0 = N (out of board), bit 1 = W (out), bit 4 = NW (out)
    assert int(lookup.adj_status_cells[anchor, 0]) == NULL_CELL  # N
    assert int(lookup.adj_status_cells[anchor, 1]) == NULL_CELL  # W
    assert int(lookup.adj_status_cells[anchor, 4]) == NULL_CELL  # NW
    # On-board: E, S, SE
    assert int(lookup.adj_status_cells[anchor, 2]) == 1   # E = (0, 1)
    assert int(lookup.adj_status_cells[anchor, 3]) == BOARD_SIZE  # S = (1, 0)
    assert int(lookup.adj_status_cells[anchor, 5]) == BOARD_SIZE + 1  # SE = (1, 1)


# ---------------------------------------------------------------------------
# Correctness: candidate-list contents
# ---------------------------------------------------------------------------

def test_candidate_lists_have_correct_move_ids(tables, lookup) -> None:
    """For every (anchor, adj_status, piece), every move ID in the
    candidate list satisfies: (a) anchor is one of its footprint cells,
    (b) it uses that piece, (c) it doesn't occupy any cell flagged by
    adj_status.

    Note (a) — the lookup table indexes moves by *every* cell of their
    footprint, not just the bounding-box top-left. This matches
    Pentobi's design: ``placement_points`` (attach points) can be any
    cell the player anchors a piece on. Runtime uses a dedup set to
    avoid double-emitting moves reachable from multiple attach points.
    """
    rng = np.random.default_rng(seed=0)
    # Sampling to keep the test fast.
    sample_anchors = rng.integers(0, NUM_CELLS, size=500)
    sample_statuses = rng.integers(0, ADJ_STATUS_COUNT, size=500)
    sample_pieces = rng.integers(0, 21, size=500)

    for anchor, status, piece_idx in zip(sample_anchors, sample_statuses,
                                          sample_pieces, strict=True):
        begin = int(lookup.begin[anchor, status, piece_idx])
        sz = int(lookup.size[anchor, status, piece_idx])
        adj_cells = lookup.adj_status_cells[anchor]
        forbidden_set = {
            int(adj_cells[bit]) for bit in range(ADJ_STATUS_BITS)
            if (status >> bit) & 1 and int(adj_cells[bit]) != NULL_CELL
        }
        for move_id in lookup.move_ids[begin : begin + sz]:
            mv = int(move_id)
            footprint = set(int(c) for c in tables.cells[mv, : tables.n_cells[mv]])
            assert anchor in footprint, (
                f"Move {mv} at slot ({anchor},{status},{piece_idx}): "
                f"anchor {anchor} is not in the move's footprint {sorted(footprint)}"
            )
            assert int(tables.piece[mv]) == piece_idx + 1, (
                f"Move {mv} has piece {int(tables.piece[mv])}, expected {piece_idx+1}"
            )
            overlap = footprint & forbidden_set
            assert not overlap, (
                f"Move {mv} at ({anchor},{status},{piece_idx}) occupies "
                f"forbidden cells {overlap}"
            )


def test_status_zero_returns_all_moves_for_anchor_piece(tables, lookup) -> None:
    """When adj_status == 0 (no neighbour cells forbidden), the
    candidate list at (anchor, 0, piece) must contain *every* move ID
    whose footprint includes ``anchor`` and uses ``piece``."""
    from collections import defaultdict
    moves_by_ap: dict[tuple[int, int], set[int]] = defaultdict(set)
    for mv in range(tables.num_moves):
        p = int(tables.piece[mv]) - 1
        n = int(tables.n_cells[mv])
        for cell in tables.cells[mv, :n]:
            moves_by_ap[(int(cell), p)].add(mv)

    for (anchor, piece_idx), expected in moves_by_ap.items():
        begin = int(lookup.begin[anchor, 0, piece_idx])
        sz = int(lookup.size[anchor, 0, piece_idx])
        got = set(int(m) for m in lookup.move_ids[begin : begin + sz])
        assert got == expected, (
            f"(anchor={anchor}, status=0, piece={piece_idx+1}): "
            f"expected {len(expected)} moves, got {len(got)}; "
            f"diff: only in got = {got - expected}, "
            f"only in expected = {expected - got}"
        )


def test_status_filtering_only_removes_moves(tables, lookup) -> None:
    """For any (anchor, status, piece), the candidate list must be a
    SUBSET of the full (anchor, *, piece) move set (where "anchor in
    move's footprint")."""
    from collections import defaultdict
    moves_by_ap: dict[tuple[int, int], set[int]] = defaultdict(set)
    for mv in range(tables.num_moves):
        p = int(tables.piece[mv]) - 1
        n = int(tables.n_cells[mv])
        for cell in tables.cells[mv, :n]:
            moves_by_ap[(int(cell), p)].add(mv)

    rng = np.random.default_rng(seed=1)
    sample_anchors = rng.integers(0, NUM_CELLS, size=500)
    sample_statuses = rng.integers(1, ADJ_STATUS_COUNT, size=500)  # avoid 0
    sample_pieces = rng.integers(0, 21, size=500)
    for anchor, status, piece_idx in zip(sample_anchors, sample_statuses,
                                          sample_pieces, strict=True):
        begin = int(lookup.begin[anchor, status, piece_idx])
        sz = int(lookup.size[anchor, status, piece_idx])
        got = set(int(m) for m in lookup.move_ids[begin : begin + sz])
        full = moves_by_ap.get((anchor, piece_idx), set())
        assert got.issubset(full), (
            f"Candidate list at ({anchor},{status},{piece_idx}) is NOT a "
            f"subset of (anchor, *, piece). Extra: {got - full}"
        )
