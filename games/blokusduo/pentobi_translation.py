"""Translation between AlphaBlokus moves and Pentobi GTP move notation (H1).

Pentobi's GTP interface describes a move purely by the **cells the piece occupies**
(e.g. ``e9,d10,e10,f10,e11``) — it never names the piece or orientation; the engine
deduces them. So translation is asymmetric:

- **out (Action → cells):** place the piece's orientation array at the action's anchor
  and emit the occupied cells (reuses the same machinery as ``BlokusDuoBoard.with_piece``).
- **in (cells → Action):** match the cell *shape* against the 91 piece-orientations. Each
  orientation is a distinct polyomino shape, so a shape (normalised to its bounding box)
  maps to exactly one ``(piece_id, orientation)`` — a bijection built once at init.

Coordinate frames are nearly identical — both bottom-left origin. Pentobi columns are
letters ``a``–``n`` (a=0…n=13), rows are 1-indexed (1=bottom): so ``x = letter``,
``y = number - 1``. Array indices (top-left origin) are handled by ``CoordinateIndexDecoder``.

See docs/06-INTERFACES.md §1 for the design.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from games.blokusduo.board import Action, CoordinateIndexDecoder

if TYPE_CHECKING:
    from games.blokusduo.board import BlokusDuoBoard
    from games.blokusduo.game import BlokusDuoGame

PASS = "pass"


class PentobiMoveTranslator:
    """Converts AlphaBlokus actions ↔ Pentobi GTP move strings for one board size."""

    def __init__(self, game: BlokusDuoGame) -> None:
        self._pm = game.piece_manager
        self._codec = game.action_codec
        self._n = game.board_size
        # CoordinateIndexDecoder is a pure function of board size — build our own
        # rather than reach into the game's private one.
        self._decoder = CoordinateIndexDecoder(self._n)

        # Bijection: normalised occupied shape -> (piece_id, orientation, r0, c0).
        # r0/c0 are the orientation array's own min offsets, kept so the anchor can
        # be recovered exactly (robust even if an orientation array isn't tight).
        self._shape_to_po: dict[frozenset[tuple[int, int]], tuple[int, object, int, int]] = {}
        for piece_id, orientation in self._pm.all_piece_id_basis_orientations():
            filled = self._pm.get_filled_cells(piece_id, orientation)
            r0 = int(filled[:, 0].min())
            c0 = int(filled[:, 1].min())
            shape = frozenset((int(r) - r0, int(c) - c0) for r, c in filled)
            if shape in self._shape_to_po:
                raise ValueError(
                    "Piece-orientation shape collision — the shape→orientation map is "
                    "not a bijection; cells→Action would be ambiguous.",
                )
            self._shape_to_po[shape] = (piece_id, orientation, r0, c0)

    # -- coordinates -----------------------------------------------------------

    def coord_to_pentobi(self, x: int, y: int) -> str:
        """AlphaBlokus board coord (x, y) → Pentobi cell, e.g. (4, 8) → 'e9'."""
        return f"{chr(ord('a') + x)}{y + 1}"

    def pentobi_to_coord(self, token: str) -> tuple[int, int]:
        """Pentobi cell → AlphaBlokus board coord, e.g. 'e9' → (4, 8)."""
        token = token.strip()
        return ord(token[0]) - ord("a"), int(token[1:]) - 1

    # -- moves -----------------------------------------------------------------

    def action_to_cells(self, action: Action) -> str:
        """Action → comma-separated Pentobi cell list (sorted for determinism)."""
        filled = self._pm.get_filled_cells(action.piece_id, action.orientation)
        length_idx, width_idx = self._decoder.to_idx(
            (action.x_coordinate, action.y_coordinate),
        )
        coords: list[tuple[int, int]] = []
        for r, c in filled:
            x, y = self._decoder.to_coordinate((length_idx + int(r), width_idx + int(c)))
            coords.append((x, y))
        coords.sort()
        return ",".join(self.coord_to_pentobi(x, y) for x, y in coords)

    def cells_to_action(self, cells: str) -> Action:
        """Pentobi cell list → Action by matching the shape against the 91 orientations."""
        coords = [self.pentobi_to_coord(tok) for tok in cells.split(",")]
        arr_cells = [(self._n - 1 - y, x) for x, y in coords]  # (row, col) array indices
        min_r = min(r for r, _ in arr_cells)
        min_c = min(c for _, c in arr_cells)
        shape = frozenset((r - min_r, c - min_c) for r, c in arr_cells)
        match = self._shape_to_po.get(shape)
        if match is None:
            raise ValueError(f"No piece-orientation matches Pentobi cells {cells!r}.")
        piece_id, orientation, r0, c0 = match
        # cells = anchor + orientation offsets, so min cell = anchor + (r0, c0).
        x, y = self._decoder.to_coordinate((min_r - r0, min_c - c0))
        return Action(piece_id, orientation, x, y)

    # -- action-index level (what Player / Arena work with) --------------------

    def action_index_to_pentobi(self, action_index: int) -> str:
        """Action index → Pentobi move string ('pass' for the pass action)."""
        if self._codec.is_pass(action_index):
            return PASS
        return self.action_to_cells(self._codec.decode(action_index))

    def pentobi_to_action_index(self, move: str) -> int:
        """Pentobi move string → action index ('pass' → the pass action)."""
        if move.strip().lower() == PASS:
            return self._codec.pass_action_index
        return self._codec.encode(self.cells_to_action(move))

    # -- debugging / cross-validation -----------------------------------------

    def render_board(self, board: BlokusDuoBoard) -> str:
        """ASCII board (Pentobi-style a–n / 1–14 grid) for debugging and the H4
        per-move cross-validation against Pentobi's ``showboard``."""
        grid = board.as_2d  # (N, N) int8, +1 white / -1 black / 0 empty, array-indexed
        header = "    " + " ".join(chr(ord("a") + x) for x in range(self._n))
        lines = [header]
        for i in range(self._n):
            row_label = self._n - i  # array row 0 is the top (y = N-1)
            cells = ["W" if v > 0 else "B" if v < 0 else "." for v in grid[i]]
            lines.append(f"{row_label:>2}  " + " ".join(cells))
        return "\n".join(lines)
