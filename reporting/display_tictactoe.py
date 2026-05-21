"""HTML rendering for TicTacToe — implements :class:`IBoardRenderer`.

Stylistic convention follows the Blokus Duo renderer:
- Blue (``#636efa``) = White / X / player +1
- Red (``#EF553B``) = Black / O / player -1
- Empty cells have a white background and no glyph
- Row/column labels at the table edges, never inside cells
- Same monospace, light-grey grid lines, small-cap labels above the board
"""
from __future__ import annotations

from games.tictactoe.board import Board

# Matched to the Blokus palette so both games look like they belong in the
# same report. Slight transparency on the "last-move" highlight so the colour
# pops without screaming.
_X_BG = "#636efa"  # blue (White / +1)
_O_BG = "#EF553B"  # red (Black / -1)
_X_LAST_RING = "#2546c8"  # darker blue outline for the just-played cell
_O_LAST_RING = "#b8362a"

_CELL_SIZE_PX = 56
_BOARD_CLASS = "ttt-board"


class TicTacToeRenderer:
    """Renders a 3×3 TTT board as an HTML table.

    Two render modes:

    1. :meth:`render_board_html` — the "actual" board. X cells blue, O cells
       red, empty cells blank, optional outline highlight on the most recent
       move.
    2. :meth:`render_policy_html` — the same board layout but with the
       player's policy probabilities written into the *empty* cells. Already-
       occupied cells render exactly as in mode 1, so policy/actual boards
       laid out side by side line up perfectly.
    """

    def render_board_html(
        self,
        board: Board,
        last_action: int | None = None,
        annotation: str = "",
    ) -> str:
        cells = [
            _actual_cell_html(int(board.as_2d[divmod(a, 3)]), a == last_action)
            for a in range(9)
        ]
        return _wrap_table(cells, annotation)

    def render_policy_html(
        self,
        board: Board,
        action_probs: dict[int, float],
        annotation: str = "",
        current_player: int = 1,
    ) -> str:
        """Render the board with policy percentages overlaid on empty cells.

        Args:
            board: the actual (non-canonical) board state at this turn.
                Cells already containing pieces show those pieces; only empty
                cells get a probability number.
            action_probs: mapping ``action_index -> probability``. Any action
                not in the dict is treated as 0%. Probabilities should sum to
                ~1 over the legal actions but we don't strictly require it.
            annotation: header text shown above the table.
            current_player: ``+1`` (White) tints empty-cell backgrounds blue,
                ``-1`` (Black) tints them red. Makes it obvious at a glance
                whose policy is being shown.
        """
        arr = board.as_2d
        cells = []
        for action in range(9):
            col, row = divmod(action, 3)
            v = int(arr[col, row])
            if v != 0:
                cells.append(_actual_cell_html(v, is_last=False))
            else:
                prob = float(action_probs.get(action, 0.0))
                cells.append(_policy_cell_html(prob, current_player))
        return _wrap_table(cells, annotation)


# -- internal cell renderers --------------------------------------------------


def _actual_cell_html(value: int, is_last: bool) -> str:
    """Render one cell of the "actual" board view."""
    if value == 1:
        bg, fg = _X_BG, "#ffffff"
        glyph = "X"
        ring = f"box-shadow: inset 0 0 0 3px {_X_LAST_RING};" if is_last else ""
    elif value == -1:
        bg, fg = _O_BG, "#ffffff"
        glyph = "O"
        ring = f"box-shadow: inset 0 0 0 3px {_O_LAST_RING};" if is_last else ""
    else:
        bg, fg, glyph, ring = "#ffffff", "#1f2937", "", ""
    return _cell_html(glyph, bg=bg, fg=fg, extra=ring, font_size_px=22)


def _policy_cell_html(prob: float, current_player: int) -> str:
    """Render one cell of the policy view — % in empty cells, blank otherwise.

    Background opacity scales with probability so a quick glance shows which
    cells the model favoured without having to read every number. Tint matches
    the player whose policy is being shown — blue for White (+1), red for
    Black (-1) — so the colour alone tells you who's "thinking".
    """
    if prob <= 0.001:
        text = "0%"
        intensity = 0.0
    else:
        text = f"{prob * 100:.0f}%"
        intensity = max(0.08, min(0.55, prob))  # 8%-55% bg opacity range
    # rgb of _X_BG = (99,110,250) blue; _O_BG = (239,85,59) red
    if current_player == -1:
        bg = f"rgba(239, 85, 59, {intensity:.2f})"
    else:
        bg = f"rgba(99, 110, 250, {intensity:.2f})"
    return _cell_html(text, bg=bg, fg="#1f2937", extra="", font_size_px=15)


def _cell_html(
    content: str,
    *,
    bg: str,
    fg: str,
    extra: str,
    font_size_px: int,
) -> str:
    return (
        f'<td style="background:{bg};color:{fg};'
        f"width:{_CELL_SIZE_PX}px;height:{_CELL_SIZE_PX}px;"
        f"text-align:center;vertical-align:middle;"
        f"border:1px solid #d1d5db;font-weight:600;font-size:{font_size_px}px;"
        f'font-family:monospace;{extra}">{content}</td>'
    )


def _wrap_table(cells: list[str], annotation: str) -> str:
    """Wrap 9 cell <td>s into the standard board layout with edge labels.

    Layout — three rows of three cells, with a column header above and a row
    label to the left of each row. Matches the Blokus convention: light-grey
    small labels on the table edges, nothing inside the playing cells beyond
    the glyph / probability text.
    """
    col_headers = "".join(
        f"<th>{c}</th>" for c in range(3)
    )
    header_row = f'<tr><th class="corner"></th>{col_headers}</tr>'

    rows_html = []
    for row in range(3):
        # Layout: action a → col a//3, row a%3. To draw row r left-to-right,
        # we step through cols 0..2 picking cell at action = col*3 + r.
        row_cells = "".join(cells[col * 3 + row] for col in range(3))
        rows_html.append(f'<tr><th class="row-label">{row}</th>{row_cells}</tr>')
    table_html = (
        '<table class="ttt-grid">'
        f'<thead>{header_row}</thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
    )

    if annotation:
        annotation_html = f'<div class="board-annotation">{annotation}</div>'
    else:
        annotation_html = ""

    return f'<div class="{_BOARD_CLASS}">{annotation_html}{table_html}</div>'


# CSS the report should include so the TTT board renders consistently.
TICTACTOE_CSS = """
.ttt-board { display: inline-block; margin: 8px; }
.ttt-board table.ttt-grid { border-collapse: collapse; }
.ttt-board td { padding: 0; }
.ttt-board th {
    padding: 2px 6px;
    font-size: 10px;
    color: #9ca3af;
    background: none;
    border: none;
    font-weight: normal;
    text-align: center;
}
.ttt-board th.corner { width: 16px; }
.ttt-board th.row-label { text-align: right; padding-right: 6px; }
.ttt-board .board-annotation {
    font-size: 12px;
    color: #4b5563;
    text-align: center;
    margin-bottom: 4px;
    font-weight: 600;
}
"""
