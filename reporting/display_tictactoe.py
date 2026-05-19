"""HTML rendering for TicTacToe — implements :class:`IBoardRenderer`."""
from __future__ import annotations

from games.tictactoe.board import Board


_CELL_SIZE_PX = 48
_BOARD_CSS_CLASS = "ttt-board"


class TicTacToeRenderer:
    """Renders a 3×3 TTT board as an HTML table with X/O glyphs.

    Empty cells show their action index so a reader can correlate a "model
    played 4" with the centre cell. The most recent action is highlighted
    with a soft accent background.
    """

    def render_board_html(
        self,
        board: Board,
        last_action: int | None = None,
        annotation: str = "",
    ) -> str:
        arr = board.as_2d  # shape (3, 3); arr[col, row]
        cells: list[str] = []
        for action in range(9):
            col, row = divmod(action, 3)
            v = int(arr[col, row])
            is_last = action == last_action

            if v == 1:
                glyph, fg = "X", "#1f2937"
                bg = "#dbeafe" if is_last else "#f3f4f6"
            elif v == -1:
                glyph, fg = "O", "#1f2937"
                bg = "#fecaca" if is_last else "#f3f4f6"
            else:
                glyph, fg = str(action), "#9ca3af"
                bg = "#ffffff"
            cells.append(_cell_html(glyph, bg, fg))

        rows_html = ""
        for row in range(3):
            row_cells = "".join(cells[row::3])  # actions 0,3,6 then 1,4,7 then 2,5,8
            rows_html += f"<tr>{row_cells}</tr>"

        annotation_html = (
            f'<div class="board-annotation">{annotation}</div>' if annotation else ""
        )
        return (
            f'<div class="{_BOARD_CSS_CLASS}">'
            f"{annotation_html}"
            f'<table>{rows_html}</table>'
            f"</div>"
        )

    def render_top_k_moves_html(
        self,
        board: Board,
        actions: list[int],
        probs: list[float],
    ) -> str:
        """Render TTT top-K as a mirror board with rank badges on chosen cells.

        For each top-K action we overlay a ranked badge ``① ② ③ …`` on its
        cell along with the probability percentage. Cells not in top-K stay
        as their normal action-index hint.
        """
        arr = board.as_2d
        rank_glyphs = ["①", "②", "③", "④", "⑤"]
        ranks: dict[int, tuple[str, float]] = {
            a: (rank_glyphs[i] if i < len(rank_glyphs) else f"{i + 1}.", p)
            for i, (a, p) in enumerate(zip(actions, probs, strict=False))
        }

        cells: list[str] = []
        for action in range(9):
            col, row = divmod(action, 3)
            v = int(arr[col, row])

            if v == 1:
                cells.append(_cell_html("X", "#f3f4f6", "#1f2937"))
                continue
            if v == -1:
                cells.append(_cell_html("O", "#f3f4f6", "#1f2937"))
                continue

            if action in ranks:
                glyph, prob = ranks[action]
                cells.append(_cell_html(
                    f'<div style="font-size:14px;">{glyph}</div>'
                    f'<div style="font-size:10px;color:#4b5563;">{prob:.0%}</div>',
                    bg="#fef3c7", fg="#1f2937",
                ))
            else:
                cells.append(_cell_html(str(action), "#ffffff", "#d1d5db"))

        rows_html = ""
        for row in range(3):
            row_cells = "".join(cells[row::3])
            rows_html += f"<tr>{row_cells}</tr>"
        return (
            f'<div class="{_BOARD_CSS_CLASS} top-k-overlay">'
            f"<table>{rows_html}</table>"
            f"</div>"
        )


def _cell_html(content: str, bg: str, fg: str) -> str:
    return (
        f'<td style="background:{bg};color:{fg};'
        f"width:{_CELL_SIZE_PX}px;height:{_CELL_SIZE_PX}px;"
        f'text-align:center;vertical-align:middle;'
        f'border:1px solid #d1d5db;font-weight:600;font-size:18px;'
        f'font-family:monospace;">{content}</td>'
    )


# CSS the report should include so the TTT board renders consistently.
TICTACTOE_CSS = """
.ttt-board { display: inline-block; margin: 8px; }
.ttt-board table { border-collapse: collapse; }
.ttt-board td { padding: 0; }
.ttt-board .board-annotation {
    font-size: 12px;
    color: #4b5563;
    text-align: center;
    margin-bottom: 4px;
}
"""
