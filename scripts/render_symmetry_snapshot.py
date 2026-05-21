"""Render a Blokus Duo position and its main-diagonal transpose side-by-side.

A visual aid for verifying that ``BlokusDuoBoard.transposed()`` does the
right thing — the kind of bug that pure-numerical assertions can miss
when the coordinate convention undoes the transform somewhere downstream
of the placement grid.

Usage::

    uv run python -m scripts.render_symmetry_snapshot

Writes ``temp/symmetry_snapshot.html`` (a self-contained file safe to
open with ``open temp/symmetry_snapshot.html``). Pairs with the
equivariance tests in ``tests/test_blokusduo/test_symmetry.py`` — those
prove correctness numerically; this lets a human confirm by eye.

Page chrome and CSS follow the existing ``move_count_analysis`` report so
the visual convention stays consistent across reporting surfaces. Each
board has the two starting squares ringed in amber and the main-diagonal
axis of reflection overlaid as a dashed line.
"""
from __future__ import annotations

from pathlib import Path

from games.blokusduo.board import BlokusDuoBoard
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pieces import Orientation
from reporting.display_blokusduo import BOARD_CSS, render_board_html


# Cell sizes used in BOARD_CSS — kept in sync with reporting/display_blokusduo.py.
_CELL_PX = 22
_ROW_LABEL_PX = 16  # approx width of the leftmost row-label column
_COL_LABEL_PX = 16  # approx height of the top column-label row

# Pentobi-aligned starting squares (array indices). Kept here as a local
# constant so the snapshot doesn't have to import them from game.py — keeps
# this script readable as a standalone document.
WHITE_START = (4, 4)
BLACK_START = (9, 9)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    game = BlokusDuoGame(
        pieces_config_path=project_root / "games" / "blokusduo" / "pieces.json",
    )

    board = _build_mid_game_position(game)
    transposed = board.transposed()

    panel_original = _snapshot_panel(
        board, game, label="Original mid-game position",
    )
    panel_transposed = _snapshot_panel(
        transposed, game, label="After BlokusDuoBoard.transposed()",
    )

    html = _wrap(panel_original, panel_transposed)
    out_path = project_root / "temp" / "symmetry_snapshot.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def _build_mid_game_position(game: BlokusDuoGame) -> BlokusDuoBoard:
    """A few pieces per side, mixing orientations to exercise the
    orientation-transpose lookup. Each player's first move is forced onto
    its starting square; follow-ups are picked to bias toward non-Identity
    orientations.
    """
    board = game.initialise_board()
    white_first = next(a for a in game.initial_actions[1] if a.piece_id == 1)
    board = board.with_piece(white_first, 1)
    black_first = next(a for a in game.initial_actions[-1] if a.piece_id == 1)
    board = board.with_piece(black_first, -1)
    for player in (1, -1, 1, -1):
        legal = game._valid_moves(board, player)
        choice = next(
            (m for m in legal if m.orientation != Orientation.Identity),
            legal[0],
        )
        board = board.with_piece(choice, player)
    return board


def _snapshot_panel(
    board: BlokusDuoBoard, game: BlokusDuoGame, *, label: str,
) -> str:
    """Wrap a single rendered board with the dashed-diagonal overlay + an
    amber ring on each starting square.

    Both annotations live in a container that sits above the standard
    ``board-grid`` table — we don't have to modify the shared renderer to
    add them.
    """
    body = render_board_html(
        board=board, game=game, current_player=1, turn=-1, action_desc=label,
        num_moves_white=len(game._valid_moves(board, 1)),
        num_moves_black=len(game._valid_moves(board, -1)),
    )
    n = BlokusDuoBoard.N
    board_px = _CELL_PX * n
    # Line goes from the top-left corner of the (0,0) cell to the bottom-
    # right corner of the (N-1,N-1) cell.
    x1 = _ROW_LABEL_PX
    y1 = _COL_LABEL_PX
    x2 = _ROW_LABEL_PX + board_px
    y2 = _COL_LABEL_PX + board_px
    overlay = f"""
    <svg class="snapshot-overlay" xmlns="http://www.w3.org/2000/svg"
         width="{x2 + 4}" height="{y2 + 4}">
      <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
            stroke="#f59e0b" stroke-width="1.5"
            stroke-dasharray="5 4" stroke-opacity="0.85"/>
    </svg>
    """
    return f'<div class="snapshot-panel">{body}{overlay}</div>'


def _wrap(panel_left: str, panel_right: str) -> str:
    """Page chrome — matches the move_count_analysis report convention so
    the visual style is consistent across reporting surfaces.
    """
    # CSS for the starting-square highlights uses nth-child to target the
    # exact cells. Rows are direct children of <table>: the column-label
    # row is the 1st <tr>, then each data row is the (i+2)-th. Inside a data
    # row the row-label <th> is the 1st child, so data cell (i,j) is at
    # column index j+2.
    wi, wj = WHITE_START
    bi, bj = BLACK_START
    start_square_css = f"""
.snapshot-panel {{ position: relative; display: inline-block; }}
.snapshot-overlay {{
    position: absolute; top: 0; left: 0;
    pointer-events: none;
}}
.snapshot-panel .board-grid tr:nth-child({wi + 2}) td:nth-child({wj + 2}) {{
    box-shadow: inset 0 0 0 3px #f59e0b;
}}
.snapshot-panel .board-grid tr:nth-child({bi + 2}) td:nth-child({bj + 2}) {{
    box-shadow: inset 0 0 0 3px #f59e0b;
}}
.snapshot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Blokus Duo — board transpose snapshot</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1280px; margin: 0 auto; padding: 24px 32px; color: #2a3f5f;
    background: #ffffff;
}}
h1 {{ border-bottom: 2px solid #636efa; padding-bottom: 8px; margin-bottom: 4px; }}
.subtitle {{ color: #6b7280; font-size: 14px; margin-bottom: 24px; }}
h2 {{ margin-top: 32px; color: #636efa; font-size: 20px; }}
.legend {{
    font-size: 12px; color: #4b5563; margin: 16px 0 24px 0;
    padding: 10px 14px; background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 6px;
}}
.legend code {{
    background: #fef3c7; padding: 1px 4px; border-radius: 3px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
{BOARD_CSS}
{start_square_css}
</style>
</head>
<body>
<h1>Blokus Duo — board transpose snapshot</h1>
<p class="subtitle">
  Visual sanity check for <code>BlokusDuoBoard.transposed()</code>. Left
  panel is a hand-built mid-game position; right panel is the same position
  after transposition.
</p>

<div class="legend">
  <strong>What to check:</strong> every piece should be reflected across the
  dashed amber diagonal (top-left → bottom-right of the array grid). The two
  starting squares — <code>(4, 4)</code> for White and <code>(9, 9)</code>
  for Black — are ringed in amber and should stay fixed because they sit on
  the diagonal itself. Placement-point markers (◇ for White, ○ for Black)
  should mirror across the same line.
</div>

<h2>Side-by-side comparison</h2>
<div class="snapshot-grid">
  <div>{panel_left}</div>
  <div>{panel_right}</div>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
