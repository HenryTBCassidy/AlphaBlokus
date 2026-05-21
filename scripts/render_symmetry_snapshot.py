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
axis of reflection drawn as a dashed line via per-cell ``::after`` SVG
overlays — anchored to the cells themselves so it always tracks the
diagonal regardless of header-label widths.
"""
from __future__ import annotations

from pathlib import Path

from games.blokusduo.board import BlokusDuoBoard
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pieces import Orientation
from reporting.display_blokusduo import BOARD_CSS, render_board_html


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
    """Wrap a single rendered board for the snapshot. All annotations
    (starting-square rings + dashed-diagonal segments) are applied via CSS
    in :func:`_wrap` — this function only emits the boilerplate container
    that the CSS selectors target.
    """
    body = render_board_html(
        board=board, game=game, current_player=1, turn=-1, action_desc=label,
        num_moves_white=len(game._valid_moves(board, 1)),
        num_moves_black=len(game._valid_moves(board, -1)),
    )
    return f'<div class="snapshot-panel">{body}</div>'


def _wrap(panel_left: str, panel_right: str) -> str:
    """Page chrome — matches the move_count_analysis report convention so
    the visual style is consistent across reporting surfaces.

    All cell-level annotations (starting-square rings, diagonal overlay)
    are applied via CSS using ``nth-child`` to target specific table cells.
    Rows are direct children of ``<table>``: the column-label row is the
    1st ``<tr>``, then each data row is the ``(i + 2)``-th. Inside a data
    row the row-label ``<th>`` is the 1st child, so data cell ``(i, j)``
    is at column index ``j + 2``. We bake the selectors out per row/col
    explicitly so the result is robust against header-label width drift.
    """
    n = BlokusDuoBoard.N
    wi, wj = WHITE_START
    bi, bj = BLACK_START

    # Each diagonal cell hosts a ``::after`` pseudo-element holding a
    # dashed amber line drawn from its top-left corner to its bottom-right
    # corner via an inline SVG. Stacked across all 14 diagonal cells the
    # segments form a continuous dashed line down the main diagonal. We
    # use a pseudo-element rather than a background-image so the existing
    # piece / placement-point colors painted inline on each cell stay
    # visible underneath.
    diagonal_svg = (
        "data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 22 22' "
        "preserveAspectRatio='none'>"
        "<line x1='0' y1='0' x2='22' y2='22' "
        "stroke='%23f59e0b' stroke-width='1.8' stroke-dasharray='4 3' "
        "stroke-opacity='0.95'/></svg>"
    )

    diagonal_position_selectors = ",\n".join(
        f".snapshot-panel .board-grid tr:nth-child({i + 2}) td:nth-child({i + 2})"
        for i in range(n)
    )
    diagonal_after_selectors = ",\n".join(
        f".snapshot-panel .board-grid tr:nth-child({i + 2}) td:nth-child({i + 2})::after"
        for i in range(n)
    )

    annotation_css = f"""
.snapshot-panel {{ display: inline-block; }}
.snapshot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}

/* Cells on the main diagonal need to host a positioned pseudo-element. */
{diagonal_position_selectors} {{
    position: relative;
}}
/* Each diagonal cell's ::after carries a single 22-px dashed segment.
   Stitching them gives a continuous dashed line along the diagonal. */
{diagonal_after_selectors} {{
    content: "";
    position: absolute;
    inset: 0;
    background-image: url("{diagonal_svg}");
    background-size: 100% 100%;
    background-repeat: no-repeat;
    pointer-events: none;
}}

/* Starting squares ringed in amber. The inset box-shadow sits inside the
   existing cell border so the layout doesn't shift. */
.snapshot-panel .board-grid tr:nth-child({wi + 2}) td:nth-child({wj + 2}),
.snapshot-panel .board-grid tr:nth-child({bi + 2}) td:nth-child({bj + 2}) {{
    box-shadow: inset 0 0 0 3px #f59e0b;
}}
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
{annotation_css}
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
