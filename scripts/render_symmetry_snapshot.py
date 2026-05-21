"""Render a Blokus Duo position and its main-diagonal transpose side-by-side.

A visual aid for verifying that ``BlokusDuoBoard.transposed()`` does the
right thing — the kind of bug that pure-numerical assertions can miss when
the coordinate convention undoes the transform somewhere downstream of
the placement grid.

Usage::

    uv run python scripts/render_symmetry_snapshot.py

Writes ``temp/symmetry_snapshot.html`` (a self-contained file safe to open
with ``open temp/symmetry_snapshot.html``). Pairs with the equivariance
tests in ``tests/test_blokusduo/test_symmetry.py`` — those prove
correctness numerically; this lets a human confirm by eye.
"""
from __future__ import annotations

from pathlib import Path

from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pieces import Orientation
from reporting.display_blokusduo import render_board_html


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    game = BlokusDuoGame(pieces_config_path=project_root / "games" / "blokusduo" / "pieces.json")

    # Build a mid-game position with a few moves per side. Use the
    # initial-actions list to honour the starting-square constraint, then
    # follow up with legal moves picked to land on a mix of orientations.
    board = game.initialise_board()
    white_first = next(a for a in game.initial_actions[1] if a.piece_id == 1)
    board = board.with_piece(white_first, 1)
    black_first = next(a for a in game.initial_actions[-1] if a.piece_id == 1)
    board = board.with_piece(black_first, -1)
    # Four follow-up plies — two per player. Bias toward non-Identity
    # orientations so the visual demonstrates orientation transposes too.
    for player in (1, -1, 1, -1):
        legal = game._valid_moves(board, player)
        choice = next(
            (m for m in legal if m.orientation != Orientation.Identity),
            legal[0],
        )
        board = board.with_piece(choice, player)

    transposed = board.transposed()

    body_original = render_board_html(
        board=board, game=game, current_player=1, turn=-1,
        action_desc="Original mid-game position",
        num_moves_white=len(game._valid_moves(board, 1)),
        num_moves_black=len(game._valid_moves(board, -1)),
    )
    body_transposed = render_board_html(
        board=transposed, game=game, current_player=1, turn=-1,
        action_desc="Same position after BlokusDuoBoard.transposed()",
        num_moves_white=len(game._valid_moves(transposed, 1)),
        num_moves_black=len(game._valid_moves(transposed, -1)),
    )

    html = _wrap(body_original, body_transposed)

    out_dir = project_root / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "symmetry_snapshot.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def _wrap(left_body: str, right_body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Blokus Duo — board transpose snapshot</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; padding: 24px;
         background: #fafafa; color: #1f2937; }}
  h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
  .subtitle {{ color: #4b5563; font-size: 14px; margin-bottom: 24px; }}
  .pair {{ display: flex; gap: 32px; align-items: flex-start; flex-wrap: wrap; }}
  .panel {{ background: white; padding: 16px; border: 1px solid #e5e7eb;
            border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .panel h2 {{ margin: 0 0 12px 0; font-size: 16px; color: #2a3f5f; }}
</style>
</head>
<body>
<h1>Blokus Duo — main-diagonal transpose snapshot</h1>
<p class="subtitle">
  Left: a hand-built mid-game position. Right: the same position after
  <code>BlokusDuoBoard.transposed()</code>. Verify by eye that the pieces
  have reflected across the main diagonal (top-left → bottom-right) of the
  array grid, that both starting squares stayed in place, and that the
  placement-point dots (◇/○) and danger zones look like the reflection of
  the original.
</p>
<div class="pair">
  <div class="panel"><h2>Original</h2>{left_body}</div>
  <div class="panel"><h2>Transposed</h2>{right_body}</div>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
