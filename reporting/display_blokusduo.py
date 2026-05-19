"""Board state rendering for Blokus Duo — ASCII and HTML.

Implements :class:`~reporting.display.IBoardRenderer` for Blokus Duo via the
:class:`BlokusDuoRenderer` class, plus exposes the older free-function API
kept around for the ``scripts/move_count_analysis.py`` script.
"""
from __future__ import annotations

from collections import defaultdict

from games.blokusduo.board import Action, BlokusDuoBoard, CoordinateIndexDecoder, PlayerSide
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pieces import Orientation

_decoder = CoordinateIndexDecoder(14)


class BlokusDuoRenderer:
    """Renders Blokus Duo boards and candidate moves to HTML.

    Implements :class:`~reporting.display.IBoardRenderer`. Heavy lifting is
    delegated to the module-level :func:`render_board_html` and
    :func:`build_game_replay_html` functions, which predate this protocol.

    The class is a thin namespace adapter so generic reporting code can do
    ``renderer.render_board_html(board)`` without needing Blokus-specific
    metadata or imports.
    """

    def __init__(self, game: BlokusDuoGame | None = None) -> None:
        # Game is needed to count valid moves for the header annotation.
        # Lazily create one with the default pieces config if not provided.
        if game is None:
            from pathlib import Path
            game = BlokusDuoGame(Path("games/blokusduo/pieces.json"))
        self._game = game

    def render_board_html(
        self,
        board: BlokusDuoBoard,
        last_action: int | None = None,
        annotation: str = "",
    ) -> str:
        wm = len(self._game._valid_moves(board, 1))
        bm = len(self._game._valid_moves(board, -1))
        return render_board_html(
            board=board,
            game=self._game,
            current_player=1,
            turn=-1,
            action_desc=annotation,
            num_moves_white=wm,
            num_moves_black=bm,
        )

    def render_top_k_moves_html(
        self,
        board: BlokusDuoBoard,
        actions: list[int],
        probs: list[float],
    ) -> str:
        """Render top-K candidate moves.

        Stub — full thumbnail rendering is part of R25. Returns a text-only
        listing so the interface is complete and the protocol type-checks.
        """
        if not actions:
            return '<div class="board-annotation">No candidate moves.</div>'
        rank_glyphs = ["①", "②", "③", "④", "⑤"]
        lines = []
        for i, (action, prob) in enumerate(zip(actions, probs, strict=False)):
            glyph = rank_glyphs[i] if i < len(rank_glyphs) else f"{i + 1}."
            lines.append(f"<div>{glyph} action {action} — {prob:.0%}</div>")
        return '<div class="blokus-top-k">' + "".join(lines) + "</div>"


# -- ASCII rendering ----------------------------------------------------------


def dump_board(
    game: BlokusDuoGame,
    board: BlokusDuoBoard,
    player: PlayerSide = 1,
    show_moves: bool = True,
    max_moves_per_point: int = 10,
) -> None:
    """Print board state with placement points and legal moves.

    Args:
        game: The game instance (needed for move generation).
        board: Current board state.
        player: Which player's perspective to show (1=White, -1=Black).
        show_moves: Whether to print legal moves grouped by placement point.
        max_moves_per_point: Max moves to list per placement point (0=unlimited).
    """
    board_2d = board.as_2d
    n = BlokusDuoBoard.N
    points = set(board.placement_points(player).keys())
    opp_points = set(board.placement_points(-player).keys())

    player_name = "White" if player == 1 else "Black"
    opp_name = "Black" if player == 1 else "White"

    # Header
    print(f"\n{'=' * 50}")
    print(f"  Perspective: {player_name} (player {player})")
    print(f"  Remaining pieces: {sorted(board.remaining_piece_ids(player))}")
    print(f"  Placement points: {len(points)}")

    if show_moves:
        moves = game._valid_moves(board, player)
        print(f"  Legal moves: {len(moves)}")
        piece_ids_with_moves = sorted({m.piece_id for m in moves})
        print(f"  Pieces with moves: {piece_ids_with_moves}")
    print(f"{'=' * 50}")

    # Board grid
    print(f"\n     {'  '.join(f'{j:2d}' for j in range(n))}")
    print(f"    {'----' * n}-")

    for i in range(n):
        row_chars = []
        for j in range(n):
            cell = board_2d[i, j]
            if cell > 0:
                row_chars.append(" W ")
            elif cell < 0:
                row_chars.append(" B ")
            elif (i, j) in points and (i, j) in opp_points:
                row_chars.append(" * ")
            elif (i, j) in points:
                row_chars.append(f" {'◇' if player == 1 else '○'} ")
            elif (i, j) in opp_points:
                row_chars.append(f" {'○' if player == 1 else '◇'} ")
            else:
                row_chars.append(" · ")
        print(f" {i:2d} |{'|'.join(row_chars)}|")

    print(f"    {'----' * n}-")
    print(f"  Legend: W=White  B=Black  ◇={player_name} point  ○={opp_name} point  *=Both  ·=Empty")

    # Moves grouped by placement point
    if not show_moves:
        return

    moves = game._valid_moves(board, player)
    if not moves:
        print(f"\n  No legal moves for {player_name}.")
        return

    moves_by_point: dict[tuple[int, int], list[Action]] = defaultdict(list)
    for move in moves:
        piece_array = game.piece_manager.get_piece_orientation_array(move.piece_id, move.orientation)
        ins_i, ins_j = _decoder.to_idx((move.x_coordinate, move.y_coordinate))
        p_len, p_wid = piece_array.shape

        for di in range(p_len):
            for dj in range(p_wid):
                if piece_array[di, dj] == 1:
                    pos = (ins_i + di, ins_j + dj)
                    if pos in points:
                        moves_by_point[pos].append(move)
                        break
            else:
                continue
            break

    print(f"\n  Moves by placement point:")
    for (pi, pj) in sorted(moves_by_point.keys()):
        point_moves = moves_by_point[(pi, pj)]
        print(f"\n  ({pi:2d}, {pj:2d}) — {len(point_moves)} moves:")
        display_moves = point_moves[:max_moves_per_point] if max_moves_per_point else point_moves
        for m in display_moves:
            piece_name = game.piece_manager.pieces[m.piece_id].name
            print(f"    piece {m.piece_id:2d} ({piece_name:12s}) {str(m.orientation):8s} at ({m.x_coordinate}, {m.y_coordinate})")
        if max_moves_per_point and len(point_moves) > max_moves_per_point:
            print(f"    ... and {len(point_moves) - max_moves_per_point} more")


# -- HTML rendering -----------------------------------------------------------


def render_board_html(
    board: BlokusDuoBoard,
    game: BlokusDuoGame,
    current_player: int,
    turn: int,
    action_desc: str,
    num_moves_white: int,
    num_moves_black: int,
) -> str:
    """Render a board state as an HTML table with coloured cells."""
    board_2d = board.as_2d
    n = BlokusDuoBoard.N
    white_points = set(board.placement_points(1).keys())
    black_points = set(board.placement_points(-1).keys())

    def cell_style(i: int, j: int) -> str:
        val = board_2d[i, j]
        if val > 0:
            return "background:#636efa;color:white;font-weight:700;"
        elif val < 0:
            return "background:#ef553b;color:white;font-weight:700;"

        in_white = (i, j) in white_points
        in_black = (i, j) in black_points
        if in_white and in_black:
            return "background:#ffd700;"
        elif in_white:
            return "background:#c5cae9;"
        elif in_black:
            return "background:#ffcdd2;"
        return ""

    def cell_text(i: int, j: int) -> str:
        val = board_2d[i, j]
        if val > 0:
            return f"{abs(int(board._piece_placement_board[i, j]))}"
        elif val < 0:
            return f"{abs(int(board._piece_placement_board[i, j]))}"

        in_white = (i, j) in white_points
        in_black = (i, j) in black_points
        if in_white and in_black:
            return "*"
        elif in_white:
            return "\u25c7"
        elif in_black:
            return "\u25cb"
        return "\u00b7"

    player_name = "White" if current_player == 1 else "Black"

    rows_html = ""
    rows_html += "<tr><th></th>"
    for j in range(n):
        rows_html += f"<th>{j}</th>"
    rows_html += "</tr>\n"

    for i in range(n):
        rows_html += f"<tr><th>{i}</th>"
        for j in range(n):
            style = cell_style(i, j)
            text = cell_text(i, j)
            rows_html += f'<td style="{style}">{text}</td>'
        rows_html += "</tr>\n"

    return f"""
    <div class="board-turn">
        <div class="turn-header">
            <strong>Turn {turn}</strong> — {player_name}'s move — {action_desc}
            <span class="move-counts">W:{num_moves_white} moves | B:{num_moves_black} moves</span>
        </div>
        <table class="board-grid">{rows_html}</table>
        <div class="board-legend">
            <span style="background:#636efa;color:white;padding:2px 6px;">W</span> White
            <span style="background:#ef553b;color:white;padding:2px 6px;">B</span> Black
            <span style="background:#c5cae9;padding:2px 6px;">\u25c7</span> White point
            <span style="background:#ffcdd2;padding:2px 6px;">\u25cb</span> Black point
        </div>
    </div>"""


def build_game_replay_html(game: BlokusDuoGame, actions: list[dict], game_id: int) -> str:
    """Build HTML for a single game replay with board at each turn.

    Args:
        game: The game instance.
        actions: List of action dicts from the game (as saved by move_count_analysis).
        game_id: Game identifier for the heading.
    """
    board = game.initialise_board()
    boards_html = []

    wm = len(game._valid_moves(board, 1))
    bm = len(game._valid_moves(board, -1))
    boards_html.append(render_board_html(board, game, 1, -1, "Initial board", wm, bm))

    for action_data in actions:
        turn = action_data["turn"]
        player = action_data["player"]

        if action_data["pass"]:
            action_desc = "Pass"
            wm = len(game._valid_moves(board, 1))
            bm = len(game._valid_moves(board, -1))
            boards_html.append(render_board_html(board, game, player, turn, action_desc, wm, bm))
            continue

        action = Action(
            piece_id=action_data["piece_id"],
            orientation=Orientation(action_data["orientation"]),
            x_coordinate=action_data["x"],
            y_coordinate=action_data["y"],
        )
        piece_name = game.piece_manager.pieces[action.piece_id].name
        action_desc = f"Piece {action.piece_id} ({piece_name}) {action.orientation.value} at ({action.x_coordinate},{action.y_coordinate})"

        board = board.with_piece(action, player_side=player)
        wm = len(game._valid_moves(board, 1))
        bm = len(game._valid_moves(board, -1))
        boards_html.append(render_board_html(board, game, player, turn, action_desc, wm, bm))

    return f"""
    <details>
        <summary>Game {game_id} — {len(actions)} turns</summary>
        <div class="game-replay">{"".join(boards_html)}</div>
    </details>"""


# -- CSS for HTML board grids (include in report <style>) ---------------------

BOARD_CSS = """
.game-replay { display: flex; flex-wrap: wrap; gap: 16px; padding: 12px 0; }
.board-turn { flex: 0 0 auto; }
.turn-header { font-size: 12px; margin-bottom: 4px; color: #2a3f5f; }
.move-counts { color: #6b7280; margin-left: 8px; }
.board-grid {
    border-collapse: collapse; font-size: 11px; font-family: monospace;
}
.board-grid th {
    padding: 2px 4px; font-size: 10px; color: #6b7280;
    background: none; border: none; font-weight: normal;
}
.board-grid td {
    width: 22px; height: 22px; text-align: center; vertical-align: middle;
    border: 1px solid #e5e7eb; padding: 0; font-size: 10px;
}
.board-legend { font-size: 11px; color: #6b7280; margin-top: 4px; }
.board-legend span { border-radius: 2px; margin-right: 4px; font-size: 10px; }
"""
