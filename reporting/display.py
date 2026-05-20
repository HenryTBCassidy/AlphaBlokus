"""Game-agnostic board rendering protocol + factory.

This module exposes the :class:`IBoardRenderer` protocol that each
game-specific renderer implements (`display_tictactoe.py`,
`display_blokusduo.py`, etc), plus a :func:`get_renderer` factory that maps
a game name to its renderer instance.

The renderers themselves do all the actual drawing — this file is the thin
abstraction layer so reporting code can render any game without importing
game-specific symbols.
"""
from __future__ import annotations

from typing import Protocol

from core.interfaces import IBoard


class IBoardRenderer(Protocol):
    """Renders boards and candidate moves to HTML for the training report.

    All implementations must:
    1. Accept ``IBoard`` instances (not raw arrays) — every game has a
       different multi-channel encoding and the board object knows its own.
    2. Produce HTML fragments — strings, not files — so the caller can embed
       them anywhere in the report.
    3. Be stateless / safe to instantiate many times.

    Implementations live in ``reporting/display_<game>.py`` files alongside
    this protocol, and are wired up in :func:`get_renderer`.
    """

    def render_board_html(
        self,
        board: IBoard,
        last_action: int | None = None,
        annotation: str = "",
    ) -> str:
        """Render a single board state as an HTML fragment.

        Args:
            board: The board state to draw.
            last_action: If provided, the action index that was just played.
                Renderers should highlight this cell/piece.
            annotation: Optional caption text (e.g. "Turn 5 — White's move").
        """
        ...

    def render_policy_html(
        self,
        board: IBoard,
        action_probs: dict[int, float],
        annotation: str = "",
    ) -> str:
        """Render the board with the player's policy probabilities overlaid.

        The resulting HTML lines up cell-for-cell with the output of
        :meth:`render_board_html` for the same board — they're designed to
        sit side-by-side and look like the "same" board showing different
        things (one the actual position, one the policy).

        Args:
            board: The board state at the moment the policy was produced
                (i.e. before the move was played).
            action_probs: Mapping action-index → probability. Cells whose
                action is not in this dict are treated as 0%.
            annotation: Caption text shown above the table.
        """
        ...

    def render_top_k_moves_html(
        self,
        board: IBoard,
        actions: list[int],
        probs: list[float],
    ) -> str:
        """[Deprecated] Render top-K candidate moves as a small HTML fragment.

        Older method, kept around so Blokus's stub implementation still
        type-checks. New code should prefer :meth:`render_policy_html` which
        takes the full action distribution and renders a board that lines up
        with :meth:`render_board_html`.
        """
        ...


def get_renderer(game_name: str) -> IBoardRenderer:
    """Return the renderer for the named game.

    Local imports to keep ``reporting/display.py`` free of game-specific
    dependencies — the factory only loads what's asked for.
    """
    if game_name == "tictactoe":
        from reporting.display_tictactoe import TicTacToeRenderer
        return TicTacToeRenderer()
    if game_name == "blokusduo":
        from reporting.display_blokusduo import BlokusDuoRenderer
        return BlokusDuoRenderer()
    raise ValueError(f"No renderer registered for game {game_name!r}")
