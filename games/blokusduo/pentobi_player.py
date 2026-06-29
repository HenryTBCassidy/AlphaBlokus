"""Pentobi as an Arena ``Player`` (H4).

Wraps the GTP adapter (H3) + the move translation (H1) so a ``pentobi-gtp`` engine
plugs into the existing :class:`core.arena.Arena` like any other player. Net-vs-Pentobi
games then run through the *same* machinery as the training arena — including
``GameRecord`` capture and the replay rendering — with no new game-loop code.

Implements the Player contract (``__call__(board) -> action_index``) plus the Arena's
optional ``startGame`` / ``notify`` / ``endGame`` hooks.

**Colour inference.** ``Arena.play_games`` swaps player1/player2 between halves, so this
object plays *both* colours across a match and can't fix its colour at construction.
Instead it infers per game from the first event (whoever moves first is White):
- if our ``__call__`` fires before any ``notify`` → we move first → we're White (GTP ``b``);
- if ``notify`` fires first → the opponent moved first → we're Black (GTP ``w``).
(White=+1 ↔ ``b`` is the mapping pinned in H2.)

**Correctness.** ``canonical()`` is colour-only (no coordinate change), so every action
index carries absolute coords and H1 translates directly. Two always-on desync guards:
the engine raises ``GtpError`` if we ``play`` a move it considers illegal, and the Arena
asserts every move we return is legal in our own game. A mismatch surfaces immediately
rather than silently corrupting the result.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from games.blokusduo.pentobi_gtp import PentobiGtp
from games.blokusduo.pentobi_translation import PASS, PentobiMoveTranslator

if TYPE_CHECKING:
    from core.interfaces import IBoard
    from games.blokusduo.game import BlokusDuoGame


class PentobiPlayer:
    """An Arena ``Player`` backed by a single ``pentobi-gtp`` engine process."""

    def __init__(
        self,
        game: BlokusDuoGame,
        level: int,
        *,
        binary: str | None = None,
        threads: int = 1,
        seed: int | None = None,
    ) -> None:
        self._translator = PentobiMoveTranslator(game)
        self._engine = PentobiGtp(level, binary=binary, threads=threads, seed=seed)
        self._my_color: str | None = None
        self._opp_color: str | None = None

    def _assign_colors(self, i_am_white: bool) -> None:
        if self._my_color is None:
            self._my_color, self._opp_color = ("b", "w") if i_am_white else ("w", "b")

    # -- Arena hooks -----------------------------------------------------------

    def startGame(self) -> None:  # noqa: N802 — Arena hook name
        """Reset for a new game (called by the Arena before each game)."""
        self._engine.clear_board()
        self._my_color = None
        self._opp_color = None

    def notify(self, board: IBoard, action: int) -> None:  # noqa: ARG002 — board unused
        """Opponent just played ``action`` — relay it to the engine.

        First ``notify`` before our first move ⇒ the opponent moved first ⇒ we're Black.
        The action index carries absolute coords (canonical is colour-only), so H1
        translates it directly with no board needed.

        A **pass** is *not* relayed: Pentobi has no ``play <c> pass`` (it expresses passes
        only via ``genmove`` returning ``pass``), and a pass places nothing on the board,
        so skipping it keeps both boards in sync. (Verified: Pentobi генmoves the other
        colour fine without being told about the opponent's pass.)
        """
        self._assign_colors(i_am_white=False)
        move = self._translator.action_index_to_pentobi(int(action))
        if move == PASS:
            return
        self._engine.play(self._opp_color, move)  # GtpError here = desync

    def endGame(self) -> None:  # noqa: N802 — Arena hook name
        """No-op: keep the engine alive across games; ``startGame`` resets it."""

    # -- Player contract -------------------------------------------------------

    def __call__(self, canonical_board: IBoard) -> int:  # noqa: ARG002 — engine tracks state
        """Our turn: ask the engine for a move and translate it to an action index.

        First call before any ``notify`` ⇒ we move first ⇒ we're White.
        """
        self._assign_colors(i_am_white=True)
        return self._translator.pentobi_to_action_index(self._engine.genmove(self._my_color))

    # -- lifecycle / helpers ---------------------------------------------------

    def final_score(self) -> str:
        """Engine's final score string (``B+N`` = our White; ``W+N`` = our Black)."""
        return self._engine.final_score()

    def close(self) -> None:
        self._engine.close()

    def __enter__(self) -> PentobiPlayer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
