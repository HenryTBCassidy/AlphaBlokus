"""Full-game correctness tests: net (RandomPlayer stand-in) vs Pentobi via the Arena (H4).

Each game is itself a strong correctness check on the whole pipeline (translation +
adapter + colour wiring): the game can only run to completion if Pentobi accepts every
move we send (else ``GtpError``) and our game accepts every move Pentobi returns (else the
Arena's validity assert fires). On top of that we cross-check the **winner** against
Pentobi's own ``final_score`` — a silent coordinate/colour error would flip it.

Skips when the binary isn't built (build per docs/plans/pentobi-harness.md H2).
"""

from __future__ import annotations

import pytest

from alphablokus.evaluation.arena import Arena
from alphablokus.evaluation.players import RandomPlayer
from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.gtp import find_pentobi_gtp
from alphablokus.games.blokusduo.pentobi.player import PentobiPlayer
from alphablokus.games.blokusduo.pieces import default_pieces_path

_PIECES = default_pieces_path()

pytestmark = pytest.mark.skipif(
    find_pentobi_gtp() is None,
    reason="pentobi-gtp binary not built (set $PENTOBI_GTP_PATH or build per H2)",
)


def _pentobi_winner(score: str) -> int:
    """Map a final_score string to absolute side: +1 White, -1 Black, 0 draw.

    ``B`` (Blue) = our White (+1); ``W`` (the GTP 'white' colour) = our Black (−1).
    A zero margin is a draw.
    """
    score = score.strip()
    if score in ("0", "0.0") or score.endswith("+0"):
        return 0
    if score.startswith("B"):
        return 1
    if score.startswith("W"):
        return -1
    return 0


@pytest.fixture
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=_PIECES)


@pytest.mark.parametrize("pentobi_is_player1", [False, True])
def test_full_game_completes_and_winner_agrees(
    game: BlokusDuoGame,
    pentobi_is_player1: bool,
) -> None:
    """A full RandomPlayer-vs-Pentobi game (both colour assignments) runs to completion
    with no desync, and our outcome matches Pentobi's final_score."""
    net = RandomPlayer(game)
    pentobi = PentobiPlayer(game, level=1, seed=1)
    try:
        p1, p2 = (pentobi, net) if pentobi_is_player1 else (net, pentobi)
        arena = Arena(p1, p2, game)
        outcome, record = arena.play_game(record=True)

        # outcome is player1-perspective; player1 always plays White in play_game,
        # so outcome already reads as absolute side (+1 White won, -1 Black won).
        assert outcome in (-1, 0, 1)
        assert record is not None and len(record.moves) > 0
        assert outcome == _pentobi_winner(pentobi.final_score()), (
            f"winner disagreement: our outcome={outcome}, pentobi final_score={pentobi.final_score()!r}"
        )
    finally:
        pentobi.close()


@pytest.mark.parametrize("pentobi_is_player1", [False, True])
def test_pentobi_resignation_scores_as_net_win(
    game: BlokusDuoGame,
    pentobi_is_player1: bool,
) -> None:
    """When Pentobi resigns, the Arena awards the win to the net — no crash.

    Stubs only the GTP ``genmove`` response to ``resign`` (I/O stubbing, not game-logic
    mocking): the real engine, translator, and Arena game loop all run. Reproduces the
    2026-07-05 crash scenario (Pentobi resigning mid-ladder) and locks the win-crediting."""
    net = RandomPlayer(game)
    pentobi = PentobiPlayer(game, level=1, seed=1)
    pentobi._engine.genmove = lambda color: "resign"  # type: ignore[method-assign]  # noqa: ARG005, SLF001
    try:
        p1, p2 = (pentobi, net) if pentobi_is_player1 else (net, pentobi)
        net_first_outcome, _ = Arena(p1, p2, game).play_game()
        # play_game outcome is player1-perspective; the resigner (Pentobi) always loses.
        expected = -1 if pentobi_is_player1 else 1
        assert net_first_outcome == expected
    finally:
        pentobi.close()


def test_play_games_handles_colour_swap(game: BlokusDuoGame) -> None:
    """play_games swaps player1/player2 between halves; PentobiPlayer must re-infer its
    colour each game. Two games (one as each colour) should both complete cleanly."""
    net = RandomPlayer(game)
    pentobi = PentobiPlayer(game, level=1, seed=1)
    try:
        one, two, draws, records = Arena(net, pentobi, game).play_games(2, record=True)
        assert one + two + draws == 2
        assert len(records) == 2
        assert all(len(r.moves) > 0 for r in records)
    finally:
        pentobi.close()
