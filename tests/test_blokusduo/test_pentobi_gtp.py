"""Integration tests for the Pentobi GTP adapter (H3).

These spawn the *real* ``pentobi-gtp`` binary, so they skip when it isn't built
(CI / other machines). Build it per docs/plans/pentobi-harness.md (H2) or set
``$PENTOBI_GTP_PATH`` to run them.
"""
from __future__ import annotations

import pytest

from games.blokusduo.pentobi_gtp import GtpError, PentobiGtp, find_pentobi_gtp

pytestmark = pytest.mark.skipif(
    find_pentobi_gtp() is None,
    reason="pentobi-gtp binary not built (set $PENTOBI_GTP_PATH or build per H2)",
)


def test_starts_and_identifies() -> None:
    """Construction spawns the engine and verifies 'name' == 'Pentobi'."""
    with PentobiGtp(level=1, seed=1) as engine:
        assert engine.send("name") == "Pentobi"
        assert engine.send("version").startswith("31")


def test_play_genmove_and_score() -> None:
    """A short scripted game: place both start pieces, generate a reply, score it."""
    with PentobiGtp(level=1, seed=1) as engine:
        engine.clear_board()
        engine.play("b", "e10")   # our White's start square
        engine.play("w", "j5")    # our Black's start square
        board = engine.showboard()
        assert "X" in board and "O" in board  # both pieces on the board
        move = engine.genmove("b")
        assert move and move != "pass"  # blue has plenty of legal moves here
        assert all(c.isalnum() or c == "," for c in move)  # cell-list shape, e.g. "h7,g8,..."
        score = engine.final_score()
        assert score[0] in ("B", "W") or score == "0"  # e.g. "B+5"


def test_play_pass_is_rejected() -> None:
    """Pentobi has no ``play <c> pass`` — passes are expressed only via ``genmove``
    returning ``pass``. A play-pass always errors; the adapter surfaces it as GtpError.
    (PentobiPlayer relies on this: it never sends a pass, it just skips telling the
    engine, since a pass places nothing.)"""
    with PentobiGtp(level=1, seed=1) as engine, pytest.raises(GtpError):
        engine.clear_board()
        engine.play("b", "pass")


def test_bad_command_raises() -> None:
    with PentobiGtp(level=1, seed=1) as engine, pytest.raises(GtpError):
        engine.send("not_a_real_gtp_command")


def test_close_is_idempotent() -> None:
    engine = PentobiGtp(level=1, seed=1)
    engine.close()
    engine.close()  # second close must be a no-op, not an error
