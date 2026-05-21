"""Tests for the TicTacToe perfect-play minimax opponent.

TTT is a forced draw under perfect play, so the minimax player is supposed to
*never* lose to anything. These tests defend that invariant — an earlier
implementation had an alpha-beta + memoisation bug that occasionally lost to
random opponents (~1 in 200 games), which is exactly the kind of subtle
regression we want a stress test to catch.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.arena import Arena
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.minimax import MinimaxTicTacToePlayer


class _RandomPlayer:
    def __init__(self, game: TicTacToeGame) -> None:
        self._game = game

    def __call__(self, board) -> int:  # type: ignore[no-untyped-def]
        valids = self._game.valid_move_masking(board, 1)
        return int(np.random.choice(np.flatnonzero(valids)))


@pytest.fixture
def game() -> TicTacToeGame:
    return TicTacToeGame()


def test_minimax_never_loses_to_random(game: TicTacToeGame) -> None:
    """Stress test: 200 games vs random, zero losses for minimax."""
    np.random.seed(0)
    arena = Arena(MinimaxTicTacToePlayer(game), _RandomPlayer(game), game)
    minimax_wins, random_wins, draws, _ = arena.play_games(200)
    assert random_wins == 0, (
        f"Minimax should never lose to a random player in TTT, "
        f"but random won {random_wins} games (W={minimax_wins}, L={random_wins}, D={draws})."
    )


def test_minimax_vs_minimax_always_draws(game: TicTacToeGame) -> None:
    """Two perfect-play agents must always draw (TTT is a forced draw)."""
    arena = Arena(MinimaxTicTacToePlayer(game), MinimaxTicTacToePlayer(game), game)
    wins, losses, draws, _ = arena.play_games(20)
    assert wins == 0 and losses == 0 and draws == 20, (
        f"Minimax-vs-minimax must always draw, got W{wins} L{losses} D{draws}."
    )


def test_minimax_blocks_immediate_loss(game: TicTacToeGame) -> None:
    """When the opponent threatens an immediate win, minimax must block."""
    # Board state: X has two in the top row, O (us, in canonical) must block.
    # Action layout: action a places at column a//3, row a%3.
    # X cells: action 0 (col 0,row 0) and action 3 (col 1,row 0)
    # Threat: X would win with action 6 (col 2,row 0). O must play 6 to block.
    board = game.initialise_board()
    # X plays action 0 (top-left), O plays elsewhere, X plays action 3 (top-middle), now O must block.
    board, _ = game.get_next_state(board, 1, 0)   # X
    board, _ = game.get_next_state(board, -1, 4)  # O centre — doesn't matter where
    board, _ = game.get_next_state(board, 1, 3)   # X creates threat on top row
    canonical = game.get_canonical_form(board, -1)  # O's turn

    minimax = MinimaxTicTacToePlayer(game)
    chosen = minimax(canonical)
    # action 6 is the blocking cell (col 2, row 0).
    assert chosen == 6, f"Minimax should block at action 6, but chose {chosen}."
