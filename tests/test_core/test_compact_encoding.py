"""Validate the compact-board interface seam (R1 of the replay-buffer refactor).

The contract for the ``IBoard.to_compact`` / ``IGame.encode_compact`` pair is:
``game.encode_compact(board.to_compact())`` must be *exactly* equal to
``board.as_multi_channel(1)``. These tests assert that over a spread of positions
for both TicTacToe and Blokus Duo, so storing the compact form and re-encoding
lazily is provably lossless.
"""

from __future__ import annotations

import numpy as np

from games.blokusduo.board import BlokusDuoBoard
from games.blokusduo.game import BlokusDuoGame
from games.tictactoe.board import Board
from games.tictactoe.game import TicTacToeGame


def _play_sequence(game: TicTacToeGame | BlokusDuoGame, board: Board | BlokusDuoBoard, moves: int) -> list:
    """Play up to ``moves`` deterministic moves, collecting every board passed through."""
    boards = [board]
    player = 1
    for _ in range(moves):
        if game.get_game_ended(board, player) != 0:
            break
        valids = game.valid_move_masking(board, player)
        legal = np.where(valids)[0]
        if len(legal) == 0:
            break
        action = int(legal[len(legal) // 2])  # deterministic, not always the first
        board, player = game.get_next_state(board, player, action)
        boards.append(board)
    return boards


def test_ttt_encode_compact_matches_multi_channel(ttt_game: TicTacToeGame) -> None:
    boards = _play_sequence(ttt_game, ttt_game.initialise_board(), moves=9)
    assert len(boards) > 3, "expected a spread of TTT positions"
    for board in boards:
        rebuilt = ttt_game.encode_compact(board.to_compact())
        expected = board.as_multi_channel(1)
        assert np.array_equal(rebuilt, expected)
        assert rebuilt.dtype == expected.dtype


def test_blokus_encode_compact_matches_multi_channel(blokus_game: BlokusDuoGame) -> None:
    boards = _play_sequence(blokus_game, blokus_game.initialise_board(), moves=12)
    assert len(boards) > 3, "expected a spread of Blokus positions"
    for board in boards:
        rebuilt = blokus_game.encode_compact(board.to_compact())
        expected = board.as_multi_channel(1)
        assert np.array_equal(rebuilt, expected)
        assert rebuilt.dtype == expected.dtype


def test_blokus_compact_holds_for_symmetry_augmented_boards(blokus_game: BlokusDuoGame) -> None:
    """get_symmetries returns real board objects; to_compact must work on both."""
    boards = _play_sequence(blokus_game, blokus_game.initialise_board(), moves=6)
    pi = np.zeros(blokus_game.get_action_size(), dtype=np.float32)
    for board in boards:
        for sym_board, _ in blokus_game.get_symmetries(board, pi):
            rebuilt = blokus_game.encode_compact(sym_board.to_compact())
            assert np.array_equal(rebuilt, sym_board.as_multi_channel(1))
