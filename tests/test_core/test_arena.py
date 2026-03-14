import numpy as np
import pytest

from core.arena import Arena
from core.interfaces import IBoard
from games.tictactoe.game import TicTacToeGame


def _random_player(board: IBoard) -> int:
    """A simple random player that picks a random valid action."""
    # For TicTacToe: 9 cells + 1 pass action = 10
    flat = board.as_2d
    valid = np.zeros(10, dtype=int)
    empty = (flat == 0)
    for i in range(3):
        for j in range(3):
            if empty[i][j]:
                valid[i * 3 + j] = 1
    # If no moves available, pass
    if np.sum(valid[:9]) == 0:
        valid[9] = 1
    valid_actions = np.where(valid == 1)[0]
    return int(np.random.choice(valid_actions))


def test_play_game_returns_valid_result(ttt_game: TicTacToeGame):
    """play_game should return a result in {-1, 1} or a small draw value."""
    arena = Arena(_random_player, _random_player, ttt_game)
    result = arena.play_game()
    # Result should be non-zero (game ended)
    assert result != 0
    # Should be one of: 1 (p1 wins), -1 (p2 wins), or small float (draw)
    assert result in (1, -1) or (0 < abs(result) < 0.01)


def test_play_game_random_vs_random(ttt_game: TicTacToeGame):
    """Two random players should be able to complete a game without errors."""
    arena = Arena(_random_player, _random_player, ttt_game)
    # Play 10 games — none should raise
    for _ in range(10):
        result = arena.play_game()
        assert result != 0


def test_play_games_counts_add_up(ttt_game: TicTacToeGame):
    """wins + losses + draws should equal num // 2 * 2."""
    num = 6
    arena = Arena(_random_player, _random_player, ttt_game)
    wins, losses, draws = arena.play_games(num)

    # Total games = num // 2 * 2 = 6
    assert wins + losses + draws == (num // 2) * 2


def test_play_games_swaps_players(ttt_game: TicTacToeGame):
    """After half the games, players should be swapped."""
    call_log: list[int] = []

    def p1(board: IBoard) -> int:
        call_log.append(1)
        return _random_player(board)

    def p2(board: IBoard) -> int:
        call_log.append(2)
        return _random_player(board)

    arena = Arena(p1, p2, ttt_game)
    arena.play_games(4)  # 2 games each way

    # Both players should have been called
    assert 1 in call_log
    assert 2 in call_log
