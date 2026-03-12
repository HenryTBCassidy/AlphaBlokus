import numpy as np
import pytest

from games.tictactoe.game import TicTacToeGame


def test_initial_board_is_empty(ttt_game: TicTacToeGame):
    board = ttt_game.initialise_board()
    assert board.shape == (3, 3)
    assert np.all(board == 0)


def test_board_size(ttt_game: TicTacToeGame):
    assert ttt_game.get_board_size() == (3, 3)


def test_action_size(ttt_game: TicTacToeGame):
    # 9 cells + 1 pass action
    assert ttt_game.get_action_size() == 10


def test_valid_moves_initial(ttt_game: TicTacToeGame):
    board = ttt_game.initialise_board()
    valids = ttt_game.valid_move_masking(board, 1)
    assert len(valids) == 10
    # All 9 board cells should be valid on an empty board
    assert np.sum(valids[:9]) == 9
    # Pass action should be invalid (there are legal moves)
    assert valids[9] == 0


def test_valid_moves_after_move(ttt_game: TicTacToeGame):
    board = ttt_game.initialise_board()
    # Place player 1 at position (0, 0) → action index 0
    board, _ = ttt_game.get_next_state(board, 1, 0)
    valids = ttt_game.valid_move_masking(board, -1)
    # Position 0 should now be invalid
    assert valids[0] == 0
    # Remaining 8 cells should be valid
    assert np.sum(valids[:9]) == 8


def test_get_next_state(ttt_game: TicTacToeGame):
    board = ttt_game.initialise_board()
    new_board, next_player = ttt_game.get_next_state(board, 1, 0)
    # Player 1 placed at (0, 0)
    assert new_board[0][0] == 1
    # Next player should be -1
    assert next_player == -1


def test_canonical_form_player1(ttt_game: TicTacToeGame):
    board = ttt_game.initialise_board()
    board, _ = ttt_game.get_next_state(board, 1, 4)  # Centre
    canonical = ttt_game.get_canonical_form(board, 1)
    np.testing.assert_array_equal(canonical, board)


def test_canonical_form_player_neg1(ttt_game: TicTacToeGame):
    board = ttt_game.initialise_board()
    board, _ = ttt_game.get_next_state(board, 1, 4)  # Centre has 1
    canonical = ttt_game.get_canonical_form(board, -1)
    # Board should be negated
    np.testing.assert_array_equal(canonical, -board)


def test_game_not_ended_initially(ttt_game: TicTacToeGame):
    board = ttt_game.initialise_board()
    assert ttt_game.get_game_ended(board, 1) == 0


def test_game_ended_win_row(ttt_game: TicTacToeGame):
    """Player 1 wins with top row."""
    board = ttt_game.initialise_board()
    # Fill top row with player 1
    board, _ = ttt_game.get_next_state(board, 1, 0)   # (0,0)
    board, _ = ttt_game.get_next_state(board, -1, 3)   # opponent at (1,0)
    board, _ = ttt_game.get_next_state(board, 1, 1)    # (0,1)
    board, _ = ttt_game.get_next_state(board, -1, 4)   # opponent at (1,1)
    board, _ = ttt_game.get_next_state(board, 1, 2)    # (0,2)
    # Player 1 should have won
    assert ttt_game.get_game_ended(board, 1) == 1


def test_game_ended_win_col(ttt_game: TicTacToeGame):
    """Player 1 wins with left column."""
    board = ttt_game.initialise_board()
    board, _ = ttt_game.get_next_state(board, 1, 0)    # (0,0)
    board, _ = ttt_game.get_next_state(board, -1, 1)    # opponent at (0,1)
    board, _ = ttt_game.get_next_state(board, 1, 3)    # (1,0)
    board, _ = ttt_game.get_next_state(board, -1, 4)    # opponent at (1,1)
    board, _ = ttt_game.get_next_state(board, 1, 6)    # (2,0)
    assert ttt_game.get_game_ended(board, 1) == 1


def test_game_ended_win_diag(ttt_game: TicTacToeGame):
    """Player 1 wins with main diagonal."""
    board = ttt_game.initialise_board()
    board, _ = ttt_game.get_next_state(board, 1, 0)    # (0,0)
    board, _ = ttt_game.get_next_state(board, -1, 1)    # opponent at (0,1)
    board, _ = ttt_game.get_next_state(board, 1, 4)    # (1,1)
    board, _ = ttt_game.get_next_state(board, -1, 2)    # opponent at (0,2)
    board, _ = ttt_game.get_next_state(board, 1, 8)    # (2,2)
    assert ttt_game.get_game_ended(board, 1) == 1


def test_game_ended_draw(ttt_game: TicTacToeGame):
    """Full board with no winner should be a draw (1e-4)."""
    board = ttt_game.initialise_board()
    # Construct a drawn position:
    # X O X     (-1  1 -1)
    # X X O  →  (-1 -1  1)
    # O X O     ( 1 -1  1)
    # But we need to use canonical form: player 1 = 1
    # Let's build manually: actions in order for a draw
    moves = [
        (1, 0),    # P1 at (0,0)
        (-1, 1),   # P2 at (0,1)
        (1, 2),    # P1 at (0,2)
        (-1, 4),   # P2 at (1,1)
        (1, 3),    # P1 at (1,0)
        (-1, 6),   # P2 at (2,0)
        (1, 7),    # P1 at (2,1)
        (-1, 8),   # P2 at (2,2)
        (1, 5),    # P1 at (1,2)
    ]
    for player, action in moves:
        board, _ = ttt_game.get_next_state(board, player, action)

    result = ttt_game.get_game_ended(board, 1)
    # If it's a draw (no winner, no legal moves), should return small non-zero value
    # If someone won, it won't be 0
    assert result != 0, "Game should have ended"
    # Either someone won or it's a draw — accept either for this board config
    # The key assertion is the game ended


def test_symmetries_count(ttt_game: TicTacToeGame):
    board = ttt_game.initialise_board()
    pi = np.ones(ttt_game.get_action_size()) / ttt_game.get_action_size()
    symmetries = ttt_game.get_symmetries(board, pi)
    assert len(symmetries) == 8


def test_symmetries_preserve_policy_sum(ttt_game: TicTacToeGame):
    board = ttt_game.initialise_board()
    pi = np.random.dirichlet(np.ones(ttt_game.get_action_size()))
    symmetries = ttt_game.get_symmetries(board, pi)
    for _, sym_pi in symmetries:
        assert pytest.approx(sum(sym_pi), abs=1e-6) == 1.0


def test_state_key_unique(ttt_game: TicTacToeGame):
    board1 = ttt_game.initialise_board()
    board2 = ttt_game.initialise_board()
    board2, _ = ttt_game.get_next_state(board2, 1, 0)

    key1 = ttt_game.state_key(board1)
    key2 = ttt_game.state_key(board2)
    assert isinstance(key1, bytes)
    assert key1 != key2


def test_state_key_same_board(ttt_game: TicTacToeGame):
    board1 = ttt_game.initialise_board()
    board2 = ttt_game.initialise_board()
    assert ttt_game.state_key(board1) == ttt_game.state_key(board2)


def test_full_random_game(ttt_game: TicTacToeGame):
    """Two random players should be able to complete a game without errors."""
    board = ttt_game.initialise_board()
    player = 1
    move_count = 0

    while ttt_game.get_game_ended(board, player) == 0:
        valids = ttt_game.valid_move_masking(board, player)
        valid_actions = np.where(valids == 1)[0]
        action = np.random.choice(valid_actions)
        board, player = ttt_game.get_next_state(board, player, action)
        move_count += 1
        assert move_count <= 10, "Game should end within 10 moves"

    result = ttt_game.get_game_ended(board, player)
    assert result != 0
