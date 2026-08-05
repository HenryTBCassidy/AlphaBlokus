import numpy as np

from alphablokus.evaluation.arena import Arena
from alphablokus.games.tictactoe.game import TicTacToeGame
from alphablokus.interfaces import RESIGN_ACTION, IBoard


def _resigner(board: IBoard) -> int:  # noqa: ARG001 — resigns regardless of the board
    """A player that immediately concedes by returning the resign sentinel."""
    return RESIGN_ACTION


def _random_player(board: IBoard) -> int:
    """A simple random player that picks a random valid action."""
    # For TicTacToe: 9 cells + 1 pass action = 10
    flat = board.as_2d
    valid = np.zeros(10, dtype=int)
    empty = flat == 0
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
    result, _ = arena.play_game()
    # Result should be non-zero (game ended)
    assert result != 0
    # Should be one of: 1 (p1 wins), -1 (p2 wins), or small float (draw)
    assert result in (1, -1) or (0 < abs(result) < 0.01)


def test_play_game_random_vs_random(ttt_game: TicTacToeGame):
    """Two random players should be able to complete a game without errors."""
    arena = Arena(_random_player, _random_player, ttt_game)
    # Play 10 games — none should raise
    for _ in range(10):
        result, _ = arena.play_game()
        assert result != 0


def test_play_games_counts_add_up(ttt_game: TicTacToeGame):
    """wins + losses + draws should equal num // 2 * 2."""
    num = 6
    arena = Arena(_random_player, _random_player, ttt_game)
    wins, losses, draws, _ = arena.play_games(num)

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


def test_resign_scores_as_loss_for_resigner(ttt_game: TicTacToeGame):
    """A resign ends the game immediately with the win going to the opponent.

    Pins the sign convention (outcome from player1's perspective) independent of Pentobi:
    resigner as player1 → -1; resigner as player2 → +1."""
    # Resigner is player1 → player1 loses → outcome -1.
    result, _ = Arena(_resigner, _random_player, ttt_game).play_game()
    assert result == -1

    # Resigner is player2 → player1 wins → outcome +1.
    result, _ = Arena(_random_player, _resigner, ttt_game).play_game()
    assert result == 1


def test_resign_record_is_truncated_and_signed(ttt_game: TicTacToeGame):
    """A recorded resigned game returns a GameRecord: no resign 'move', outcome signed."""
    result, record = Arena(_resigner, _random_player, ttt_game).play_game(record=True)
    assert result == -1
    assert record is not None
    assert record.outcome == -1
    # The resigner resigns on move 1, so no moves were recorded before the concession.
    assert record.moves == ()


def test_resign_colour_swap_correctness(ttt_game: TicTacToeGame):
    """Across the halftime colour swap, an always-resigning player loses every game.

    Resigner is player1: it should win 0 games regardless of which colour it plays."""
    one_won, two_won, draws, _ = Arena(_resigner, _random_player, ttt_game).play_games(4)
    assert one_won == 0  # the resigner never wins
    assert two_won == 4  # the opponent wins all games (both halves)
    assert draws == 0


def test_play_games_by_colour_splits_the_halves(ttt_game: TicTacToeGame):
    """The colour split must be measured, not reconstructed.

    ``play_games`` pools the two halves, and the split is unrecoverable from the pooled
    tally — yet Blokus Duo's first mover takes ~75% of decisive games, so an unbiased
    Elo cannot be fitted without it. An always-resigning player1 makes the expected
    split exact: it loses every game as either colour.
    """
    as_white, as_black, _ = Arena(_resigner, _random_player, ttt_game).play_games_by_colour(6)
    assert (as_white.games, as_white.wins, as_white.losses, as_white.draws) == (3, 0, 3, 0)
    assert (as_black.games, as_black.wins, as_black.losses, as_black.draws) == (3, 0, 3, 0)
    assert as_white.score == 0.0


def test_play_games_by_colour_agrees_with_the_pooled_tally(ttt_game: TicTacToeGame):
    """The split halves must sum to exactly what ``play_games`` reports."""
    np.random.seed(7)
    pooled = Arena(_random_player, _random_player, ttt_game).play_games(8)
    np.random.seed(7)
    as_white, as_black, _ = Arena(_random_player, _random_player, ttt_game).play_games_by_colour(8)
    assert (as_white.wins + as_black.wins) == pooled[0]
    assert (as_white.losses + as_black.losses) == pooled[1]
    assert (as_white.draws + as_black.draws) == pooled[2]
    assert as_white.games + as_black.games == 8


def test_colour_tally_scores_a_draw_as_half():
    from alphablokus.evaluation.arena import ColourTally

    assert ColourTally(games=10, wins=4, losses=4, draws=2).score == 0.5
    assert ColourTally(games=0, wins=0, losses=0, draws=0).score == 0.0


def test_records_are_tagged_with_the_colour_they_were_played_as(ttt_game: TicTacToeGame):
    """``player1_was_white`` still marks the half, so replays stay interpretable."""
    _, _, records = Arena(_random_player, _random_player, ttt_game).play_games_by_colour(4, record=True)
    assert [r.player1_was_white for r in records] == [True, True, False, False]
