"""Paired colour-swapped arena play + config-selectable gate modes.

Covers docs/plans/fix-arena-colour-pinning.md S1/S3:

- ``sample_opening_prefix`` + ``Arena.play_games_paired`` replay one identical
  opening prefix across both halves of a pair.
- The paired scoring rule (a) maps {both-win, split, both-lose} to the expected
  ``[0, 1]`` values (which — being a linear aggregation of per-pair
  differentials — is the ordinary score over the paired games).
- ``is_accepted`` dispatches the three gate modes correctly.

Real objects throughout: a real TicTacToe game and deterministic function
players (no mocks), so the outcomes are exact and colour-independent where the
assertion needs them to be.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphablokus.evaluation.acceptance import is_accepted, is_accepted_score_rule
from alphablokus.evaluation.arena import Arena
from alphablokus.evaluation.players import sample_opening_prefix
from alphablokus.interfaces import RESIGN_ACTION

if TYPE_CHECKING:
    from alphablokus.games.tictactoe.game import TicTacToeGame
    from alphablokus.interfaces import IBoard


def _first_legal_player(game: TicTacToeGame):
    """Deterministic player: always take the lowest-indexed legal action."""

    def player(board: IBoard) -> int:
        valids = game.valid_move_masking(board, 1)
        return int(next(i for i, v in enumerate(valids) if v))

    return player


def _resigner(board: IBoard) -> int:  # noqa: ARG001 — resigns regardless of the board
    return RESIGN_ACTION


# ---------------------------------------------------------------------------
# S1 — shared opening prefix
# ---------------------------------------------------------------------------


def test_sample_opening_prefix_is_deterministic(ttt_game: TicTacToeGame) -> None:
    """A deterministic sampler yields the deterministic first-k plies."""
    sampler = _first_legal_player(ttt_game)
    assert sample_opening_prefix(ttt_game, sampler, 2) == (0, 1)
    # Non-positive length short-circuits to an empty prefix.
    assert sample_opening_prefix(ttt_game, sampler, 0) == ()


def test_paired_games_replay_identical_prefix(ttt_game: TicTacToeGame) -> None:
    """Both halves of a pair open with the same forced prefix (S1)."""
    player = _first_legal_player(ttt_game)
    arena = Arena(player, player, ttt_game)

    prefix = sample_opening_prefix(ttt_game, _first_legal_player(ttt_game), 2)
    _, _, _, records = arena.play_games_paired(
        num=2,  # one pair
        prefix_sampler=_first_legal_player(ttt_game),
        opening_moves=2,
        record=True,
        top_k=5,
    )
    assert len(records) == 2  # one pair -> two games
    for rec in records:
        opening = tuple(m.action for m in rec.moves[: len(prefix)])
        assert opening == prefix
    # The two games open identically because they replay the same prefix.
    first = tuple(m.action for m in records[0].moves[: len(prefix)])
    second = tuple(m.action for m in records[1].moves[: len(prefix)])
    assert first == second == prefix


# ---------------------------------------------------------------------------
# S1 — paired scoring rule (a)
# ---------------------------------------------------------------------------


def test_paired_scoring_both_win_and_both_lose(ttt_game: TicTacToeGame) -> None:
    """A player that wins both games of every pair scores 1.0; its opponent 0.0.

    The resigner concedes every game regardless of colour, so player1 wins both
    games of each pair (both-win) and the resigner loses both (both-lose).
    """
    arena = Arena(_first_legal_player(ttt_game), _resigner, ttt_game)
    p1_wins, p2_wins, draws, _ = arena.play_games_paired(
        num=4,  # two pairs
        prefix_sampler=_first_legal_player(ttt_game),
        opening_moves=0,
    )
    assert (p1_wins, p2_wins, draws) == (4, 0, 0)
    # Rule (a) == ordinary score over the paired games.
    assert is_accepted_score_rule(p1_wins, p2_wins, draws, threshold=0.5) is True  # score 1.0
    assert is_accepted_score_rule(p2_wins, p1_wins, draws, threshold=0.5) is False  # score 0.0


def test_paired_scoring_split_is_half(ttt_game: TicTacToeGame) -> None:
    """When only colour decides, each pair splits 1-1 and the score is 0.5.

    Two identical deterministic players: whoever moves first wins, so player1
    wins as White (game A) and loses as Black (game B) — a clean split per pair.
    """
    player = _first_legal_player(ttt_game)
    arena = Arena(player, player, ttt_game)
    p1_wins, p2_wins, draws, _ = arena.play_games_paired(
        num=4,  # two pairs
        prefix_sampler=_first_legal_player(ttt_game),
        opening_moves=0,
    )
    assert (p1_wins, p2_wins, draws) == (2, 2, 0)  # split every pair
    score = (p1_wins + 0.5 * draws) / (p1_wins + p2_wins + draws)
    assert score == 0.5


# ---------------------------------------------------------------------------
# S3 — gate modes
# ---------------------------------------------------------------------------


def test_gate_mode_regression_guard() -> None:
    """regression_guard accepts a near-parity 0.49 but rejects a clear 0.40."""
    # 49 wins / 51 losses -> score 0.49, above the 0.48 floor -> accept.
    assert is_accepted("regression_guard", 49, 51, 0, threshold=0.55, guard_floor=0.48) is True
    # 40 wins / 60 losses -> score 0.40, below the floor -> reject.
    assert is_accepted("regression_guard", 40, 60, 0, threshold=0.55, guard_floor=0.48) is False


def test_gate_mode_always_accepts_anything() -> None:
    """always adopts the candidate regardless of the arena result."""
    assert is_accepted("always", 0, 100, 0, threshold=0.55, guard_floor=0.48) is True
    assert is_accepted("always", 0, 0, 0, threshold=0.55, guard_floor=0.48) is True


def test_gate_mode_threshold_matches_legacy_rule() -> None:
    """threshold reproduces the historical ``is_accepted_score_rule`` behaviour."""
    for wins, losses, draws in [(56, 44, 0), (54, 46, 0), (50, 40, 10), (0, 0, 0)]:
        dispatched = is_accepted("threshold", wins, losses, draws, threshold=0.55, guard_floor=0.48)
        legacy = is_accepted_score_rule(wins, losses, draws, threshold=0.55)
        assert dispatched == legacy


def test_gate_modes_reject_when_no_games_except_always() -> None:
    """threshold / regression_guard need games; always does not."""
    assert is_accepted("threshold", 0, 0, 0, threshold=0.55, guard_floor=0.48) is False
    assert is_accepted("regression_guard", 0, 0, 0, threshold=0.55, guard_floor=0.48) is False
    assert is_accepted("always", 0, 0, 0, threshold=0.55, guard_floor=0.48) is True


# ---------------------------------------------------------------------------
# S4 — per-generation colour split
# ---------------------------------------------------------------------------


def test_colour_split_counts_white_and_black_wins() -> None:
    """``_colour_split`` reduces GameRecords to (white_wins, black_wins) (S4a)."""
    from alphablokus.evaluation.arena import GameRecord
    from alphablokus.training.coach import _colour_split

    records = [
        GameRecord(moves=(), outcome=1, player1_was_white=True),  # player1 (White) won -> White
        GameRecord(moves=(), outcome=-1, player1_was_white=True),  # player1 (White) lost -> Black
        GameRecord(moves=(), outcome=1, player1_was_white=False),  # player1 (Black) won -> Black
        GameRecord(moves=(), outcome=-1, player1_was_white=False),  # player1 (Black) lost -> White
        GameRecord(moves=(), outcome=0, player1_was_white=True),  # draw -> neither
    ]
    assert _colour_split(records) == (2, 2)
