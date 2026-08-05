"""Tests for the colour-conditional value diagnostic.

The skill tests are built on predictors whose skill is known by construction: a
predictor that outputs exactly each colour's mean outcome must score ~0 (it *is*
the baseline), a perfect predictor must score ~1, and a predictor that ignores the
position while being miscalibrated must score below 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphablokus.evaluation.colour_value import (
    BLACK_TO_MOVE,
    COLOUR_UNKNOWN,
    PHASE_EARLY,
    PHASE_LATE,
    PHASE_MID,
    WHITE_TO_MOVE,
    compute_colour_value_diagnostic,
    game_phase,
    infer_mover_colour,
)

N = 14


def _board(mine: int, theirs: int, *, size: int = N) -> np.ndarray:
    """A compact board with ``mine``/``theirs`` distinct piece ids placed.

    Piece ids are positive for the side to move, negative for the opponent —
    the canonical sign convention ``IBoard.to_compact`` produces.
    """
    board = np.zeros((size, size), dtype=np.int8)
    cell = 0
    for piece_id in range(1, mine + 1):
        board[cell // size, cell % size] = piece_id
        cell += 1
    for piece_id in range(1, theirs + 1):
        board[cell // size, cell % size] = -piece_id
        cell += 1
    return board


# --- colour inference -----------------------------------------------------


def test_equal_piece_counts_means_white_to_move() -> None:
    assert infer_mover_colour(_board(3, 3)) == WHITE_TO_MOVE
    assert infer_mover_colour(_board(0, 0)) == WHITE_TO_MOVE


def test_opponent_one_ahead_means_black_to_move() -> None:
    assert infer_mover_colour(_board(2, 3)) == BLACK_TO_MOVE


def test_broken_parity_is_unknown() -> None:
    """After a pass the parity breaks and the colour is not recoverable."""
    assert infer_mover_colour(_board(5, 3)) == COLOUR_UNKNOWN
    assert infer_mover_colour(_board(2, 5)) == COLOUR_UNKNOWN


# --- phase ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("mine", "theirs", "expected"),
    [
        (1, 1, PHASE_EARLY),
        (7, 7, PHASE_EARLY),
        (8, 7, PHASE_MID),
        (14, 14, PHASE_MID),
        (15, 14, PHASE_LATE),
        (21, 21, PHASE_LATE),
    ],
)
def test_phase_buckets_by_pieces_placed(mine: int, theirs: int, expected: str) -> None:
    assert game_phase(_board(mine, theirs)) == expected


# --- synthetic population -------------------------------------------------


def _population(
    n_games: int = 40,
    per_game: int = 4,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Games alternating White/Black to move, one shared outcome per game.

    White-to-move positions win 80% of the time and Black-to-move lose 80% — the
    large first-mover skew that makes the colour prior such a strong baseline.
    Returns ``(targets, colours, boards, game_ids)``.
    """
    rng = np.random.default_rng(seed)
    targets: list[float] = []
    colours: list[int] = []
    boards: list[np.ndarray] = []
    game_ids: list[int] = []
    for game in range(n_games):
        white_to_move = game % 2 == 0
        # One outcome for the whole game — the correlation the bootstrap exists for.
        # The mover wins 80% of the time, so the sign follows the mover's colour.
        favoured = 1.0 if white_to_move else -1.0
        outcome = favoured if rng.random() < 0.8 else -favoured
        for position in range(per_game):
            pieces = 2 + position * 5
            boards.append(_board(pieces, pieces) if white_to_move else _board(pieces - 1, pieces))
            targets.append(outcome)
            colours.append(WHITE_TO_MOVE if white_to_move else BLACK_TO_MOVE)
            game_ids.append(game)
    return np.array(targets), np.array(colours), boards, np.array(game_ids)


def _colour_mean_predictor(targets: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """Predict each colour's mean outcome — exactly the colour-only baseline."""
    predictions = np.empty_like(targets)
    for colour in np.unique(colours):
        mask = colours == colour
        predictions[mask] = targets[mask].mean()
    return predictions


def test_colour_prior_predictor_scores_zero_skill() -> None:
    """A predictor that *is* the colour baseline must score 0 with an interval at 0."""
    targets, colours, boards, game_ids = _population()
    predictions = _colour_mean_predictor(targets, colours)

    result = compute_colour_value_diagnostic(predictions, targets, boards, game_ids, n_resamples=400, seed=1)

    assert result is not None
    assert result.skill_vs_colour.point == pytest.approx(0.0, abs=1e-9)
    # The interval sits at zero. Its upper end is a hair *below* zero rather than
    # straddling it, because the baseline is refit inside each resample and so
    # enjoys a small in-sample advantage — see the module docstring. The bias must
    # stay negligible; if it grows, the interval has stopped meaning what it says.
    assert result.skill_vs_colour.lo < 0.0
    assert result.skill_vs_colour.hi == pytest.approx(0.0, abs=0.01)


def test_refit_baseline_bias_stays_negligible_and_conservative() -> None:
    """Pin the direction and scale of the refit-baseline bias.

    It must be negative (conservative: it cannot manufacture apparent skill) and
    it must shrink as the number of games grows, since it scales like
    baseline-cells / games.
    """
    widths = []
    for n_games in (20, 80):
        targets, colours, boards, game_ids = _population(n_games=n_games, seed=n_games)
        predictions = _colour_mean_predictor(targets, colours)
        result = compute_colour_value_diagnostic(predictions, targets, boards, game_ids, n_resamples=400, seed=1)
        assert result is not None
        assert result.skill_vs_colour.hi <= 0.0  # never invents skill
        widths.append(abs(result.skill_vs_colour.hi))

    assert widths[1] < widths[0]
    assert widths[0] < 0.02


def test_perfect_predictor_scores_full_skill() -> None:
    targets, _colours, boards, game_ids = _population()

    result = compute_colour_value_diagnostic(targets.copy(), targets, boards, game_ids, n_resamples=400, seed=2)

    assert result is not None
    assert result.skill_vs_colour.point == pytest.approx(1.0)
    assert result.value_mse == pytest.approx(0.0)


def test_miscalibrated_colour_predictor_scores_negative_skill() -> None:
    """Worse than guessing from the colour must read as negative, not merely low."""
    targets, colours, boards, game_ids = _population()
    # Exaggerate the colour prior past the actual base rates.
    predictions = np.where(colours == WHITE_TO_MOVE, 1.0, -1.0)

    result = compute_colour_value_diagnostic(predictions, targets, boards, game_ids, n_resamples=400, seed=3)

    assert result is not None
    assert result.skill_vs_colour.point < 0.0


def test_intervals_are_game_clustered_not_position_level() -> None:
    """Duplicating every position must not shrink the interval.

    Position-level intervals narrow by ~sqrt(2) when each position is duplicated,
    because they see twice the rows. A game-cluster interval must not move: the
    number of independent games is unchanged.
    """
    targets, colours, boards, game_ids = _population()
    rng = np.random.default_rng(5)
    predictions = targets + rng.normal(scale=0.5, size=targets.size)

    single = compute_colour_value_diagnostic(predictions, targets, boards, game_ids, n_resamples=600, seed=7)
    doubled = compute_colour_value_diagnostic(
        np.concatenate([predictions, predictions]),
        np.concatenate([targets, targets]),
        boards + boards,
        np.concatenate([game_ids, game_ids]),
        n_resamples=600,
        seed=7,
    )

    assert single is not None and doubled is not None
    assert doubled.n_games == single.n_games
    assert doubled.n_positions == 2 * single.n_positions
    single_width = single.skill_vs_colour.hi - single.skill_vs_colour.lo
    doubled_width = doubled.skill_vs_colour.hi - doubled.skill_vs_colour.lo
    assert doubled_width == pytest.approx(single_width, rel=0.15)


def test_colour_phase_baseline_is_at_least_as_strong_as_colour_only() -> None:
    """Adding phase can only reduce the baseline's error, so skill can only fall."""
    targets, _colours, boards, game_ids = _population()
    rng = np.random.default_rng(11)
    predictions = targets + rng.normal(scale=0.4, size=targets.size)

    result = compute_colour_value_diagnostic(predictions, targets, boards, game_ids, n_resamples=400, seed=4)

    assert result is not None
    assert result.colour_phase_mse <= result.colour_only_mse + 1e-12
    assert result.skill_vs_colour_phase.point <= result.skill_vs_colour.point + 1e-12


def test_slices_report_per_colour_bias() -> None:
    targets, colours, boards, game_ids = _population()
    predictions = np.where(colours == WHITE_TO_MOVE, 1.0, -1.0)

    result = compute_colour_value_diagnostic(predictions, targets, boards, game_ids, n_resamples=200, seed=6)

    assert result is not None
    assert [s.colour for s in result.slices] == [WHITE_TO_MOVE, BLACK_TO_MOVE]
    white = result.slices[0]
    assert white.n_positions + result.slices[1].n_positions == result.n_positions
    assert white.n_games > 1
    # Predicting +1 for White when White wins 80% of the time over-predicts.
    assert white.bias > 0.0


def test_prediction_tracks_colour_more_than_outcomes_do() -> None:
    """The headline pathology: the net is more certain about colour than the game."""
    targets, colours, boards, game_ids = _population()
    predictions = np.where(colours == WHITE_TO_MOVE, 0.95, -0.95)

    result = compute_colour_value_diagnostic(predictions, targets, boards, game_ids, n_resamples=200, seed=8)

    assert result is not None
    assert abs(result.colour_prediction_correlation) > abs(result.colour_target_correlation)


def test_ambiguous_positions_are_excluded_and_counted() -> None:
    targets, colours, boards, game_ids = _population(n_games=20)
    # Break parity on a handful of boards (simulating post-pass positions).
    boards = list(boards)
    for i in (0, 1, 2):
        boards[i] = _board(6, 2)
    predictions = _colour_mean_predictor(targets, colours)

    result = compute_colour_value_diagnostic(predictions, targets, boards, game_ids, n_resamples=200, seed=9)

    assert result is not None
    assert result.n_excluded == 3
    assert result.n_positions == len(targets) - 3


def test_single_colour_returns_none() -> None:
    """The colour-only baseline degenerates to the global mean — not the claim."""
    boards = [_board(p, p) for p in range(1, 9)]  # all White-to-move
    targets = np.array([1.0, -1.0] * 4)
    game_ids = np.repeat(np.arange(4), 2)

    assert compute_colour_value_diagnostic(targets, targets, boards, game_ids, n_resamples=50) is None


def test_all_ambiguous_returns_none() -> None:
    boards = [_board(6, 2) for _ in range(6)]
    targets = np.ones(6)
    game_ids = np.arange(6)

    assert compute_colour_value_diagnostic(targets, targets, boards, game_ids, n_resamples=50) is None


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_colour_value_diagnostic(np.ones(3), np.ones(4), [_board(1, 1)] * 3, np.arange(3), n_resamples=10)


def test_payload_is_json_shaped() -> None:
    targets, colours, boards, game_ids = _population(n_games=16)
    predictions = _colour_mean_predictor(targets, colours)

    result = compute_colour_value_diagnostic(predictions, targets, boards, game_ids, n_resamples=200, seed=12)

    assert result is not None
    payload = result.as_payload()
    assert set(payload) >= {
        "n_positions",
        "n_games",
        "n_excluded",
        "value_mse",
        "colour_only_mse",
        "colour_phase_mse",
        "skill_vs_colour",
        "skill_vs_colour_phase",
        "slices",
        "phase_skill",
    }
    assert payload["skill_vs_colour"]["ci"][0] <= payload["skill_vs_colour"]["ci"][1]  # type: ignore[index]
    import json

    json.dumps(payload)  # must be strictly serialisable


def test_degenerate_eval_set_returns_no_reading_instead_of_raising() -> None:
    """An eval set whose outcome is perfectly predicted by colour yields None, not a crash.

    The colour baseline's error is exactly zero when every colour group is
    internally constant, so ``_skill`` is undefined (nan) in every resample and
    the bootstrap has nothing to build an interval from. This is a *diagnostic*:
    it must report "no reading" and leave training alone. Before this was handled,
    the ValueError escaped through ``BaseNNetWrapper.train`` and killed the run.
    """
    boards: list[np.ndarray] = []
    targets: list[float] = []
    game_ids: list[int] = []
    for game in range(8):
        white_to_move = game % 2 == 0
        for position in range(3):
            pieces = 2 + position * 5
            boards.append(_board(pieces, pieces) if white_to_move else _board(pieces - 1, pieces))
            # Outcome is a pure function of mover colour -> zero baseline error.
            targets.append(1.0 if white_to_move else -1.0)
            game_ids.append(game)

    result = compute_colour_value_diagnostic(
        np.zeros(len(targets)),
        np.array(targets),
        boards,
        np.array(game_ids),
        n_resamples=200,
        seed=0,
    )
    assert result is None


def test_single_position_per_colour_phase_group_does_not_raise() -> None:
    """A tiny eval set makes the colour x phase groups singletons — still no crash.

    The colour x phase baseline has many more groups than the colour-only one, so
    it is the fragile one: with one position per group its error is zero by
    construction, whatever the outcomes are.
    """
    boards = [_board(2, 2), _board(7, 7), _board(1, 2), _board(6, 7)]
    targets = np.array([1.0, -1.0, -1.0, 1.0])
    game_ids = np.array([0, 1, 2, 3])
    result = compute_colour_value_diagnostic(
        np.array([0.5, -0.5, 0.25, -0.25]),
        targets,
        boards,
        game_ids,
        n_resamples=200,
        seed=0,
    )
    assert result is None or np.isfinite(result.skill_vs_colour.point)
