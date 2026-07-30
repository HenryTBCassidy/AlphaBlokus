"""Tests for game-level holdout splits + out-of-sample fit metrics (capacity probe)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
from alphablokus.storage.sparse_policy import sparsify
from alphablokus.training.holdout import evaluate_holdout, evaluate_imitation_diagnostics, split_games_holdout

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.games.tictactoe.game import TicTacToeGame
    from alphablokus.selfplay.episode import GameExamples


def _fake_game(game_id: int, positions: int = 3) -> GameExamples:
    """A distinguishable fake game: the board array carries the game id."""
    return [
        (
            np.full((3, 3), game_id, dtype=np.int8),
            (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32)),
            1.0,
        )
        for _ in range(positions)
    ]


def _game_ids(games: list[GameExamples]) -> set[int]:
    return {int(game[0][0][0, 0]) for game in games}


def test_split_is_a_partition_at_game_granularity() -> None:
    games = [_fake_game(i) for i in range(20)]
    train, holdout = split_games_holdout(games, holdout_fraction=0.2, seed=7)

    assert len(train) + len(holdout) == 20
    assert len(holdout) == 4  # round(20 * 0.2)
    # No game straddles the split, and none is lost or duplicated.
    assert _game_ids(train).isdisjoint(_game_ids(holdout))
    assert _game_ids(train) | _game_ids(holdout) == set(range(20))
    # Positions within each held-out game stayed together.
    assert all(len(game) == 3 for game in holdout)


def test_split_is_deterministic_in_seed() -> None:
    games = [_fake_game(i) for i in range(30)]
    first = split_games_holdout(games, 0.1, seed=42)
    second = split_games_holdout(games, 0.1, seed=42)
    other_seed = split_games_holdout(games, 0.1, seed=43)

    assert _game_ids(first[1]) == _game_ids(second[1])
    assert _game_ids(first[1]) != _game_ids(other_seed[1])  # 3-of-30 collision is (30C3)⁻¹-unlikely


def test_split_nonzero_fraction_holds_out_at_least_one_game() -> None:
    games = [_fake_game(i) for i in range(3)]
    _train, holdout = split_games_holdout(games, holdout_fraction=0.01, seed=1)
    assert len(holdout) == 1


def test_split_zero_fraction_holds_out_nothing() -> None:
    games = [_fake_game(i) for i in range(3)]
    train, holdout = split_games_holdout(games, holdout_fraction=0.0, seed=1)
    assert holdout == [] and len(train) == 3


def test_split_validates_inputs() -> None:
    with pytest.raises(ValueError, match="holdout_fraction"):
        split_games_holdout([_fake_game(0)], holdout_fraction=1.0, seed=1)
    with pytest.raises(ValueError, match="at least one game"):
        split_games_holdout([], holdout_fraction=0.1, seed=1)


class _UniformPredictor:
    """Real (non-mock) minimal predictor: uniform policy, constant value."""

    def __init__(self, action_size: int, value: float) -> None:
        self._action_size = action_size
        self._value = value

    def predict_encoded(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = planes.shape[0]
        policies = np.full((n, self._action_size), 1.0 / self._action_size, dtype=np.float32)
        values = np.full((n,), self._value, dtype=np.float32)
        return policies, values


def test_evaluate_holdout_matches_closed_form() -> None:
    """One-hot targets vs a uniform predictor have CE = ln(A) and KL = ln(A)."""
    action_size = 10
    examples = [
        (
            np.zeros((3, 3), dtype=np.int8),
            (np.array([i % action_size], dtype=np.int32), np.array([1.0], dtype=np.float32)),
            1.0,
        )
        for i in range(5)
    ]
    predictor = _UniformPredictor(action_size, value=0.5)

    metrics = evaluate_holdout(
        predictor,
        examples,
        encode_fn=lambda board: board[np.newaxis].astype(np.float32),
        action_size=action_size,
        batch_size=2,  # exercises the batching path (5 examples → 3 batches)
    )

    assert metrics.policy_ce == pytest.approx(math.log(action_size))
    assert metrics.target_entropy == pytest.approx(0.0)  # one-hot targets
    assert metrics.policy_kl == pytest.approx(math.log(action_size))
    assert metrics.value_mse == pytest.approx(0.25)  # (1.0 − 0.5)²
    assert metrics.n_positions == 5


def test_evaluate_holdout_on_a_real_wrapper(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """A real TicTacToe wrapper produces finite metrics consistent with its own outputs."""
    wrapper = NNetWrapper(ttt_game, test_config)
    board = ttt_game.initialise_board()
    compact = ttt_game.get_canonical_form(board, 1).to_compact()
    action_size = ttt_game.get_action_size()
    uniform = np.full(action_size, 1.0 / action_size)
    examples = [(compact, sparsify(uniform), 0.0)]

    metrics = evaluate_holdout(
        wrapper,
        examples,
        encode_fn=ttt_game.encode_compact,
        action_size=action_size,
    )

    # Cross-check against the wrapper's own forward pass.
    policies, values = wrapper.predict_encoded(ttt_game.encode_compact(compact)[np.newaxis])
    expected_ce = float(-(uniform * np.log(policies[0])).sum())
    assert metrics.policy_ce == pytest.approx(expected_ce, rel=1e-5)
    assert metrics.value_mse == pytest.approx(float(values[0] ** 2), rel=1e-5)
    assert metrics.policy_kl == pytest.approx(metrics.policy_ce - metrics.target_entropy)
    assert np.isfinite(metrics.policy_ce) and np.isfinite(metrics.value_mse)


def test_evaluate_holdout_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        evaluate_holdout(
            _UniformPredictor(4, 0.0),
            [],
            encode_fn=lambda board: board,
            action_size=4,
        )


# --------------------------------------------------------------------------- #
# Imitation diagnostics (SL distillation, plan D7)
# --------------------------------------------------------------------------- #


class _TablePredictor:
    """Real (non-mock) scripted predictor: per-position outputs looked up by the
    row id the fake board carries, so every metric is hand-computable."""

    def __init__(self, policies: np.ndarray, values: np.ndarray) -> None:
        self._policies = policies
        self._values = values

    def predict_encoded(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = planes[:, 0, 0, 0].astype(int)
        return self._policies[ids], self._values[ids]


def _imitation_fixture() -> tuple[list, list[int], list[int], _TablePredictor]:
    """Four positions with known legal sets, expert moves, colours, and predictions.

    Position 0's *global* policy argmax (action 0) is illegal — its legal argmax is
    the expert move, so it must count as a top-1 hit (legal-restricted on purpose).
    """
    action_size = 10
    legal_sets = [[1, 2, 3], [4, 5], [0, 9], [6, 7]]
    expert_actions = [2, 4, 9, 6]
    players = [1, 1, -1, -1]
    outcomes = [1.0, -1.0, 1.0, -1.0]
    policies = np.zeros((4, action_size), dtype=np.float32)
    policies[0, [0, 1, 2, 3]] = [0.5, 0.1, 0.3, 0.1]  # global argmax illegal; legal argmax = expert → hit
    policies[1, [4, 5]] = [0.4, 0.6]  # legal argmax 5 != expert 4 → miss
    policies[2, [0, 9]] = [0.1, 0.9]  # hit
    policies[3, [6, 7]] = [0.6, 0.4]  # hit
    predicted_values = np.array([0.95, -0.05, 0.5, -0.99], dtype=np.float32)
    examples = [
        (
            np.full((3, 3), i, dtype=np.int8),
            (np.array(legal, dtype=np.int32), np.full(len(legal), 1.0 / len(legal), dtype=np.float32)),
            outcome,
        )
        for i, (legal, outcome) in enumerate(zip(legal_sets, outcomes, strict=True))
    ]
    return examples, expert_actions, players, _TablePredictor(policies, predicted_values)


def _encode_fake(board: np.ndarray) -> np.ndarray:
    return board[np.newaxis].astype(np.float32)


def test_imitation_diagnostics_top1_is_legal_restricted() -> None:
    examples, expert_actions, players, predictor = _imitation_fixture()
    diagnostics = evaluate_imitation_diagnostics(
        predictor,
        examples,
        expert_actions,
        players,
        encode_fn=_encode_fake,
        batch_size=3,  # exercises the batching path (4 examples → 2 batches)
    )
    assert diagnostics.n_positions == 4
    assert diagnostics.top1_accuracy == pytest.approx(3 / 4)


def test_imitation_diagnostics_calibration_is_colour_conditional() -> None:
    examples, expert_actions, players, predictor = _imitation_fixture()
    diagnostics = evaluate_imitation_diagnostics(predictor, examples, expert_actions, players, encode_fn=_encode_fake)

    black, white = diagnostics.calibration
    assert (black.player, white.player) == (-1, 1)  # ordered by player ascending

    # White (+1): predicted [0.95, -0.05] vs outcomes [1, -1].
    assert white.n_positions == 2
    assert white.mean_predicted == pytest.approx(0.45)
    assert white.mean_outcome == pytest.approx(0.0)
    assert white.value_mse == pytest.approx((0.05**2 + 0.95**2) / 2)
    assert white.bucket_counts[9] == 1 and white.bucket_mean_outcomes[9] == pytest.approx(1.0)  # 0.95 → [0.8, 1.0]
    assert white.bucket_counts[4] == 1 and white.bucket_mean_outcomes[4] == pytest.approx(-1.0)  # -0.05 → [-0.2, 0.0)
    assert sum(white.bucket_counts) == 2
    assert all(mean is None for i, mean in enumerate(white.bucket_mean_outcomes) if i not in (4, 9))

    # Black (-1): predicted [0.5, -0.99] vs outcomes [1, -1].
    assert black.n_positions == 2
    assert black.bucket_counts[7] == 1 and black.bucket_mean_outcomes[7] == pytest.approx(1.0)  # 0.5 → [0.4, 0.6)
    assert black.bucket_counts[0] == 1 and black.bucket_mean_outcomes[0] == pytest.approx(-1.0)  # -0.99 → [-1, -0.8)
    assert black.mean_predicted == pytest.approx((0.5 - 0.99) / 2)


def test_imitation_diagnostics_accepts_dense_policies_and_validates_inputs() -> None:
    examples, expert_actions, players, predictor = _imitation_fixture()
    # A dense target works too: its nonzero support is the legal set.
    board, (indices, values), outcome = examples[0]
    dense = np.zeros(10, dtype=np.float32)
    dense[indices] = values
    diagnostics = evaluate_imitation_diagnostics(
        predictor, [(board, dense, outcome)], expert_actions[:1], players[:1], encode_fn=_encode_fake
    )
    assert diagnostics.top1_accuracy == 1.0

    with pytest.raises(ValueError, match="at least one example"):
        evaluate_imitation_diagnostics(predictor, [], [], [], encode_fn=_encode_fake)
    with pytest.raises(ValueError, match="misaligned"):
        evaluate_imitation_diagnostics(predictor, examples, expert_actions[:2], players, encode_fn=_encode_fake)


def test_imitation_diagnostics_report_the_colour_only_value_floor() -> None:
    """A value head must beat "guess from whose turn it is" to be reading the board.

    Blokus Duo has a severe first-player advantage, so the outcome is largely predictable
    from the side to move alone — measured on real v2 data, White-to-move positions are
    79% wins and Black-to-move 78% losses, and guessing purely from the colour scores
    0.30 MSE against 0.84 for always predicting a draw. The per-position win/loss split
    looks reassuringly balanced (43/41) but that is a mechanical consequence of players
    alternating and carries no information. Without this floor reported alongside it, a
    value head that has learnt nothing but the colour prior is indistinguishable from one
    that works.
    """
    examples, expert_actions, players, predictor = _imitation_fixture()
    diagnostics = evaluate_imitation_diagnostics(predictor, examples, expert_actions, players, encode_fn=_encode_fake)

    # The fixture's outcomes are [1, -1] for each colour, so each colour's mean is 0 and
    # the colour-only predictor scores the outcome variance.
    outcomes = np.array([value for _board, _policy, value in examples])
    assert diagnostics.colour_only_value_mse == pytest.approx(float(np.mean(outcomes**2)))
    assert diagnostics.value_mse > 0.0
    assert diagnostics.value_skill == pytest.approx(1.0 - diagnostics.value_mse / diagnostics.colour_only_value_mse)


def test_colour_only_floor_is_high_when_one_side_usually_wins() -> None:
    """The floor rises with the first-player advantage — which is the whole point."""
    from alphablokus.training.holdout import ImitationDiagnostics

    # A head that exactly reproduces the colour prior has zero skill by construction.
    no_skill = ImitationDiagnostics(
        top1_accuracy=0.5, n_positions=100, calibration=(), value_mse=0.30, colour_only_value_mse=0.30
    )
    assert no_skill.value_skill == pytest.approx(0.0)

    reads_the_board = ImitationDiagnostics(
        top1_accuracy=0.5, n_positions=100, calibration=(), value_mse=0.15, colour_only_value_mse=0.30
    )
    assert reads_the_board.value_skill == pytest.approx(0.5)

    worse_than_guessing = ImitationDiagnostics(
        top1_accuracy=0.5, n_positions=100, calibration=(), value_mse=0.45, colour_only_value_mse=0.30
    )
    assert worse_than_guessing.value_skill < 0.0


# --------------------------------------------------------------------------- #
# Auxiliary score head (docs/plans/score-auxiliary-target.md S6)
# --------------------------------------------------------------------------- #


class _AuxPredictor:
    """Scripted predictor exposing the diagnostics-only auxiliary surface.

    ``aux`` maps a head's short name to its per-position outputs, sliced batch by batch
    in call order — so a metric that fed the batches in the wrong order, or fed masked
    positions it should have skipped, reads the wrong rows and the numbers move.
    An empty mapping is a net that built no auxiliary head at all.
    """

    def __init__(self, aux: dict[str, np.ndarray]) -> None:
        self._aux = aux
        self._cursor = 0

    def predict_encoded_aux(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        n = planes.shape[0]
        policies = np.full((n, 4), 0.25, dtype=np.float32)
        values = np.zeros((n,), dtype=np.float32)
        chunk = {name: values_[self._cursor : self._cursor + n] for name, values_ in self._aux.items()}
        self._cursor += n
        return policies, values, chunk


def _scoring_predictor(scores: np.ndarray | None) -> _AuxPredictor:
    """A predictor with just the score head, or (``None``) with no head at all."""
    return _AuxPredictor({} if scores is None else {"score": scores.astype(np.float32)})


def _score_examples(count: int) -> list:
    return [
        (
            np.zeros((3, 3), dtype=np.int8),
            (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32)),
            1.0,
        )
        for _ in range(count)
    ]


def test_evaluate_score_head_matches_the_closed_form_mse() -> None:
    """Targets are ``tanh(margin / scale)``; the MSE is against exactly those."""
    from alphablokus.training.holdout import evaluate_score_head

    margins: list[float | None] = [25.0, -25.0, 0.0]
    target = math.tanh(1.0)
    predictor = _scoring_predictor(np.array([target, -target, 0.5]))

    metrics = evaluate_score_head(
        predictor,
        _score_examples(3),
        margins,
        score_scale=25.0,
        encode_fn=_encode_fake,
        batch_size=2,  # exercise batching
    )

    assert metrics is not None
    assert metrics.n_positions == 3
    assert metrics.n_skipped == 0
    assert metrics.score_mse == pytest.approx(0.25 / 3)  # only the third row is wrong


def test_evaluate_score_head_skips_positions_without_a_margin() -> None:
    """Opening rows (no margin) must be excluded, not scored against a zero."""
    from alphablokus.training.holdout import evaluate_score_head

    predictor = _scoring_predictor(np.array([0.0, 0.0, 0.0, 0.0]))
    margins: list[float | None] = [0.0, None, 0.0, None]

    metrics = evaluate_score_head(predictor, _score_examples(4), margins, score_scale=25.0, encode_fn=_encode_fake)

    assert metrics is not None
    assert (metrics.n_positions, metrics.n_skipped) == (2, 2)
    assert metrics.score_mse == pytest.approx(0.0)


def test_evaluate_score_head_returns_none_without_a_head_or_without_margins() -> None:
    from alphablokus.training.holdout import evaluate_score_head

    headless = _scoring_predictor(None)
    assert (
        evaluate_score_head(headless, _score_examples(2), [1.0, 2.0], score_scale=25.0, encode_fn=_encode_fake) is None
    )

    no_margins = _scoring_predictor(np.zeros(2))
    assert (
        evaluate_score_head(no_margins, _score_examples(2), [None, None], score_scale=25.0, encode_fn=_encode_fake)
        is None
    )


def test_score_skill_is_measured_against_a_constant_predictor() -> None:
    """A head emitting the mean target scores zero skill however small its MSE."""
    from alphablokus.training.holdout import evaluate_score_head

    margins: list[float | None] = [25.0, -25.0]
    mean_target = 0.0  # tanh is odd, so these two average to zero
    constant_head = _scoring_predictor(np.array([mean_target, mean_target]))

    metrics = evaluate_score_head(constant_head, _score_examples(2), margins, score_scale=25.0, encode_fn=_encode_fake)

    assert metrics is not None
    assert metrics.score_skill == pytest.approx(0.0)
    assert metrics.score_mse == pytest.approx(metrics.constant_mse)


def test_evaluate_score_head_rejects_misaligned_margins() -> None:
    from alphablokus.training.holdout import evaluate_score_head

    with pytest.raises(ValueError, match="index-aligned"):
        evaluate_score_head(
            _scoring_predictor(np.zeros(3)),
            _score_examples(3),
            [1.0],
            score_scale=25.0,
            encode_fn=_encode_fake,
        )


def test_imitation_diagnostics_top3_is_a_rank_three_window() -> None:
    """Top-3 must be *the top three legal moves*, not "any legal move".

    Both positions have five legal moves and the same ranking, so the only thing
    separating them is where the expert's move sits in it — third (inside the window)
    versus fourth (outside). A "top-k over the whole legal set" bug scores 100% here,
    and a top-3 that silently aliased top-1 scores 0%.
    """
    from alphablokus.training.holdout import evaluate_imitation_diagnostics as evaluate

    legal = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    policies = np.zeros((2, 8), dtype=np.float32)
    policies[:, legal] = [0.30, 0.25, 0.20, 0.15, 0.10]  # ranking is 0, 1, 2, 3, 4
    predictor = _TablePredictor(policies, np.zeros(2, dtype=np.float32))
    examples = [(np.full((3, 3), i, dtype=np.int8), (legal, np.full(5, 0.2, dtype=np.float32)), 0.0) for i in range(2)]

    diagnostics = evaluate(predictor, examples, [2, 3], [1, -1], encode_fn=_encode_fake)

    assert diagnostics.top1_accuracy == 0.0
    assert diagnostics.top3_accuracy == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Auxiliary ownership head (plan N4)
# --------------------------------------------------------------------------- #


def _ownership_fixture() -> tuple[list, list, _AuxPredictor]:
    """Three 2x2 positions — two labelled, one with no final board — hand-computable.

    Position 0 predicts class 0 everywhere at 0.6 and is labelled class 0 everywhere;
    position 1 predicts class 2 at 0.6 and is labelled ``[2, 2, 1, 1]``. So the head is
    right on 6 of 8 labelled cells and every cell's cross-entropy is ``−ln 0.6`` or
    ``−ln 0.3``.
    """
    probabilities = np.zeros((3, 3, 2, 2), dtype=np.float32)
    probabilities[0] = np.array([0.6, 0.3, 0.1], dtype=np.float32)[:, None, None]
    probabilities[1] = np.array([0.1, 0.3, 0.6], dtype=np.float32)[:, None, None]
    probabilities[2] = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)[:, None, None]
    ownership = [
        np.array([[-1, -1], [-1, -1]], dtype=np.int8),  # classes 0,0,0,0
        np.array([[1, 1], [0, 0]], dtype=np.int8),  # classes 2,2,1,1
        None,  # no final board — masked, and never shown to the predictor
    ]
    examples = [
        (
            np.full((3, 3), i, dtype=np.int8),
            (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32)),
            0.0,
        )
        for i in range(3)
    ]
    return examples, ownership, _AuxPredictor({"ownership": probabilities})


def test_evaluate_ownership_head_matches_the_closed_form() -> None:
    from alphablokus.training.holdout import evaluate_ownership_head

    examples, ownership, predictor = _ownership_fixture()

    metrics = evaluate_ownership_head(predictor, examples, ownership, encode_fn=_encode_fake, batch_size=1)

    assert metrics is not None
    assert (metrics.n_positions, metrics.n_skipped) == (2, 1)
    assert metrics.cross_entropy == pytest.approx((6 * -math.log(0.6) + 2 * -math.log(0.3)) / 8)
    assert metrics.accuracy == pytest.approx(0.75)
    # Label marginal over the 8 labelled cells is [0.5, 0.25, 0.25].
    assert metrics.marginal_cross_entropy == pytest.approx(-(0.5 * math.log(0.5) + 0.5 * math.log(0.25)))
    assert metrics.skill == pytest.approx(1.0 - metrics.cross_entropy / metrics.marginal_cross_entropy)


def test_evaluate_ownership_head_returns_none_without_a_head_or_a_final_board() -> None:
    from alphablokus.training.holdout import evaluate_ownership_head

    examples, ownership, _predictor = _ownership_fixture()

    headless = _AuxPredictor({})
    assert evaluate_ownership_head(headless, examples, ownership, encode_fn=_encode_fake) is None

    _, _, predictor = _ownership_fixture()
    assert evaluate_ownership_head(predictor, examples, [None] * 3, encode_fn=_encode_fake) is None


def test_evaluate_ownership_head_rejects_misaligned_targets() -> None:
    from alphablokus.training.holdout import evaluate_ownership_head

    examples, ownership, predictor = _ownership_fixture()
    with pytest.raises(ValueError, match="index-aligned"):
        evaluate_ownership_head(predictor, examples, ownership[:2], encode_fn=_encode_fake)


# --------------------------------------------------------------------------- #
# Auxiliary opponent-reply head (plan N5)
# --------------------------------------------------------------------------- #


def _reply_fixture() -> tuple[list, list, _AuxPredictor]:
    """Three positions, the last with no next ply, over a 4-action space."""
    predicted = np.array(
        [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]],
        dtype=np.float32,
    )
    replies: list[object | None] = [
        (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32)),  # top-1 hit
        (np.array([1], dtype=np.int32), np.array([1.0], dtype=np.float32)),  # top-1 miss
        None,  # a game's final position
    ]
    examples = [
        (
            np.full((3, 3), i, dtype=np.int8),
            (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32)),
            0.0,
        )
        for i in range(3)
    ]
    return examples, replies, _AuxPredictor({"reply": predicted})


def test_evaluate_reply_head_matches_the_closed_form() -> None:
    from alphablokus.training.holdout import evaluate_reply_head

    examples, replies, predictor = _reply_fixture()

    metrics = evaluate_reply_head(predictor, examples, replies, action_size=4, encode_fn=_encode_fake, batch_size=2)

    assert metrics is not None
    assert (metrics.n_positions, metrics.n_skipped) == (2, 1)
    assert metrics.policy_ce == pytest.approx((-math.log(0.7) + -math.log(0.1)) / 2)
    assert metrics.target_entropy == pytest.approx(0.0)  # one-hot replies
    assert metrics.policy_kl == pytest.approx(metrics.policy_ce)
    assert metrics.top1_accuracy == pytest.approx(0.5)


def test_evaluate_reply_head_returns_none_without_a_head_or_a_next_ply() -> None:
    from alphablokus.training.holdout import evaluate_reply_head

    examples, replies, _predictor = _reply_fixture()

    headless = _AuxPredictor({})
    assert evaluate_reply_head(headless, examples, replies, action_size=4, encode_fn=_encode_fake) is None

    _, _, predictor = _reply_fixture()
    assert evaluate_reply_head(predictor, examples, [None] * 3, action_size=4, encode_fn=_encode_fake) is None


def test_evaluate_reply_head_rejects_misaligned_targets() -> None:
    from alphablokus.training.holdout import evaluate_reply_head

    examples, replies, predictor = _reply_fixture()
    with pytest.raises(ValueError, match="index-aligned"):
        evaluate_reply_head(predictor, examples, replies[:2], action_size=4, encode_fn=_encode_fake)
