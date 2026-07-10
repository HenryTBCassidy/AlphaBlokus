"""Tests for shared ``BaseNNetWrapper`` behaviour (LR scheduler)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.base_wrapper import PVC_TOP_K, BaseNNetWrapper
from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
from alphablokus.interfaces import IPolicyValuePredictor
from alphablokus.storage.metrics import EvalSet, MetricsCollector
from alphablokus.storage.sparse_policy import sparsify

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphablokus.config import RunConfig
    from alphablokus.games.tictactoe.game import TicTacToeGame
    from alphablokus.interfaces import IBoard


def _lr_sequence(config: RunConfig, game: TicTacToeGame, steps: int) -> list[float]:
    """LR seen at each scheduler step (start value first, then after each step)."""
    wrapper = NNetWrapper(game, config)
    assert wrapper.scheduler is not None
    seq = [wrapper.optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        wrapper.scheduler.step()
        seq.append(wrapper.optimizer.param_groups[0]["lr"])
    return seq


def test_cosine_floor_never_drops_below_eta_min(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """With a non-zero ``lr_eta_min`` the cosine LR never anneals below it."""
    eta_min = 1e-4
    net_config = replace(test_config.net_config, lr_scheduler="cosine", lr_eta_min=eta_min)
    config = replace(test_config, num_generations=60, net_config=net_config)

    seq = _lr_sequence(config, ttt_game, steps=config.num_generations)

    assert min(seq) >= eta_min, f"LR dropped below floor {eta_min}: min={min(seq)}"
    # The final LR sits right at the floor (cosine reaches eta_min at T_max).
    assert seq[-1] == eta_min


def test_cosine_default_eta_min_is_unchanged(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """Default ``lr_eta_min=0.0`` reproduces the prior schedule exactly.

    The reference is a bare ``CosineAnnealingLR`` with the old hardcoded
    ``eta_min=0`` — the schedule must be identical value-for-value.
    """
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    net_config = replace(test_config.net_config, lr_scheduler="cosine")  # lr_eta_min defaults to 0.0
    config = replace(test_config, num_generations=60, net_config=net_config)
    assert config.net_config.lr_eta_min == 0.0

    seq = _lr_sequence(config, ttt_game, steps=config.num_generations)

    # Reference: the pre-change construction (eta_min=0, same T_max, same peak LR).
    ref_optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=config.net_config.learning_rate)
    ref_scheduler = CosineAnnealingLR(ref_optimizer, T_max=config.num_generations * config.net_config.epochs)
    ref = [ref_optimizer.param_groups[0]["lr"]]
    for _ in range(config.num_generations):
        ref_scheduler.step()
        ref.append(ref_optimizer.param_groups[0]["lr"])

    assert seq == ref, "Default lr_eta_min=0.0 changed the cosine schedule"


def test_train_logs_actual_learning_rate(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """train() records the optimizer's actual LR once per epoch (L2).

    The logged value is the LR *before* the epoch's scheduler step — what the
    epoch actually trained at — and for a constant schedule it is the config LR.
    """
    net_config = replace(test_config.net_config, epochs=2)  # lr_scheduler defaults to None (constant)
    config = replace(test_config, net_config=net_config)
    wrapper = NNetWrapper(ttt_game, config)

    compacts = _ttt_eval_positions(ttt_game, 4)
    examples = [(compact, sparsify(_uniform_over_legal(ttt_game, compact)), 0.0) for compact in compacts]
    metrics = MetricsCollector(config=config)
    wrapper.train(examples, generation=3, metrics=metrics)

    records = metrics._learning_rate_records
    assert len(records) == config.net_config.epochs, "expected one LR record per epoch"
    assert all(r["generation"] == 3 for r in records)
    assert {r["epoch"] for r in records} == set(range(config.net_config.epochs))
    # Constant schedule: every epoch trains at the configured LR.
    assert all(r["learning_rate"] == config.net_config.learning_rate for r in records)


def _cosine_wrapper(ttt_game: TicTacToeGame, test_config: RunConfig) -> tuple[NNetWrapper, RunConfig]:
    net_config = replace(test_config.net_config, lr_scheduler="cosine", lr_eta_min=1e-4)
    config = replace(test_config, num_generations=10, net_config=net_config)
    wrapper = NNetWrapper(ttt_game, config)
    config.net_directory.mkdir(parents=True, exist_ok=True)
    return wrapper, config


def test_reject_reload_does_not_rewind_lr_schedule(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """After a rejection, the LR clock is where the next generation expects it,
    not rewound to before this generation's training (L3).

    Weights are still reverted (the gate's job); only the LR clock is preserved.
    """
    import torch

    wrapper, _ = _cosine_wrapper(ttt_game, test_config)
    assert wrapper.scheduler is not None

    # Coach cycle: save temp before training, then training steps the scheduler.
    wrapper.save_checkpoint("temp.pth.tar")
    reference_param = next(iter(wrapper.nnet.parameters())).detach().clone()
    lr_before_training = wrapper.optimizer.param_groups[0]["lr"]

    wrapper.scheduler.step()  # one generation of training advances the clock
    last_epoch_after_step = wrapper.scheduler.last_epoch
    lr_after_step = wrapper.optimizer.param_groups[0]["lr"]
    assert lr_after_step != lr_before_training, "schedule should have moved"

    # Perturb the weights so we can confirm the reject-reload reverts them.
    with torch.no_grad():
        for param in wrapper.nnet.parameters():
            param.add_(1.0)

    # Rejection: reload the pre-training checkpoint, keeping the LR clock.
    wrapper.load_checkpoint("temp.pth.tar", restore_lr_schedule=False)

    # Weights reverted...
    assert torch.equal(next(iter(wrapper.nnet.parameters())).detach(), reference_param)
    # ...but the LR clock did NOT rewind: it stays at the post-training position,
    # and the optimizer LR is re-synced to the scheduler (not the pre-step value).
    assert wrapper.scheduler.last_epoch == last_epoch_after_step
    assert wrapper.optimizer.param_groups[0]["lr"] == lr_after_step


def test_resume_reload_restores_lr_schedule(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """The default (``--resume``) path restores the saved scheduler position.

    Contrast with the reject path: a resume must continue the exact schedule.
    """
    wrapper, _ = _cosine_wrapper(ttt_game, test_config)
    assert wrapper.scheduler is not None

    # Advance the schedule, then checkpoint at that position (latest.pth.tar).
    for _ in range(3):
        wrapper.scheduler.step()
    saved_last_epoch = wrapper.scheduler.last_epoch
    saved_lr = wrapper.optimizer.param_groups[0]["lr"]
    wrapper.save_checkpoint("latest.pth.tar")

    # Advance further to prove the reload rewinds to the saved position.
    for _ in range(2):
        wrapper.scheduler.step()
    assert wrapper.scheduler.last_epoch != saved_last_epoch

    wrapper.load_checkpoint("latest.pth.tar")  # default restore_lr_schedule=True
    assert wrapper.scheduler.last_epoch == saved_last_epoch
    assert wrapper.optimizer.param_groups[0]["lr"] == saved_lr


def test_load_weights_yields_fresh_optimizer_and_scheduler(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """Warm start (``load_weights``) loads weights only: the optimizer LR and
    scheduler clock start fresh at the config LR, not the donor's annealed state (L4).
    """
    import torch

    # Donor: advance the schedule deep into the anneal, then checkpoint.
    donor, config = _cosine_wrapper(ttt_game, test_config)
    assert donor.scheduler is not None
    for _ in range(8):
        donor.scheduler.step()
    donor_lr = donor.optimizer.param_groups[0]["lr"]
    assert donor_lr < config.net_config.learning_rate, "donor should be mid-anneal"
    with torch.no_grad():
        for param in donor.nnet.parameters():
            param.add_(0.5)  # make the donor weights distinctive
    donor.save_checkpoint("best.pth.tar")
    donor_param = next(iter(donor.nnet.parameters())).detach().clone()

    # Recipient: a fresh run that warm-starts from the donor's weights.
    recipient = NNetWrapper(ttt_game, config)
    recipient.load_weights("best.pth.tar")

    # Weights adopted...
    assert torch.equal(next(iter(recipient.nnet.parameters())).detach(), donor_param)
    # ...but the optimisation is fresh: first generation trains at the peak LR,
    # and the scheduler is back at its initial position.
    assert recipient.optimizer.param_groups[0]["lr"] == config.net_config.learning_rate
    assert recipient.scheduler is not None
    assert recipient.scheduler.last_epoch == 0


def test_constant_scheduler_is_none(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """Both None and the explicit "constant" alias build no scheduler (L5)."""
    for value in (None, "constant"):
        net_config = replace(test_config.net_config, lr_scheduler=value)
        config = replace(test_config, net_config=net_config)
        wrapper = NNetWrapper(ttt_game, config)
        assert wrapper.scheduler is None
        assert wrapper.optimizer.param_groups[0]["lr"] == config.net_config.learning_rate


def test_step_scheduler_decays_at_milestones(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """The "step" scheduler multiplies LR by lr_gamma at each milestone (L5)."""
    net_config = replace(
        test_config.net_config,
        lr_scheduler="step",
        lr_milestones=(2, 4),
        lr_gamma=0.1,
        epochs=1,
    )
    config = replace(test_config, num_generations=6, net_config=net_config)

    seq = _lr_sequence(config, ttt_game, steps=config.num_generations)

    base = config.net_config.learning_rate
    # seq[i] is the LR after i steps: constant until milestone 2, then ×0.1,
    # then ×0.1 again at milestone 4.
    assert seq[0] == base
    assert seq[2] == pytest.approx(base * 0.1)
    assert seq[4] == pytest.approx(base * 0.01)


def test_step_scheduler_requires_milestones(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """ "step" with empty lr_milestones is a config error (L5)."""
    net_config = replace(test_config.net_config, lr_scheduler="step")  # lr_milestones defaults to ()
    config = replace(test_config, net_config=net_config)
    with pytest.raises(ValueError, match="lr_milestones"):
        NNetWrapper(ttt_game, config)


def test_unknown_scheduler_raises(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """An unrecognised lr_scheduler value is rejected (L5)."""
    net_config = replace(test_config.net_config, lr_scheduler="nope")
    config = replace(test_config, net_config=net_config)
    with pytest.raises(ValueError, match="Unknown lr_scheduler"):
        NNetWrapper(ttt_game, config)


def _ttt_eval_positions(game: TicTacToeGame, count: int) -> list[np.ndarray]:
    """A few distinct canonical TTT compact boards from random short games."""
    rng = np.random.default_rng(0)
    positions: list[np.ndarray] = []
    for _ in range(count):
        board = game.initialise_board()
        player = 1
        for _ in range(rng.integers(1, 4)):
            legal = np.flatnonzero(game.valid_move_masking(board, player))
            board, player = game.get_next_state(board, player, int(rng.choice(legal)))
        positions.append(game.get_canonical_form(board, player).to_compact())
    return positions


def _uniform_over_legal(game: TicTacToeGame, compact: np.ndarray) -> np.ndarray:
    board = game.board_from_compact(compact)
    valids = game.valid_move_masking(board, 1)
    total = valids.sum()
    return (valids / total).astype(np.float32) if total > 0 else valids.astype(np.float32)


def _make_eval_set(game: TicTacToeGame, compacts: list[np.ndarray], *, with_compact: bool) -> EvalSet:
    boards = np.array([game.encode_compact(c) for c in compacts])
    policies = np.array([_uniform_over_legal(game, c) for c in compacts])
    values = np.zeros(len(compacts), dtype=np.float32)
    return EvalSet(
        boards=boards,
        target_policies=policies,
        target_values=values,
        compact_boards=np.array(compacts) if with_compact else None,
    )


def test_mcts_agreement_returns_none_without_compact_boards(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """No compact boards ⇒ nothing to re-search ⇒ the diagnostic is skipped."""
    wrapper = NNetWrapper(ttt_game, test_config)
    compacts = _ttt_eval_positions(ttt_game, 4)
    eval_set = _make_eval_set(ttt_game, compacts, with_compact=False)
    assert wrapper._compute_mcts_agreement(eval_set) is None


def test_mcts_agreement_is_computed_and_logged(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """With compact boards the net-vs-own-MCTS agreement is computed and logged."""
    wrapper = NNetWrapper(ttt_game, test_config)
    compacts = _ttt_eval_positions(ttt_game, 4)
    eval_set = _make_eval_set(ttt_game, compacts, with_compact=True)

    # Direct diagnostic: fractions in [0, 1].
    agreement = wrapper._compute_mcts_agreement(eval_set)
    assert agreement is not None
    top1, top5 = agreement
    assert 0.0 <= top1 <= top5 <= 1.0

    # End-to-end: train one generation and confirm the new series is persisted.
    examples = [(compact, sparsify(_uniform_over_legal(ttt_game, compact)), 0.0) for compact in compacts]
    metrics = MetricsCollector(config=test_config)
    wrapper.train(examples, generation=1, metrics=metrics, eval_set=eval_set)

    records = metrics._policy_accuracy_records
    assert records, "no policy-accuracy records logged"
    assert all("mcts_top1_accuracy" in r and "mcts_top5_accuracy" in r for r in records)
    assert all(0.0 <= r["mcts_top1_accuracy"] <= 1.0 for r in records)


# ── Policy–Value Consistency (S1/S4) ───────────────────────────────────────


class _ScriptedPredictor(IPolicyValuePredictor):
    """A predictor with hand-set policy/value outputs keyed by board state.

    Not a mock of *game logic* (the real ``TicTacToeGame`` drives all rules) —
    only the network surface is scripted, which is exactly what lets us assert
    the PVC negamax/terminal/ranking maths against known-good outputs.
    """

    def __init__(
        self,
        action_size: int,
        *,
        policy_by_key: dict[bytes, NDArray],
        value_by_key: dict[bytes, float],
        default_value: float = 0.0,
    ) -> None:
        self._action_size = action_size
        self._policy_by_key = policy_by_key
        self._value_by_key = value_by_key
        self._default_value = default_value

    def predict(self, board: IBoard) -> tuple[NDArray, float]:
        key = board.state_key
        policy = self._policy_by_key.get(key)
        if policy is None:
            policy = np.full(self._action_size, 1.0 / self._action_size, dtype=np.float64)
        return policy, self._value_by_key.get(key, self._default_value)

    def predict_batch(self, boards) -> tuple[list[NDArray], list[float]]:  # type: ignore[no-untyped-def]
        results = [self.predict(board) for board in boards]
        return [p for p, _ in results], [v for _, v in results]


def _descending_policy_over(game: TicTacToeGame, legal: NDArray) -> NDArray:
    """Policy that assigns strictly descending probability to ``legal`` in order."""
    policy = np.zeros(game.get_action_size(), dtype=np.float64)
    weights = np.arange(len(legal), 0, -1, dtype=np.float64)
    policy[legal] = weights / weights.sum()
    return policy


def test_spearman_agree_reverse_and_undefined() -> None:
    """The rank-correlation helper: +1 agree, −1 reverse, NaN when undefined."""
    assert BaseNNetWrapper._spearman(np.array([3.0, 2.0, 1.0]), np.array([9.0, 5.0, 1.0])) == pytest.approx(1.0)
    assert BaseNNetWrapper._spearman(np.array([3.0, 2.0, 1.0]), np.array([1.0, 5.0, 9.0])) == pytest.approx(-1.0)
    # Ties are averaged, so identical-with-ties still correlates perfectly.
    assert BaseNNetWrapper._spearman(np.array([1.0, 1.0, 2.0]), np.array([4.0, 4.0, 9.0])) == pytest.approx(1.0)
    # Fewer than two items, or zero variance, is undefined → NaN (excluded, not poisoning).
    assert np.isnan(BaseNNetWrapper._spearman(np.array([1.0]), np.array([1.0])))
    assert np.isnan(BaseNNetWrapper._spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))


def _non_terminal_ttt_board(game: TicTacToeGame) -> tuple[IBoard, NDArray]:
    """A canonical, non-terminal TTT board and its legal actions.

    Player 1 (+1) to move with no immediate winning move, so *every* child is
    non-terminal and the scripted value head fully controls Q₁.
    """
    compact = np.array([[1, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.int8)
    board = game.board_from_compact(compact)
    legal = np.flatnonzero(game.valid_move_masking(board, 1))
    assert len(legal) >= 2
    for action in legal:
        child, child_player = game.get_next_state(board, 1, int(action))
        assert game.get_game_ended(child, child_player) == 0.0, "test board must have no terminal children"
    return board, legal


def test_one_ply_q_values_negamax_ranks_best_child_top(ttt_game: TicTacToeGame) -> None:
    """Q₁ = −V(child): the move into the worst child-value ranks best for the mover."""
    board, legal = _non_terminal_ttt_board(ttt_game)
    policy = _descending_policy_over(ttt_game, legal)

    # Give the LAST legal move (lowest policy prob) the most negative child value,
    # so its Q₁ = −V is the largest — the one-ply-best move despite low prior.
    value_by_key: dict[bytes, float] = {}
    best_action = int(legal[-1])
    for rank, action in enumerate(legal):
        child, child_player = ttt_game.get_next_state(board, 1, int(action))
        canonical_child = ttt_game.get_canonical_form(child, child_player)
        value_by_key[canonical_child.state_key] = -2.0 if int(action) == best_action else float(rank)

    predictor = _ScriptedPredictor(
        ttt_game.get_action_size(),
        policy_by_key={board.state_key: policy},
        value_by_key=value_by_key,
    )
    result = BaseNNetWrapper._one_ply_q_values(ttt_game, predictor, board, PVC_TOP_K)
    assert result is not None
    candidate_actions, _, q1_values = result
    assert int(candidate_actions[int(np.argmax(q1_values))]) == best_action


def test_one_ply_q_values_terminal_child_uses_result_not_value(ttt_game: TicTacToeGame) -> None:
    """A winning move scores Q₁ = +1 from the true result, ignoring the value head."""
    # Player 1 (+1) has an immediate winning move; value head is set adversarially.
    compact = np.array([[1, 1, 0], [-1, -1, 0], [0, 0, 0]], dtype=np.int8)
    board = ttt_game.board_from_compact(compact)
    legal = np.flatnonzero(ttt_game.valid_move_masking(board, 1))

    winning_actions = [
        int(a) for a in legal if ttt_game.get_game_ended(*ttt_game.get_next_state(board, 1, int(a))) != 0.0
    ]
    assert len(winning_actions) == 1, "expected exactly one immediate winning move"
    winning_action = winning_actions[0]

    # Every child (incl. the winning one) gets a misleading value; the winning
    # move is terminal so its Q₁ must come from the result (+1), not −V.
    predictor = _ScriptedPredictor(
        ttt_game.get_action_size(),
        policy_by_key={board.state_key: _descending_policy_over(ttt_game, legal)},
        value_by_key={},
        default_value=0.5,  # would give Q₁ = −0.5 if V were (wrongly) used
    )
    result = BaseNNetWrapper._one_ply_q_values(ttt_game, predictor, board, PVC_TOP_K)
    assert result is not None
    candidate_actions, _, q1_values = result
    winning_slot = int(np.flatnonzero(candidate_actions == winning_action)[0])
    assert q1_values[winning_slot] == pytest.approx(1.0)
    # ...and it is the top move by Q₁ (a win beats any non-terminal −0.5).
    assert int(np.argmax(q1_values)) == winning_slot


def test_policy_value_consistency_perfect_agreement(ttt_game: TicTacToeGame) -> None:
    """Policy ranking == Q₁ ranking → argmax-match 1, Spearman +1."""
    board, legal = _non_terminal_ttt_board(ttt_game)
    policy = _descending_policy_over(ttt_game, legal)

    value_by_key: dict[bytes, float] = {}
    num = len(legal)
    for rank, action in enumerate(legal):
        child, child_player = ttt_game.get_next_state(board, 1, int(action))
        canonical_child = ttt_game.get_canonical_form(child, child_player)
        # Q₁ = −V descending in the same order as the policy → V = rank − num.
        value_by_key[canonical_child.state_key] = float(rank - num)

    predictor = _ScriptedPredictor(
        ttt_game.get_action_size(),
        policy_by_key={board.state_key: policy},
        value_by_key=value_by_key,
    )
    compacts = np.array([board.to_compact()])
    result = BaseNNetWrapper._policy_value_consistency(ttt_game, predictor, compacts, PVC_TOP_K)
    assert result is not None
    assert result["pvc_argmax_match"] == pytest.approx(1.0)
    assert result["pvc_spearman"] == pytest.approx(1.0)


def test_policy_value_consistency_reversed(ttt_game: TicTacToeGame) -> None:
    """Policy ranking reversed vs Q₁ → argmax-match 0, Spearman −1."""
    board, legal = _non_terminal_ttt_board(ttt_game)
    policy = _descending_policy_over(ttt_game, legal)

    value_by_key: dict[bytes, float] = {}
    for rank, action in enumerate(legal):
        child, child_player = ttt_game.get_next_state(board, 1, int(action))
        canonical_child = ttt_game.get_canonical_form(child, child_player)
        # Q₁ = −V ascending while policy descends → perfect anti-correlation.
        value_by_key[canonical_child.state_key] = float(-(rank + 1))

    predictor = _ScriptedPredictor(
        ttt_game.get_action_size(),
        policy_by_key={board.state_key: policy},
        value_by_key=value_by_key,
    )
    compacts = np.array([board.to_compact()])
    result = BaseNNetWrapper._policy_value_consistency(ttt_game, predictor, compacts, PVC_TOP_K)
    assert result is not None
    assert result["pvc_argmax_match"] == pytest.approx(0.0)
    assert result["pvc_spearman"] == pytest.approx(-1.0)


def test_policy_value_consistency_skips_positions_below_two_moves(ttt_game: TicTacToeGame) -> None:
    """A full (single legal move) board is skipped; all-skipped → None (no crash, no NaN)."""
    # Classic drawn, full board: the only legal action is pass ⇒ < 2 candidates.
    full = np.array([[1, -1, 1], [1, -1, -1], [-1, 1, 1]], dtype=np.int8)
    board = ttt_game.board_from_compact(full)
    assert int(np.count_nonzero(ttt_game.valid_move_masking(board, 1))) == 1

    predictor = _ScriptedPredictor(ttt_game.get_action_size(), policy_by_key={}, value_by_key={})
    result = BaseNNetWrapper._policy_value_consistency(ttt_game, predictor, np.array([full]), PVC_TOP_K)
    assert result is None


def test_value_symmetry_mae_matches_hand_computed(ttt_game: TicTacToeGame) -> None:
    """MAE = mean|V(s) − V(reflect(s))| with the identity excluded."""
    # Asymmetric board so its symmetric variants are genuinely different states.
    compact = np.array([[1, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
    board = ttt_game.board_from_compact(compact)

    # V=+1 for the source only; every other state defaults to −1, so each
    # non-identity variant contributes |1 − (−1)| = 2.
    predictor = _ScriptedPredictor(
        ttt_game.get_action_size(),
        policy_by_key={},
        value_by_key={board.state_key: 1.0},
        default_value=-1.0,
    )
    mae = BaseNNetWrapper._value_symmetry_mae(ttt_game, predictor, np.array([compact]))
    assert mae == pytest.approx(2.0)


def test_value_symmetry_mae_none_for_fully_symmetric_position(ttt_game: TicTacToeGame) -> None:
    """The empty board is its own image under every symmetry ⇒ no variants ⇒ None."""
    empty = np.zeros((ttt_game.N, ttt_game.N), dtype=np.int8)
    predictor = _ScriptedPredictor(ttt_game.get_action_size(), policy_by_key={}, value_by_key={})
    assert BaseNNetWrapper._value_symmetry_mae(ttt_game, predictor, np.array([empty])) is None


def test_pvc_computed_and_logged_end_to_end(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """A real wrapper computes valid PVC + value-symmetry ranges and persists them."""
    wrapper = NNetWrapper(ttt_game, test_config)
    compacts = _ttt_eval_positions(ttt_game, 4)
    eval_set = _make_eval_set(ttt_game, compacts, with_compact=True)

    pvc = wrapper._compute_policy_value_consistency(eval_set)
    assert pvc is not None
    assert 0.0 <= pvc["pvc_argmax_match"] <= 1.0
    assert np.isnan(pvc["pvc_spearman"]) or -1.0 <= pvc["pvc_spearman"] <= 1.0
    mae = wrapper._compute_value_symmetry_mae(eval_set)
    assert mae is not None and mae >= 0.0

    examples = [(compact, sparsify(_uniform_over_legal(ttt_game, compact)), 0.0) for compact in compacts]
    metrics = MetricsCollector(config=test_config)
    wrapper.train(examples, generation=1, metrics=metrics, eval_set=eval_set)

    records = metrics._policy_value_consistency_records
    assert records, "no policy-value-consistency records logged"
    assert all("pvc_argmax_match" in r and "pvc_spearman" in r for r in records)
    assert all("value_symmetry_mae" in r for r in records)


def test_pvc_returns_none_without_compact_boards(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """No compact boards ⇒ no children to build ⇒ the diagnostic is skipped."""
    wrapper = NNetWrapper(ttt_game, test_config)
    compacts = _ttt_eval_positions(ttt_game, 4)
    eval_set = _make_eval_set(ttt_game, compacts, with_compact=False)
    assert wrapper._compute_policy_value_consistency(eval_set) is None
    assert wrapper._compute_value_symmetry_mae(eval_set) is None
