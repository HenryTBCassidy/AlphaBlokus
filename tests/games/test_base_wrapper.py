"""Tests for shared ``BaseNNetWrapper`` behaviour (LR scheduler)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
from alphablokus.storage.metrics import EvalSet, MetricsCollector
from alphablokus.storage.sparse_policy import sparsify

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.games.tictactoe.game import TicTacToeGame


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
    from alphablokus.storage.sparse_policy import sparsify

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
