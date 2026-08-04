"""Optimizer-continuation semantics across the Coach's generation cycle (bug-sweep C5).

``tests/games/test_base_wrapper.py`` already pins the LR-clock behaviour
(reject keeps the schedule position, resume restores it) and the weight-decay
re-assert. What it does NOT pin is the Adam-moment contract itself:

- moments must persist (not reset) across an accepted generation, and
- an arena reject-reload must revert the moments EXACTLY to the pre-training
  snapshot alongside the weights — a partial revert (weights back, moments
  forward) would make every post-rejection generation train with moments
  accumulated from weights that no longer exist.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper


def _make_examples(game, n: int = 64):
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        board = game.initialise_board()
        pi = rng.dirichlet(np.ones(game.get_action_size()))
        out.append(
            (
                board.to_compact(),
                (np.arange(game.get_action_size(), dtype=np.int32), pi.astype(np.float32)),
                1.0,
            )
        )
    return out


def _moments(wrapper) -> dict:
    state = wrapper.optimizer.state_dict()["state"]
    return {k: {kk: vv.clone() if torch.is_tensor(vv) else vv for kk, vv in v.items()} for k, v in state.items()}


def _moments_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        if a[k].keys() != b[k].keys():
            return False
        for kk in a[k]:
            va, vb = a[k][kk], b[k][kk]
            if torch.is_tensor(va):
                if not torch.equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


@pytest.fixture
def wrapper(ttt_game, test_config):
    torch.manual_seed(0)
    net_config = replace(test_config.net_config, epochs=1, lr_scheduler="cosine", lr_eta_min=1e-4)
    return NNetWrapper(ttt_game, replace(test_config, net_config=net_config, num_generations=8))


def test_adam_moments_persist_across_accepted_generations(wrapper, ttt_game) -> None:
    examples = _make_examples(ttt_game)
    wrapper.train(examples, generation=1)
    after_g1 = _moments(wrapper)
    assert after_g1, "training must populate Adam state"
    wrapper.train(examples, generation=2)
    after_g2 = _moments(wrapper)
    assert not _moments_equal(after_g1, after_g2), "moments must keep accumulating, not reset"


def test_reject_reload_reverts_weights_and_moments_together(wrapper, ttt_game) -> None:
    examples = _make_examples(ttt_game)
    wrapper.train(examples, generation=1)
    pre_training_moments = _moments(wrapper)

    # Coach's cycle: preserve the incumbent BEFORE training the candidate...
    wrapper.save_checkpoint(filename="temp.pth.tar")
    incumbent_weights = {k: v.clone() for k, v in wrapper.nnet.state_dict().items()}
    wrapper.train(examples, generation=2)
    assert not _moments_equal(pre_training_moments, _moments(wrapper))

    # ...and on rejection revert weights + moments but never the LR clock.
    clock_before = wrapper.scheduler.last_epoch
    wrapper.load_checkpoint(filename="temp.pth.tar", restore_lr_schedule=False)

    assert _moments_equal(_moments(wrapper), pre_training_moments), (
        "arena reject must restore the Adam moments saved with the incumbent"
    )
    for key, value in wrapper.nnet.state_dict().items():
        assert torch.equal(value, incumbent_weights[key]), f"weight {key} not reverted"
    assert wrapper.scheduler.last_epoch == clock_before, "reject must not rewind the LR clock"
