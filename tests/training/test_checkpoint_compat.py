"""``load_state_dict_compat``: tolerant about the score head, strict about everything else.

The score head has to cross the checkpoint boundary in both directions (plan
``docs/plans/score-auxiliary-target.md`` S3) — but making the load non-strict *in general*
would silently accept a genuinely wrong checkpoint, which is a far worse failure than the
one being fixed. These tests pin both halves of that contract on plain modules, with the
real-net round trips living in ``tests/games/blokusduo/nn/test_score_head.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from loguru import logger

from alphablokus.training.checkpoint_compat import load_state_dict_compat


class _Body(nn.Module):
    """A trunk with no score head."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 3)


class _BodyWithScore(nn.Module):
    """The same trunk plus a score head, registered under the tolerated prefix."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 3)
        self.score_head = nn.Linear(3, 1)


def test_old_checkpoint_into_a_score_head_module_loads_and_leaves_the_head_fresh() -> None:
    donor = _Body()
    target = _BodyWithScore()
    fresh_head = {k: v.clone() for k, v in target.score_head.state_dict().items()}

    load_state_dict_compat(target, donor.state_dict())

    assert torch.equal(target.trunk.weight, donor.trunk.weight)
    assert torch.equal(target.trunk.bias, donor.trunk.bias)
    for key, value in target.score_head.state_dict().items():
        assert torch.equal(value, fresh_head[key]), f"score head tensor {key} was overwritten"


def test_score_head_checkpoint_into_a_plain_module_ignores_the_extra_tensors() -> None:
    donor = _BodyWithScore()
    target = _Body()

    load_state_dict_compat(target, donor.state_dict())

    assert torch.equal(target.trunk.weight, donor.trunk.weight)


def test_missing_tensors_that_are_not_the_score_head_still_raise() -> None:
    """The guard that stops half a net loading silently — e.g. an fc/conv head swap."""
    target = _Body()
    partial = {"trunk.bias": torch.zeros(3)}

    with pytest.raises(RuntimeError, match="trunk.weight"):
        load_state_dict_compat(target, partial)


def test_unexpected_tensors_that_are_not_the_score_head_still_raise() -> None:
    target = _Body()
    state = dict(target.state_dict())
    state["some_other_head.weight"] = torch.zeros(2, 2)

    with pytest.raises(RuntimeError, match="some_other_head.weight"):
        load_state_dict_compat(target, state)


def test_a_shape_mismatch_still_raises() -> None:
    """Wrong-sized tensors are torch's own error and must not be swallowed."""
    target = _Body()
    state = dict(target.state_dict())
    state["trunk.weight"] = torch.zeros(3, 99)

    with pytest.raises(RuntimeError):
        load_state_dict_compat(target, state)


def test_missing_score_head_tensors_are_logged_by_name() -> None:
    """A tolerated mismatch is still *loud* — the operator sees exactly what was skipped."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message), level="INFO")
    try:
        load_state_dict_compat(_BodyWithScore(), _Body().state_dict())
    finally:
        logger.remove(sink_id)

    joined = "".join(messages)
    assert "score_head.weight" in joined
    assert "score_head.bias" in joined
