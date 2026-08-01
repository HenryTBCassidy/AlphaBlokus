"""Tests for the distillation trainer's ``--net-size`` sweep + auxiliary-head switches."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace

import pytest

from alphablokus.config import load_args
from scripts.distill_sl import _arm_config

_BASE_CONFIG = "run_configurations/test_run.json"


def _args(net_size: str | None, **overrides: object) -> Namespace:
    defaults: dict[str, object] = {
        "lr": 1e-4,
        "batch_size": 256,
        "net_size": net_size,
        "score_head": False,
        "score_loss_weight": 0.15,
        "score_scale": 25.0,
        "ownership_head": False,
        "ownership_loss_weight": 0.15,
        "reply_head": False,
        "reply_loss_weight": 0.15,
    }
    return Namespace(**{**defaults, **overrides})


def test_net_size_override_reshapes_the_net() -> None:
    base = load_args(_BASE_CONFIG)
    reshaped = _arm_config(base, _args("160x10")).net_config
    assert reshaped.num_filters == 160
    assert reshaped.num_residual_blocks == 10


def test_no_net_size_keeps_the_config_shape() -> None:
    base = load_args(_BASE_CONFIG)
    kept = _arm_config(base, _args(None)).net_config
    assert kept.num_filters == base.net_config.num_filters
    assert kept.num_residual_blocks == base.net_config.num_residual_blocks


def test_score_head_flags_reach_the_arm_net_config() -> None:
    """``--score-head`` is the S7 A/B switch, so it must survive the arm reshape."""
    base = load_args(_BASE_CONFIG)

    off = _arm_config(base, _args(None)).net_config
    assert off.score_head is False

    on = _arm_config(base, _args(None, score_head=True, score_loss_weight=0.3, score_scale=15.0)).net_config
    assert on.score_head is True
    assert on.score_loss_weight == 0.3
    assert on.score_scale == 15.0


@pytest.mark.parametrize("head", ["score_head", "ownership_head", "reply_head"])
def test_the_two_ab_arms_differ_only_by_the_head_under_test(head: str) -> None:
    """Everything else about the arm config must be identical, or the A/B is unreadable.

    Run for every auxiliary head, not just the score head: each is an A/B switch, and
    each must be the *only* field its arm changes.
    """
    base = load_args(_BASE_CONFIG)
    off = _arm_config(base, _args(None)).net_config
    on = _arm_config(base, _args(None, **{head: True})).net_config

    assert replace(on, **{head: False}) == off


def test_auxiliary_head_flags_reach_the_arm_net_config() -> None:
    """The N4/N5 switches must survive the arm reshape, weights included."""
    base = load_args(_BASE_CONFIG)

    off = _arm_config(base, _args(None)).net_config
    assert off.ownership_head is False
    assert off.reply_head is False

    on = _arm_config(
        base,
        _args(None, ownership_head=True, ownership_loss_weight=0.4, reply_head=True, reply_loss_weight=0.2),
    ).net_config
    assert on.ownership_head is True
    assert on.ownership_loss_weight == 0.4
    assert on.reply_head is True
    assert on.reply_loss_weight == 0.2


def test_unset_head_flags_inherit_the_base_config() -> None:
    """``None`` means "the config decides" — an argparse default must not overwrite it.

    Letting the flag default to ``False`` would quietly run the control arm for a
    config JSON that pinned the treatment on, which is exactly the sort of silent
    mis-run that makes an A/B unreadable.
    """
    base = load_args(_BASE_CONFIG)
    pinned = replace(base, net_config=replace(base.net_config, ownership_head=True, reply_head=True))

    inherited = _arm_config(pinned, _args(None, ownership_head=None, reply_head=None)).net_config
    assert inherited.ownership_head is True
    assert inherited.reply_head is True

    overridden = _arm_config(pinned, _args(None, ownership_head=False, reply_head=False)).net_config
    assert overridden.ownership_head is False
    assert overridden.reply_head is False
