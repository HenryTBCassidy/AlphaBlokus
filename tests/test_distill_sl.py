"""Tests for the distillation trainer's ``--net-size`` sweep override."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace

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


def test_the_two_ab_arms_differ_only_by_the_score_head() -> None:
    """Everything else about the arm config must be identical, or the A/B is unreadable."""
    base = load_args(_BASE_CONFIG)
    off = _arm_config(base, _args(None)).net_config
    on = _arm_config(base, _args(None, score_head=True)).net_config

    assert replace(on, score_head=False) == off
