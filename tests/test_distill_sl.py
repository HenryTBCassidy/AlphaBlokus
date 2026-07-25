"""Tests for the distillation trainer's ``--net-size`` sweep override."""

from __future__ import annotations

from argparse import Namespace

from alphablokus.config import load_args
from scripts.distill_sl import _arm_config

_BASE_CONFIG = "run_configurations/test_run.json"


def _args(net_size: str | None) -> Namespace:
    return Namespace(lr=1e-4, batch_size=256, net_size=net_size)


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
