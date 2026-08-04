"""Tests for ``validate_active_path_knobs`` (A1) — the ignored-knob guard.

``dataclass_wizard`` silently discards unknown JSON keys, so a stale
``dirichlet_epsilon`` in a Gumbel config is invisible: it implies root-noise
exploration that the Gumbel path never performs, and anyone who tunes it tunes
nothing. This guard turns that silence into a load-time error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphablokus.config import WARM_CONTINUATION_MAX_LR, load_args, validate_active_path_knobs

GUMBEL_MCTS = {"num_mcts_sims": 8, "cpuct": 1.0, "search_policy": "gumbel", "gumbel_max_considered": 16}
PUCT_MCTS = {"num_mcts_sims": 8, "cpuct": 1.0}
NET = {
    "learning_rate": 0.00025,
    "dropout": 0.0,
    "epochs": 1,
    "batch_size": 8,
    "cuda": False,
    "num_filters": 16,
    "num_residual_blocks": 1,
}


def _config(**overrides) -> dict:
    base = {
        "game": "blokusduo",
        "run_name": "validator_test",
        "num_generations": 1,
        "num_eps": 1,
        "update_threshold": 0.55,
        "num_arena_matches": 2,
        "root_directory": "./temp",
        "load_model": False,
        "selfplay_backend": "jax",
        "mcts_config": dict(GUMBEL_MCTS),
        "net_config": dict(NET),
    }
    base.update(overrides)
    return base


# --- ignored knobs under Gumbel -------------------------------------------


@pytest.mark.parametrize("knob", ["dirichlet_epsilon", "dirichlet_alpha"])
def test_gumbel_config_rejects_dirichlet_knobs(knob: str) -> None:
    config = _config()
    config["mcts_config"][knob] = 0.25

    with pytest.raises(ValueError, match=knob):
        validate_active_path_knobs(config)


def test_gumbel_config_rejects_temp_threshold() -> None:
    with pytest.raises(ValueError, match="temp_threshold"):
        validate_active_path_knobs(_config(temp_threshold=12))


def test_error_names_every_offending_knob_at_once() -> None:
    config = _config(temp_threshold=12)
    config["mcts_config"]["dirichlet_epsilon"] = 0.25
    config["mcts_config"]["dirichlet_alpha"] = 0.03

    with pytest.raises(ValueError) as excinfo:
        validate_active_path_knobs(config)

    message = str(excinfo.value)
    for knob in ("dirichlet_epsilon", "dirichlet_alpha", "temp_threshold"):
        assert knob in message
    # The message must say what to do, not just what is wrong.
    assert "Delete the ignored keys" in message


def test_error_explains_why_the_knob_is_inert() -> None:
    config = _config()
    config["mcts_config"]["dirichlet_epsilon"] = 0.25

    with pytest.raises(ValueError, match="root_log_pi = log_pi"):
        validate_active_path_knobs(config)


def test_clean_gumbel_config_passes() -> None:
    validate_active_path_knobs(_config())


def test_source_is_named_in_the_error() -> None:
    config = _config(temp_threshold=12)

    with pytest.raises(ValueError, match="my_config.json"):
        validate_active_path_knobs(config, source="my_config.json")


# --- the same knobs are REQUIRED on the PUCT path -------------------------


def test_puct_config_keeps_dirichlet_knobs() -> None:
    """These are live on the python/PUCT path — the guard must not touch them."""
    config = _config(selfplay_backend="python", temp_threshold=12)
    config["mcts_config"] = dict(PUCT_MCTS, dirichlet_epsilon=0.25, dirichlet_alpha=0.03)

    validate_active_path_knobs(config)


def test_puct_config_requires_temp_threshold() -> None:
    config = _config(selfplay_backend="python")
    config["mcts_config"] = dict(PUCT_MCTS)

    with pytest.raises(ValueError, match="temp_threshold is required"):
        validate_active_path_knobs(config)


def test_puct_config_rejects_null_temp_threshold() -> None:
    config = _config(selfplay_backend="python", temp_threshold=None)
    config["mcts_config"] = dict(PUCT_MCTS)

    with pytest.raises(ValueError, match="temp_threshold is required"):
        validate_active_path_knobs(config)


# --- warm-continuation learning rate --------------------------------------


def test_warm_continuation_at_peak_lr_is_refused() -> None:
    config = _config(load_model=True)
    config["net_config"]["learning_rate"] = 1e-3

    with pytest.raises(ValueError, match="warm continuation"):
        validate_active_path_knobs(config)


def test_warm_continuation_at_the_continuation_rate_passes() -> None:
    config = _config(load_model=True)
    config["net_config"]["learning_rate"] = WARM_CONTINUATION_MAX_LR

    validate_active_path_knobs(config)


def test_from_scratch_run_may_use_the_peak_rate() -> None:
    config = _config(load_model=False)
    config["net_config"]["learning_rate"] = 1e-3

    validate_active_path_knobs(config)


def test_explicit_opt_in_allows_a_high_warm_start_rate() -> None:
    """The escape hatch the B3 sweep arms use — deliberate, and recorded in the config."""
    config = _config(load_model=True)
    config["net_config"]["learning_rate"] = 1e-3
    config["net_config"]["allow_high_warm_start_lr"] = True

    validate_active_path_knobs(config)


# --- end to end through load_args -----------------------------------------


def test_load_args_applies_the_validator(tmp_path: Path) -> None:
    config = _config(temp_threshold=12)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="temp_threshold"):
        load_args(path)


def test_load_args_accepts_a_clean_gumbel_config(tmp_path: Path) -> None:
    path = tmp_path / "good.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")

    loaded = load_args(path)

    assert loaded.temp_threshold is None
    assert loaded.mcts_config.search_policy == "gumbel"


def test_every_shipped_config_passes_the_validator() -> None:
    """No committed run config may set a knob its own search path ignores.

    ``full_run.json`` is excluded: it predates the ``game`` field and fails to
    load for an unrelated, pre-existing reason.
    """
    root = Path(__file__).resolve().parent.parent / "run_configurations"
    failures = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "full_run.json":
            continue
        try:
            load_args(path)
        except Exception as err:  # noqa: BLE001 - collecting all failures to report at once
            failures.append(f"{path.relative_to(root)}: {err}")

    assert not failures, "shipped configs failed validation:\n" + "\n".join(failures)


def test_sampling_temp_threshold_raises_when_unset(tmp_path: Path) -> None:
    """A Gumbel config has no temp_threshold; asking for one must fail loudly."""
    path = tmp_path / "gumbel.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    loaded = load_args(path)

    with pytest.raises(ValueError, match="samples moves and needs it"):
        _ = loaded.sampling_temp_threshold
