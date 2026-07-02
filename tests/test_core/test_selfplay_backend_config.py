"""G1: the selfplay_backend config field parses and defaults correctly."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from dataclass_wizard import fromdict

from core.config import RunConfig, load_args


def test_default_backend_is_python(test_config: RunConfig) -> None:
    assert test_config.selfplay_backend == "python"


def test_backend_parses_from_json(test_config: RunConfig, tmp_path: Path) -> None:
    payload = dataclasses.asdict(test_config)
    payload["selfplay_backend"] = "jax"
    payload["root_directory"] = str(payload["root_directory"])
    config = fromdict(RunConfig, json.loads(json.dumps(payload, default=str)))
    assert config.selfplay_backend == "jax"


def test_existing_configs_still_load() -> None:
    config = load_args(Path("run_configurations/test_run.json"))
    assert config.selfplay_backend == "python"
