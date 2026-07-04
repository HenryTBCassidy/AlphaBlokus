import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from alphablokus.config import MCTSConfig, NetConfig, RunConfig, TrainingPerfConfig, load_args


def test_load_args_from_test_run_json():
    """load_args should parse the test_run.json file into a valid RunConfig."""
    config = load_args("run_configurations/test_run.json")
    assert isinstance(config, RunConfig)
    assert config.run_name == "test_run"
    assert config.num_generations == 2
    assert config.num_eps == 10


def test_config_directory_properties():
    """Directory properties should be derived from root_directory and run_name."""
    config = load_args("run_configurations/test_run.json")
    run_dir = config.run_directory

    assert config.log_directory == run_dir / "Logs"
    assert config.timings_directory == run_dir / "Timings"
    assert config.self_play_history_directory == run_dir / "SelfPlayHistory"
    assert config.net_directory == run_dir / "Nets"
    assert config.training_data_directory == run_dir / "TrainingData"
    assert config.arena_data_directory == run_dir / "ArenaData"
    assert config.report_directory == run_dir / "Reporting"


def test_config_frozen():
    """RunConfig is a frozen dataclass — attribute assignment should raise."""
    config = load_args("run_configurations/test_run.json")
    with pytest.raises(FrozenInstanceError):
        config.run_name = "hacked"  # type: ignore[misc]


def test_mcts_config_fields():
    """MCTSConfig fields should load correctly from JSON."""
    config = load_args("run_configurations/test_run.json")
    assert isinstance(config.mcts_config, MCTSConfig)
    assert config.mcts_config.num_mcts_sims == 2
    assert config.mcts_config.cpuct == 1


def test_net_config_fields():
    """NetConfig fields should load correctly from JSON."""
    config = load_args("run_configurations/test_run.json")
    assert isinstance(config.net_config, NetConfig)
    assert config.net_config.learning_rate == 0.001
    assert config.net_config.dropout == 0.3
    assert config.net_config.epochs == 1
    assert config.net_config.batch_size == 10
    assert config.net_config.num_filters == 512
    assert config.net_config.num_residual_blocks == 1


def test_training_perf_defaults_are_off():
    """Configs that don't mention perf get the all-off TrainingPerfConfig — current behaviour."""
    config = load_args("run_configurations/test_run.json")
    assert config.net_config.perf == TrainingPerfConfig()
    perf = config.net_config.perf
    assert perf.autocast_dtype == "off"
    assert perf.tf32 is False
    assert perf.cudnn_benchmark is False
    assert perf.channels_last is False
    assert perf.compile is False
    assert perf.dataloader_workers == 0
    assert perf.pin_memory is False
    assert perf.persistent_workers is False
    assert perf.log_every_batches == 1


def test_training_perf_loads_from_json(tmp_path):
    """A net_config.perf block in the JSON populates TrainingPerfConfig."""
    raw = json.loads(Path("run_configurations/test_run.json").read_text())
    raw["net_config"]["perf"] = {
        "autocast_dtype": "bf16",
        "tf32": True,
        "channels_last": True,
        "compile": True,
        "dataloader_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "log_every_batches": 25,
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))
    config = load_args(path)
    perf = config.net_config.perf
    assert perf.autocast_dtype == "bf16"
    assert perf.tf32 is True
    assert perf.channels_last is True
    assert perf.compile is True
    assert perf.dataloader_workers == 4
    assert perf.pin_memory is True
    assert perf.persistent_workers is True
    assert perf.log_every_batches == 25
