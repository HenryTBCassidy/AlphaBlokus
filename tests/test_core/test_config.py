from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.config import MCTSConfig, NetConfig, RunConfig, load_args


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
    assert config.net_config.num_channels == 512
    assert config.net_config.num_residual_blocks == 1
