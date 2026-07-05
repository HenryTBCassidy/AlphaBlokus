import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from alphablokus.config import MCTSConfig, NetConfig, RunConfig, TournamentConfig, TrainingPerfConfig, load_args


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


def _write_config_with_net(tmp_path, net_config: dict):
    """test_run.json with its net-size keys dropped and ``net_config`` merged in."""
    raw = json.loads(Path("run_configurations/test_run.json").read_text())
    base = {k: v for k, v in raw["net_config"].items() if k not in ("num_filters", "num_residual_blocks")}
    raw["net_config"] = {**base, **net_config}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))
    return path


def test_step_scheduler_fields_load_from_json(tmp_path):
    """lr_milestones (JSON list → tuple) and lr_gamma parse for the "step" scheduler."""
    path = _write_config_with_net(
        tmp_path,
        {
            "lr_scheduler": "step",
            "lr_milestones": [20, 40],
            "lr_gamma": 0.3,
            "num_filters": 32,
            "num_residual_blocks": 1,
        },
    )
    config = load_args(path)
    assert config.net_config.lr_scheduler == "step"
    assert config.net_config.lr_milestones == (20, 40)
    assert config.net_config.lr_gamma == 0.3


def test_step_scheduler_field_defaults():
    """Configs that don't mention the step knobs get empty milestones + 0.1 gamma."""
    config = load_args("run_configurations/test_run.json")
    assert config.net_config.lr_milestones == ()
    assert config.net_config.lr_gamma == 0.1


def test_net_preset_fills_size_fields(tmp_path):
    """ "preset": "large" supplies num_filters/num_residual_blocks."""
    raw = json.loads(Path("run_configurations/test_run.json").read_text())
    del raw["net_config"]["num_filters"]
    del raw["net_config"]["num_residual_blocks"]
    raw["net_config"]["preset"] = "large"
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))
    config = load_args(path)
    assert config.net_config.preset == "large"
    assert config.net_config.num_filters == 192
    assert config.net_config.num_residual_blocks == 12


def test_explicit_size_keys_win_over_preset(tmp_path):
    path = _write_config_with_net(tmp_path, {"preset": "large", "num_filters": 96})
    config = load_args(path)
    assert config.net_config.num_filters == 96  # explicit key wins
    assert config.net_config.num_residual_blocks == 12  # preset fills the rest


def test_unknown_preset_raises(tmp_path):
    path = _write_config_with_net(tmp_path, {"preset": "gigantic"})
    with pytest.raises(ValueError, match="Unknown net preset"):
        load_args(path)


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


def test_tournament_defaults_apply_when_block_absent():
    """A config without a tournament block gets the default TournamentConfig."""
    config = load_args("run_configurations/test_run.json")
    assert config.tournament == TournamentConfig()
    assert config.tournament.games_per_pairing == 30
    assert config.tournament.back_ref_offsets == (1, 2, 4, 8, 16, 32)
    assert config.tournament.include_first_last is True
    assert config.tournament.max_checkpoints is None
    assert config.tournament_directory == config.run_directory / "Tournament"


def test_tournament_loads_from_json(tmp_path):
    """A tournament block in the JSON populates TournamentConfig."""
    raw = json.loads(Path("run_configurations/test_run.json").read_text())
    raw["tournament"] = {
        "games_per_pairing": 12,
        "back_ref_offsets": [1, 3, 9],
        "include_first_last": False,
        "prior_games": 4.0,
        "anchor_rating": 400.0,
        "max_checkpoints": 20,
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))
    config = load_args(path)
    tour = config.tournament
    assert tour.games_per_pairing == 12
    assert tour.back_ref_offsets == (1, 3, 9)
    assert tour.include_first_last is False
    assert tour.prior_games == 4.0
    assert tour.anchor_rating == 400.0
    assert tour.max_checkpoints == 20


def test_blokus_cloud_v2_config_loads_and_round_trips():
    """The next-run config parses and carries the analysis-driven deltas."""
    from alphablokus.config import ObjectStoreConfig

    config = load_args("run_configurations/blokus_cloud_v2.json")
    assert isinstance(config, RunConfig)
    assert config.run_name == "blokus_cloud_v2"

    # N1 LR floor is set (default is 0.0; v2 floors at 1e-4).
    assert config.net_config.lr_scheduler == "cosine"
    assert config.net_config.lr_eta_min == 0.0001

    # Analysis §4 config deltas.
    assert config.load_model is True
    assert config.replay_buffer_games == 60000
    assert config.num_arena_matches == 100
    assert config.mcts_config.num_mcts_sims == 128
    assert config.mcts_config.gumbel_max_considered == 32

    # Unchanged from blokus_cloud.json.
    assert config.net_config.preset == "large"
    assert config.net_config.num_filters == 192
    assert config.net_config.num_residual_blocks == 12
    assert config.num_eps == 10000
    assert config.mcts_config.dirichlet_alpha == 0.03
    assert config.update_threshold == 0.55
    assert config.net_config.epochs == 1
    assert config.net_config.batch_size == 1024
    assert config.jax_selfplay.top_k == 64
    assert config.jax_selfplay.wave_plies == 32

    # Data-safety protocol: object_store block present (creds stay env-only).
    assert isinstance(config.object_store, ObjectStoreConfig)
    assert config.object_store.bucket
    assert config.wandb is not None and config.wandb.mode == "online"
