from pathlib import Path

import pandas as pd
import pytest

from core.config import RunConfig
from core.metrics import CycleStage, MetricsCollector


@pytest.fixture
def collector() -> MetricsCollector:
    return MetricsCollector()


def test_log_training_appends(collector: MetricsCollector):
    """log_training should accumulate records in _training_records."""
    collector.log_training(
        generation=1, epoch=0, batch_number=0,
        pi_loss=0.5, v_loss=0.3, total_loss=0.8,
        avg_pi_loss=0.5, avg_v_loss=0.3,
    )
    assert len(collector._training_records) == 1

    collector.log_training(
        generation=1, epoch=0, batch_number=1,
        pi_loss=0.4, v_loss=0.2, total_loss=0.6,
        avg_pi_loss=0.45, avg_v_loss=0.25,
    )
    assert len(collector._training_records) == 2


def test_log_arena_appends(collector: MetricsCollector):
    """log_arena should accumulate records in _arena_records."""
    collector.log_arena(generation=1, wins=3, losses=1, draws=0)
    assert len(collector._arena_records) == 1


def test_log_timing_appends(collector: MetricsCollector):
    """log_timing should accumulate records in _timing_records."""
    collector.log_timing(generation=1, cycle_stage=CycleStage.SELF_PLAY, time_elapsed=12.5)
    assert len(collector._timing_records) == 1


def test_flush_writes_training_parquet(collector: MetricsCollector, test_config: RunConfig):
    """flush should create a parquet file under TrainingData/generation=1/."""
    collector.log_training(
        generation=1, epoch=0, batch_number=0,
        pi_loss=0.5, v_loss=0.3, total_loss=0.8,
        avg_pi_loss=0.5, avg_v_loss=0.3,
    )
    collector.flush(test_config, generation=1)

    parquet_path = test_config.training_data_directory / "generation=1" / "data.parquet"
    assert parquet_path.exists()


def test_flush_writes_arena_parquet(collector: MetricsCollector, test_config: RunConfig):
    """flush should create a parquet file under ArenaData/generation=1/."""
    collector.log_arena(generation=1, wins=3, losses=1, draws=0)
    collector.flush(test_config, generation=1)

    parquet_path = test_config.arena_data_directory / "generation=1" / "arena.parquet"
    assert parquet_path.exists()


def test_flush_writes_timings_parquet(collector: MetricsCollector, test_config: RunConfig):
    """flush should create a parquet file under Timings/generation=1/."""
    collector.log_timing(generation=1, cycle_stage=CycleStage.TRAINING, time_elapsed=5.0)
    collector.flush(test_config, generation=1)

    parquet_path = test_config.timings_directory / "generation=1" / "timings.parquet"
    assert parquet_path.exists()


def test_flush_clears_buffers(collector: MetricsCollector, test_config: RunConfig):
    """After flush, all record lists should be empty."""
    collector.log_training(
        generation=1, epoch=0, batch_number=0,
        pi_loss=0.5, v_loss=0.3, total_loss=0.8,
        avg_pi_loss=0.5, avg_v_loss=0.3,
    )
    collector.log_arena(generation=1, wins=3, losses=1, draws=0)
    collector.log_timing(generation=1, cycle_stage=CycleStage.ARENA, time_elapsed=8.0)

    collector.flush(test_config, generation=1)

    assert len(collector._training_records) == 0
    assert len(collector._arena_records) == 0
    assert len(collector._timing_records) == 0


def test_flush_multiple_generations(collector: MetricsCollector, test_config: RunConfig):
    """Flushing gen 1 then gen 2 should create two partition directories."""
    collector.log_training(
        generation=1, epoch=0, batch_number=0,
        pi_loss=0.5, v_loss=0.3, total_loss=0.8,
        avg_pi_loss=0.5, avg_v_loss=0.3,
    )
    collector.flush(test_config, generation=1)

    collector.log_training(
        generation=2, epoch=0, batch_number=0,
        pi_loss=0.4, v_loss=0.2, total_loss=0.6,
        avg_pi_loss=0.4, avg_v_loss=0.2,
    )
    collector.flush(test_config, generation=2)

    assert (test_config.training_data_directory / "generation=1" / "data.parquet").exists()
    assert (test_config.training_data_directory / "generation=2" / "data.parquet").exists()


def test_hive_partitioned_read_back(collector: MetricsCollector, test_config: RunConfig):
    """pd.read_parquet(root_dir) should reconstruct the generation column."""
    collector.log_training(
        generation=1, epoch=0, batch_number=0,
        pi_loss=0.5, v_loss=0.3, total_loss=0.8,
        avg_pi_loss=0.5, avg_v_loss=0.3,
    )
    collector.flush(test_config, generation=1)

    collector.log_training(
        generation=2, epoch=0, batch_number=0,
        pi_loss=0.4, v_loss=0.2, total_loss=0.6,
        avg_pi_loss=0.4, avg_v_loss=0.2,
    )
    collector.flush(test_config, generation=2)

    df = pd.read_parquet(test_config.training_data_directory)
    assert "generation" in df.columns
    assert set(df["generation"].unique()) == {1, 2}


def test_training_data_has_average_loss(collector: MetricsCollector, test_config: RunConfig):
    """Flushed training data should include the computed average_loss column."""
    collector.log_training(
        generation=1, epoch=0, batch_number=0,
        pi_loss=0.5, v_loss=0.3, total_loss=0.8,
        avg_pi_loss=0.5, avg_v_loss=0.3,
    )
    collector.flush(test_config, generation=1)

    df = pd.read_parquet(test_config.training_data_directory)
    assert "average_loss" in df.columns
    # average_loss = avg_pi_loss + avg_v_loss = 0.5 + 0.3 = 0.8
    assert pytest.approx(df["average_loss"].iloc[0], abs=1e-6) == 0.8


def test_flush_noop_when_empty(collector: MetricsCollector, test_config: RunConfig):
    """Flushing with no records should not create any directories."""
    collector.flush(test_config, generation=1)
    assert not test_config.training_data_directory.exists()
    assert not test_config.arena_data_directory.exists()
    assert not test_config.timings_directory.exists()
