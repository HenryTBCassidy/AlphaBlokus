import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from core.config import RunConfig


class CycleStage(StrEnum):
    """Stages of the training cycle, used for timing records."""
    SELF_PLAY = "SelfPlay"
    TRAINING = "Training"
    ARENA = "Arena"
    WHOLE_CYCLE = "WholeCycle"


@dataclass
class MetricsCollector:
    """Collects metrics from all components during a training run.

    Components call log_* methods during execution. Call flush() at generation
    boundaries to write the current generation's data to hive-partitioned
    parquet files and clear the buffers.

    Directory structure after multiple generations::

        TrainingData/generation=1/data.parquet
        TrainingData/generation=2/data.parquet
        ArenaData/generation=1/arena.parquet
        ...

    Reading back: ``pd.read_parquet(directory)`` automatically discovers all
    partitions and reconstructs the ``generation`` column from directory names.
    """

    _training_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _arena_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _timing_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _self_play_profiling_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _resource_usage_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _training_throughput_records: list[dict] = field(default_factory=list, init=False, repr=False)

    def log_training(
        self,
        generation: int,
        epoch: int,
        batch_number: int,
        pi_loss: float,
        v_loss: float,
        total_loss: float,
        avg_pi_loss: float,
        avg_v_loss: float,
    ) -> None:
        """Record metrics from a single training batch."""
        self._training_records.append({
            "generation": generation,
            "epoch": epoch,
            "batch_number": batch_number,
            "pi_loss": pi_loss,
            "v_loss": v_loss,
            "total_loss": total_loss,
            "average_pi_loss": avg_pi_loss,
            "average_v_loss": avg_v_loss,
        })

    def log_arena(
        self,
        generation: int,
        wins: int,
        losses: int,
        draws: int,
    ) -> None:
        """Record arena evaluation results for a generation."""
        self._arena_records.append({
            "generation": generation,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        })

    def log_timing(
        self,
        generation: int,
        cycle_stage: CycleStage,
        time_elapsed: float,
    ) -> None:
        """Record execution time of a training stage."""
        self._timing_records.append({
            "generation": generation,
            "cycle_stage": cycle_stage,
            "time_elapsed": time_elapsed,
        })

    def log_self_play_profiling(
        self,
        generation: int,
        episode: int,
        num_moves: int,
        total_sims: int,
        total_search_time_s: float,
        total_inference_time_s: float,
        num_leaf_expansions: int,
        tree_size: int,
    ) -> None:
        """Record MCTS profiling data for a single self-play episode."""
        sims_per_second = total_sims / total_search_time_s if total_search_time_s > 0 else 0.0
        inference_fraction = (
            total_inference_time_s / total_search_time_s if total_search_time_s > 0 else 0.0
        )
        self._self_play_profiling_records.append({
            "generation": generation,
            "episode": episode,
            "num_moves": num_moves,
            "total_sims": total_sims,
            "total_search_time_s": total_search_time_s,
            "total_inference_time_s": total_inference_time_s,
            "num_leaf_expansions": num_leaf_expansions,
            "tree_size": tree_size,
            "sims_per_second": sims_per_second,
            "inference_fraction": inference_fraction,
        })

    def log_resource_usage(
        self,
        generation: int,
        cycle_stage: CycleStage,
        process_rss_bytes: int,
        gpu_memory_bytes: float | None = None,
    ) -> None:
        """Record a memory usage snapshot at a point in the training cycle."""
        self._resource_usage_records.append({
            "generation": generation,
            "cycle_stage": cycle_stage,
            "process_rss_bytes": process_rss_bytes,
            "gpu_memory_bytes": gpu_memory_bytes,
        })

    def log_training_throughput(
        self,
        generation: int,
        epoch: int,
        num_examples: int,
        epoch_time_s: float,
    ) -> None:
        """Record training throughput for a single epoch."""
        samples_per_second = num_examples / epoch_time_s if epoch_time_s > 0 else 0.0
        self._training_throughput_records.append({
            "generation": generation,
            "epoch": epoch,
            "num_examples": num_examples,
            "epoch_time_s": epoch_time_s,
            "samples_per_second": samples_per_second,
        })

    def flush(self, config: RunConfig, generation: int) -> None:
        """Write buffered metrics for the current generation and clear buffers.

        Each generation's data is written to a hive-partitioned directory
        (e.g. ``TrainingData/generation=N/data.parquet``). The ``generation``
        column is dropped from the parquet data since it's encoded in the
        directory name and restored automatically on read.

        Buffers are cleared after a successful write so memory stays bounded.
        """
        start = time.perf_counter()
        count = 0

        if self._training_records:
            df = pd.DataFrame(self._training_records).assign(
                average_loss=lambda df: df.average_pi_loss + df.average_v_loss
            ).astype(
                {"pi_loss": "float64", "v_loss": "float64", "total_loss": "float64"}
            )
            _write_partition(df, config.training_data_directory, generation, "data.parquet")
            count += len(self._training_records)
            self._training_records.clear()

        if self._arena_records:
            _write_partition(
                pd.DataFrame(self._arena_records),
                config.arena_data_directory,
                generation,
                "arena.parquet",
            )
            count += len(self._arena_records)
            self._arena_records.clear()

        if self._timing_records:
            _write_partition(
                pd.DataFrame(self._timing_records),
                config.timings_directory,
                generation,
                "timings.parquet",
            )
            count += len(self._timing_records)
            self._timing_records.clear()

        if self._self_play_profiling_records:
            _write_partition(
                pd.DataFrame(self._self_play_profiling_records),
                config.self_play_profiling_directory,
                generation,
                "profiling.parquet",
            )
            count += len(self._self_play_profiling_records)
            self._self_play_profiling_records.clear()

        if self._resource_usage_records:
            _write_partition(
                pd.DataFrame(self._resource_usage_records),
                config.resource_usage_directory,
                generation,
                "resources.parquet",
            )
            count += len(self._resource_usage_records)
            self._resource_usage_records.clear()

        if self._training_throughput_records:
            _write_partition(
                pd.DataFrame(self._training_throughput_records),
                config.training_throughput_directory,
                generation,
                "throughput.parquet",
            )
            count += len(self._training_throughput_records)
            self._training_throughput_records.clear()

        elapsed = time.perf_counter() - start
        logger.info(f"Flushed {count} metric records for generation {generation} in {elapsed:.2f}s")


def _write_partition(
    df: pd.DataFrame,
    root_dir: Path,
    generation: int,
    filename: str,
) -> None:
    """Write a DataFrame to a hive-partitioned parquet directory.

    Creates ``root_dir/generation=N/filename``. The ``generation`` column is
    dropped from the data since it's encoded in the directory name and
    restored automatically by ``pd.read_parquet(root_dir)``.
    """
    partition_dir = root_dir / f"generation={generation}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.drop(columns=["generation"], errors="ignore")
    pq.write_table(pa.Table.from_pandas(df_out), partition_dir / filename)
