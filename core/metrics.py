import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd
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
    boundaries to write accumulated data to parquet files.

    Each flush writes ALL data accumulated so far (does not clear buffers),
    so files always contain the complete run history. This provides crash
    resilience — if a run fails, the most recent flush contains all data
    from completed generations.
    """

    _training_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _arena_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _timing_records: list[dict] = field(default_factory=list, init=False, repr=False)

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

    def flush(self, config: RunConfig) -> None:
        """Write all accumulated metrics to parquet files.

        Each call overwrites previous files with the complete run history.
        Buffers are NOT cleared, enabling incremental flushes at generation
        boundaries while always producing complete output files.

        Column names and directory layout match the existing reporting module
        expectations so reports continue to work without changes.
        """
        start = time.perf_counter()
        count = 0

        if self._training_records:
            config.training_data_directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self._training_records).assign(
                average_loss=lambda df: df.average_pi_loss + df.average_v_loss
            ).astype(
                {"pi_loss": "float64", "v_loss": "float64", "total_loss": "float64"}
            ).to_parquet(config.training_data_directory / "data.parquet", index=False)
            count += len(self._training_records)

        if self._arena_records:
            config.arena_data_directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self._arena_records).to_parquet(
                config.arena_data_directory / "arena.parquet", index=False
            )
            count += len(self._arena_records)

        if self._timing_records:
            config.timings_directory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self._timing_records).to_parquet(
                config.timings_directory / "timings.parquet", index=False
            )
            count += len(self._timing_records)

        elapsed = time.perf_counter() - start
        logger.info(f"Flushed {count} metric records in {elapsed:.2f}s")
