"""Hive-partitioned parquet metrics plus W&B mirroring.

:class:`MetricsCollector` is a stateful buffer that accumulates
hive-partitioned metrics (training loss, arena results, timings, profiling,
resources, throughput) and writes them to disk on ``flush()``. Self-play
history persistence lives in :mod:`alphablokus.storage.selfplay_store`.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphablokus.config import RunConfig
    from alphablokus.evaluation.arena import GameRecord
    from alphablokus.evaluation.colour_value import ColourValueDiagnostic


# Training-loss columns written only by runs with that auxiliary head on
# (:mod:`alphablokus.aux_heads`). Absent from every other run's parquet, so both the
# W&B mirror and the report treat them as optional series.
AUXILIARY_LOSS_COLUMNS: tuple[str, ...] = ("score_loss", "ownership_loss", "reply_loss")


# ---------------------------------------------------------------------------
# Public enum
# ---------------------------------------------------------------------------


class CycleStage(StrEnum):
    """Stages of the training cycle, used for timing and resource records."""

    SELF_PLAY = "SelfPlay"
    SAVE = "Save"  # self-play history persistence (resource snapshots only)
    TRAINING = "Training"
    ARENA = "Arena"
    WHOLE_CYCLE = "WholeCycle"


@dataclass(frozen=True)
class EvalSet:
    """Held-out positions used for per-epoch network diagnostics.

    Sampled from the replay buffer and rebuilt every
    ``RunConfig.eval_set_rebuild_every`` generations (0 = build once and freeze,
    the historical behaviour). The fields are aligned by index:

    - ``boards[i]``: model-channel encoded board at position i
    - ``target_policies[i]``: MCTS-improved policy that was actually used to
      generate that example. Used as the "ground truth" for top-1/top-5
      policy accuracy.
    - ``target_values[i]``: the final game outcome from that position's
      perspective, in {-1, 0, +1}. Used for value calibration.
    - ``compact_boards[i]`` (optional): the compact int8 board array (canonical
      form) that ``boards[i]`` was encoded from. Present for runs built after
      the eval set started persisting it; lets diagnostics rebuild a playable
      board (``IGame.board_from_compact``) and search it with the current net's
      MCTS. ``None`` for older eval sets, in which case the MCTS-agreement
      diagnostic is skipped.
    - ``source_game_ids[i]`` (optional): which self-play game position i came
      from. Positions from one game share an outcome label, so **every
      confidence interval over this set must be a game-cluster bootstrap over
      these ids** (:mod:`alphablokus.bootstrap`), never a
      position-level resample. ``None`` for eval sets built before provenance
      was recorded, in which case interval-bearing diagnostics are skipped
      rather than computed wrongly.

    Attributes:
        source_fingerprints: Content hashes of the source games, which the replay
            buffer withholds from training (``ReplayBuffer.exclude_games``). This
            is what makes the set genuinely held out: dropping only the sampled
            positions would leave their symmetry twins and same-game siblings —
            carrying the same outcome label — in training. Empty for eval sets
            built before this existed, which are therefore **not** held out.
        built_at_generation: Generation whose buffer the set was sampled from.
            ``None`` for eval sets predating this field. Diagnostics from
            different vintages are **not** comparable — the positions differ —
            so this is logged alongside every metric computed from the set.
    """

    boards: NDArray
    target_policies: NDArray
    target_values: NDArray
    compact_boards: NDArray | None = None
    source_game_ids: NDArray | None = None
    source_fingerprints: tuple[str, ...] = ()
    built_at_generation: int | None = None

    def __len__(self) -> int:
        return len(self.boards)

    @property
    def n_source_games(self) -> int | None:
        """Distinct source games — the effective sample size for diagnostics.

        ``None`` when provenance was not recorded. This is usually far smaller
        than ``len(self)``: symmetry augmentation stores each position twice and
        a game contributes many positions, so a 200-position set can carry a
        small fraction of that many independent observations.
        """
        if self.source_game_ids is None:
            return None
        return int(np.unique(self.source_game_ids).size)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dataclass_to_jsonable(obj: Any) -> Any:
    """Recursively convert a dataclass tree into JSON-serialisable primitives.

    ``dataclasses.asdict`` preserves types like ``pathlib.Path`` which W&B's
    config serialiser rejects. This helper flattens those to strings while
    leaving plain dataclass fields, lists, tuples, dicts, and primitives
    untouched.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_jsonable(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# MetricsCollector (stateful buffer → hive-partitioned parquet)
# ---------------------------------------------------------------------------


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

    If ``config.wandb`` is set, the collector also mirrors each ``log_*`` call
    to Weights & Biases. The W&B run is initialised in ``__post_init__`` and
    finalised by ``close()`` (call it from the owning component's shutdown
    path — typically a ``try/finally`` around the training loop).
    """

    config: RunConfig | None = None
    # When set, a resumed run re-attaches to this existing W&B run id instead of
    # starting a second run for the same logical training run.
    resume_wandb_run_id: str | None = None

    _training_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _arena_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _timing_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _self_play_profiling_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _resource_usage_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _training_throughput_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _learning_rate_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _training_entropy_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _policy_accuracy_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _value_calibration_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _policy_value_consistency_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _colour_value_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _run_progress_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _rolling_elo_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _minimax_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _arena_replay_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _symmetry_diagnostic_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _wandb_run: Any | None = field(default=None, init=False, repr=False)

    # Cumulative counters used as W&B step_metric values for per-episode /
    # per-batch metrics. Incrementing globally instead of resetting at gen
    # boundaries gives W&B a monotonic x-axis — otherwise its auto-step would
    # interpolate huge straight lines across the training+arena+elo phases
    # where neither self_play nor training metrics are logged.
    _global_episode: int = field(default=0, init=False, repr=False)
    _global_batch: int = field(default=0, init=False, repr=False)
    # Running acceptance counter for the arena dashboard. Accepted = the new
    # net beat the previous net by ``update_threshold``.
    _arena_accepts: int = field(default=0, init=False, repr=False)
    _arena_attempts: int = field(default=0, init=False, repr=False)
    # Run wall-clock anchor — every W&B publish includes elapsed seconds so
    # the "Run progress" section can show time-axis charts and an ETA.
    _run_start_perf: float = field(default_factory=time.perf_counter, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config is None or self.config.wandb is None:
            return
        self._init_wandb()

    @property
    def wandb_run_id(self) -> str | None:
        """The active W&B run id (for persisting in the resume marker), or None."""
        return getattr(self._wandb_run, "id", None) if self._wandb_run else None

    def _init_wandb(self) -> None:
        """Initialise a W&B run using the active ``WandbConfig``.

        Imported lazily so the heavy wandb dependency only loads when actually
        used. The full ``RunConfig`` is captured as the W&B run config so
        hyperparameters appear alongside the metrics in the dashboard.

        The W&B run name appends a UTC timestamp suffix (``_YYYYMMDD_HHMMSS``)
        so multiple launches of the same config produce distinguishable runs
        in the dashboard instead of a wall of identical names.

        ``define_metric`` is used so the ``*_per_gen/*`` namespaces use
        ``generation`` as their x-axis — gives clean per-generation trend
        charts in the dashboard instead of the noisy auto-step view that the
        per-episode / per-batch metrics produce.
        """
        from datetime import UTC, datetime

        import wandb  # lazy import — wandb is a heavy dep

        assert self.config is not None and self.config.wandb is not None  # narrowed by caller
        wandb_config = self.config.wandb

        # Loud warning: an OFFLINE multi-generation run can't be watched live and
        # its data lives only on the (ephemeral) container disk — lost if a cloud
        # pod is terminated (exactly the blokus_cloud_60 gap). Online is the
        # default; offline is for throwaway local tests only. Not a hard failure —
        # a deliberate, warned choice. See docs/plans/archive/harden-long-runs.md H3.
        if wandb_config.mode == "offline" and self.config.num_generations > 1:
            logger.warning(
                "W&B is OFFLINE for a {}-generation run: metrics will NOT stream to the "
                "dashboard and offline data is lost if the pod is terminated. Set "
                "WANDB_API_KEY and wandb.mode='online' for any run you care about.",
                self.config.num_generations,
            )

        run_name = f"{self.config.run_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        # On resume, re-attach to the original run id so the dashboard shows one
        # continuous run; resume="allow" creates it if the id isn't found.
        resume_kwargs: dict[str, Any] = (
            {"id": self.resume_wandb_run_id, "resume": "allow"} if self.resume_wandb_run_id else {}
        )
        # A W&B init failure (e.g. missing WANDB_API_KEY in online mode, or no
        # network on the pod) must never sink the training run — parquet metrics,
        # which drive the report, are unaffected. Degrade to no-W&B and continue.
        try:
            self._wandb_run = wandb.init(
                project=wandb_config.project,
                entity=wandb_config.entity,
                tags=list(wandb_config.tags),
                mode=wandb_config.mode,
                name=run_name,
                config=_dataclass_to_jsonable(self.config),
                **resume_kwargs,
            )
        except Exception as err:
            logger.warning(
                "W&B init failed ({}); continuing without W&B (parquet metrics unaffected).",
                err,
            )
            self._wandb_run = None
            return

        if self._wandb_run is None:
            logger.warning("W&B init returned no run; continuing without W&B (parquet metrics unaffected).")
            return

        if getattr(self._wandb_run, "url", None):
            logger.info("Initialised W&B run: {}", self._wandb_run.url)
        else:
            logger.info("Initialised W&B run in {} mode", wandb_config.mode)

        # ───── W&B metric registration ────────────────────────────────────
        # The dashboard is laid out as: Run progress → Self-play → Training
        # loss → Learning quality → Arena → Strength → Operational. Each
        # namespace below is wired to the right step_metric so charts plot
        # against meaningful x-axes (generation / global episode / global
        # batch / wall-clock) instead of W&B's auto-incrementing internal
        # step counter.

        # Step metrics (the available x-axes). All monotonic.
        self._wandb_run.define_metric("generation")
        self._wandb_run.define_metric("global_episode")
        self._wandb_run.define_metric("global_batch")
        self._wandb_run.define_metric("progress/wall_clock_seconds")

        # Run progress — visible "where are we" panels, plotted against time.
        # The bare ``epoch`` / ``episode`` / ``batch`` keys we used to log are
        # mirrored into ``progress/*`` so they can be charted, while the
        # originals are kept around as hidden step values only.
        self._wandb_run.define_metric("progress/*", step_metric="progress/wall_clock_seconds")
        self._wandb_run.define_metric("epoch", hidden=True)
        self._wandb_run.define_metric("episode", hidden=True)
        self._wandb_run.define_metric("batch", hidden=True)

        # Self-play diagnostics — per-episode against the cumulative episode
        # counter; per-gen aggregates against generation.
        self._wandb_run.define_metric("self_play/*", step_metric="global_episode")
        self._wandb_run.define_metric("self_play_per_gen/*", step_metric="generation")

        # Training — per-batch against cumulative batch counter; per-gen
        # aggregates against generation.
        self._wandb_run.define_metric("training/*", step_metric="global_batch")
        self._wandb_run.define_metric("training_per_gen/*", step_metric="generation")

        # Arena, Elo, throughput, timing — all reported per generation.
        self._wandb_run.define_metric("arena/*", step_metric="generation")
        self._wandb_run.define_metric("elo/*", step_metric="generation")
        self._wandb_run.define_metric("throughput/*", step_metric="generation")
        self._wandb_run.define_metric("timing/*", step_metric="generation")

        # Minimax oracle only exists for TTT — register conditionally so
        # Blokus dashboards aren't littered with empty panels.
        if self.config.game == "tictactoe":
            self._wandb_run.define_metric("minimax/*", step_metric="generation")

        # Symmetry diagnostic — logged per (gen, reference position) tuple.
        # Headline scalar ``learning_quality/symmetry_kl_mean`` is the mean
        # across all reference positions for that generation.
        self._wandb_run.define_metric("learning_quality/*", step_metric="generation")

        # Policy–value consistency (one-ply lookahead agreement between the two
        # heads) — one point per generation.
        self._wandb_run.define_metric("pvc/*", step_metric="generation")

    def _publish(self, payload: dict) -> None:
        """Mirror a metrics payload to W&B if a run is active.

        Augments every payload with two things automatically:

        1. ``progress/wall_clock_seconds`` — elapsed seconds since
           ``MetricsCollector`` was constructed. Acts as the natural x-axis
           for the "Run progress" panel.
        2. ``progress/*`` mirrors of any bare counter keys (``generation``,
           ``epoch``, ``episode``, ``batch``) the caller included. The bare
           keys are registered as ``hidden=True`` so they don't auto-chart;
           the ``progress/*`` mirrors are the visible panels.

        No-op when W&B is disabled. Keeps the ``log_*`` methods small and
        lets W&B's own batching/throttling handle the network side.
        """
        if self._wandb_run is None:
            return
        augmented = dict(payload)
        augmented["progress/wall_clock_seconds"] = time.perf_counter() - self._run_start_perf
        for key in ("generation", "epoch", "episode", "batch"):
            if key in payload:
                augmented[f"progress/{key}"] = payload[key]
        self._wandb_run.log(augmented)

    def close(self) -> None:
        """Finalise the W&B run if one is active. Safe to call multiple times."""
        if self._wandb_run is None:
            return
        import wandb

        wandb.finish()
        self._wandb_run = None

    def log_progress(
        self,
        generation: int,
        total_generations: int,
    ) -> None:
        """Publish "Run progress" headline metrics at a generation boundary.

        Records the gen counter as a fraction of the total run plus an ETA
        derived from elapsed wall-clock divided by completed generations.
        Called once at the start of each generation, before self-play begins,
        so the dashboard's progress panel updates as soon as a gen begins
        rather than only when the first batch is logged.
        """
        elapsed = time.perf_counter() - self._run_start_perf
        completed = max(generation - 1, 0)
        if completed > 0:
            per_gen_s = elapsed / completed
            eta_s = per_gen_s * (total_generations - completed)
        else:
            eta_s = float("nan")  # unknown until gen 1 completes
        self._publish(
            {
                "progress/generation_fraction": completed / max(total_generations, 1),
                "progress/eta_seconds": eta_s,
                "generation": generation,
            }
        )

    def log_training(
        self,
        generation: int,
        epoch: int,
        batch_number: int,
        pi_loss: float,
        v_loss: float,
        total_loss: float,
        score_loss: float | None = None,
        ownership_loss: float | None = None,
        reply_loss: float | None = None,
    ) -> None:
        """Record raw per-batch policy, value, and total loss.

        Earlier versions also logged ``average_pi_loss`` and ``average_v_loss``
        running means within the epoch. Those were dropped because they reset
        every epoch — producing characteristic upward spikes at epoch starts
        that misled the eye. The reporting layer now smooths the raw per-batch
        losses visually instead (EWM in HTML, native in W&B).

        ``score_loss`` / ``ownership_loss`` / ``reply_loss`` are the auxiliary heads'
        raw losses (plans docs/plans/score-auxiliary-target.md S4 and
        docs/plans/supervised-network-improvements.md N4/N5), present only for runs
        with that head on; each column is omitted otherwise, so existing runs' parquet
        schema is unchanged and the report simply has one fewer series to draw.
        """
        record: dict[str, Any] = {
            "generation": generation,
            "epoch": epoch,
            "batch_number": batch_number,
            "pi_loss": pi_loss,
            "v_loss": v_loss,
            "total_loss": total_loss,
        }
        auxiliary = {"score_loss": score_loss, "ownership_loss": ownership_loss, "reply_loss": reply_loss}
        record.update({name: value for name, value in auxiliary.items() if value is not None})
        self._training_records.append(record)
        self._global_batch += 1
        published: dict[str, Any] = {
            "training/pi_loss": pi_loss,
            "training/v_loss": v_loss,
            "training/total_loss": total_loss,
            "global_batch": self._global_batch,
            "generation": generation,
            "epoch": epoch,
            "batch": batch_number,
        }
        published.update({f"training/{name}": value for name, value in auxiliary.items() if value is not None})
        self._publish(published)

    def log_arena(
        self,
        generation: int,
        wins: int,
        losses: int,
        draws: int,
        accepted: bool | None = None,
        white_wins: int | None = None,
        black_wins: int | None = None,
    ) -> None:
        """Record arena evaluation results for a generation.

        ``accepted`` reports whether the new network passed the acceptance test.
        When supplied we maintain a running ``arena/acceptance_rate`` over the
        run — the most useful "is training producing improvements?" headline.

        ``white_wins`` / ``black_wins`` are the per-colour decisive-game counts
        across the generation's arena games (from ``GameRecord.player1_was_white``;
        see :func:`alphablokus.training.coach._colour_split`). Logging them
        permanently (S4) surfaces first-mover pinning — the failure mode that
        froze ``blokus_search_harder`` (96% of decisive games won by White would
        have been visible three runs earlier). ``None`` for callers that don't
        pass a colour split (e.g. older code paths); the columns are then omitted.
        """
        record: dict[str, Any] = {
            "generation": generation,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "accepted": accepted,
        }
        if white_wins is not None:
            record["white_wins"] = white_wins
        if black_wins is not None:
            record["black_wins"] = black_wins
        self._arena_records.append(record)
        total = wins + losses + draws
        win_rate = wins / total if total > 0 else 0.0
        payload = {
            "arena/wins": wins,
            "arena/losses": losses,
            "arena/draws": draws,
            "arena/win_rate": win_rate,
            "generation": generation,
        }
        if white_wins is not None and black_wins is not None:
            decisive = white_wins + black_wins
            payload["arena/white_wins"] = white_wins
            payload["arena/black_wins"] = black_wins
            payload["arena/white_win_rate"] = white_wins / decisive if decisive > 0 else 0.0
        if accepted is not None:
            self._arena_attempts += 1
            if accepted:
                self._arena_accepts += 1
            payload["arena/accepted"] = int(accepted)
            payload["arena/acceptance_rate"] = self._arena_accepts / self._arena_attempts
        self._publish(payload)

    def log_timing(
        self,
        generation: int,
        cycle_stage: CycleStage,
        time_elapsed: float,
    ) -> None:
        """Record execution time of a training stage."""
        self._timing_records.append(
            {
                "generation": generation,
                "cycle_stage": cycle_stage,
                "time_elapsed": time_elapsed,
            }
        )
        self._publish(
            {
                f"timing/{cycle_stage.value}_s": time_elapsed,
                "generation": generation,
            }
        )

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
        mean_policy_entropy: float = 0.0,
        total_valid_moves_time_s: float = 0.0,
        total_game_ended_time_s: float = 0.0,
        num_valid_moves_calls: int = 0,
        num_game_ended_calls: int = 0,
    ) -> None:
        """Record MCTS profiling data for a single self-play episode.

        ``mean_policy_entropy`` is the per-episode mean of the raw MCTS visit
        distribution's entropy (in nats), computed move-by-move on the
        pre-temperature distribution. Falls to zero as the model becomes more
        confident in its move choice.
        """
        sims_per_second = total_sims / total_search_time_s if total_search_time_s > 0 else 0.0
        inference_fraction = total_inference_time_s / total_search_time_s if total_search_time_s > 0 else 0.0
        record = {
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
            "mean_policy_entropy": mean_policy_entropy,
        }
        if total_valid_moves_time_s > 0 or total_game_ended_time_s > 0:
            valid_moves_fraction = total_valid_moves_time_s / total_search_time_s if total_search_time_s > 0 else 0.0
            record.update(
                {
                    "total_valid_moves_time_s": total_valid_moves_time_s,
                    "total_game_ended_time_s": total_game_ended_time_s,
                    "num_valid_moves_calls": num_valid_moves_calls,
                    "num_game_ended_calls": num_game_ended_calls,
                    "valid_moves_fraction": valid_moves_fraction,
                }
            )
        self._self_play_profiling_records.append(record)
        self._global_episode += 1
        self._publish(
            {
                "self_play/num_moves": num_moves,
                "self_play/total_sims": total_sims,
                "self_play/search_time_s": total_search_time_s,
                "self_play/inference_time_s": total_inference_time_s,
                "self_play/sims_per_second": sims_per_second,
                "self_play/inference_fraction": inference_fraction,
                "self_play/leaf_expansions": num_leaf_expansions,
                "self_play/tree_size": tree_size,
                "self_play/policy_entropy": mean_policy_entropy,
                "global_episode": self._global_episode,
                "generation": generation,
                "episode": episode,
            }
        )

    def log_resource_usage(
        self,
        generation: int,
        cycle_stage: CycleStage,
        process_rss_bytes: int,
        gpu_memory_bytes: float | None = None,
        peak_rss_bytes: int | None = None,
    ) -> None:
        """Record a memory usage snapshot at a point in the training cycle.

        ``peak_rss_bytes`` is the process's high-water RSS (``ru_maxrss``) —
        the number the OOM killer acts on, which point-in-time RSS reads can
        miss. Logged to console as well as parquet/W&B so a memory spike is
        visible in the run, not just post-mortem (oom-hardening O8).
        """
        self._resource_usage_records.append(
            {
                "generation": generation,
                "cycle_stage": cycle_stage,
                "process_rss_bytes": process_rss_bytes,
                "gpu_memory_bytes": gpu_memory_bytes,
                "peak_rss_bytes": peak_rss_bytes,
            }
        )
        logger.info(
            "Memory after {} (gen {}): RSS {:.2f} GB, peak RSS {} GB, GPU {}",
            cycle_stage.value,
            generation,
            process_rss_bytes / (1024**3),
            f"{peak_rss_bytes / (1024**3):.2f}" if peak_rss_bytes is not None else "n/a",
            f"{gpu_memory_bytes / (1024**3):.2f} GB" if gpu_memory_bytes is not None else "n/a",
        )
        payload: dict[str, Any] = {
            f"resources/{cycle_stage.value}_rss_mb": process_rss_bytes / (1024**2),
            "generation": generation,
        }
        if peak_rss_bytes is not None:
            payload[f"resources/{cycle_stage.value}_peak_rss_mb"] = peak_rss_bytes / (1024**2)
        if gpu_memory_bytes is not None:
            payload[f"resources/{cycle_stage.value}_gpu_mb"] = gpu_memory_bytes / (1024**2)
        self._publish(payload)

    def log_training_entropy(
        self,
        generation: int,
        epoch: int,
        mean_entropy: float,
        std_entropy: float,
        eval_set_size: int,
        eval_set_generation: int | None = None,
    ) -> None:
        """Record the network's mean policy entropy on the held-out eval set.

        Computed by forward-passing the network (no MCTS) over a frozen set of
        positions sampled once from gen-1 self-play. Falls over training as the
        network internalises stronger move selection. The papers' headline
        "is the network learning?" curve.
        """
        self._training_entropy_records.append(
            {
                "generation": generation,
                "epoch": epoch,
                "mean_entropy": mean_entropy,
                "std_entropy": std_entropy,
                "eval_set_size": eval_set_size,
                "eval_set_generation": eval_set_generation,
            }
        )
        self._publish(
            {
                "training/network_policy_entropy": mean_entropy,
                "training/network_policy_entropy_std": std_entropy,
                "generation": generation,
                "epoch": epoch,
            }
        )

    def log_policy_accuracy(
        self,
        generation: int,
        epoch: int,
        top1_accuracy: float,
        top5_accuracy: float,
        eval_set_size: int,
        mcts_top1_accuracy: float | None = None,
        mcts_top5_accuracy: float | None = None,
        eval_set_generation: int | None = None,
    ) -> None:
        """Record network policy agreement on the frozen eval set.

        Two independent agreement series (each top-1 / top-5):

        - ``top1_accuracy`` / ``top5_accuracy``: agreement with the **frozen
          gen-1 MCTS targets** (or the minimax oracle for TTT). This *decays*
          for Blokus as the net surpasses gen-1's search — a stale-eval-set
          artifact, not a strength signal (see
          docs/research/blokus-cloud-60-analysis.md §1).
        - ``mcts_top1_accuracy`` / ``mcts_top5_accuracy`` (optional): agreement
          of the raw policy with the **current net's own MCTS** on the same
          positions — the net-vs-own-search gap, which should hold/rise as
          training works. ``None`` when the eval set can't be re-searched
          (older eval sets without persisted compact boards).
        """
        record: dict[str, Any] = {
            "generation": generation,
            "epoch": epoch,
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "eval_set_size": eval_set_size,
        }
        if mcts_top1_accuracy is not None:
            record["mcts_top1_accuracy"] = mcts_top1_accuracy
        if mcts_top5_accuracy is not None:
            record["mcts_top5_accuracy"] = mcts_top5_accuracy
        record["eval_set_generation"] = eval_set_generation
        self._policy_accuracy_records.append(record)
        payload: dict[str, Any] = {
            "training/network_top1_accuracy": top1_accuracy,
            "training/network_top5_accuracy": top5_accuracy,
            "generation": generation,
            "epoch": epoch,
        }
        if mcts_top1_accuracy is not None:
            payload["training/network_mcts_top1_accuracy"] = mcts_top1_accuracy
        if mcts_top5_accuracy is not None:
            payload["training/network_mcts_top5_accuracy"] = mcts_top5_accuracy
        self._publish(payload)

    def log_value_calibration(
        self,
        generation: int,
        epoch: int,
        bucket_centers: NDArray,
        bucket_means: NDArray,
        bucket_counts: NDArray,
        eval_set_generation: int | None = None,
    ) -> None:
        """Record a reliability diagram for the value head.

        Predicted v ∈ [-1, 1] is binned into 10 equal buckets. For each
        bucket we record (a) its centre, (b) the mean *actual* outcome of
        positions whose predicted v fell in this bucket, (c) the bucket
        count. A perfectly calibrated value head has bucket_mean ≈ bucket_centre
        (the y=x diagonal of the reliability plot).
        """
        for i, (centre, mean_v, count) in enumerate(
            zip(bucket_centers, bucket_means, bucket_counts, strict=True),
        ):
            self._value_calibration_records.append(
                {
                    "generation": generation,
                    "epoch": epoch,
                    "bucket_idx": i,
                    "bucket_center": float(centre),
                    "bucket_mean_actual": float(mean_v) if not np.isnan(mean_v) else None,
                    "bucket_count": int(count),
                    "eval_set_generation": eval_set_generation,
                }
            )

        # W&B summary: log the mean absolute calibration error across populated
        # buckets — a single scalar that tracks "how off is the value head?"
        populated = bucket_counts > 0
        if populated.any():
            errs = np.abs(bucket_means[populated] - bucket_centers[populated])
            self._publish(
                {
                    "training/value_calibration_error": float(errs.mean()),
                    "generation": generation,
                    "epoch": epoch,
                }
            )

    def log_policy_value_consistency(
        self,
        generation: int,
        epoch: int,
        pvc_argmax_match: float,
        pvc_spearman: float,
        eval_set_size: int,
        value_symmetry_mae: float | None = None,
        eval_set_generation: int | None = None,
    ) -> None:
        """Record policy–value consistency on the frozen eval set.

        Two agreement series between the policy head and a one-ply value
        lookahead (``Q₁(a) = −V(child)``), over each position's top-K policy
        moves:

        - ``pvc_argmax_match``: fraction of positions where the policy's best
          move is also the ``Q₁``-best move.
        - ``pvc_spearman``: mean Spearman rank correlation between ``π`` and
          ``Q₁`` across the candidate moves.

        Read as a **trend**, not a target: the policy sees deeper than one ply,
        so a healthy net plateaus *below* perfect agreement. A late drop or a
        persistently low level flags a value/policy imbalance (see
        docs/plans/archive/policy-value-consistency-metric.md).

        ``value_symmetry_mae`` (optional) is ``mean|V(s) − V(reflect(s))|`` over
        the eval set — the value head should be invariant under the game's
        symmetry group, so this sits near 0; a rising value means the value head
        isn't respecting the symmetry.
        """
        record: dict[str, Any] = {
            "generation": generation,
            "epoch": epoch,
            "pvc_argmax_match": pvc_argmax_match,
            "pvc_spearman": pvc_spearman,
            "eval_set_size": eval_set_size,
        }
        if value_symmetry_mae is not None:
            record["value_symmetry_mae"] = value_symmetry_mae
        record["eval_set_generation"] = eval_set_generation
        self._policy_value_consistency_records.append(record)
        payload: dict[str, Any] = {
            "pvc/argmax_match": pvc_argmax_match,
            "pvc/spearman": pvc_spearman,
            "generation": generation,
        }
        if value_symmetry_mae is not None:
            payload["pvc/value_symmetry_mae"] = value_symmetry_mae
        self._publish(payload)

    def log_run_progress(
        self,
        generation: int,
        total_games: int,
        total_positions: int,
        total_optimiser_steps: int,
        buffer_games: int,
        buffer_positions: int,
        passes_per_position: float,
        epochs: int,
    ) -> None:
        """Record the run's cumulative budget in comparable units.

        "Generation" bundles games per generation, epochs and buffer staleness into
        one word, so no two runs' generations mean the same thing — which is a large
        part of why the run ledger is hard to reason about. These are the units that
        do compare across runs:

        - ``total_games``: self-play games generated so far.
        - ``total_optimiser_steps``: gradient steps taken so far — the actual amount
          of training, independent of how it was chunked.
        - ``passes_per_position``: how many times a position is trained on over its
          lifetime in the buffer, ``epochs × (buffer_games / games_per_generation)``.
          This is the data-reuse regime, and it is emergent rather than a knob: the
          run that degraded was doing ~12 passes per position.

        Unlike the older W&B-only ``log_training_dynamics``, this is persisted to
        parquet, and the totals survive a resume (the Coach restores them from the
        progress marker) — so a resumed run reports its lineage's budget rather
        than restarting from zero.
        """
        self._run_progress_records.append(
            {
                "generation": generation,
                "total_games": total_games,
                "total_positions": total_positions,
                "total_optimiser_steps": total_optimiser_steps,
                "buffer_games": buffer_games,
                "buffer_positions": buffer_positions,
                "passes_per_position": passes_per_position,
                "epochs": epochs,
            }
        )
        self._publish(
            {
                "progress/total_games": total_games,
                "progress/total_positions": total_positions,
                "progress/total_optimiser_steps": total_optimiser_steps,
                "progress/passes_per_position": passes_per_position,
                "generation": generation,
            }
        )

    def log_colour_value_diagnostic(
        self,
        generation: int,
        epoch: int,
        diagnostic: ColourValueDiagnostic,
        eval_set_generation: int | None = None,
    ) -> None:
        """Record the colour-conditional value diagnostic (one row per colour).

        The question: does the value head read the position, or has it only learnt
        that the first mover usually wins? ``skill_vs_colour`` is
        ``1 - mse/colour_only_mse``; ``skill_vs_colour_phase`` also removes game
        phase, and is the honest number. Zero means the head has learnt the colour
        prior and nothing else.

        Both come with a **game-cluster** confidence interval — positions within a
        game share an outcome label, so a position-level interval is roughly
        ``sqrt(positions per game)`` too narrow, and the kill criteria in the plan
        are read through these numbers.

        One row per side-to-move (plus the pooled statistics repeated on each, so a
        single row is self-contained), following the row-per-sub-entity shape of
        ``log_value_calibration``. ``eval_set_generation`` records which vintage of
        the eval set the numbers were measured against — rebuilt sets are different
        positions, so rows with different vintages must not be read as one curve.
        """
        skill = diagnostic.skill_vs_colour
        skill_phase = diagnostic.skill_vs_colour_phase
        for slice_ in diagnostic.slices:
            record: dict[str, Any] = {
                "generation": generation,
                "epoch": epoch,
                "colour": slice_.colour,
                "n_positions": slice_.n_positions,
                "n_games": slice_.n_games,
                "mean_prediction": slice_.mean_prediction,
                "mean_target": slice_.mean_target,
                "bias": slice_.bias,
                "colour_value_mse": slice_.value_mse,
                # Pooled statistics, repeated per row so a row stands alone.
                "value_mse": diagnostic.value_mse,
                "colour_only_mse": diagnostic.colour_only_mse,
                "colour_phase_mse": diagnostic.colour_phase_mse,
                "skill_vs_colour": skill.point,
                "skill_vs_colour_lo": skill.lo,
                "skill_vs_colour_hi": skill.hi,
                "skill_vs_colour_phase": skill_phase.point,
                "skill_vs_colour_phase_lo": skill_phase.lo,
                "skill_vs_colour_phase_hi": skill_phase.hi,
                "colour_target_correlation": diagnostic.colour_target_correlation,
                "colour_prediction_correlation": diagnostic.colour_prediction_correlation,
                "total_positions": diagnostic.n_positions,
                "total_games": diagnostic.n_games,
                "n_excluded": diagnostic.n_excluded,
            }
            if eval_set_generation is not None:
                record["eval_set_generation"] = eval_set_generation
            self._colour_value_records.append(record)

        self._publish(
            {
                "colour_value/skill_vs_colour": skill.point,
                "colour_value/skill_vs_colour_lo": skill.lo,
                "colour_value/skill_vs_colour_hi": skill.hi,
                "colour_value/skill_vs_colour_phase": skill_phase.point,
                "colour_value/colour_prediction_correlation": diagnostic.colour_prediction_correlation,
                "colour_value/colour_target_correlation": diagnostic.colour_target_correlation,
                "colour_value/n_games": diagnostic.n_games,
                "generation": generation,
            }
        )

    def log_rolling_elo(
        self,
        generation: int,
        rolling_elo: float,
        incumbent_elo: float,
        elo_delta: float,
        score_rate: float,
        wins: int,
        losses: int,
        draws: int,
        accepted: bool,
    ) -> None:
        """Record the candidate's rolling arena-derived Elo for a generation.

        The arena already played candidate-vs-incumbent, so the candidate's
        absolute Elo is ``incumbent_elo + 400·log10(s/(1−s))`` where ``s`` is the
        candidate's score rate (:func:`alphablokus.evaluation.elo.compute_elo`,
        clamped to ``[0.001, 0.999]``). Unlike the retired frozen-gen-0 metric
        this never saturates, because on acceptance the incumbent rolls forward
        to the candidate — so each generation is rated against an opponent of
        comparable strength.

        ``accepted`` is stored alongside every point so the report can split the
        accepted line from the rejected scatter without joining ArenaData — a
        rejected generation still logs its provisional candidate Elo but does not
        advance the benchmark. Logged for *every* generation.
        """
        self._rolling_elo_records.append(
            {
                "generation": generation,
                "rolling_elo": rolling_elo,
                "incumbent_elo": incumbent_elo,
                "elo_delta": elo_delta,
                "score_rate": score_rate,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "accepted": accepted,
            }
        )
        self._publish(
            {
                "elo/rolling": rolling_elo,
                "elo/incumbent": incumbent_elo,
                "elo/delta_vs_incumbent": elo_delta,
                "elo/score_rate": score_rate,
                "elo/accepted": int(accepted),
                "generation": generation,
            }
        )

    def log_arena_game(
        self,
        generation: int,
        game_idx: int,
        record: GameRecord,
    ) -> None:
        """Record one arena game's move-by-move history with top-K policies.

        Flattens the ``GameRecord`` into one row per move so the parquet stays
        tabular. Outcome and "player1 played White" flag are denormalised onto
        every row of the game for easy filtering at render time.

        Not mirrored to W&B (this is bulky structured data; the dashboard
        already has the aggregate W/L/D outcomes from ``log_arena``).
        """
        outcome = float(record.outcome)
        p1_white = bool(record.player1_was_white)
        for move_idx, move in enumerate(record.moves):
            self._arena_replay_records.append(
                {
                    "generation": generation,
                    "game_idx": game_idx,
                    "move_idx": move_idx,
                    "player": move.player,
                    "action": move.action,
                    "top_k_actions": list(move.top_k_actions),
                    "top_k_probs": list(move.top_k_probs),
                    "played_prob": float(move.played_prob),
                    "outcome": outcome,
                    "player1_was_white": p1_white,
                }
            )

    def log_minimax(
        self,
        generation: int,
        wins: int,
        losses: int,
        draws: int,
        games: int,
    ) -> None:
        """Record results vs a perfect-play minimax opponent (TTT only).

        Against perfect play, the *best* a model can do is draw every game
        (since TTT is fully solved as a forced draw). ``draw_rate`` rising
        toward 1.0 with ``loss_rate`` collapsing to 0 is the "is this model
        optimal?" signal.
        """
        draw_rate = draws / games if games > 0 else 0.0
        loss_rate = losses / games if games > 0 else 0.0
        win_rate = wins / games if games > 0 else 0.0
        self._minimax_records.append(
            {
                "generation": generation,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": games,
                "win_rate": win_rate,
                "draw_rate": draw_rate,
                "loss_rate": loss_rate,
            }
        )
        self._publish(
            {
                "minimax/win_rate": win_rate,
                "minimax/draw_rate": draw_rate,
                "minimax/loss_rate": loss_rate,
                "minimax/wins": wins,
                "minimax/losses": losses,
                "minimax/draws": draws,
                "generation": generation,
            }
        )

    def log_symmetry_diagnostic(
        self,
        generation: int,
        position_results: list[tuple[int, float, list[float], list[bool]]],
    ) -> None:
        """Record raw NN-policy symmetry diagnostic for one generation.

        Each entry of ``position_results`` is
        ``(position_index, mean_kl, kl_divergences, top1_matches)`` — one
        per reference position. Per-position KLs are stored individually
        (one row per (gen, position, symmetry-idx) so we can plot per-
        position lines later) and the mean across positions is published as
        the headline W&B scalar ``learning_quality/symmetry_kl_mean``.

        Zero is the target; larger values mean the network has internalised
        asymmetric biases that aren't averaged out by augmentation.
        """
        if not position_results:
            return
        overall_kls: list[float] = []
        overall_matches: list[bool] = []
        for pos_idx, mean_kl, kls, matches in position_results:
            overall_kls.append(mean_kl)
            overall_matches.extend(matches)
            for sym_idx, kl in enumerate(kls):
                self._symmetry_diagnostic_records.append(
                    {
                        "generation": generation,
                        "position_idx": pos_idx,
                        "symmetry_idx": sym_idx,
                        "kl_divergence": float(kl),
                        "top1_match": bool(matches[sym_idx]) if sym_idx < len(matches) else False,
                    }
                )
        self._publish(
            {
                "learning_quality/symmetry_kl_mean": float(np.mean(overall_kls)),
                "learning_quality/symmetry_kl_max": float(np.max(overall_kls)),
                "learning_quality/symmetry_top1_match_rate": (
                    float(np.mean(overall_matches)) if overall_matches else 0.0
                ),
                "generation": generation,
            }
        )

    def log_training_throughput(
        self,
        generation: int,
        epoch: int,
        num_examples: int,
        epoch_time_s: float,
    ) -> None:
        """Record training throughput for a single epoch."""
        samples_per_second = num_examples / epoch_time_s if epoch_time_s > 0 else 0.0
        self._training_throughput_records.append(
            {
                "generation": generation,
                "epoch": epoch,
                "num_examples": num_examples,
                "epoch_time_s": epoch_time_s,
                "samples_per_second": samples_per_second,
            }
        )
        self._publish(
            {
                "throughput/num_examples": num_examples,
                "throughput/epoch_time_s": epoch_time_s,
                "throughput/samples_per_second": samples_per_second,
                "generation": generation,
                "epoch": epoch,
            }
        )

    def log_learning_rate(
        self,
        generation: int,
        epoch: int,
        learning_rate: float,
    ) -> None:
        """Record the optimizer's actual learning rate for one training epoch.

        This is ``optimizer.param_groups[0]["lr"]`` read *before* the epoch's
        ``scheduler.step()`` — i.e. the LR the epoch actually trained at. It is
        the visibility gap that let ``blokus_cloud_60``'s real LR trajectory go
        unmeasured (docs/research/blokus-cloud-60-analysis.md §3 addendum);
        logging it makes any schedule experiment reviewable. Keyed on
        ``generation`` for W&B (``training_per_gen/learning_rate``).
        """
        self._learning_rate_records.append(
            {
                "generation": generation,
                "epoch": epoch,
                "learning_rate": learning_rate,
            }
        )
        self._publish(
            {
                "training_per_gen/learning_rate": learning_rate,
                "generation": generation,
                "epoch": epoch,
            }
        )

    def log_training_dynamics(
        self,
        generation: int,
        epochs: int,
        buffer_games: int,
        buffer_capacity_games: int,
        buffer_positions: int,
        staleness_gens: float,
        emergent_reuse: float,
    ) -> None:
        """Publish the per-generation rolling-buffer data regime to W&B.

        Surfaces how the data is being used so over-reuse / staleness are
        visible at a glance: the buffer fill (``buffer_games`` vs capacity), the
        staleness in generations (``B/F``), and the *emergent* reuse
        (``epochs × B/F``) — reuse is not a knob, just the consequence of full-
        pass training over a games-sized buffer. W&B-only; the HTML report shows
        the governing knobs in its config summary instead.
        """
        self._publish(
            {
                "training_per_gen/epochs": epochs,
                "training_per_gen/emergent_reuse": emergent_reuse,
                "training_per_gen/staleness_gens": staleness_gens,
                "training_per_gen/buffer_games": buffer_games,
                "training_per_gen/buffer_positions": buffer_positions,
                "training_per_gen/buffer_fill_fraction": (
                    buffer_games / buffer_capacity_games if buffer_capacity_games > 0 else 0.0
                ),
                "generation": generation,
            }
        )

    def flush(self, config: RunConfig, generation: int) -> None:
        """Write buffered metrics for the current generation and clear buffers.

        Each generation's data is written to a hive-partitioned directory
        (e.g. ``TrainingData/generation=N/data.parquet``). The ``generation``
        column is dropped from the parquet data since it's encoded in the
        directory name and restored automatically on read.

        Per-generation aggregates of the noisy per-episode/per-batch buffers
        are also published to W&B (under ``self_play_per_gen/*`` and
        ``training_per_gen/*``) before the buffers are cleared. These give
        the dashboard a clean per-generation trend view alongside the
        existing fine-grained per-episode / per-batch view.

        Buffers are cleared after a successful write so memory stays bounded.
        """
        start = time.perf_counter()
        count = 0

        # W&B per-gen aggregates first — must run before we clear the buffers.
        self._publish_self_play_per_gen()
        self._publish_training_per_gen()

        if self._training_records:
            df = pd.DataFrame(self._training_records).astype(
                {"pi_loss": "float64", "v_loss": "float64", "total_loss": "float64"}
            )
            self._write_partition(df, config.training_data_directory, generation, "data.parquet")
            count += len(self._training_records)
            self._training_records.clear()

        if self._arena_records:
            self._write_partition(
                pd.DataFrame(self._arena_records),
                config.arena_data_directory,
                generation,
                "arena.parquet",
            )
            count += len(self._arena_records)
            self._arena_records.clear()

        if self._timing_records:
            self._write_partition(
                pd.DataFrame(self._timing_records),
                config.timings_directory,
                generation,
                "timings.parquet",
            )
            count += len(self._timing_records)
            self._timing_records.clear()

        if self._self_play_profiling_records:
            self._write_partition(
                pd.DataFrame(self._self_play_profiling_records),
                config.self_play_profiling_directory,
                generation,
                "profiling.parquet",
            )
            count += len(self._self_play_profiling_records)
            self._self_play_profiling_records.clear()

        if self._resource_usage_records:
            self._write_partition(
                pd.DataFrame(self._resource_usage_records),
                config.resource_usage_directory,
                generation,
                "resources.parquet",
            )
            count += len(self._resource_usage_records)
            self._resource_usage_records.clear()

        if self._training_throughput_records:
            self._write_partition(
                pd.DataFrame(self._training_throughput_records),
                config.training_throughput_directory,
                generation,
                "throughput.parquet",
            )
            count += len(self._training_throughput_records)
            self._training_throughput_records.clear()

        if self._learning_rate_records:
            self._write_partition(
                pd.DataFrame(self._learning_rate_records),
                config.learning_rate_directory,
                generation,
                "learning_rate.parquet",
            )
            count += len(self._learning_rate_records)
            self._learning_rate_records.clear()

        if self._training_entropy_records:
            self._write_partition(
                pd.DataFrame(self._training_entropy_records),
                config.training_entropy_directory,
                generation,
                "entropy.parquet",
            )
            count += len(self._training_entropy_records)
            self._training_entropy_records.clear()

        if self._policy_accuracy_records:
            self._write_partition(
                pd.DataFrame(self._policy_accuracy_records),
                config.policy_accuracy_directory,
                generation,
                "accuracy.parquet",
            )
            count += len(self._policy_accuracy_records)
            self._policy_accuracy_records.clear()

        if self._value_calibration_records:
            self._write_partition(
                pd.DataFrame(self._value_calibration_records),
                config.value_calibration_directory,
                generation,
                "calibration.parquet",
            )
            count += len(self._value_calibration_records)
            self._value_calibration_records.clear()

        if self._policy_value_consistency_records:
            self._write_partition(
                pd.DataFrame(self._policy_value_consistency_records),
                config.policy_value_consistency_directory,
                generation,
                "consistency.parquet",
            )
            count += len(self._policy_value_consistency_records)
            self._policy_value_consistency_records.clear()

        if self._colour_value_records:
            self._write_partition(
                pd.DataFrame(self._colour_value_records),
                config.colour_value_directory,
                generation,
                "colour_value.parquet",
            )
            count += len(self._colour_value_records)
            self._colour_value_records.clear()

        if self._run_progress_records:
            self._write_partition(
                pd.DataFrame(self._run_progress_records),
                config.run_progress_directory,
                generation,
                "progress.parquet",
            )
            count += len(self._run_progress_records)
            self._run_progress_records.clear()

        if self._rolling_elo_records:
            self._write_partition(
                pd.DataFrame(self._rolling_elo_records),
                config.rolling_elo_directory,
                generation,
                "rolling.parquet",
            )
            count += len(self._rolling_elo_records)
            self._rolling_elo_records.clear()

        if self._minimax_records:
            self._write_partition(
                pd.DataFrame(self._minimax_records),
                config.minimax_results_directory,
                generation,
                "minimax.parquet",
            )
            count += len(self._minimax_records)
            self._minimax_records.clear()

        if self._symmetry_diagnostic_records:
            self._write_partition(
                pd.DataFrame(self._symmetry_diagnostic_records),
                config.symmetry_diagnostic_directory,
                generation,
                "symmetry.parquet",
            )
            count += len(self._symmetry_diagnostic_records)
            self._symmetry_diagnostic_records.clear()

        if self._arena_replay_records:
            self._write_partition(
                pd.DataFrame(self._arena_replay_records),
                config.arena_replays_directory,
                generation,
                "games.parquet",
            )
            count += len(self._arena_replay_records)
            self._arena_replay_records.clear()

        elapsed = time.perf_counter() - start
        logger.info(f"Flushed {count} metric records for generation {generation} in {elapsed:.2f}s")

    def _publish_self_play_per_gen(self) -> None:
        """Publish per-generation aggregates of self-play profiling metrics to
        W&B, keyed by generation. No-op if W&B isn't active or buffer is empty.
        """
        if self._wandb_run is None or not self._self_play_profiling_records:
            return
        df = pd.DataFrame(self._self_play_profiling_records)
        for gen, group in df.groupby("generation"):
            payload = {
                "self_play_per_gen/policy_entropy_mean": float(group["mean_policy_entropy"].mean()),
                "self_play_per_gen/policy_entropy_std": float(group["mean_policy_entropy"].std() or 0.0),
                "self_play_per_gen/num_moves_mean": float(group["num_moves"].mean()),
                "self_play_per_gen/tree_size_mean": float(group["tree_size"].mean()),
                "self_play_per_gen/sims_per_second_mean": float(group["sims_per_second"].mean()),
                "self_play_per_gen/inference_fraction_mean": float(group["inference_fraction"].mean()),
                "generation": int(gen),
            }
            self._wandb_run.log(payload)

    def _publish_training_per_gen(self) -> None:
        """Publish per-generation aggregates of training-side metrics to W&B.

        Covers loss (per-batch records → last epoch's mean) plus the per-epoch
        network diagnostics (entropy, top-K accuracy, value-calibration error)
        — each reduced to one point per generation so the dashboard reads as
        a clean trend instead of a sparse spiky line.
        """
        if self._wandb_run is None:
            return

        # Per-gen payload assembled across multiple buffers, keyed by gen.
        per_gen_payload: dict[int, dict[str, float]] = {}

        if self._training_records:
            df = pd.DataFrame(self._training_records)
            for gen, group in df.groupby("generation"):
                last_epoch = group["epoch"].max()
                last = group[group["epoch"] == last_epoch]
                per_gen_payload.setdefault(int(gen), {}).update(
                    {
                        "training_per_gen/pi_loss": float(last["pi_loss"].mean()),
                        "training_per_gen/v_loss": float(last["v_loss"].mean()),
                        "training_per_gen/total_loss": float(last["total_loss"].mean()),
                    }
                )
                for column in AUXILIARY_LOSS_COLUMNS:
                    if column in last and last[column].notna().any():
                        per_gen_payload[int(gen)][f"training_per_gen/{column}"] = float(last[column].mean())

        if self._training_entropy_records:
            ent = pd.DataFrame(self._training_entropy_records)
            for gen, group in ent.groupby("generation"):
                last = group[group["epoch"] == group["epoch"].max()]
                per_gen_payload.setdefault(int(gen), {}).update(
                    {
                        "training_per_gen/network_policy_entropy": float(last["mean_entropy"].iloc[0]),
                    }
                )

        if self._policy_accuracy_records:
            acc = pd.DataFrame(self._policy_accuracy_records)
            for gen, group in acc.groupby("generation"):
                last = group[group["epoch"] == group["epoch"].max()]
                gen_payload = {
                    "training_per_gen/network_top1_accuracy": float(last["top1_accuracy"].iloc[0]),
                    "training_per_gen/network_top5_accuracy": float(last["top5_accuracy"].iloc[0]),
                }
                if "mcts_top1_accuracy" in last and pd.notna(last["mcts_top1_accuracy"].iloc[0]):
                    gen_payload["training_per_gen/network_mcts_top1_accuracy"] = float(
                        last["mcts_top1_accuracy"].iloc[0]
                    )
                if "mcts_top5_accuracy" in last and pd.notna(last["mcts_top5_accuracy"].iloc[0]):
                    gen_payload["training_per_gen/network_mcts_top5_accuracy"] = float(
                        last["mcts_top5_accuracy"].iloc[0]
                    )
                per_gen_payload.setdefault(int(gen), {}).update(gen_payload)

        if self._value_calibration_records:
            calib = pd.DataFrame(self._value_calibration_records)
            for gen, group in calib.groupby("generation"):
                last = group[group["epoch"] == group["epoch"].max()].dropna(subset=["bucket_mean_actual"])
                if last.empty:
                    continue
                errs = (last["bucket_mean_actual"] - last["bucket_center"]).abs()
                per_gen_payload.setdefault(int(gen), {}).update(
                    {
                        "training_per_gen/value_calibration_error": float(errs.mean()),
                    }
                )

        # Publish one combined payload per generation, keyed by generation
        # via the define_metric step_metric wiring set up in _init_wandb.
        for gen, payload in sorted(per_gen_payload.items()):
            payload["generation"] = gen
            self._wandb_run.log(payload)

    @staticmethod
    def _write_partition(
        df: pd.DataFrame,
        root_dir: Path,
        generation: int,
        filename: str,
    ) -> None:
        """Write a DataFrame to a hive-partitioned parquet directory.

        Creates ``root_dir/generation=N/filename``. The ``generation`` column
        is dropped from the data since it's encoded in the directory name and
        restored automatically by ``pd.read_parquet(root_dir)``.
        """
        partition_dir = root_dir / f"generation={generation}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        df_out = df.drop(columns=["generation"], errors="ignore")
        pq.write_table(pa.Table.from_pandas(df_out), partition_dir / filename)
