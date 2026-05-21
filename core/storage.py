"""Parquet I/O for all training data: metrics and self-play history.

This module is the single point of contact for all parquet reads and writes
in the project.  It consolidates two storage classes:

- :class:`MetricsCollector` — stateful buffer that accumulates hive-partitioned
  metrics (training loss, arena results, timings, profiling, resources,
  throughput) and writes them to disk on ``flush()``.
- :class:`SelfPlayStore` — handles per-generation self-play files containing
  board + policy arrays stored as byte-serialised numpy data with shape
  metadata in the parquet schema.
"""

import dataclasses
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from numpy.typing import NDArray

from core.config import RunConfig

if TYPE_CHECKING:
    from core.arena import GameRecord


# ---------------------------------------------------------------------------
# Public enum
# ---------------------------------------------------------------------------

class CycleStage(StrEnum):
    """Stages of the training cycle, used for timing records."""

    SELF_PLAY = "SelfPlay"
    TRAINING = "Training"
    ARENA = "Arena"
    WHOLE_CYCLE = "WholeCycle"


# ---------------------------------------------------------------------------
# Public type alias — data contract for self-play persistence
# ---------------------------------------------------------------------------

ProcessedExample: TypeAlias = tuple[NDArray, NDArray, float]  # (board, policy, value)


@dataclass(frozen=True)
class EvalSet:
    """Frozen held-out positions used for per-epoch network diagnostics.

    Sampled once from the first generation's self-play and reused unchanged for
    every subsequent epoch. The three fields are aligned by index:

    - ``boards[i]``: model-channel encoded board at position i
    - ``target_policies[i]``: MCTS-improved policy that was actually used to
      generate that example. Used as the "ground truth" for top-1/top-5
      policy accuracy.
    - ``target_values[i]``: the final game outcome from that position's
      perspective, in {-1, 0, +1}. Used for value calibration.
    """

    boards: NDArray
    target_policies: NDArray
    target_values: NDArray

    def __len__(self) -> int:
        return len(self.boards)


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

    _training_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _arena_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _timing_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _self_play_profiling_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _resource_usage_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _training_throughput_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _training_entropy_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _policy_accuracy_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _value_calibration_records: list[dict] = field(default_factory=list, init=False, repr=False)
    _elo_records: list[dict] = field(default_factory=list, init=False, repr=False)
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
        import wandb  # lazy import — wandb is a heavy dep
        from datetime import UTC, datetime

        assert self.config is not None and self.config.wandb is not None  # narrowed by caller
        wandb_config = self.config.wandb
        run_name = f"{self.config.run_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        self._wandb_run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            tags=list(wandb_config.tags),
            mode=wandb_config.mode,
            name=run_name,
            config=_dataclass_to_jsonable(self.config),
        )
        if self._wandb_run is not None and getattr(self._wandb_run, "url", None):
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
        self._publish({
            "progress/generation_fraction": completed / max(total_generations, 1),
            "progress/eta_seconds": eta_s,
            "generation": generation,
        })

    def log_training(
        self,
        generation: int,
        epoch: int,
        batch_number: int,
        pi_loss: float,
        v_loss: float,
        total_loss: float,
    ) -> None:
        """Record raw per-batch policy, value, and total loss.

        Earlier versions also logged ``average_pi_loss`` and ``average_v_loss``
        running means within the epoch. Those were dropped because they reset
        every epoch — producing characteristic upward spikes at epoch starts
        that misled the eye. The reporting layer now smooths the raw per-batch
        losses visually instead (EWM in HTML, native in W&B).
        """
        self._training_records.append({
            "generation": generation,
            "epoch": epoch,
            "batch_number": batch_number,
            "pi_loss": pi_loss,
            "v_loss": v_loss,
            "total_loss": total_loss,
        })
        self._global_batch += 1
        self._publish({
            "training/pi_loss": pi_loss,
            "training/v_loss": v_loss,
            "training/total_loss": total_loss,
            "global_batch": self._global_batch,
            "generation": generation,
            "epoch": epoch,
            "batch": batch_number,
        })

    def log_arena(
        self,
        generation: int,
        wins: int,
        losses: int,
        draws: int,
        accepted: bool | None = None,
    ) -> None:
        """Record arena evaluation results for a generation.

        ``accepted`` reports whether the new network passed the
        ``update_threshold`` acceptance test. When supplied we maintain a
        running ``arena/acceptance_rate`` over the run — the most useful
        "is training producing improvements?" headline.
        """
        self._arena_records.append({
            "generation": generation,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "accepted": accepted,
        })
        total = wins + losses + draws
        win_rate = wins / total if total > 0 else 0.0
        payload = {
            "arena/wins": wins,
            "arena/losses": losses,
            "arena/draws": draws,
            "arena/win_rate": win_rate,
            "generation": generation,
        }
        if accepted is not None:
            self._arena_attempts += 1
            if accepted:
                self._arena_accepts += 1
            payload["arena/accepted"] = int(accepted)
            payload["arena/acceptance_rate"] = (
                self._arena_accepts / self._arena_attempts
            )
        self._publish(payload)

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
        self._publish({
            f"timing/{cycle_stage.value}_s": time_elapsed,
            "generation": generation,
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
        inference_fraction = (
            total_inference_time_s / total_search_time_s if total_search_time_s > 0 else 0.0
        )
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
            valid_moves_fraction = (
                total_valid_moves_time_s / total_search_time_s if total_search_time_s > 0 else 0.0
            )
            record.update({
                "total_valid_moves_time_s": total_valid_moves_time_s,
                "total_game_ended_time_s": total_game_ended_time_s,
                "num_valid_moves_calls": num_valid_moves_calls,
                "num_game_ended_calls": num_game_ended_calls,
                "valid_moves_fraction": valid_moves_fraction,
            })
        self._self_play_profiling_records.append(record)
        self._global_episode += 1
        self._publish({
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
        payload: dict[str, Any] = {
            f"resources/{cycle_stage.value}_rss_mb": process_rss_bytes / (1024 ** 2),
            "generation": generation,
        }
        if gpu_memory_bytes is not None:
            payload[f"resources/{cycle_stage.value}_gpu_mb"] = gpu_memory_bytes / (1024 ** 2)
        self._publish(payload)

    def log_training_entropy(
        self,
        generation: int,
        epoch: int,
        mean_entropy: float,
        std_entropy: float,
        eval_set_size: int,
    ) -> None:
        """Record the network's mean policy entropy on the held-out eval set.

        Computed by forward-passing the network (no MCTS) over a frozen set of
        positions sampled once from gen-1 self-play. Falls over training as the
        network internalises stronger move selection. The papers' headline
        "is the network learning?" curve.
        """
        self._training_entropy_records.append({
            "generation": generation,
            "epoch": epoch,
            "mean_entropy": mean_entropy,
            "std_entropy": std_entropy,
            "eval_set_size": eval_set_size,
        })
        self._publish({
            "training/network_policy_entropy": mean_entropy,
            "training/network_policy_entropy_std": std_entropy,
            "generation": generation,
            "epoch": epoch,
        })

    def log_policy_accuracy(
        self,
        generation: int,
        epoch: int,
        top1_accuracy: float,
        top5_accuracy: float,
        eval_set_size: int,
    ) -> None:
        """Record network top-1 / top-5 policy accuracy vs MCTS targets on the
        frozen eval set. ``top1_accuracy`` is the fraction of eval positions
        where the network's argmax matches the MCTS target's argmax; ``top5``
        is the fraction where the MCTS argmax is in the network's top-5.
        Should rise toward 1.0 as the network internalises MCTS preferences.
        """
        self._policy_accuracy_records.append({
            "generation": generation,
            "epoch": epoch,
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "eval_set_size": eval_set_size,
        })
        self._publish({
            "training/network_top1_accuracy": top1_accuracy,
            "training/network_top5_accuracy": top5_accuracy,
            "generation": generation,
            "epoch": epoch,
        })

    def log_value_calibration(
        self,
        generation: int,
        epoch: int,
        bucket_centers: NDArray,
        bucket_means: NDArray,
        bucket_counts: NDArray,
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
            self._value_calibration_records.append({
                "generation": generation,
                "epoch": epoch,
                "bucket_idx": i,
                "bucket_center": float(centre),
                "bucket_mean_actual": float(mean_v) if not np.isnan(mean_v) else None,
                "bucket_count": int(count),
            })

        # W&B summary: log the mean absolute calibration error across populated
        # buckets — a single scalar that tracks "how off is the value head?"
        populated = bucket_counts > 0
        if populated.any():
            errs = np.abs(bucket_means[populated] - bucket_centers[populated])
            self._publish({
                "training/value_calibration_error": float(errs.mean()),
                "generation": generation,
                "epoch": epoch,
            })

    def log_elo(
        self,
        generation: int,
        elo_diff: float,
        baseline_rating: int,
        score_rate: float,
        wins: int,
        losses: int,
        draws: int,
        games: int,
    ) -> None:
        """Record the new network's Elo rating vs the frozen gen-0 baseline.

        Math: ``score_rate = (wins + 0.5·draws) / games``;
        ``elo_diff = 400 · log10(score_rate / (1 − score_rate))`` with score_rate
        clamped to ``[0.001, 0.999]``. ``baseline_rating`` is the *display
        anchor* for the random-init network (default 1000 — set in
        ``RunConfig.elo_baseline_rating``). Absolute rating = baseline + diff,
        which is what AlphaZero-style papers actually plot.

        Logged unconditionally for *every* generation, including ones where
        the new network was rejected in arena.
        """
        elo_absolute = baseline_rating + elo_diff
        self._elo_records.append({
            "generation": generation,
            "elo_rating": elo_absolute,
            "elo_diff": elo_diff,
            "baseline_rating": baseline_rating,
            "score_rate": score_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": games,
        })
        self._publish({
            "elo/rating": elo_absolute,
            "elo/diff_vs_baseline": elo_diff,
            "elo/baseline_rating": baseline_rating,
            "elo/score_rate": score_rate,
            "elo/wins": wins,
            "elo/losses": losses,
            "elo/draws": draws,
            "generation": generation,
        })

    def log_arena_game(
        self,
        generation: int,
        game_idx: int,
        record: "GameRecord",
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
            self._arena_replay_records.append({
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
            })

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
        self._minimax_records.append({
            "generation": generation,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": games,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
        })
        self._publish({
            "minimax/win_rate": win_rate,
            "minimax/draw_rate": draw_rate,
            "minimax/loss_rate": loss_rate,
            "minimax/wins": wins,
            "minimax/losses": losses,
            "minimax/draws": draws,
            "generation": generation,
        })

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
                self._symmetry_diagnostic_records.append({
                    "generation": generation,
                    "position_idx": pos_idx,
                    "symmetry_idx": sym_idx,
                    "kl_divergence": float(kl),
                    "top1_match": bool(matches[sym_idx]) if sym_idx < len(matches) else False,
                })
        self._publish({
            "learning_quality/symmetry_kl_mean": float(np.mean(overall_kls)),
            "learning_quality/symmetry_kl_max": float(np.max(overall_kls)),
            "learning_quality/symmetry_top1_match_rate": (
                float(np.mean(overall_matches)) if overall_matches else 0.0
            ),
            "generation": generation,
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
        self._publish({
            "throughput/num_examples": num_examples,
            "throughput/epoch_time_s": epoch_time_s,
            "throughput/samples_per_second": samples_per_second,
            "generation": generation,
            "epoch": epoch,
        })

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

        if self._elo_records:
            self._write_partition(
                pd.DataFrame(self._elo_records),
                config.elo_ratings_directory,
                generation,
                "elo.parquet",
            )
            count += len(self._elo_records)
            self._elo_records.clear()

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
                per_gen_payload.setdefault(int(gen), {}).update({
                    "training_per_gen/pi_loss": float(last["pi_loss"].mean()),
                    "training_per_gen/v_loss": float(last["v_loss"].mean()),
                    "training_per_gen/total_loss": float(last["total_loss"].mean()),
                })

        if self._training_entropy_records:
            ent = pd.DataFrame(self._training_entropy_records)
            for gen, group in ent.groupby("generation"):
                last = group[group["epoch"] == group["epoch"].max()]
                per_gen_payload.setdefault(int(gen), {}).update({
                    "training_per_gen/network_policy_entropy": float(last["mean_entropy"].iloc[0]),
                })

        if self._policy_accuracy_records:
            acc = pd.DataFrame(self._policy_accuracy_records)
            for gen, group in acc.groupby("generation"):
                last = group[group["epoch"] == group["epoch"].max()]
                per_gen_payload.setdefault(int(gen), {}).update({
                    "training_per_gen/network_top1_accuracy": float(last["top1_accuracy"].iloc[0]),
                    "training_per_gen/network_top5_accuracy": float(last["top5_accuracy"].iloc[0]),
                })

        if self._value_calibration_records:
            calib = pd.DataFrame(self._value_calibration_records)
            for gen, group in calib.groupby("generation"):
                last = group[group["epoch"] == group["epoch"].max()].dropna(subset=["bucket_mean_actual"])
                if last.empty:
                    continue
                errs = (last["bucket_mean_actual"] - last["bucket_center"]).abs()
                per_gen_payload.setdefault(int(gen), {}).update({
                    "training_per_gen/value_calibration_error": float(errs.mean()),
                })

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


# ---------------------------------------------------------------------------
# SelfPlayStore (per-generation self-play parquet files)
# ---------------------------------------------------------------------------

class SelfPlayStore:
    """Read and write per-generation self-play training data.

    Each generation is stored as a single flat parquet file.  Board and policy
    arrays are serialised as raw bytes, with shape and dtype metadata stored
    in the parquet file-level schema so they can be reconstructed on load.

    Args:
        directory: Root directory for self-play history files.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def save(self, examples: deque[ProcessedExample], generation: int) -> None:
        """Save one generation's self-play examples to a parquet file.

        Args:
            examples: The generation's training examples.
            generation: Generation number (used in the filename).
        """
        if not examples:
            return

        self._directory.mkdir(parents=True, exist_ok=True)

        boards, policies, values = zip(*examples)

        df = pd.DataFrame({
            "board": [b.tobytes() for b in boards],
            "policy": [p.tobytes() for p in policies],
            "value": list(values),
        })

        sample_board = boards[0]
        sample_policy = policies[0]
        metadata = {
            "board_shape": ",".join(str(d) for d in sample_board.shape),
            "board_dtype": str(sample_board.dtype),
            "policy_size": str(sample_policy.shape[0]),
            "policy_dtype": str(sample_policy.dtype),
        }

        table = pa.Table.from_pandas(df)
        merged_metadata = {
            **(table.schema.metadata or {}),
            **{k.encode(): v.encode() for k, v in metadata.items()},
        }
        table = table.replace_schema_metadata(merged_metadata)

        filepath = self._directory / self._filename(generation)
        pq.write_table(table, filepath)
        logger.info(f"Saved {len(df)} self-play examples to {filepath.name}")

    def load(self, generation: int) -> deque[ProcessedExample] | None:
        """Load a single generation's self-play examples from a parquet file.

        Returns ``None`` if the file does not exist (caller decides how to
        handle missing data).

        Args:
            generation: Generation number to load.

        Returns:
            A deque of ``ProcessedExample`` tuples, or ``None`` if the file
            is missing.
        """
        filepath = self._directory / self._filename(generation)
        if not filepath.exists():
            return None

        table = pq.read_table(filepath)
        metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

        board_shape = tuple(int(d) for d in metadata["board_shape"].split(","))
        board_dtype = np.dtype(metadata["board_dtype"])
        policy_size = int(metadata["policy_size"])
        policy_dtype = np.dtype(metadata["policy_dtype"])

        df = table.to_pandas()
        examples: deque[ProcessedExample] = deque()
        for _, row in df.iterrows():
            board = np.frombuffer(row["board"], dtype=board_dtype).reshape(board_shape).copy()
            policy = np.frombuffer(row["policy"], dtype=policy_dtype).reshape(policy_size).copy()
            examples.append((board, policy, float(row["value"])))

        logger.info(f"Loaded {len(examples)} examples from {filepath.name}")
        return examples

    def load_window(
        self,
        up_to_generation: int,
        window_size: int,
    ) -> list[deque[ProcessedExample]]:
        """Load self-play examples for a sliding window of generations.

        Loads generations from ``max(0, up_to_generation - window_size)``
        through ``up_to_generation`` (inclusive), skipping any files that do
        not exist.

        Args:
            up_to_generation: The most recent generation to include.
            window_size: How many past generations to look back.

        Returns:
            A list of deques, one per loaded generation, in generation order.
            Empty list if the directory does not exist or no files are found.
        """
        if not self._directory.exists():
            logger.warning(f"Self-play history directory not found: {self._directory}")
            return []

        start_gen = max(0, up_to_generation - window_size)
        history: list[deque[ProcessedExample]] = []

        for gen in range(start_gen, up_to_generation + 1):
            loaded = self.load(gen)
            if loaded is not None:
                history.append(loaded)

        logger.info(
            f"Loaded {sum(len(e) for e in history)} total examples "
            f"from {len(history)} generations"
        )
        return history

    @staticmethod
    def _filename(generation: int) -> str:
        """Generate filename for a single generation's self-play data."""
        return f"self_play_{generation}.parquet"
