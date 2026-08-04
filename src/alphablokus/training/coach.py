from __future__ import annotations

import hashlib
import json
import os
import time
from functools import partial
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd
import torch
from loguru import logger
from numpy.typing import NDArray

from alphablokus.evaluation.acceptance import acceptance_score
from alphablokus.evaluation.arena import Arena, GameRecord
from alphablokus.evaluation.elo import compute_elo
from alphablokus.evaluation.ladder_selection import (
    detect_drift,
    ladder_point_from_payload,
    select_best,
)
from alphablokus.evaluation.players import NetworkPlayer
from alphablokus.interfaces import IBoard, IGame, INeuralNetWrapper
from alphablokus.registry import resolve_oracle
from alphablokus.reporting.pentobi_ladder import load_ladder_results
from alphablokus.search.mcts import MCTS
from alphablokus.selfplay.generate import generate_games
from alphablokus.storage.metrics import (
    CycleStage,
    EvalSet,
    MetricsCollector,
)
from alphablokus.storage.object_store import create_object_store, sync_up_guarded
from alphablokus.training.diagnostics import check_ram_budget, get_memory_snapshot
from alphablokus.training.eval_set import build_or_load_eval_set, should_rebuild
from alphablokus.training.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.config import RunConfig
    from alphablokus.search.stats import MCTSEpisodeStats

# Type aliases for improved readability
TrainingExample: TypeAlias = tuple[IBoard, int, NDArray, float | None]  # (board, player, policy, value)

# Resume marker: written atomically at the end of every completed generation and
# read by ``alphablokus --resume`` to continue a crashed run in place. Lives in the
# run's log directory (already created early in ``main``).
PROGRESS_MARKER_FILENAME = "progress.json"


def _colour_split(records: list[GameRecord]) -> tuple[int, int]:
    """Count decisive arena games won by White vs Black across ``records``.

    ``GameRecord.outcome`` is from player1's perspective and
    ``player1_was_white`` says which colour player1 played, so a game is a White
    win iff ``(outcome > 0 and player1_was_white)`` or
    ``(outcome < 0 and not player1_was_white)``; the other decisive games are
    Black wins. Draws (``outcome == 0``) count as neither. Logged per generation
    (S4) so first-mover pinning — the failure that froze ``blokus_search_harder``
    — is visible in the report instead of latent in the raw replays.
    """
    white_wins = 0
    black_wins = 0
    for rec in records:
        if rec.outcome > 0:
            white_wins += 1 if rec.player1_was_white else 0
            black_wins += 0 if rec.player1_was_white else 1
        elif rec.outcome < 0:
            white_wins += 0 if rec.player1_was_white else 1
            black_wins += 1 if rec.player1_was_white else 0
    return white_wins, black_wins


def read_progress_marker(config: RunConfig) -> dict | None:
    """Return the resume marker for ``config``'s run, or ``None`` if absent.

    The marker records the last *fully completed* generation (and the W&B run id,
    if any), so resume continues from the next generation without re-running or
    overwriting completed work.
    """
    path = config.log_directory / PROGRESS_MARKER_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_benchmark_elo(config: RunConfig) -> float:
    """Rolling-Elo benchmark to resume from = the last *accepted* net's Elo.

    Self-healing: reads the persisted rolling-Elo history rather than adding
    checkpoint state. Returns the ``rolling_elo`` of the highest generation
    flagged ``accepted`` (that net *is* the current arena incumbent). Falls back
    to ``config.elo_baseline_rating`` when no history exists yet or nothing has
    been accepted — matching a fresh run's anchor.

    The parquet only ever holds completed generations, so the last accepted row
    is the correct benchmark; a resume landing after a rejection streak still
    picks the last accepted net, not the last logged (rejected) point.
    """
    anchor = float(config.elo_baseline_rating)
    rolling_dir = config.rolling_elo_directory
    if not rolling_dir.exists():
        return anchor
    try:
        history = pd.read_parquet(rolling_dir)
    except (FileNotFoundError, ValueError, OSError):
        return anchor
    if history.empty or "accepted" not in history.columns:
        return anchor
    accepted = history[history["accepted"].astype(bool)]
    if accepted.empty:
        logger.info("Resume: no accepted generation yet — rolling Elo benchmark = anchor {:.0f}", anchor)
        return anchor
    last = accepted.sort_values("generation").iloc[-1]
    benchmark = float(last["rolling_elo"])
    logger.info("Resume: rolling Elo benchmark = {:.0f} (from gen {})", benchmark, int(last["generation"]))
    return benchmark


class Coach:
    """The generation loop: self-play → train → arena → strength eval.

    Each complete cycle is a *generation*. The freshly trained network only
    replaces the previous best if it wins the arena at
    ``config.update_threshold`` or better; either way the carried-forward
    net is checkpointed so ``--resume`` can continue from any completed
    generation.
    """

    def __init__(
        self,
        game: IGame,
        nnet: INeuralNetWrapper,
        config: RunConfig,
        *,
        resume: bool = False,
        resume_wandb_run_id: str | None = None,
    ) -> None:
        """
        Initialize the training coordinator.

        Args:
            game: Game implementation providing rules and mechanics
            nnet: Neural network for policy and value predictions
            config: Configuration parameters for the training process
            resume: When True, this is a continuation of a crashed/stopped run —
                the existing frozen Elo baseline is reused rather than re-frozen
                from the (already-trained) net, keeping the Elo curve comparable.
            resume_wandb_run_id: W&B run id to re-attach to (from the resume
                marker), so the dashboard shows one continuous run.
        """
        if config.mcts_config.search_policy == "gumbel" and config.selfplay_backend != "jax":
            raise ValueError(
                "search_policy 'gumbel' is only implemented by the jax self-play backend "
                "(the python MCTS is PUCT-only); set selfplay_backend: 'jax' or search_policy: 'puct'."
            )
        self.resume = resume
        # Seed everything FIRST — before any wrapper / MCTS instance is built,
        # so weight init, replay shuffles, MCTS tie-breaks and the global
        # ``np.random`` calls scattered through self-play all see the same
        # generator state. Without this, runs at the same config drift and
        # ablations can't be compared cleanly.
        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            logger.info("Seeded numpy + torch with seed={}", config.seed)

        self.game = game
        self.nnet = nnet
        # Concrete wrappers share the (game, config) constructor; Protocols can't declare __init__.
        self.pnet = self.nnet.__class__(self.game, config)  # type: ignore[call-arg]
        self.config = config

        # Pre-flight guard: refuse configs whose estimated peak RAM cannot fit
        # this machine, instead of OOM-killing the box hours in (O8).
        check_ram_budget(config)

        self.replay_buffer = ReplayBuffer(config, game)
        self._oracle = resolve_oracle(config, game)
        self.metrics = MetricsCollector(config=config, resume_wandb_run_id=resume_wandb_run_id)

        # Optional S3-compatible mirror of the run directory; None (default)
        # keeps pure local-FS behaviour. Synced after every completed
        # generation (see ``_write_progress_marker``) so a killed cloud
        # instance loses at most its in-flight generation. Public so the CLI
        # can reuse the same (incremental) store for the final post-report sync.
        self.object_store = create_object_store(config)

        # Frozen held-out positions for per-epoch network diagnostics (policy
        # entropy, top-K accuracy, value calibration). Built lazily from gen
        # 1's self-play examples; saved to disk so resumed runs use the same set.
        # Deliberately small and pinned: the eval set is held DENSE, so it must
        # never scale with the buffer (bounded by
        # ``base_wrapper.MAX_EVAL_SET_POSITIONS``).
        self._eval_set: EvalSet | None = None
        self._eval_set_size: int = 200

        # Cumulative budget in units that compare across runs (see
        # ``MetricsCollector.log_run_progress``). Restored from the progress marker
        # on resume so a continued run reports its lineage's totals, not this
        # process's.
        self._total_games: int = 0
        self._total_positions: int = 0

        # Reference positions for the post-training symmetry diagnostic.
        # Lazily built on first call to ``_evaluate_symmetry_diagnostic`` and
        # reused across generations so the per-gen KL trend is comparable.
        self._symmetry_diagnostic_positions: list[IBoard] | None = None

        # Gen-0 anchor checkpoint. Freeze the starting network (random-init, or
        # the warm-start donor) to ``Nets/elo_baseline.pth.tar``. We no longer
        # play per-generation games against it — the rolling arena-derived Elo
        # below replaced that eval — but the file is retained because the
        # post-hoc pool BayesElo tournament anchors the whole pool on it
        # (``scripts/tournament_elo.py``, ``_ANCHOR_FILENAME``). No in-memory
        # opponent net is loaded now, only the file on disk.
        baseline_path = self.config.net_directory / "elo_baseline.pth.tar"
        # On resume, reuse the original gen-0 anchor if it's on disk — re-saving
        # would re-anchor it to the already-trained net and break the pool
        # tournament's gen-0 reference.
        if not (self.resume and baseline_path.exists()):
            self.nnet.save_checkpoint(filename="elo_baseline.pth.tar")
        elif self.resume:
            logger.info("Resume: reusing existing gen-0 anchor checkpoint {}", baseline_path)

        # Rolling arena-derived Elo (docs/plans/archive/arena-derived-elo.md).
        # The arena already plays candidate-vs-incumbent, so the incumbent is a
        # rolling benchmark: the candidate's Elo is ``_benchmark_elo +
        # compute_elo(arena_result)``, and on acceptance the benchmark rolls
        # forward to the candidate. Anchored at ``elo_baseline_rating``; this is
        # the non-saturating live strength curve that replaced the frozen-gen-0
        # eval. On resume, reconstruct it from the persisted history so the
        # chained estimate continues seamlessly (last accepted net's Elo).
        self._benchmark_elo: float = (
            reconstruct_benchmark_elo(self.config) if self.resume else float(self.config.elo_baseline_rating)
        )

        # Anchor provenance: what "Elo = elo_baseline_rating" means for this run
        # (scratch vs warm-start donor). Recorded once so cross-run curves can be
        # spliced via the donor's weight hash (S3).
        self._write_anchor_provenance()

    def learn(self, start_generation: int = 1) -> None:
        """Run the generation loop, finalising metrics/W&B even on crash.

        Args:
            start_generation: 1 for a fresh run; ``last_completed + 1`` when
                resuming (artifacts are keyed by generation, so a partial run
                appends rather than overwrites).
        """
        try:
            self._learn_loop(start_generation=start_generation)
        finally:
            # Ensure W&B (if active) is finalised even on crash/interrupt.
            self.metrics.close()

    def _learn_loop(self, start_generation: int = 1) -> None:
        """Inner training loop. Separated so ``learn`` can wrap it in try/finally.

        ``start_generation`` is 1 for a fresh run and ``last_completed + 1`` when
        resuming; every per-generation artifact is keyed by generation number, so
        starting partway through appends rather than overwriting earlier work.
        """
        for generation in range(start_generation, self.config.num_generations + 1):
            logger.info(f"Starting Generation #{generation} ...")
            generation_start = time.perf_counter()
            self.metrics.log_progress(generation, self.config.num_generations)

            logger.info(f"Starting Self-Play For Generation #{generation} ...")
            self_play_start = time.perf_counter()

            # Each completed game streams straight into the rolling buffer
            # (oldest auto-evict via maxlen) — the generation is never
            # accumulated separately alongside the buffer (oom-hardening O6).
            self.replay_buffer.begin_generation()
            generate_games(
                self.config,
                self.game,
                self.nnet,
                generation,
                log_stats=partial(self._log_self_play_stats, generation),
                sink=self.replay_buffer.add_game,
            )

            self_play_end = time.perf_counter()
            self.metrics.log_timing(generation, CycleStage.SELF_PLAY, self_play_end - self_play_start)

            # Count what was actually generated, not what was configured — a
            # generation can come up short (a crashed worker, a capped wave).
            self._total_games += self.replay_buffer.fresh_game_count
            self._total_positions += self.replay_buffer.fresh_position_count

            self._log_memory_snapshot(generation, CycleStage.SELF_PLAY)

            # Persist this generation's fresh games (file index = generation - 1).
            self.save_self_play_history(generation - 1)

            # The save used to be the peak-RSS moment (whole-generation densify);
            # snapshot it so any regression shows up in the run, not post-mortem.
            self._log_memory_snapshot(generation, CycleStage.SAVE)

            # Build/load the eval set used for per-epoch network diagnostics, and
            # withhold its source games from training. This has to happen BEFORE
            # the buffer is flattened, or the "held-out" positions end up in the
            # training set — which is exactly what used to happen.
            self._ensure_eval_set(generation)

            train_examples = self.replay_buffer.flat_examples()

            # Preserve current best network
            self.nnet.save_checkpoint(filename="temp.pth.tar")
            self.pnet.load_checkpoint(filename="temp.pth.tar")
            MCTS(self.game, self.pnet, self.config.mcts_config)

            # Train network on the whole buffer (epochs full passes — use all the
            # data). Log the emergent reuse (epochs × B/F) and staleness (B/F) so
            # the data regime is visible without it being a tunable knob.
            self._log_training_dynamics(generation, len(train_examples))
            logger.info(f"Starting Training For Generation #{generation} ...")
            training_start = time.perf_counter()
            self.nnet.train(
                train_examples,
                generation,
                metrics=self.metrics,
                eval_set=self._eval_set,
            )
            training_end = time.perf_counter()
            self.metrics.log_timing(generation, CycleStage.TRAINING, training_end - training_start)

            # Memory snapshot after training phase
            self._log_memory_snapshot(generation, CycleStage.TRAINING)

            # Arena: accept or reject the newly trained network.
            logger.info(f"Evaluating Against Previous Version For Generation #{generation} ...")
            arena_start = time.perf_counter()
            # ``top_k`` capped at the action-space size (TTT only has 10
            # actions; for Blokus's 17,837 actions 20 is plenty to capture
            # the meaningful head). Recording at least 20 also guarantees
            # the played action is in the recorded list even when MCTS is
            # uniform across many tied actions.
            top_k_to_record = min(self.game.get_action_size(), 20)

            if self.config.num_parallel_workers > 1:
                nwins, pwins, draws, game_records = self._run_arena_parallel(
                    generation,
                    top_k_to_record,
                )
            else:
                nwins, pwins, draws, game_records = self._run_arena_serial(
                    top_k_to_record,
                )

            arena_end = time.perf_counter()
            accepted = self._should_accept_new_network(nwins, pwins, draws)
            self._check_arena_for_crash(generation, nwins, pwins, draws)
            white_wins, black_wins = _colour_split(game_records)
            self.metrics.log_arena(
                generation,
                wins=nwins,
                losses=pwins,
                draws=draws,
                accepted=accepted,
                white_wins=white_wins,
                black_wins=black_wins,
            )
            self.metrics.log_timing(generation, CycleStage.ARENA, arena_end - arena_start)

            # Persist arena game replays for offline inspection in the HTML
            # report and via `scripts/replay.py`. Recorded for every gen.
            for game_idx, record in enumerate(game_records):
                self.metrics.log_arena_game(
                    generation=generation,
                    game_idx=game_idx,
                    record=record,
                )

            # Memory snapshot after arena phase
            self._log_memory_snapshot(generation, CycleStage.ARENA)

            # Accept or reject new network
            logger.info(f"NEW/PREV WINS : {nwins}/{pwins}; DRAWS : {draws}")
            if accepted:
                logger.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(filename=f"accepted_{generation}.pth.tar")
                self.nnet.save_checkpoint(filename="best.pth.tar")
            else:
                logger.info("REJECTING NEW MODEL")
                self.nnet.save_checkpoint(filename=f"rejected_{generation}.pth.tar")
                # Revert weights + Adam moments to the pre-training net (the
                # gate's job) but do NOT rewind the LR-schedule clock: the LR
                # must advance once per generation regardless of accept/reject,
                # so a rejection streak no longer freezes the schedule (L3).
                self.nnet.load_checkpoint(filename="temp.pth.tar", restore_lr_schedule=False)

            # Rolling arena-derived Elo: derive the candidate's Elo from the
            # arena score it just played against the incumbent, and roll the
            # benchmark forward on acceptance. Zero extra games — reuses the
            # arena result above.
            self._record_rolling_elo(generation, nwins, pwins, draws, accepted)

            # PHASE 4: Strength evaluation against fixed baselines.
            # The new network this gen is measured against the frozen gen-0
            # baseline (Elo) and, for TTT, a perfect-play minimax opponent.
            # Logged whether or not the new network was accepted in arena.
            self._evaluate_strength_vs_baselines(generation)

            # Record total generation time and flush metrics
            generation_end = time.perf_counter()
            self.metrics.log_timing(generation, CycleStage.WHOLE_CYCLE, generation_end - generation_start)
            self.metrics.flush(self.config, generation)

            # Mark this generation fully complete — written last, after all of
            # its data is on disk, so `--resume` always restarts from a clean
            # boundary (never a half-finished generation).
            self._write_progress_marker(generation)

            # Promotion and catastrophe-stop are read from the Pentobi ladder, not
            # from the arena. Done after the marker so a stop always leaves a
            # resumable, fully-committed generation behind.
            if self._check_ladder_and_drift(generation):
                logger.error(
                    "Drift circuit-breaker tripped — stopping the run at generation {}. "
                    "Everything up to and including this generation is on disk and resumable.",
                    generation,
                )
                return

    def _log_memory_snapshot(self, generation: int, stage: CycleStage) -> None:
        """Snapshot RSS / peak-RSS / GPU memory after ``stage`` → console + metrics.

        Peak RSS is the high-water mark the OOM killer acts on; logging it at
        every phase transition makes a memory spike visible in the run rather
        than post-mortem (oom-hardening O8).
        """
        snapshot = get_memory_snapshot()
        self.metrics.log_resource_usage(
            generation,
            stage,
            snapshot.process_rss_bytes,
            snapshot.gpu_bytes,
            peak_rss_bytes=snapshot.process_peak_rss_bytes,
        )

    def _write_progress_marker(self, generation: int) -> None:
        """Persist the carried-forward net + record the last completed generation.

        At this point ``self.nnet`` is the network carried into the next
        generation (the just-accepted net, or the reverted previous best). Saved
        as ``latest.pth.tar`` so ``--resume`` always has the exact continuation
        net — unlike ``best.pth.tar``, which doesn't exist until the first accept.

        The marker is written last, write-to-temp + ``os.replace`` (atomic on the
        same filesystem), so a crash mid-write can never leave a truncated marker
        and resume always restarts from a clean generation boundary.
        """
        self.nnet.save_checkpoint(filename="latest.pth.tar")
        path = self.config.log_directory / PROGRESS_MARKER_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_completed_generation": generation,
            "wandb_run_id": self.metrics.wandb_run_id,
            # Cumulative budget, so a resumed run continues the count instead of
            # restarting it (these were previously in-memory only and reset on
            # resume, which made the totals unusable for a run that crashed).
            "total_games": self._total_games,
            "total_positions": self._total_positions,
            "total_optimiser_steps": getattr(self.nnet, "optimiser_steps", 0),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

        # Everything this generation produced (checkpoints, parquet, the marker
        # itself) is now on local disk — mirror it. Best-effort by design:
        # object-storage trouble never kills training.
        sync_up_guarded(self.object_store, self.config.run_directory, f"generation {generation}")

    def _write_anchor_provenance(self) -> None:
        """Record what the rolling-Elo anchor (Elo = ``elo_baseline_rating``) is.

        The anchor is run-specific: for a scratch run it's the random-init net;
        for a warm-start run (``load_model``) it's the donor net loaded into
        ``elo_baseline.pth.tar``. Writing the donor weights' SHA-256 lets a
        reader splice this run's rolling curve onto the donor's pooled Elo (if
        the donor checkpoint's rating is known) — the only way to compare
        chained curves across runs. Donor ``run_name``/``generation`` aren't
        knowable from a weights-only warm start, so they're left null; the hash
        is the cross-run key. Left untouched on resume (written by the original
        run) so the recorded provenance stays stable.
        """
        anchor_path = self.config.net_directory / "elo_anchor.json"
        if self.resume and anchor_path.exists():
            return
        baseline_path = self.config.net_directory / "elo_baseline.pth.tar"
        source = "warm_start" if self.config.load_model else "scratch"
        payload = {
            "anchor_rating": self.config.elo_baseline_rating,
            "source": source,
            "checkpoint": "elo_baseline.pth.tar",
            "weights_sha256": self._sha256(baseline_path) if baseline_path.exists() else None,
            "donor_run_name": None,
            "donor_generation": None,
        }
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Elo anchor: {} run, rating {} → {}",
            source,
            self.config.elo_baseline_rating,
            anchor_path,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        """SHA-256 of a file, read in 1 MiB chunks (checkpoints are large)."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_self_play_history_for_resume(self, last_completed_generation: int) -> None:
        """Refill the rolling replay buffer to resume training at ``last + 1``."""
        self.replay_buffer.load_for_resume(last_completed_generation)

    def _log_self_play_stats(self, generation: int, episode_idx: int, stats: MCTSEpisodeStats) -> None:
        """One episode's MCTS profiling → metrics, identical schema for every
        self-play backend so downstream reports don't care which path produced
        the data."""
        self.metrics.log_self_play_profiling(
            generation=generation,
            episode=episode_idx,
            num_moves=stats.num_moves,
            total_sims=stats.total_sims,
            total_search_time_s=stats.total_search_time_s,
            total_inference_time_s=stats.total_inference_time_s,
            num_leaf_expansions=stats.num_leaf_expansions,
            tree_size=stats.tree_size,
            mean_policy_entropy=stats.mean_policy_entropy,
            total_valid_moves_time_s=stats.total_valid_moves_time_s,
            total_game_ended_time_s=stats.total_game_ended_time_s,
            num_valid_moves_calls=stats.num_valid_moves_calls,
            num_game_ended_calls=stats.num_game_ended_calls,
        )

    def _run_arena_serial(
        self,
        top_k_to_record: int,
    ) -> tuple[int, int, int, list]:
        """Sequential arena loop. Returns
        ``(new_wins, prev_wins, draws, game_records)``.
        """
        prev_player = NetworkPlayer(
            game=self.game,
            nnet=self.pnet,
            mcts_config=self.config.mcts_config,
            temp=0.0,
            opening_temp=self.config.arena_opening_temp,
            opening_moves=self.config.arena_opening_moves,
        )
        new_player = NetworkPlayer(
            game=self.game,
            nnet=self.nnet,
            mcts_config=self.config.mcts_config,
            temp=0.0,
            opening_temp=self.config.arena_opening_temp,
            opening_moves=self.config.arena_opening_moves,
        )
        arena = Arena(prev_player, new_player, self.game)
        if self.config.paired_arena:
            # Paired colour-swapped gate: each pair shares one opening prefix
            # sampled from the incumbent (self.pnet) at ``arena_opening_temp``, so
            # the first-mover advantage cancels within the pair (S1/S2). Player1 =
            # prev, player2 = new, so the returned tally maps to (pwins, nwins).
            prefix_sampler = NetworkPlayer(
                game=self.game,
                nnet=self.pnet,
                mcts_config=self.config.mcts_config,
                temp=self.config.arena_opening_temp,
            )
            pwins, nwins, draws, records = arena.play_games_paired(
                self.config.num_arena_matches,
                prefix_sampler=prefix_sampler,
                opening_moves=self.config.arena_opening_moves,
                record=True,
                top_k=top_k_to_record,
            )
        else:
            pwins, nwins, draws, records = arena.play_games(
                self.config.num_arena_matches,
                record=True,
                top_k=top_k_to_record,
            )
        return nwins, pwins, draws, records

    def _run_arena_parallel(
        self,
        generation: int,
        top_k_to_record: int,
    ) -> tuple[int, int, int, list]:
        """Parallel arena across the worker pool.

        Returns ``(new_wins, prev_wins, draws, game_records)`` —
        identical shape to ``_run_arena_serial``.

        **Convention** (matters for record fields): the orchestrator's
        ``A`` is mapped to ``self.pnet`` (the previous-best network)
        and ``B`` to ``self.nnet`` (the new candidate). This matches
        the serial path which constructs ``Arena(prev_player,
        new_player)`` with player1 = prev. The resulting GameRecords
        carry ``outcome`` from prev's perspective (so ``+1`` =
        previous net won) and ``player1_was_white`` tracking whether
        prev was white — which is what
        the arena-replay viewer (``reporting/arena_replays.py``) expects when
        it labels winners.

        Getting this wrong would silently flip the "new net wins" /
        "previous net wins" labels in the HTML report. The convention
        is enforced by a unit test in
        ``tests/parallel/test_pool.py``.
        """
        from alphablokus.parallel.pool import (
            PHASE_ARENA,
            run_two_player_games_parallel,
        )

        new_checkpoint = "parallel_arena_new.pth.tar"
        prev_checkpoint = "parallel_arena_prev.pth.tar"
        self.nnet.save_checkpoint(filename=new_checkpoint)
        self.pnet.save_checkpoint(filename=prev_checkpoint)

        prev_wins, new_wins, draws, records = run_two_player_games_parallel(
            config=self.config,
            generation=generation,
            checkpoint_a_path=prev_checkpoint,  # A = prev (matches serial player1)
            checkpoint_b_path=new_checkpoint,  # B = new
            num_games=self.config.num_arena_matches,
            num_workers=self.config.num_parallel_workers,
            phase=PHASE_ARENA,
            record=True,
            top_k=top_k_to_record,
            desc="Arena",
            # Paired colour-swapped play (S2): each pair shares one opening prefix
            # sampled by net A (= prev/incumbent) at ``arena_opening_temp``, so the
            # two colour-swapped games cancel the first-mover advantage. Inert when
            # ``paired_arena`` is False (unpaired path, unchanged).
            paired=self.config.paired_arena,
            opening_moves=self.config.arena_opening_moves,
            opening_temp=self.config.arena_opening_temp,
        )
        return new_wins, prev_wins, draws, records

    def _evaluate_strength_vs_baselines(self, generation: int) -> None:
        """Play the new network this gen against fixed baselines, log results.

        - **Perfect-play oracle** (only when the game has one and
          ``minimax_games_per_gen > 0``): draw rate rising to 1.0 with loss
          rate falling to 0 means the model has internalised optimal play.
        - **Symmetry diagnostic** (when ``symmetry_diagnostic_positions > 0``).

        The old per-generation Elo-vs-frozen-gen-0 eval was removed here: it
        saturated once the net ≫ gen-0 and cost extra games each generation. The
        live strength signal is now the rolling arena-derived Elo (recorded in
        ``_record_rolling_elo`` from the accept/reject arena, zero extra games);
        the rigorous non-saturating curve is the end-of-run pool tournament.

        The oracle arena uses the same MCTS sim count as the accept/reject
        arena. The new network's MCTS tree is reset between games via the
        :class:`NetworkPlayer.startGame` hook.
        """
        if self._oracle is not None and self.config.minimax_games_per_gen > 0:
            self._evaluate_vs_oracle(generation)
        if self.config.symmetry_diagnostic_positions > 0:
            self._evaluate_symmetry_diagnostic(generation)

    def _evaluate_vs_oracle(self, generation: int) -> None:
        from alphablokus.evaluation.players import NetworkPlayer

        n = self.config.minimax_games_per_gen
        logger.info(f"Evaluating vs perfect-play oracle ({n} games) ...")
        mm_start = time.perf_counter()

        new_player = NetworkPlayer(
            game=self.game,
            nnet=self.nnet,
            mcts_config=self.config.mcts_config,
            temp=0.0,
            opening_temp=self.config.arena_opening_temp,
            opening_moves=self.config.arena_opening_moves,
        )
        assert self._oracle is not None  # caller-guarded in _evaluate_strength_vs_baselines
        oracle_player = self._oracle.make_player()
        arena = Arena(new_player, oracle_player, self.game)
        wins, losses, draws, _ = arena.play_games(n)
        elapsed = time.perf_counter() - mm_start
        logger.info(
            "Gen {} vs minimax: W{} L{} D{} (draw_rate {:.2f}, {:.1f}s)",
            generation,
            wins,
            losses,
            draws,
            draws / max(wins + losses + draws, 1),
            elapsed,
        )
        self.metrics.log_minimax(
            generation=generation,
            wins=wins,
            losses=losses,
            draws=draws,
            games=wins + losses + draws,
        )

    def _evaluate_symmetry_diagnostic(self, generation: int) -> None:
        """Measure whether the trained network plays equivariantly under
        the game's symmetry group.

        For each of ``config.symmetry_diagnostic_positions`` deterministic
        reference positions, compute the KL divergence between
        ``nnet.predict(s(board))`` and ``s(nnet.predict(board))`` across
        all non-identity symmetries. Zero is the target. Lazily-built
        reference positions are stable across generations so the per-gen
        metric is directly comparable.
        """
        from alphablokus.evaluation.symmetry import (
            build_diagnostic_positions,
            compute_symmetry_diagnostic,
        )

        if self._symmetry_diagnostic_positions is None:
            self._symmetry_diagnostic_positions = build_diagnostic_positions(
                self.game,
                n=self.config.symmetry_diagnostic_positions,
            )

        start = time.perf_counter()
        position_results: list[tuple[int, float, list[float], list[bool]]] = []
        for idx, board in enumerate(self._symmetry_diagnostic_positions):
            result = compute_symmetry_diagnostic(self.nnet, self.game, board, idx)
            position_results.append(
                (idx, result.mean_kl, result.kl_divergences, result.top1_matches),
            )

        if position_results:
            mean_of_means = float(np.mean([m for _, m, _, _ in position_results]))
            logger.info(
                "Gen {} symmetry diagnostic: mean KL = {:.4f} across {} positions ({:.2f}s)",
                generation,
                mean_of_means,
                len(position_results),
                time.perf_counter() - start,
            )
        self.metrics.log_symmetry_diagnostic(generation, position_results)

    def _ensure_eval_set(self, generation: int) -> None:
        """Build/load the eval set and withhold its source games from training.

        Three things this has to get right, each of which was previously wrong:

        - **Held out.** The set's source games are excluded from
          ``ReplayBuffer.flat_examples`` for as long as the set is in use.
          Previously the eval positions were sampled from the training buffer and
          then trained on — for ``replay_buffer_games / num_eps`` generations at
          ``epochs`` passes each — so every "held-out" per-epoch diagnostic was
          in-sample early in a run and then silently changed meaning as those
          positions aged out. An eval set from disk with no recorded fingerprints
          cannot be held out, so it is rebuilt rather than trusted.
        - **Refreshed.** With ``eval_set_rebuild_every`` set, the set is resampled
          every N generations instead of being frozen from generation 1's
          weakest-ever data. Diagnostics carry the vintage
          (``EvalSet.built_at_generation``) because different vintages measure
          different positions and must not be read as one curve.
        - **Clustered.** Every position records its source game so intervals can be
          game-cluster bootstraps (:mod:`alphablokus.bootstrap`).
        """
        rebuild = should_rebuild(generation, self.config.eval_set_rebuild_every)
        if self._eval_set is not None and not rebuild:
            return
        if rebuild:
            logger.info(
                "Rebuilding the eval set from generation #{}'s buffer (cadence every {} generations)",
                generation,
                self.config.eval_set_rebuild_every,
            )
        eval_set = build_or_load_eval_set(
            self.config,
            self.game,
            self._oracle,
            self.replay_buffer.games,
            self._eval_set_size,
            generation=generation,
            force_rebuild=rebuild,
        )

        # A set loaded from disk without fingerprints predates the holdout fix: its
        # source games cannot be identified, so it is *not* held out. Reporting its
        # numbers as held-out is what made a degrading run look healthy, so rebuild
        # instead of inheriting the defect.
        if eval_set is not None and not eval_set.source_fingerprints and self.replay_buffer.games:
            logger.warning(
                "The eval set on disk records no source-game fingerprints, so its games cannot be "
                "withheld from training and its diagnostics would be in-sample. Rebuilding it from "
                "generation #{}'s buffer.",
                generation,
            )
            eval_set = build_or_load_eval_set(
                self.config,
                self.game,
                self._oracle,
                self.replay_buffer.games,
                self._eval_set_size,
                generation=generation,
                force_rebuild=True,
            )

        self._eval_set = eval_set
        if eval_set is not None:
            self.replay_buffer.exclude_games(set(eval_set.source_fingerprints))
            logger.info(
                "Eval set: {} positions from {} source games (vintage gen {}); {} of those games are "
                "in the buffer and withheld from training.",
                len(eval_set),
                eval_set.n_source_games,
                eval_set.built_at_generation,
                self.replay_buffer.held_out_game_count(),
            )

    def _check_arena_for_crash(self, generation: int, nwins: int, pwins: int, draws: int) -> None:
        """Use the arena as a crash detector rather than a promotion signal.

        With ``gate_mode: "always"`` the arena score no longer decides anything —
        which is correct, because it cannot resolve strength in this game (~96% of
        decisive deterministic games go to White, and per-generation scores collapse
        into 0.485–0.530 where independent play would give a spread four times
        wider). But a score far *below* that floor is not "slightly worse": it means
        something broke — a checkpoint that failed to load, a policy collapsed to
        all-pass, a net serving garbage. That is worth a loud warning, and it never
        rejects a candidate.
        """
        floor = self.config.arena_crash_floor
        if floor <= 0:
            return
        total = nwins + pwins + draws
        if total == 0:
            return
        score = acceptance_score(nwins, pwins, draws)
        if score < floor:
            logger.error(
                "ARENA CRASH DETECTOR: generation {} scored {:.3f} over {} games, below the {:.2f} floor. "
                "The arena cannot resolve small strength differences, so a score this low almost "
                "certainly means something is broken (checkpoint load, degenerate policy) rather than "
                "a slightly weaker candidate. Investigate before trusting this generation.",
                generation,
                score,
                total,
                floor,
            )

    def _check_ladder_and_drift(self, generation: int) -> bool:
        """Consume ladder results: record keep-best, and arm the drift breaker.

        The ladder itself runs out-of-process (``scripts/mini_ladder.py``) because it
        needs the ``pentobi-gtp`` binary and its own CPU workers. This method is the
        training loop's side of that contract: on the configured cadence it reads
        whatever ladder results exist, records which checkpoint is currently best by
        ladder, and evaluates the drift circuit-breaker.

        Both mechanisms already existed and had never run in a real run
        (``evaluation/ladder_selection.py``; ``docs/plans/archive/post-regression-recovery.md``
        P3/P4/P7 — P7, the wiring, was the unticked row). This arms them without
        reimplementing them.

        Returns:
            True when the drift breaker has tripped and the run should stop.
        """
        cadence = self.config.ladder_check_every
        if cadence <= 0 or generation % cadence != 0:
            return False

        results = load_ladder_results(self.config.pentobi_ladder_directory)
        points = [ladder_point_from_payload(r) for r in results if "weighted_score" in r.get("metrics", {})]
        if not points:
            logger.info(
                "Ladder check at generation {}: no ladder results in {} yet. Run "
                "scripts/mini_ladder.py against this run's checkpoints to populate it.",
                generation,
                self.config.pentobi_ladder_directory,
            )
            return False

        best = select_best(points)
        logger.info(
            "Ladder check at generation {}: best by ladder is {} (weighted {:.3f}) over {} laddered checkpoint(s)",
            generation,
            best.label,
            best.weighted_score,
            len(points),
        )
        # Record the choice next to the checkpoints. The run's product is this
        # file's answer, not ``best.pth.tar`` — with the gate off, ``best.pth.tar``
        # is simply the latest candidate.
        selection_path = self.config.net_directory / "best_by_ladder.json"
        try:
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(
                json.dumps(
                    {
                        "checked_at_generation": generation,
                        "label": best.label,
                        "generation": best.generation,
                        "weighted_score": best.weighted_score,
                        "pentobi_level": best.pentobi_level,
                        "laddered_checkpoints": len(points),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as err:
            logger.warning("Could not record the keep-best selection to {} ({}).", selection_path, err)

        alarm = detect_drift(
            points,
            drop=self.config.ladder_drift_drop,
            consecutive=self.config.ladder_drift_consecutive,
        )
        if alarm is None:
            return False
        logger.error(
            "DRIFT CIRCUIT-BREAKER: ladder fell to {:.3f} at {} after a best of {:.3f} at {} "
            "({} consecutive drops of >= {:.2f}). Resume from {} rather than the latest checkpoint.",
            alarm.tripped_at.weighted_score,
            alarm.tripped_at.label,
            alarm.best_before.weighted_score,
            alarm.best_before.label,
            alarm.consecutive_drops,
            self.config.ladder_drift_drop,
            alarm.best_before.label,
        )
        return True

    def restore_cumulative_totals(self, marker: dict[str, Any]) -> None:
        """Continue the run's game/position/optimiser-step counts across a resume.

        Called on the ``--resume`` path with the progress marker. Absent keys mean
        the marker predates these counters, in which case the totals stay at zero
        and the run under-reports its lineage rather than inventing a number.
        """
        self._total_games = int(marker.get("total_games", 0) or 0)
        self._total_positions = int(marker.get("total_positions", 0) or 0)
        steps = int(marker.get("total_optimiser_steps", 0) or 0)
        # ``optimiser_steps`` lives on ``BaseNNetWrapper``, not on the
        # ``INeuralNetWrapper`` protocol, so probe for it rather than widening the
        # protocol for a counter.
        if hasattr(self.nnet, "optimiser_steps"):
            self.nnet.optimiser_steps = steps
        logger.info(
            "Resumed cumulative budget: {} games, {} positions, {} optimiser steps",
            self._total_games,
            self._total_positions,
            steps,
        )

    def _log_training_dynamics(self, generation: int, buffer_positions: int) -> None:
        """Log the emergent reuse / staleness of the rolling-buffer data regime.

        Reuse is not a knob: every position is trained ``epochs`` times per
        generation and lives in the buffer for ``B/F`` generations, so its
        lifetime reuse is ``epochs × B/F``. Staleness (oldest game's age) is
        ``B/F`` generations. Both are computed from config and the current buffer
        fill, then surfaced to the console and W&B.
        """
        epochs = self.config.net_config.epochs
        buffer_capacity_games = self.replay_buffer.capacity_games
        fresh_games = max(self.config.num_eps, 1)
        staleness_gens = buffer_capacity_games / fresh_games
        emergent_reuse = epochs * staleness_gens
        buffer_games = len(self.replay_buffer)
        logger.info(
            "Gen {} data regime: epochs={} buffer={}/{} games ({} positions), "
            "staleness ≈{:.1f} gens, emergent reuse ≈{:.1f} (epochs × B/F)",
            generation,
            epochs,
            buffer_games,
            buffer_capacity_games,
            buffer_positions,
            staleness_gens,
            emergent_reuse,
        )
        self.metrics.log_training_dynamics(
            generation=generation,
            epochs=epochs,
            buffer_games=buffer_games,
            buffer_capacity_games=buffer_capacity_games,
            buffer_positions=buffer_positions,
            staleness_gens=staleness_gens,
            emergent_reuse=emergent_reuse,
        )
        # The same regime in units that compare across runs, persisted to parquet.
        self.metrics.log_run_progress(
            generation=generation,
            total_games=self._total_games,
            total_positions=self._total_positions,
            total_optimiser_steps=getattr(self.nnet, "optimiser_steps", 0),
            buffer_games=buffer_games,
            buffer_positions=buffer_positions,
            passes_per_position=emergent_reuse,
            epochs=epochs,
        )
        logger.info(
            "Gen {} budget so far: {} games, {} positions, {} optimiser steps, ~{:.1f} passes/position",
            generation,
            self._total_games,
            self._total_positions,
            getattr(self.nnet, "optimiser_steps", 0),
            emergent_reuse,
        )

    def _record_rolling_elo(
        self,
        generation: int,
        new_wins: int,
        prev_wins: int,
        draws: int,
        accepted: bool,
    ) -> float:
        """Derive + log the candidate's rolling Elo, rolling the benchmark on accept.

        ``compute_elo`` returns the candidate's Elo delta vs the incumbent from
        the arena score (candidate-first orientation, clamped to ``[0.001,
        0.999]`` so a 100-0 sweep saturates at ~+1200 rather than diverging).
        The candidate's absolute Elo is ``self._benchmark_elo + delta``. Every
        generation logs a point; only an accepted generation advances
        ``self._benchmark_elo`` to the candidate — a rejected generation leaves
        the benchmark untouched so the next candidate is still measured against
        the last accepted net. Returns the candidate's Elo (for tests / logging).
        """
        elo_delta, score_rate = compute_elo(new_wins, prev_wins, draws)
        candidate_elo = self._benchmark_elo + elo_delta
        logger.info(
            "Gen {} rolling Elo: {:.0f} ({:+.0f} vs incumbent {:.0f}) — score {:.3f}{}",
            generation,
            candidate_elo,
            elo_delta,
            self._benchmark_elo,
            score_rate,
            "" if accepted else " (rejected — benchmark held)",
        )
        self.metrics.log_rolling_elo(
            generation=generation,
            rolling_elo=candidate_elo,
            incumbent_elo=self._benchmark_elo,
            elo_delta=elo_delta,
            score_rate=score_rate,
            wins=new_wins,
            losses=prev_wins,
            draws=draws,
            accepted=accepted,
        )
        if accepted:
            self._benchmark_elo = candidate_elo
        return candidate_elo

    def _should_accept_new_network(
        self,
        new_wins: int,
        prev_wins: int,
        draws: int = 0,
    ) -> bool:
        """Decide whether to accept the newly trained network.

        Thin wrapper around :func:`alphablokus.evaluation.acceptance.is_accepted`,
        which dispatches on ``config.gate_mode`` (``threshold`` |
        ``regression_guard`` | ``always``). Single source of truth lives there so
        reporting code can never diverge from the training-time decision — see
        ``evaluation/acceptance.py`` for the full rationale.
        """
        from alphablokus.evaluation.acceptance import is_accepted

        return is_accepted(
            mode=self.config.gate_mode,
            new_wins=new_wins,
            prev_wins=prev_wins,
            draws=draws,
            threshold=self.config.update_threshold,
            guard_floor=self.config.guard_floor,
        )

    def save_self_play_history(self, file_index: int) -> None:
        """Persist this generation's fresh games (see :meth:`ReplayBuffer.save_fresh`)."""
        self.replay_buffer.save_fresh(file_index)

    def load_self_play_history(self, up_to_generation: int) -> None:
        """Warm-start refill of the buffer (see :meth:`ReplayBuffer.load_recent`)."""
        self.replay_buffer.load_recent(up_to_generation)
