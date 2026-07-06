from __future__ import annotations

import json
import os
import time
from functools import partial
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray

from alphablokus.evaluation.arena import Arena
from alphablokus.evaluation.elo import compute_elo
from alphablokus.evaluation.players import NetworkPlayer
from alphablokus.interfaces import IBoard, IGame, INeuralNetWrapper
from alphablokus.registry import resolve_oracle
from alphablokus.search.mcts import MCTS
from alphablokus.selfplay.generate import generate_games
from alphablokus.storage.metrics import (
    CycleStage,
    EvalSet,
    MetricsCollector,
)
from alphablokus.storage.object_store import create_object_store, sync_up_guarded
from alphablokus.training.diagnostics import check_ram_budget, get_memory_snapshot
from alphablokus.training.eval_set import build_or_load_eval_set
from alphablokus.training.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.search.stats import MCTSEpisodeStats
    from alphablokus.selfplay.episode import ProcessedExample

# Type aliases for improved readability
TrainingExample: TypeAlias = tuple[IBoard, int, NDArray, float | None]  # (board, player, policy, value)

# Resume marker: written atomically at the end of every completed generation and
# read by ``alphablokus --resume`` to continue a crashed run in place. Lives in the
# run's log directory (already created early in ``main``).
PROGRESS_MARKER_FILENAME = "progress.json"


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

        # Reference positions for the post-training symmetry diagnostic.
        # Lazily built on first call to ``_evaluate_symmetry_diagnostic`` and
        # reused across generations so the per-gen KL trend is comparable.
        self._symmetry_diagnostic_positions: list[IBoard] | None = None

        # Elo evaluation: freeze the random-init network as the anchor opponent.
        # ``elo_baseline_net`` is a separate wrapper instance with that frozen
        # state so the current ``self.nnet`` can train without disturbing it.
        # Saved to disk under ``Nets/elo_baseline.pth.tar`` so resumed runs use
        # the same baseline.
        if self.config.elo_games_per_gen > 0:
            baseline_path = self.config.net_directory / "elo_baseline.pth.tar"
            # On resume, reuse the original gen-0 baseline if it's on disk —
            # re-saving here would re-anchor it to the already-trained net and
            # make the resumed Elo numbers incomparable to the pre-crash portion.
            if not (self.resume and baseline_path.exists()):
                self.nnet.save_checkpoint(filename="elo_baseline.pth.tar")
            elif self.resume:
                logger.info("Resume: reusing existing Elo baseline {}", baseline_path)
            self.elo_baseline_net: INeuralNetWrapper | None = self.nnet.__class__(self.game, config)  # type: ignore[call-arg]
            self.elo_baseline_net.load_checkpoint(filename="elo_baseline.pth.tar")
        else:
            self.elo_baseline_net = None

        # Rolling arena-derived Elo (docs/plans/archive/arena-derived-elo.md).
        # The arena already plays candidate-vs-incumbent, so the incumbent is a
        # rolling benchmark: the candidate's Elo is ``_benchmark_elo +
        # compute_elo(arena_result)``, and on acceptance the benchmark rolls
        # forward to the candidate. Anchored at ``elo_baseline_rating``; this is
        # the non-saturating live strength curve that replaced the frozen-gen-0
        # eval.
        self._benchmark_elo: float = float(self.config.elo_baseline_rating)

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

            self._log_memory_snapshot(generation, CycleStage.SELF_PLAY)

            # Persist this generation's fresh games (file index = generation - 1).
            self.save_self_play_history(generation - 1)

            # The save used to be the peak-RSS moment (whole-generation densify);
            # snapshot it so any regression shows up in the run, not post-mortem.
            self._log_memory_snapshot(generation, CycleStage.SAVE)

            train_examples = self.replay_buffer.flat_shuffled_examples()

            # Build/load the frozen eval set used for per-epoch network
            # entropy logging. First gen's self-play is the source.
            self._ensure_eval_set(train_examples)

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
            self.metrics.log_arena(
                generation,
                wins=nwins,
                losses=pwins,
                draws=draws,
                accepted=accepted,
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
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

        # Everything this generation produced (checkpoints, parquet, the marker
        # itself) is now on local disk — mirror it. Best-effort by design:
        # object-storage trouble never kills training.
        sync_up_guarded(self.object_store, self.config.run_directory, f"generation {generation}")

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
        )
        new_player = NetworkPlayer(
            game=self.game,
            nnet=self.nnet,
            mcts_config=self.config.mcts_config,
            temp=0.0,
        )
        arena = Arena(prev_player, new_player, self.game)
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
        )
        return new_wins, prev_wins, draws, records

    def _evaluate_strength_vs_baselines(self, generation: int) -> None:
        """Play the new network this gen against fixed baselines, log results.

        Two baselines:

        1. **Elo vs gen-0** (always, when ``elo_games_per_gen > 0``): the
           frozen random-init network from training start. Score rate +
           Elo diff are computed via :func:`alphablokus.evaluation.elo.compute_elo`
           and logged via :meth:`MetricsCollector.log_elo`.
        2. **Perfect-play oracle** (only when the game has one and
           ``minimax_games_per_gen > 0``): draw rate rising to 1.0 with loss
           rate falling to 0 means the model has internalised optimal play.

        Both arenas use the same MCTS sim count as the regular accept/reject
        arena, for consistent comparison. The new network's MCTS tree is
        reset between games via the :class:`NetworkPlayer.startGame` hook.
        """
        if self.elo_baseline_net is not None and self.config.elo_games_per_gen > 0:
            self._evaluate_elo_vs_baseline(generation)
        if self._oracle is not None and self.config.minimax_games_per_gen > 0:
            self._evaluate_vs_oracle(generation)
        if self.config.symmetry_diagnostic_positions > 0:
            self._evaluate_symmetry_diagnostic(generation)

    def _evaluate_elo_vs_baseline(self, generation: int) -> None:
        assert self.elo_baseline_net is not None
        n = self.config.elo_games_per_gen
        baseline_rating = self.config.elo_baseline_rating
        logger.info(f"Evaluating Elo vs frozen gen-0 baseline ({n} games) ...")
        elo_start = time.perf_counter()

        if self.config.num_parallel_workers > 1:
            wins, losses, draws = self._run_elo_parallel(generation, n)
        else:
            wins, losses, draws = self._run_elo_serial(n)

        elo_diff, score_rate = compute_elo(wins, losses, draws)
        absolute = baseline_rating + elo_diff
        elapsed = time.perf_counter() - elo_start
        logger.info(
            "Gen {} Elo: {:.0f} ({:+.0f} vs baseline) — W{} L{} D{}, score rate {:.3f}, {:.1f}s",
            generation,
            absolute,
            elo_diff,
            wins,
            losses,
            draws,
            score_rate,
            elapsed,
        )
        self.metrics.log_elo(
            generation=generation,
            elo_diff=elo_diff,
            baseline_rating=baseline_rating,
            score_rate=score_rate,
            wins=wins,
            losses=losses,
            draws=draws,
            games=wins + losses + draws,
        )

    def _run_elo_serial(self, n: int) -> tuple[int, int, int]:
        """Sequential Elo loop. Returns ``(new_wins, baseline_wins, draws)``."""
        assert self.elo_baseline_net is not None
        new_player = NetworkPlayer(
            game=self.game,
            nnet=self.nnet,
            mcts_config=self.config.mcts_config,
            temp=0.0,
        )
        baseline_player = NetworkPlayer(
            game=self.game,
            nnet=self.elo_baseline_net,
            mcts_config=self.config.mcts_config,
            temp=0.0,
        )
        arena = Arena(new_player, baseline_player, self.game)
        wins, losses, draws, _ = arena.play_games(n)
        return wins, losses, draws

    def _run_elo_parallel(self, generation: int, n: int) -> tuple[int, int, int]:
        """Parallel Elo across the worker pool.

        The baseline checkpoint (``elo_baseline.pth.tar``) is written
        once in ``Coach.__init__`` and never changes, so workers can
        load it directly. The new net's weights get a fresh per-gen
        checkpoint so the right network is being evaluated.
        Returns ``(new_wins, baseline_wins, draws)``.
        """
        from alphablokus.parallel.pool import (
            PHASE_ELO,
            run_two_player_games_parallel,
        )

        new_checkpoint = "parallel_elo_new.pth.tar"
        self.nnet.save_checkpoint(filename=new_checkpoint)

        a_wins, b_wins, draws, _ = run_two_player_games_parallel(
            config=self.config,
            generation=generation,
            checkpoint_a_path=new_checkpoint,
            checkpoint_b_path="elo_baseline.pth.tar",
            num_games=n,
            num_workers=self.config.num_parallel_workers,
            phase=PHASE_ELO,
            record=False,
            top_k=0,
            desc="Elo",
        )
        return a_wins, b_wins, draws

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

    def _ensure_eval_set(self, train_examples: list[ProcessedExample]) -> None:
        """Build/load the frozen eval set once (see :func:`build_or_load_eval_set`)."""
        if self._eval_set is not None:
            return
        self._eval_set = build_or_load_eval_set(
            self.config,
            self.game,
            self._oracle,
            train_examples,
            self._eval_set_size,
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

        Thin wrapper around :func:`alphablokus.evaluation.acceptance.is_accepted_score_rule`.
        Single source of truth lives there so reporting code can never
        diverge from the training-time decision — see ``evaluation/acceptance.py``
        for the full rationale.
        """
        from alphablokus.evaluation.acceptance import is_accepted_score_rule

        return is_accepted_score_rule(
            new_wins=new_wins,
            prev_wins=prev_wins,
            draws=draws,
            threshold=self.config.update_threshold,
        )

    def save_self_play_history(self, file_index: int) -> None:
        """Persist this generation's fresh games (see :meth:`ReplayBuffer.save_fresh`)."""
        self.replay_buffer.save_fresh(file_index)

    def load_self_play_history(self, up_to_generation: int) -> None:
        """Warm-start refill of the buffer (see :meth:`ReplayBuffer.load_recent`)."""
        self.replay_buffer.load_recent(up_to_generation)
