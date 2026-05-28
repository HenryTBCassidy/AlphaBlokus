import math
import time
from collections import deque
from dataclasses import dataclass
from random import shuffle
from typing import TypeAlias

import numpy as np
import psutil
import torch
from loguru import logger
from numpy.typing import NDArray
from tqdm import tqdm

from core.arena import Arena
from core.config import RunConfig
from core.interfaces import IBoard, IGame, INeuralNetWrapper
from core.mcts import MCTS
from core.players import NetworkPlayer
from core.storage import (
    CycleStage,
    EvalSet,
    MetricsCollector,
    ProcessedExample,
    SelfPlayStore,
)


def _compute_elo(wins: int, losses: int, draws: int) -> tuple[float, float]:
    """Chess-style Elo difference vs an anchor opponent.

    Score rate = (wins + 0.5·draws) / total_games, clamped to [0.001, 0.999]
    to avoid log(0). Elo difference = 400 · log₁₀(score_rate / (1−score_rate)).
    Returns ``(elo_diff, score_rate)``.
    """
    total = wins + losses + draws
    if total == 0:
        return 0.0, 0.0
    raw = (wins + 0.5 * draws) / total
    score_rate = max(0.001, min(0.999, raw))
    elo = 400 * math.log10(score_rate / (1 - score_rate))
    return elo, raw

# Type aliases for improved readability
TrainingExample: TypeAlias = tuple[IBoard, int, NDArray, float | None]  # (board, player, policy, value)
TrainingHistory: TypeAlias = list[deque[ProcessedExample]]  # List of examples per generation


@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time memory usage of the current process.

    All values are in bytes. ``gpu_bytes`` is ``None`` when no GPU is
    available or the backend doesn't expose an allocation counter.
    """

    process_rss_bytes: int
    gpu_bytes: float | None


def _get_memory_snapshot() -> MemorySnapshot:
    """Take a cross-platform snapshot of the current process's memory usage.

    Uses ``psutil`` for process RSS (works identically on macOS, Linux,
    and Windows — always returns bytes).

    For GPU memory, checks CUDA first, then MPS (Apple Silicon).
    Returns ``None`` for ``gpu_bytes`` if no GPU is available.
    """
    rss = psutil.Process().memory_info().rss

    gpu_mem: float | None = None
    if torch.cuda.is_available():
        gpu_mem = float(torch.cuda.memory_allocated())
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_mem = float(torch.mps.current_allocated_memory())

    return MemorySnapshot(process_rss_bytes=rss, gpu_bytes=gpu_mem)


class Coach:
    """
    Main training coordinator for the AlphaZero-style learning process.
    
    This class orchestrates the entire training loop:
    1. Self-play: Generate training data using MCTS + current neural network
    2. Training: Update network weights using generated data
    3. Evaluation: Compare new network against previous version
    
    The process is iterative, with each complete cycle called a 'generation'.
    The neural network improves over time by learning from self-play games,
    but only if the new version proves stronger than the previous one.
    """

    def __init__(self, game: IGame, nnet: INeuralNetWrapper, config: RunConfig) -> None:
        """
        Initialize the training coordinator.

        Args:
            game: Game implementation providing rules and mechanics
            nnet: Neural network for policy and value predictions
            config: Configuration parameters for the training process
        """
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
        self.pnet = self.nnet.__class__(self.game, config)  # Previous best network
        self.config = config
        self.mcts = MCTS(self.game, self.nnet, self.config.mcts_config)

        # Training state
        self.train_examples_history: TrainingHistory = []
        self.skip_first_self_play = False  # Can be set by load_train_examples()
        self.metrics = MetricsCollector(config=config)
        self._self_play_store = SelfPlayStore(config.self_play_history_directory)

        # Frozen held-out positions for per-epoch network diagnostics (policy
        # entropy, top-K accuracy, value calibration). Built lazily from gen
        # 1's self-play examples; saved to disk so resumed runs use the same set.
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
            self.nnet.save_checkpoint(filename="elo_baseline.pth.tar")
            self.elo_baseline_net: INeuralNetWrapper | None = self.nnet.__class__(self.game, config)
            self.elo_baseline_net.load_checkpoint(filename="elo_baseline.pth.tar")
        else:
            self.elo_baseline_net = None

    def execute_episode(self) -> list[ProcessedExample]:
        """Execute one complete self-play game to generate training data.

        Thin wrapper around :func:`core.self_play.play_self_play_episode`,
        which is the single source of truth for the episode loop. Both
        this serial entry point and the parallel worker module in
        :mod:`core.parallel_self_play` call into that function — keeping
        them bit-for-bit equivalent at the same seed (basis of the F1
        determinism test).
        """
        from core.self_play import play_self_play_episode
        return play_self_play_episode(
            self.game, self.mcts, self.config.temp_threshold,
        )

    def learn(self) -> None:
        """
        Execute the main training loop for a specified number of generations.

        Each generation consists of:
        1. Self-play: Generate new games using MCTS + current network
        2. Training: Update network weights using accumulated game data
        3. Arena: Evaluate new network against previous version
        
        The new network is only accepted if it wins >= update_threshold
        fraction of games against the previous version.

        Notes:
            - Training data from older generations is gradually discarded
            - Game data is saved after each generation
            - Timing information is collected for performance analysis
        """
        try:
            self._learn_loop()
        finally:
            # Ensure W&B (if active) is finalised even on crash/interrupt.
            self.metrics.close()

    def _learn_loop(self) -> None:
        """Inner training loop. Separated so ``learn`` can wrap it in try/finally."""
        for generation in range(1, self.config.num_generations + 1):
            logger.info(f'Starting Generation #{generation} ...')
            generation_start = time.perf_counter()
            self.metrics.log_progress(generation, self.config.num_generations)

            # PHASE 1: Generate new training data through self-play
            if not self.skip_first_self_play or generation > 1:
                iteration_examples = deque([], maxlen=self.config.max_queue_length)

                logger.info(f'Starting Self-Play For Generation #{generation} ...')
                self_play_start = time.perf_counter()

                if self.config.num_parallel_workers > 1:
                    self._run_self_play_parallel(generation, iteration_examples)
                else:
                    self._run_self_play_serial(generation, iteration_examples)

                self_play_end = time.perf_counter()
                self.metrics.log_timing(generation, CycleStage.SELF_PLAY, self_play_end - self_play_start)

                # Memory snapshot after self-play phase
                snapshot = _get_memory_snapshot()
                self.metrics.log_resource_usage(
                    generation, CycleStage.SELF_PLAY,
                    snapshot.process_rss_bytes, snapshot.gpu_bytes,
                )

                self.train_examples_history.append(iteration_examples)

            # Save generated data and manage training window
            self.save_self_play_history(generation - 1)
            self._manage_training_window(generation)

            # PHASE 2: Train neural network
            train_examples = self._prepare_training_data()

            # Build/load the frozen eval set used for per-epoch network
            # entropy logging. First gen's self-play is the source.
            self._ensure_eval_set(train_examples)

            # Preserve current best network
            self.nnet.save_checkpoint(filename='temp.pth.tar')
            self.pnet.load_checkpoint(filename='temp.pth.tar')
            MCTS(self.game, self.pnet, self.config.mcts_config)

            # Train network on accumulated data
            logger.info(f'Starting Training For Generation #{generation} ...')
            training_start = time.perf_counter()
            self.nnet.train(
                train_examples, generation,
                metrics=self.metrics, eval_set=self._eval_set,
            )
            training_end = time.perf_counter()
            self.metrics.log_timing(generation, CycleStage.TRAINING, training_end - training_start)

            # Memory snapshot after training phase
            snapshot = _get_memory_snapshot()
            self.metrics.log_resource_usage(
                generation, CycleStage.TRAINING,
                snapshot.process_rss_bytes, snapshot.gpu_bytes,
            )

            # PHASE 3: Evaluate new network
            logger.info(f'Evaluating Against Previous Version For Generation #{generation} ...')
            arena_start = time.perf_counter()
            # ``top_k`` capped at the action-space size (TTT only has 10
            # actions; for Blokus's 17,837 actions 20 is plenty to capture
            # the meaningful head). Recording at least 20 also guarantees
            # the played action is in the recorded list even when MCTS is
            # uniform across many tied actions.
            top_k_to_record = min(self.game.get_action_size(), 20)

            if self.config.num_parallel_workers > 1:
                nwins, pwins, draws, game_records = self._run_arena_parallel(
                    generation, top_k_to_record,
                )
            else:
                nwins, pwins, draws, game_records = self._run_arena_serial(
                    top_k_to_record,
                )

            arena_end = time.perf_counter()
            accepted = self._should_accept_new_network(nwins, pwins, draws)
            self.metrics.log_arena(
                generation, wins=nwins, losses=pwins, draws=draws, accepted=accepted,
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
            snapshot = _get_memory_snapshot()
            self.metrics.log_resource_usage(
                generation, CycleStage.ARENA,
                snapshot.process_rss_bytes, snapshot.gpu_bytes,
            )

            # Accept or reject new network
            logger.info(f'NEW/PREV WINS : {nwins}/{pwins}; DRAWS : {draws}')
            if accepted:
                logger.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(filename=f"accepted_{generation}.pth.tar")
                self.nnet.save_checkpoint(filename='best.pth.tar')
            else:
                logger.info('REJECTING NEW MODEL')
                self.nnet.save_checkpoint(filename=f'rejected_{generation}.pth.tar')
                self.nnet.load_checkpoint(filename='temp.pth.tar')

            # PHASE 4: Strength evaluation against fixed baselines.
            # The new network this gen is measured against the frozen gen-0
            # baseline (Elo) and, for TTT, a perfect-play minimax opponent.
            # Logged whether or not the new network was accepted in arena.
            self._evaluate_strength_vs_baselines(generation)

            # Record total generation time and flush metrics
            generation_end = time.perf_counter()
            self.metrics.log_timing(generation, CycleStage.WHOLE_CYCLE, generation_end - generation_start)
            self.metrics.flush(self.config, generation)

    def _run_self_play_serial(
        self, generation: int, iteration_examples: deque,
    ) -> None:
        """Original sequential self-play loop. Kept bit-for-bit identical
        to the pre-F1 behaviour: same MCTS instance lifecycle, same
        per-episode logging, same tqdm. The parallel codepath is opt-in
        via ``config.num_parallel_workers > 1``.
        """
        for episode_idx in tqdm(range(self.config.num_eps), desc="Self Play"):
            self.mcts = MCTS(self.game, self.nnet, self.config.mcts_config)
            iteration_examples.extend(self.execute_episode())

            stats = self.mcts.get_episode_stats()
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

    def _run_self_play_parallel(
        self, generation: int, iteration_examples: deque,
    ) -> None:
        """Parallel self-play across a process pool (F1).

        Saves the current ``self.nnet`` to a fixed checkpoint workers
        load at pool startup, dispatches ``num_eps`` per-episode tasks,
        then extends ``iteration_examples`` with the returned examples
        in submission order. Per-episode MCTS stats are logged via
        ``metrics.log_self_play_profiling`` to match the serial
        codepath's schema — downstream reports don't care which path
        produced the data.
        """
        from core.parallel_self_play import run_self_play_episodes_parallel

        worker_init_checkpoint = "parallel_worker_init.pth.tar"
        self.nnet.save_checkpoint(filename=worker_init_checkpoint)

        per_ep_examples, per_ep_stats = run_self_play_episodes_parallel(
            config=self.config,
            generation=generation,
            checkpoint_path=worker_init_checkpoint,
            num_workers=self.config.num_parallel_workers,
        )

        for episode_idx, (examples, stats) in enumerate(zip(per_ep_examples, per_ep_stats, strict=False)):
            iteration_examples.extend(examples)
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
        self, top_k_to_record: int,
    ) -> tuple[int, int, int, list]:
        """Original sequential arena loop. Returns
        ``(new_wins, prev_wins, draws, game_records)``.
        """
        prev_player = NetworkPlayer(
            game=self.game, nnet=self.pnet,
            mcts_config=self.config.mcts_config, temp=0.0,
        )
        new_player = NetworkPlayer(
            game=self.game, nnet=self.nnet,
            mcts_config=self.config.mcts_config, temp=0.0,
        )
        arena = Arena(prev_player, new_player, self.game)
        pwins, nwins, draws, records = arena.play_games(
            self.config.num_arena_matches, record=True, top_k=top_k_to_record,
        )
        return nwins, pwins, draws, records

    def _run_arena_parallel(
        self, generation: int, top_k_to_record: int,
    ) -> tuple[int, int, int, list]:
        """Parallel arena across the F1 worker pool.

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
        :func:`reporting.training._render_arena_replays` expects when
        it labels winners.

        Getting this wrong would silently flip the "new net wins" /
        "previous net wins" labels in the HTML report. The convention
        is enforced by a unit test in
        ``tests/test_core/test_parallel_self_play.py``.
        """
        from core.parallel_self_play import (
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
            checkpoint_b_path=new_checkpoint,   # B = new
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
           Elo diff are computed via :func:`_compute_elo` and logged via
           :meth:`MetricsCollector.log_elo`.
        2. **TTT minimax** (only when the game is TicTacToe and
           ``minimax_games_per_gen > 0``): perfect-play opponent. Draw rate
           rising to 1.0 with loss rate falling to 0 means the model has
           internalised optimal play.

        Both arenas use the same MCTS sim count as the regular accept/reject
        arena, for consistent comparison. The new network's MCTS tree is
        reset between games via the :class:`NetworkPlayer.startGame` hook.
        """
        if self.elo_baseline_net is not None and self.config.elo_games_per_gen > 0:
            self._evaluate_elo_vs_baseline(generation)
        if (
            self.config.game == "tictactoe"
            and self.config.minimax_games_per_gen > 0
        ):
            self._evaluate_minimax_tictactoe(generation)
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

        elo_diff, score_rate = _compute_elo(wins, losses, draws)
        absolute = baseline_rating + elo_diff
        elapsed = time.perf_counter() - elo_start
        logger.info(
            "Gen {} Elo: {:.0f} ({:+.0f} vs baseline) — W{} L{} D{}, score rate {:.3f}, {:.1f}s",
            generation, absolute, elo_diff, wins, losses, draws, score_rate, elapsed,
        )
        self.metrics.log_elo(
            generation=generation, elo_diff=elo_diff, baseline_rating=baseline_rating,
            score_rate=score_rate, wins=wins, losses=losses, draws=draws,
            games=wins + losses + draws,
        )

    def _run_elo_serial(self, n: int) -> tuple[int, int, int]:
        """Original sequential Elo loop. Returns ``(new_wins, baseline_wins, draws)``."""
        assert self.elo_baseline_net is not None
        new_player = NetworkPlayer(
            game=self.game, nnet=self.nnet,
            mcts_config=self.config.mcts_config, temp=0.0,
        )
        baseline_player = NetworkPlayer(
            game=self.game, nnet=self.elo_baseline_net,
            mcts_config=self.config.mcts_config, temp=0.0,
        )
        arena = Arena(new_player, baseline_player, self.game)
        wins, losses, draws, _ = arena.play_games(n)
        return wins, losses, draws

    def _run_elo_parallel(self, generation: int, n: int) -> tuple[int, int, int]:
        """Parallel Elo across the F1 worker pool.

        The baseline checkpoint (``elo_baseline.pth.tar``) is written
        once in ``Coach.__init__`` and never changes, so workers can
        load it directly. The new net's weights get a fresh per-gen
        checkpoint so the right network is being evaluated.
        Returns ``(new_wins, baseline_wins, draws)``.
        """
        from core.parallel_self_play import (
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

    def _evaluate_minimax_tictactoe(self, generation: int) -> None:
        from core.players import NetworkPlayer
        from games.tictactoe.minimax import MinimaxTicTacToePlayer

        n = self.config.minimax_games_per_gen
        logger.info(f"Evaluating vs TTT minimax perfect-play ({n} games) ...")
        mm_start = time.perf_counter()

        new_player = NetworkPlayer(
            game=self.game, nnet=self.nnet,
            mcts_config=self.config.mcts_config, temp=0.0,
        )
        minimax_player = MinimaxTicTacToePlayer(self.game)
        arena = Arena(new_player, minimax_player, self.game)
        wins, losses, draws, _ = arena.play_games(n)
        elapsed = time.perf_counter() - mm_start
        logger.info(
            "Gen {} vs minimax: W{} L{} D{} (draw_rate {:.2f}, {:.1f}s)",
            generation, wins, losses, draws,
            draws / max(wins + losses + draws, 1), elapsed,
        )
        self.metrics.log_minimax(
            generation=generation, wins=wins, losses=losses, draws=draws,
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
        from core.symmetry_diagnostic import (
            build_diagnostic_positions,
            compute_symmetry_diagnostic,
        )

        if self._symmetry_diagnostic_positions is None:
            self._symmetry_diagnostic_positions = build_diagnostic_positions(
                self.game, n=self.config.symmetry_diagnostic_positions,
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
                generation, mean_of_means, len(position_results),
                time.perf_counter() - start,
            )
        self.metrics.log_symmetry_diagnostic(generation, position_results)

    def _ensure_eval_set(self, train_examples: list[ProcessedExample]) -> None:
        """Build (or load from disk) the frozen held-out eval set.

        For TicTacToe we replace the self-play-derived targets with **minimax
        oracle** targets: each position's ``target_policies`` row is uniform
        over all game-theoretically optimal actions, and ``target_values`` is
        the true minimax value of the position (∈ ``{-1, 0, +1}``). This makes
        the per-generation top-K agreement plot answer the right question
        ("does the net pick a truly optimal move?") rather than chasing
        gen-1's noisy MCTS targets, which is what was making that curve dip
        over training.

        For other games we keep the original behaviour: the eval set targets
        are MCTS visit distributions and final game outcomes recorded during
        gen 1's self-play.

        Sampled once from gen 1's training examples and saved to
        ``config.eval_set_directory`` as three numpy files: ``boards.npy``,
        ``target_policies.npy``, ``target_values.npy``. If any of the three is
        missing on disk we re-sample (which keeps things consistent between
        old runs and current ones).
        """
        if self._eval_set is not None:
            return

        eval_dir = self.config.eval_set_directory
        boards_path = eval_dir / "boards.npy"
        policies_path = eval_dir / "target_policies.npy"
        values_path = eval_dir / "target_values.npy"
        # Marker file: tells us *how* the targets were generated. We refuse to
        # reuse an on-disk eval set whose targets don't match the current
        # scheme — otherwise an old "selfplay-targets" file would silently
        # poison the metrics on a TTT run that now expects minimax targets.
        marker_path = eval_dir / "targets_kind.txt"
        expected_kind = "minimax_v1" if self.config.game == "tictactoe" else "selfplay_v1"
        if (
            boards_path.exists() and policies_path.exists() and values_path.exists()
            and marker_path.exists() and marker_path.read_text().strip() == expected_kind
        ):
            self._eval_set = EvalSet(
                boards=np.load(boards_path),
                target_policies=np.load(policies_path),
                target_values=np.load(values_path),
            )
            logger.info("Loaded eval set ({} positions, kind={}) from {}",
                        len(self._eval_set), expected_kind, eval_dir)
            return

        if not train_examples:
            return

        # Sample positions from the training examples (capped to actual size).
        boards_all = np.array([ex[0] for ex in train_examples])
        policies_all = np.array([ex[1] for ex in train_examples])
        values_all = np.array([ex[2] for ex in train_examples])
        n = min(self._eval_set_size, len(boards_all))
        rng = np.random.default_rng(seed=self.config.seed or 0)
        idx = rng.choice(len(boards_all), size=n, replace=False)
        sampled_boards = boards_all[idx]
        target_policies = policies_all[idx]
        target_values = values_all[idx]

        if self.config.game == "tictactoe":
            target_policies, target_values = self._minimax_targets_for_eval_set(
                sampled_boards, action_size=target_policies.shape[1],
            )

        self._eval_set = EvalSet(
            boards=sampled_boards,
            target_policies=target_policies,
            target_values=target_values,
        )

        eval_dir.mkdir(parents=True, exist_ok=True)
        np.save(boards_path, self._eval_set.boards)
        np.save(policies_path, self._eval_set.target_policies)
        np.save(values_path, self._eval_set.target_values)
        marker_path.write_text(expected_kind)
        logger.info("Built eval set ({} positions, kind={}) → {}",
                    n, expected_kind, eval_dir)

    def _minimax_targets_for_eval_set(
        self, boards: NDArray, action_size: int,
    ) -> tuple[NDArray, NDArray]:
        """Overwrite eval-set targets with a perfect-play oracle (TTT only).

        Each row of ``target_policies`` becomes a uniform distribution over
        all minimax-optimal actions; each ``target_values`` entry becomes the
        position's true game-theoretic value. Boards are decoded from the
        2-channel canonical encoding back into a :class:`Board` so we can
        query the minimax solver.
        """
        from games.tictactoe.board import Board
        from games.tictactoe.minimax import MinimaxTicTacToePlayer

        minimax = MinimaxTicTacToePlayer(self.game)
        n = len(boards)
        new_policies = np.zeros((n, action_size), dtype=np.float32)
        new_values = np.zeros(n, dtype=np.float32)

        for i, encoded in enumerate(boards):
            # Decode canonical 2-channel encoding back into a canonical Board.
            # encoded[0]: side-to-move stones (=+1), encoded[1]: opponent (=-1).
            grid = (encoded[0] - encoded[1]).astype(int)
            canonical_board = Board._from_pieces(grid.tolist())

            new_values[i] = float(minimax.evaluate_position(canonical_board))
            optimal = minimax.optimal_actions(canonical_board)
            if optimal:
                weight = 1.0 / len(optimal)
                for action in optimal:
                    new_policies[i, action] = weight

        return new_policies, new_values

    def _generation_window_size(self, generation: int) -> int:
        """
        Calculate the number of past generations to retain for training.

        The window size grows linearly from 5 to max_generations_lookback,
        allowing the network to:
        - Learn quickly from recent games in early training
        - Maintain stability from more games in later training

        Args:
            generation: Current training iteration

        Returns:
            int: Number of past generations to use for training
        """
        if generation <= 5:
            return 5
            
        max_val = self.config.max_generations_lookback
        if 2 * max_val > generation:
            # Linear growth: y = mx + b
            # Points: (0,5) and (2*max_val, max_val)
            return round(generation * (max_val - 5) / (2 * max_val) + 5)
            
        return max_val

    def _manage_training_window(self, generation: int) -> None:
        """
        Remove old training data that falls outside the current window.

        Args:
            generation: Current training iteration
        """
        window_size = self._generation_window_size(generation)
        if len(self.train_examples_history) > window_size:
            logger.warning(
                f"Removing oldest training examples. "
                f"History size: {len(self.train_examples_history)}"
            )
            self.train_examples_history.pop(0)

    def _prepare_training_data(self) -> list[ProcessedExample]:
        """
        Combine and shuffle training examples from all retained generations.

        Returns:
            List[ProcessedExample]: Shuffled training examples
        """
        examples = []
        for generation_examples in self.train_examples_history:
            examples.extend(generation_examples)
        shuffle(examples)
        return examples

    def _should_accept_new_network(
        self, new_wins: int, prev_wins: int, draws: int = 0,
    ) -> bool:
        """Decide whether to accept the newly trained network.

        Thin wrapper around :func:`core.acceptance.is_accepted_score_rule`.
        Single source of truth lives there so reporting code can never
        diverge from the training-time decision — see ``core/acceptance.py``
        for the full rationale.
        """
        from core.acceptance import is_accepted_score_rule
        return is_accepted_score_rule(
            new_wins=new_wins, prev_wins=prev_wins, draws=draws,
            threshold=self.config.update_threshold,
        )

    def save_self_play_history(self, generation: int) -> None:
        """Save the current generation's self-play data to a parquet file.

        Delegates to :meth:`SelfPlayStore.save`.
        """
        if not self.train_examples_history:
            return
        latest = self.train_examples_history[-1]
        if not latest:
            return
        self._self_play_store.save(latest, generation)

    def load_self_play_history(self, up_to_generation: int) -> None:
        """Load self-play examples from parquet files for recent generations.

        Delegates to :meth:`SelfPlayStore.load_window`.
        """
        window = self._generation_window_size(up_to_generation)
        self.train_examples_history = self._self_play_store.load_window(
            up_to_generation, window,
        )
