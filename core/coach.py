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
from core.interfaces import IBoard, INeuralNetWrapper, IGame
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
        """
        Execute one complete self-play game to generate training data.

        The game is played using MCTS + current neural network, with moves chosen
        according to:
        - Temperature=1 for first temp_threshold moves (exploration)
        - Temperature≈0 afterwards (exploitation)

        For each game state encountered, we store:
        - Board position
        - MCTS policy (improved by tree search)
        - Game outcome from this position

        Additionally, we augment the data by adding symmetric positions
        to prevent the network from developing arbitrary directional preferences.

        Returns:
            List[ProcessedExample]: Training examples of the form (board, policy, value)
                                  where value is +1 for winning positions, -1 for losing
        """
        train_examples: list[TrainingExample] = []
        board = self.game.initialise_board()
        current_player = 1
        move_count = 0

        while True:
            move_count += 1
            canonical_board = self.game.get_canonical_form(board, current_player)
            temperature = int(move_count < self.config.temp_threshold)

            # Get improved policy from MCTS
            pi = self.mcts.get_action_prob(canonical_board, temp=temperature)
            
            # Add symmetric positions to training data
            symmetries = self.game.get_symmetries(canonical_board, pi)
            for symmetric_board, symmetric_pi in symmetries:
                train_examples.append((symmetric_board, current_player, symmetric_pi, None))

            # Choose and execute move
            action = np.random.choice(len(pi), p=pi)
            board, current_player = self.game.get_next_state(board, current_player, action)

            # Check if game has ended
            game_result = self.game.get_game_ended(board, current_player)
            if game_result != 0:
                # Assign values to all positions based on final outcome.
                # Convert canonical board objects → NDArrays for training.
                return [
                    (x[0].as_multi_channel(1), x[2], game_result * ((-1) ** (x[1] != current_player)))
                    for x in train_examples
                ]

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

            # PHASE 1: Generate new training data through self-play
            if not self.skip_first_self_play or generation > 1:
                iteration_examples = deque([], maxlen=self.config.max_queue_length)

                logger.info(f'Starting Self-Play For Generation #{generation} ...')
                self_play_start = time.perf_counter()

                for episode_idx in tqdm(range(self.config.num_eps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.config.mcts_config)
                    iteration_examples.extend(self.execute_episode())

                    # Log per-episode MCTS profiling data
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
            pmcts = MCTS(self.game, self.pnet, self.config.mcts_config)

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
            nmcts = MCTS(self.game, self.nnet, self.config.mcts_config)
            
            logger.info(f'Evaluating Against Previous Version For Generation #{generation} ...')
            arena_start = time.perf_counter()

            arena = Arena(
                lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)),
                self.game
            )

            pwins, nwins, draws = arena.play_games(self.config.num_arena_matches)

            arena_end = time.perf_counter()
            self.metrics.log_arena(generation, wins=nwins, losses=pwins, draws=draws)
            self.metrics.log_timing(generation, CycleStage.ARENA, arena_end - arena_start)

            # Memory snapshot after arena phase
            snapshot = _get_memory_snapshot()
            self.metrics.log_resource_usage(
                generation, CycleStage.ARENA,
                snapshot.process_rss_bytes, snapshot.gpu_bytes,
            )

            # Accept or reject new network
            logger.info(f'NEW/PREV WINS : {nwins}/{pwins}; DRAWS : {draws}')
            if self._should_accept_new_network(nwins, pwins):
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

    def _evaluate_elo_vs_baseline(self, generation: int) -> None:
        assert self.elo_baseline_net is not None
        n = self.config.elo_games_per_gen
        logger.info(f"Evaluating Elo vs frozen gen-0 baseline ({n} games) ...")
        elo_start = time.perf_counter()

        new_player = NetworkPlayer(
            game=self.game, nnet=self.nnet,
            mcts_config=self.config.mcts_config, temp=0.0,
        )
        baseline_player = NetworkPlayer(
            game=self.game, nnet=self.elo_baseline_net,
            mcts_config=self.config.mcts_config, temp=0.0,
        )
        arena = Arena(new_player, baseline_player, self.game)
        wins, losses, draws = arena.play_games(n)
        elo, score_rate = _compute_elo(wins, losses, draws)
        elapsed = time.perf_counter() - elo_start
        logger.info(
            "Gen {} Elo vs gen-0: {:+.1f} (W{} L{} D{}, score rate {:.3f}, {:.1f}s)",
            generation, elo, wins, losses, draws, score_rate, elapsed,
        )
        self.metrics.log_elo(
            generation=generation, elo=elo, score_rate=score_rate,
            wins=wins, losses=losses, draws=draws, games=wins + losses + draws,
        )

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
        wins, losses, draws = arena.play_games(n)
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

    def _ensure_eval_set(self, train_examples: list[ProcessedExample]) -> None:
        """Build (or load from disk) the frozen held-out eval set.

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
        if boards_path.exists() and policies_path.exists() and values_path.exists():
            self._eval_set = EvalSet(
                boards=np.load(boards_path),
                target_policies=np.load(policies_path),
                target_values=np.load(values_path),
            )
            logger.info("Loaded eval set ({} positions) from {}",
                        len(self._eval_set), eval_dir)
            return

        if not train_examples:
            return

        # Sample positions from the training examples (capped to actual size).
        boards_all = np.array([ex[0] for ex in train_examples])
        policies_all = np.array([ex[1] for ex in train_examples])
        values_all = np.array([ex[2] for ex in train_examples])
        n = min(self._eval_set_size, len(boards_all))
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(len(boards_all), size=n, replace=False)
        self._eval_set = EvalSet(
            boards=boards_all[idx],
            target_policies=policies_all[idx],
            target_values=values_all[idx],
        )

        eval_dir.mkdir(parents=True, exist_ok=True)
        np.save(boards_path, self._eval_set.boards)
        np.save(policies_path, self._eval_set.target_policies)
        np.save(values_path, self._eval_set.target_values)
        logger.info("Built eval set ({} positions) → {}", n, eval_dir)

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

    def _should_accept_new_network(self, new_wins: int, prev_wins: int) -> bool:
        """
        Decide whether to accept the newly trained network.

        Args:
            new_wins: Number of games won by new network
            prev_wins: Number of games won by previous network

        Returns:
            bool: True if new network should replace previous one
        """
        total_games = new_wins + prev_wins
        if total_games == 0:
            return False
        win_rate = float(new_wins) / total_games
        return win_rate >= self.config.update_threshold

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
