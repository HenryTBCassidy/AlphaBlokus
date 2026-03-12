import time
from collections import deque
from dataclasses import dataclass
from random import shuffle
from typing import TypeAlias

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from loguru import logger
from numpy.typing import NDArray
from tqdm import tqdm

from core.arena import Arena
from core.config import RunConfig
from core.mcts import MCTS
from core.metrics import CycleStage, MetricsCollector
from core.interfaces import INeuralNetWrapper, IGame

# Type aliases for improved readability
TrainingExample: TypeAlias = tuple[NDArray, int, NDArray, float | None]  # (board, player, policy, value)
ProcessedExample: TypeAlias = tuple[NDArray, NDArray, float]  # (board, policy, value)
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
        self.metrics = MetricsCollector()

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
                # Assign values to all positions based on final outcome
                return [(x[0], x[2], game_result * ((-1) ** (x[1] != current_player)))
                        for x in train_examples]

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
            
            # Preserve current best network
            self.nnet.save_checkpoint(filename='temp.pth.tar')
            self.pnet.load_checkpoint(filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.config.mcts_config)

            # Train network on accumulated data
            logger.info(f'Starting Training For Generation #{generation} ...')
            training_start = time.perf_counter()
            self.nnet.train(train_examples, generation, metrics=self.metrics)
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

            # Record total generation time and flush metrics
            generation_end = time.perf_counter()
            self.metrics.log_timing(generation, CycleStage.WHOLE_CYCLE, generation_end - generation_start)
            self.metrics.flush(self.config, generation)

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

    @staticmethod
    def _self_play_filename(generation: int) -> str:
        """Generate filename for a single generation's self-play data."""
        return f"self_play_{generation}.parquet"

    def save_self_play_history(self, generation: int) -> None:
        """
        Save the current generation's self-play data to a parquet file.

        Each generation is saved as a separate file containing flattened
        board and policy arrays as bytes columns, with shape metadata
        stored in the parquet file metadata for reconstruction.

        Args:
            generation: Training iteration number
        """
        folder = self.config.self_play_history_directory
        folder.mkdir(exist_ok=True, parents=True)

        if not self.train_examples_history:
            return

        # Save only the latest generation's examples
        latest = self.train_examples_history[-1]
        if not latest:
            return

        boards, policies, values = zip(*latest)

        # Ensure policies are numpy arrays (get_symmetries may return lists)
        policies = [np.array(p, dtype=np.float64) if not isinstance(p, np.ndarray) else p for p in policies]

        df = pd.DataFrame({
            "board": [b.tobytes() for b in boards],
            "policy": [p.tobytes() for p in policies],
            "value": list(values),
        })

        # Store array shapes in parquet metadata so we can reconstruct
        sample_board = boards[0]
        sample_policy = policies[0]
        metadata = {
            "board_shape": ",".join(str(d) for d in sample_board.shape),
            "board_dtype": str(sample_board.dtype),
            "policy_size": str(sample_policy.shape[0]),
            "policy_dtype": str(sample_policy.dtype),
        }

        table = pa.Table.from_pandas(df)
        merged_metadata = {**(table.schema.metadata or {}), **{k.encode(): v.encode() for k, v in metadata.items()}}
        table = table.replace_schema_metadata(merged_metadata)

        filepath = folder / self._self_play_filename(generation)
        pq.write_table(table, filepath)
        logger.info(f"Saved {len(df)} self-play examples to {filepath.name}")

    def load_self_play_history(self, up_to_generation: int) -> None:
        """
        Load self-play examples from parquet files for recent generations.

        Loads files for the most recent generations within the current
        training window size.

        Args:
            up_to_generation: Load generations up to and including this one.
        """
        folder = self.config.self_play_history_directory
        if not folder.exists():
            logger.warning(f"Self-play history directory not found: {folder}")
            return

        window = self._generation_window_size(up_to_generation)
        start_gen = max(0, up_to_generation - window)

        self.train_examples_history = []
        for gen in range(start_gen, up_to_generation + 1):
            filepath = folder / self._self_play_filename(gen)
            if not filepath.exists():
                continue

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

            self.train_examples_history.append(examples)
            logger.info(f"Loaded {len(examples)} examples from {filepath.name}")

        logger.info(
            f"Loaded {sum(len(e) for e in self.train_examples_history)} total examples "
            f"from {len(self.train_examples_history)} generations"
        )
