import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pickle import Pickler, Unpickler
from random import shuffle
from typing import List, Deque, Optional, TypeAlias
from numpy.typing import NDArray

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.arena import Arena
from core.config import RunConfig, LOGGER_NAME
from core.mcts import MCTS
from core.interfaces import INeuralNetWrapper, IGame

# Type aliases for improved readability
TrainingExample: TypeAlias = tuple[NDArray, int, NDArray, Optional[float]]  # (board, player, policy, value)
ProcessedExample: TypeAlias = tuple[NDArray, NDArray, float]  # (board, policy, value)
TrainingHistory: TypeAlias = List[Deque[ProcessedExample]]  # List of examples per generation

log = logging.getLogger(LOGGER_NAME)


class CycleStage(StrEnum):
    """
    Enumeration of stages in the training cycle.
    
    Each generation consists of:
    - SelfPlay: Generate new games using current network
    - Training: Update network weights using game data
    - Arena: Evaluate new network against previous version
    - WholeCycle: Complete generation (all above stages)
    """
    SelfPlay = "SelfPlay"
    Training = "Training"
    Arena = "Arena"
    WholeCycle = "WholeCycle"

    def __repr__(self) -> str:
        """Return the enum member's name for string representation."""
        return self._name_


@dataclass(frozen=True)
class TimingsLoggable:
    """
    Data class for tracking execution time of training stages.
    
    Attributes:
        generation: Training iteration number
        cycle_stage: Which stage of training is being timed
        time_elapsed: Duration of the stage in seconds
    """
    generation: int
    cycle_stage: CycleStage
    time_elapsed: float


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

    def __init__(self, game: IGame, nnet: INeuralNetWrapper, run_config: RunConfig) -> None:
        """
        Initialize the training coordinator.

        Args:
            game: Game implementation providing rules and mechanics
            nnet: Neural network for policy and value predictions
            run_config: Configuration parameters for the training process
        """
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, run_config)  # Previous best network
        self.run_config = run_config
        self.mcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)
        
        # Training state
        self.train_examples_history: TrainingHistory = []
        self.skip_first_self_play = False  # Can be set by load_train_examples()
        self.timings: List[TimingsLoggable] = []

    def execute_episode(self) -> List[ProcessedExample]:
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
        train_examples: List[TrainingExample] = []
        board = self.game.initialise_board()
        current_player = 1
        move_count = 0

        while True:
            move_count += 1
            canonical_board = self.game.get_canonical_form(board, current_player)
            temperature = int(move_count < self.run_config.temp_threshold)

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
        for generation in range(1, self.run_config.num_generations + 1):
            log.info(f'Starting Generation #{generation} ...')
            generation_start = time.perf_counter()

            # PHASE 1: Generate new training data through self-play
            if not self.skip_first_self_play or generation > 1:
                iteration_examples = deque([], maxlen=self.run_config.max_queue_length)

                log.info(f'Starting Self-Play For Generation #{generation} ...')
                self_play_start = time.perf_counter()
                
                for _ in tqdm(range(self.run_config.num_eps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)
                    iteration_examples.extend(self.execute_episode())
                
                self_play_end = time.perf_counter()
                self.timings.append(TimingsLoggable(
                    generation=generation,
                    cycle_stage=CycleStage.SelfPlay,
                    time_elapsed=self_play_end - self_play_start
                ))
                
                self.train_examples_history.append(iteration_examples)

            # Save generated data and manage training window
            self.save_self_play_history(generation - 1)
            self._manage_training_window(generation)

            # PHASE 2: Train neural network
            train_examples = self._prepare_training_data()
            
            # Preserve current best network
            self.nnet.save_checkpoint(filename='temp.pth.tar')
            self.pnet.load_checkpoint(filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.run_config.mcts_config)

            # Train network on accumulated data
            log.info(f'Starting Training For Generation #{generation} ...')
            training_start = time.perf_counter()
            self.nnet.train(train_examples, generation)
            training_end = time.perf_counter()
            
            self.timings.append(TimingsLoggable(
                generation=generation,
                cycle_stage=CycleStage.Training,
                time_elapsed=training_end - training_start
            ))

            # PHASE 3: Evaluate new network
            nmcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)
            
            log.info(f'Evaluating Against Previous Version For Generation #{generation} ...')
            arena_start = time.perf_counter()
            
            arena = Arena(
                lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)),
                self.game
            )

            pwins, nwins, draws = arena.play_games(
                self.run_config.num_arena_matches,
                generation=generation,
                directory=self.run_config.arena_data_directory
            )

            arena_end = time.perf_counter()
            self.timings.append(TimingsLoggable(
                generation=generation,
                cycle_stage=CycleStage.Arena,
                time_elapsed=arena_end - arena_start
            ))

            # Accept or reject new network
            log.info(f'NEW/PREV WINS : {nwins}/{pwins}; DRAWS : {draws}')
            if self._should_accept_new_network(nwins, pwins):
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(filename=f"accepted_{generation}.pth.tar")
                self.nnet.save_checkpoint(filename='best.pth.tar')
            else:
                log.info('REJECTING NEW MODEL')
                self.nnet.save_checkpoint(filename=f'rejected_{generation}.pth.tar')
                self.nnet.load_checkpoint(filename='temp.pth.tar')

            # Record total generation time
            generation_end = time.perf_counter()
            self.timings.append(TimingsLoggable(
                generation=generation,
                cycle_stage=CycleStage.WholeCycle,
                time_elapsed=generation_end - generation_start
            ))

        self._write_timings()

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
            
        max_val = self.run_config.max_generations_lookback
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
            log.warning(
                f"Removing oldest training examples. "
                f"History size: {len(self.train_examples_history)}"
            )
            self.train_examples_history.pop(0)

    def _prepare_training_data(self) -> List[ProcessedExample]:
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
        return win_rate >= self.run_config.update_threshold

    def _write_timings(self) -> None:
        """
        Save timing data for all training stages to a parquet file.
        """
        start = time.perf_counter()
        self.run_config.timings_directory.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([t.__dict__ for t in self.timings]).to_parquet(
            self.run_config.timings_directory / "timings.parquet"
        )

        end = time.perf_counter()
        logging.info(f"Took {end - start} seconds to write timing data!")

    @staticmethod
    def get_self_play_filename(generation: int) -> str:
        """
        Generate filename for saving self-play data.

        Args:
            generation: Training iteration number

        Returns:
            str: Filename for self-play data
        """
        return f"self_play_history_{generation}.pickle"

    def save_self_play_history(self, generation: int) -> None:
        """
        Save self-play training data to a pickle file.

        Args:
            generation: Training iteration number
        """
        folder = self.run_config.self_play_history_directory
        if not folder.exists():
            folder.mkdir(exist_ok=True, parents=True)
            
        filename = folder / self.get_self_play_filename(generation)
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)

    # TODO: Rename to load_self_play_history and fix
    def load_train_examples(self) -> None:
        """
        Load previously saved training examples.

        TODO: This method is currently broken and needs to be fixed.
        It should be renamed to load_self_play_history for consistency.
        """
        # TODO: Fix this it's broken
        model_file = os.path.join(self.run_config.load_folder_file[0],
                                self.run_config.load_folder_file[1])
        examples_file = model_file + ".examples"
        
        if not os.path.isfile(examples_file):
            log.warning(f'File "{examples_file}" with training examples not found!')
            response = input("Continue? [y|n]")
            if response != "y":
                sys.exit()
        else:
            log.info("Loading saved training examples...")
            with open(examples_file, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            log.info("Loading complete!")
