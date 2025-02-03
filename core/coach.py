import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.arena import Arena
from core.config import RunConfig, LOGGER_NAME
from core.mcts import MCTS
from core.interfaces import INeuralNetWrapper, IGame

log = logging.getLogger(LOGGER_NAME)


class CycleStage(StrEnum):
    SelfPlay = "SelfPlay"
    Training = "Training"
    Arena = "Arena"
    WholeCycle = "WholeCycle"

    def __repr__(self):
        return self._name_


@dataclass(frozen=True)
class TimingsLoggable:
    generation: int
    cycle_stage: CycleStage
    time_elapsed: float


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game: IGame, nnet: INeuralNetWrapper, run_config: RunConfig):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, run_config)  # the competitor network
        self.run_config = run_config
        self.mcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)

        # List of Generations, each generation is a list of every board state, pi vector and whether that won or lost
        # i.e. list[list[training_example]]
        self.train_examples_history = []
        self.skip_first_self_play = False  # Can override in loadTrainExamples()
        self.timings: list[TimingsLoggable] = []

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temp=1 if episode_step < temp_threshold, and thereafter
        uses temp=0.

        Returns:
            train_examples: A list of examples of the form [canonical_board, pi, v]
                            pi is the MCTS informed policy vector.
                            v is +1 if the player eventually won the game, else -1.
        """
        # Has form: [board, current player, pi vector, end result of game (initially not known so = None)
        train_examples = []
        board = self.game.initialise_board()
        cur_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, cur_player)
            temp = int(episode_step < self.run_config.temp_threshold)

            pi = self.mcts.get_action_prob(canonical_board, temp=temp)
            sym = self.game.get_symmetries(canonical_board, pi)

            # Because the board could have up to 4 fold rotation symmetry and flip symmetry, artificially add these
            # Positions into the network so the algorithm doesn't get a random preference for certain directions
            for b, p in sym:
                train_examples.append([b, cur_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.get_next_state(board, cur_player, action)

            r = self.game.get_game_ended(board, cur_player)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]

    def learn(self):
        """
        Performs num_generations iterations with num_eps episodes of self-play in each
        generation.

        After every generation, it retrains neural network with examples in train_examples.

        It then pits the new neural network against the old one and accepts it
        only if it wins >= update_threshold fraction of games.
        """

        for generation in range(1, self.run_config.num_generations + 1):
            # bookkeeping
            log.info(f'Starting Iter #{generation} ...')
            generation_start_time = time.perf_counter()
            # examples of the generation
            if not self.skip_first_self_play or generation > 1:
                iteration_train_examples = deque([], maxlen=self.run_config.max_queue_length)

                log.info(f'Starting Self Play For Generation #{generation} ...')
                self_play_start_time = time.perf_counter()
                for _ in tqdm(range(self.run_config.num_eps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)  # reset search tree
                    iteration_train_examples += self.execute_episode()
                self_play_end_time = time.perf_counter()
                self.timings.append(TimingsLoggable(
                    generation=generation, cycle_stage=CycleStage.SelfPlay,
                    time_elapsed=self_play_end_time - self_play_start_time
                ))
                # Save the generation examples to the history
                self.train_examples_history.append(iteration_train_examples)

            # backup history to a file
            # NB! the examples were collected using the model from the previous generation, so (i-1)
            self.save_self_play_history(generation - 1)

            if len(self.train_examples_history) > self._generation_window_size(generation):
                log.warning(
                    f"Removing the oldest entry in train_examples. len(trainExamplesHistory) = "
                    f"{len(self.train_examples_history)}")
                self.train_examples_history.pop(0)

            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)

            # Shuffle examples before training
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(filename='temp.pth.tar')
            self.pnet.load_checkpoint(filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.run_config.mcts_config)

            log.info(f'Starting Training For Generation #{generation} ...')
            training_start_time = time.perf_counter()
            self.nnet.train(train_examples, generation)
            training_end_time = time.perf_counter()

            self.timings.append(TimingsLoggable(
                generation=generation, cycle_stage=CycleStage.Training,
                time_elapsed=training_end_time - training_start_time
            ))

            nmcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)

            log.info(f'PITTING AGAINST PREVIOUS VERSION For Generation #{generation} ...')
            arena_start_time = time.perf_counter()
            arena = Arena(lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                          lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)), self.game)

            pwins, nwins, draws = arena.play_games(
                self.run_config.num_arena_matches,
                generation=generation, directory=self.run_config.arena_data_directory
            )

            arena_end_time = time.perf_counter()
            self.timings.append(TimingsLoggable(
                generation=generation, cycle_stage=CycleStage.Arena,
                time_elapsed=arena_end_time - arena_start_time
            ))

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.run_config.update_threshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.save_checkpoint(filename=f'rejected_{generation}.pth.tar')
                self.nnet.load_checkpoint(filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(filename=f"accepted_{generation}.pth.tar")
                self.nnet.save_checkpoint(filename='best.pth.tar')

            generation_end_time = time.perf_counter()
            self.timings.append(TimingsLoggable(
                generation=generation, cycle_stage=CycleStage.WholeCycle,
                time_elapsed=generation_end_time - generation_start_time
            ))

        self._write_timings()

    def _generation_window_size(self, generation: int) -> int:
        # Dynamically adjusted window size for generation lookback
        # To get window size, I've fit the line:
        # y = a*x + b to the points (x=0, y=5) and (x=2*max_window_size, y=max_window_size)
        # This allows a good combination of forgetting earlier games but retaining good memory as iterations go on

        if generation <= 5:
            return 5
        max_val = self.run_config.max_generations_lookback

        if 2 * max_val > generation:
            return round(generation * (max_val - 5) / (2 * max_val) + 5)

        return max_val

    def _write_timings(self):
        start = time.perf_counter()
        self.run_config.timings_directory.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([log_data.__dict__ for log_data in self.timings]).to_parquet(
            self.run_config.timings_directory / f"timings.parquet")

        end = time.perf_counter()
        logging.info(f"Took {end - start} seconds to write timings data!")

    @staticmethod
    def get_self_play_filename(generation) -> str:
        return f"self_play_history_{generation}.pickle"

    def save_self_play_history(self, generation):
        folder = self.run_config.self_play_history_directory
        if not folder.exists():
            folder.mkdir(exist_ok=True, parents=True)
        filename = folder / f"{self.get_self_play_filename(generation)}"
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)

    # TODO: Rename to load self play history and fix
    def load_train_examples(self):
        # TODO: Fix this it's broken
        model_file = os.path.join(self.run_config.load_folder_file[0], self.run_config.load_folder_file[1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            log.warning(f'File "{examples_file}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examples_file, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True
