import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from arena import Arena
from mcts import MCTS
from neuralnet import NeuralNet
from config import RunConfig

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet: NeuralNet, run_config: RunConfig):
        """

        :param game:
        :param nnet:
        :param run_config:
        """
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, run_config)  # the competitor network
        self.run_config = run_config
        self.mcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)
        self.train_examples_history = []  # History of examples from args.num_iters_for_train_examples_history latest iterations
        self.skip_first_self_play = False  # can be overriden in loadTrainExamples()

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
            train_examples: a list of examples of the form (canonical_board, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = []
        board = self.game.get_init_board()
        cur_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, cur_player)
            temp = int(episode_step < self.run_config.temp_threshold)

            pi = self.mcts.get_action_prob(canonical_board, temp=temp)
            sym = self.game.get_symmetries(canonical_board, pi)
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
        generation. After every generation, it retrains neural network with
        examples in train_examples (which has a maximum length of max_queue_length).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= update_threshold fraction of games.
        """

        for generation in range(1, self.run_config.num_generations + 1):
            # bookkeeping
            log.info(f'Starting Iter #{generation} ...')
            # examples of the generation
            if not self.skip_first_self_play or generation > 1:
                iteration_train_examples = deque([], maxlen=self.run_config.max_queue_length)

                log.info(f'Starting Self Play For Generation #{generation} ...')
                for _ in tqdm(range(self.run_config.num_eps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)  # reset search tree
                    iteration_train_examples += self.execute_episode()

                # save the generation examples to the history
                self.train_examples_history.append(iteration_train_examples)

            if len(self.train_examples_history) > self.run_config.num_iters_for_train_examples_history:
                log.warning(
                    f"Removing the oldest entry in train_examples. len(trainExamplesHistory) = "
                    f"{len(self.train_examples_history)}")
                self.train_examples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous generation, so (i-1)
            self.save_self_play_history(generation - 1)

            # shuffle examples before training
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(filename='temp.pth.tar')
            self.pnet.load_checkpoint(filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.run_config.mcts_config)

            self.nnet.train(train_examples, generation)
            nmcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                          lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)), self.game)

            pwins, nwins, draws = arena.play_games(
                self.run_config.num_arena_matches,
                generation=generation, directory=self.run_config.arena_data_directory
            )

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.run_config.update_threshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.save_checkpoint(filename=f'rejected_{generation}.pth.tar')
                self.nnet.load_checkpoint(filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(filename=f"accepted_{generation}.pth.tar")
                self.nnet.save_checkpoint(filename='best.pth.tar')

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
