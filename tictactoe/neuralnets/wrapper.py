import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from pickle import Pickler, Unpickler

import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm

from core.config import RunConfig, LOGGER_NAME
from core.interfaces import INeuralNetWrapper
from utils import AverageMeter

sys.path.append('..')

from tictactoe.neuralnets.tictactoenet import AlphaTicTacToe as nnet
from tictactoe.tictactoegame import TicTacToeGame as Game

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""
log = logging.getLogger(LOGGER_NAME)


@dataclass
class TrainingDataLoggable:
    generation: int
    epoch: int
    batch_number: int
    pi_loss: float  # pi loss for this batch
    v_loss: float  # value loss for this batch
    total_loss: float  # Sum of above two i.e total loss for this cycle
    average_pi_loss: float  # Average pi loss of all batches so far
    average_v_loss: float  # Average value loss of all batches so far


class NNetWrapper(INeuralNetWrapper):
    def __init__(self, game: Game, args: RunConfig):
        super().__init__(game, args)
        self.args = args
        self.config = args.net_config
        self.nnet = nnet(game, args.net_config)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        if args.net_config.cuda:
            self.nnet.cuda()

    def train(self, examples, generation: int):
        """
        Input:
            examples: List of examples, each example is of form (board, pi, v)
                      board is the canonical board, pi is the MCTS informed policy vector, v is the value from MCTS.
            generation: Current iteration number in the self-play, train and compare network stage.
        """
        optimizer = optim.Adam(self.nnet.parameters())

        # Cannot immediately have this as list of loggable type because we cannot log torch datatypes with pandas
        # Bringing them off the gpu takes a while so we do it after training
        log_data: list = []

        for epoch in range(self.config.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.config.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for batch_number, _ in enumerate(t):
                sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.config.cuda:
                    boards, target_pis, target_vs = (boards.contiguous().cuda(), target_pis.contiguous().cuda(),
                                                     target_vs.contiguous().cuda())

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                log_data.append(
                    [generation, epoch, batch_number, l_pi.detach(), l_v.detach(), total_loss.detach(),
                     deepcopy(pi_losses.avg), deepcopy(v_losses.avg)])

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        self._log_data(generation, log_data)

    def predict(self, board):
        """
        board: canonical form
        """
        # timing
        # start = time.perf_counter()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.config.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # end = time.perf_counter()
        # log.info(fr"PREDICTION TIME TAKEN : {end - start}")
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    @staticmethod
    def loss_pi(targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    @staticmethod
    def loss_v(targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def _log_data(self, generation: int, log_data: list[list]):
        # Convert to loggable after loop because moving from cpu to gpu is slow
        # NB: Pickling is faster than writing this as a df
        start = time.perf_counter()

        cpu_log_data = [TrainingDataLoggable(
            generation=i[0],
            epoch=i[1],
            batch_number=i[2],
            pi_loss=i[3].numpy(force=True),
            v_loss=i[4].numpy(force=True),
            total_loss=i[5].numpy(force=True),
            average_pi_loss=i[6],
            average_v_loss=i[7],
        ) for i in log_data]

        file_name = self.args.training_data_directory / f"train_{generation}.data"

        if not self.args.training_data_directory.exists():
            self.args.training_data_directory.mkdir(parents=True, exist_ok=True)

        with open(file_name, "wb+") as f:
            Pickler(f).dump(cpu_log_data)

        end = time.perf_counter()
        logging.info(f"Took {end - start} seconds to bring data back from GPU and write for generation # {generation}!")

    def collect_training_data(self):
        logging.info(f"Collecting pickled training data into one dataframe for whole run ...")
        start = time.perf_counter()

        def loader(filepath: Path):
            with open(filepath, "rb") as f:
                data = Unpickler(f).load()
            return pd.DataFrame(data)

        files = [file for file in os.listdir(self.args.training_data_directory) if ".data" in file]

        dataframe = pd.concat(
            [loader(self.args.training_data_directory / f) for f in files], copy=False
        ).assign(
            average_loss=lambda df: df.average_pi_loss + df.average_v_loss
        ).astype(
            {'pi_loss': 'float64', 'v_loss': 'float64', 'total_loss': 'float64'}
        )
        end = time.perf_counter()

        dataframe.to_parquet(f"{self.args.training_data_directory / 'data.parquet'}")
        logging.info(f"Took {end - start} seconds to collect, convert and write data!")

    def save_checkpoint(self, filename: str):
        folder = self.args.net_directory

        filepath = folder / filename

        if not folder.exists():
            log.info(f"Checkpoint Directory does not exist! Making directory {folder}")
            folder.mkdir(exist_ok=True, parents=True)
        else:
            log.info("Checkpoint Directory exists!")

        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, filename: str):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        folder = self.args.net_directory
        filepath = folder / filename

        if not filepath.exists():
            log.error(f"No model in path {filepath}")
            raise FileNotFoundError(f"No model in path {filepath}")

        map_location = None if self.config.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
