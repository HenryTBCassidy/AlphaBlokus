import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from pickle import Pickler, Unpickler
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch import optim, Tensor
from tqdm import tqdm

from core.config import RunConfig
from core.interfaces import IGame


class AverageMeter:
    """
    Computes and stores the average and current value.
    Originally from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self) -> None:
        """Initialise the meter with zero values."""
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def __repr__(self) -> str:
        """Return string representation of the average value."""
        return f'{self.avg:.2e}'

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.

        Args:
            val: The new value to include in the average
            n: The weight of the new value (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass(frozen=True)
class TrainingDataLoggable:
    """Data class for logging training metrics during model training."""
    generation: int
    epoch: int
    batch_number: int
    pi_loss: float
    v_loss: float
    total_loss: float
    average_pi_loss: float
    average_v_loss: float


class BaseNNetWrapper(ABC):
    """
    Base neural network wrapper implementing all shared training, prediction,
    and persistence logic. Game-specific wrappers only need to implement
    _create_network() to return their specific nn.Module.
    """

    def __init__(self, game: IGame, args: RunConfig) -> None:
        self.game = game
        self.args = args
        self.config = args.net_config
        self.nnet = self._create_network()
        self.board_x: int = self.nnet.board_x
        self.board_y: int = self.nnet.board_y

        if self.config.cuda:
            self.nnet.cuda()

    @abstractmethod
    def _create_network(self) -> nn.Module:
        """Create and return the game-specific neural network."""
        ...

    def train(self, examples: List[Tuple[np.ndarray, np.ndarray, float]], generation: int) -> None:
        """Train the neural network using provided examples."""
        optimizer = optim.Adam(self.nnet.parameters())
        log_data: List[List[Any]] = []

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

                if self.config.cuda:
                    boards, target_pis, target_vs = (boards.contiguous().cuda(),
                                                   target_pis.contiguous().cuda(),
                                                   target_vs.contiguous().cuda())

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                log_data.append(
                    [generation, epoch, batch_number, l_pi.detach(), l_v.detach(), total_loss.detach(),
                     deepcopy(pi_losses.avg), deepcopy(v_losses.avg)])

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        self._log_data(generation, log_data)

    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make a prediction for a given board state."""
        board = torch.FloatTensor(board.astype(np.float64))
        if self.config.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    @staticmethod
    def loss_pi(targets: Tensor, outputs: Tensor) -> Tensor:
        """Calculate the policy loss."""
        return -torch.sum(targets * outputs) / targets.size()[0]

    @staticmethod
    def loss_v(targets: Tensor, outputs: Tensor) -> Tensor:
        """Calculate the value loss."""
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def _log_data(self, generation: int, log_data: List[List[Any]]) -> None:
        """Log training data for the current generation."""
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
        logger.info(f"Took {end - start} seconds to bring data back from GPU and write for generation # {generation}!")

    def collect_training_data(self) -> None:
        """Collect and consolidate all training data into a single DataFrame."""
        logger.info("Collecting pickled training data into one dataframe for whole run ...")
        start = time.perf_counter()

        def loader(filepath: Path) -> pd.DataFrame:
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
        logger.info(f"Took {end - start} seconds to collect, convert and write data!")

    def save_checkpoint(self, filename: str) -> None:
        """Save the neural network state to a checkpoint file."""
        folder = self.args.net_directory
        filepath = folder / filename

        if not folder.exists():
            logger.info(f"Checkpoint Directory does not exist! Making directory {folder}")
            folder.mkdir(exist_ok=True, parents=True)
        else:
            logger.info("Checkpoint Directory exists!")

        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, filename: str) -> None:
        """Load a neural network state from a checkpoint file."""
        folder = self.args.net_directory
        filepath = folder / filename

        if not filepath.exists():
            logger.error(f"No model in path {filepath}")
            raise FileNotFoundError(f"No model in path {filepath}")

        map_location = None if self.config.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
