from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import optim, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from core.config import RunConfig
from core.interfaces import IBoard, IGame, INeuralNetWrapper
from core.storage import MetricsCollector


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


class BaseNNetWrapper(INeuralNetWrapper, ABC):
    """
    Base neural network wrapper implementing all shared training, prediction,
    and persistence logic. Game-specific wrappers only need to implement
    _create_network() to return their specific nn.Module.
    """

    def __init__(self, game: IGame, config: RunConfig) -> None:
        self.game = game
        self.config = config
        self.net_config = config.net_config
        self.nnet = self._create_network()
        self.board_rows: int = self.nnet.board_rows
        self.board_cols: int = self.nnet.board_cols

        if self.net_config.cuda:
            self.nnet.cuda()

        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.net_config.learning_rate)
        self.scheduler: LRScheduler | None = self._create_scheduler()

    def _create_scheduler(self) -> LRScheduler | None:
        """Create LR scheduler based on config. Returns None if no schedule configured."""
        match self.net_config.lr_scheduler:
            case "cosine":
                total_epochs = self.config.num_generations * self.net_config.epochs
                return CosineAnnealingLR(self.optimizer, T_max=total_epochs)
            case None:
                return None
            case unknown:
                raise ValueError(f"Unknown lr_scheduler: {unknown!r}")

    @abstractmethod
    def _create_network(self) -> nn.Module:
        """Create and return the game-specific neural network."""
        ...

    def train(
        self,
        examples: list[tuple[np.ndarray, np.ndarray, float]],
        generation: int,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Train the neural network using provided examples."""
        if not examples:
            logger.warning("No training examples provided, skipping training.")
            return

        boards_np, pis_np, vs_np = zip(*examples)

        # Validate training data at the interface boundary
        sample_board = boards_np[0]
        sample_pi = pis_np[0]
        sample_v = vs_np[0]
        expected_board_shape = (sample_board.shape[0], self.board_rows, self.board_cols)
        assert sample_board.shape == expected_board_shape, (
            f"Board shape {sample_board.shape} != expected {expected_board_shape}")
        assert abs(sample_pi.sum() - 1.0) < 0.01, (
            f"Policy vector sums to {sample_pi.sum()}, expected ~1.0")
        assert -1.0 <= sample_v <= 1.0, (
            f"Value {sample_v} outside [-1, 1]")
        dataset = TensorDataset(
            torch.tensor(np.array(boards_np), dtype=torch.float32),
            torch.tensor(np.array(pis_np), dtype=torch.float32),
            torch.tensor(np.array(vs_np), dtype=torch.float32),
        )

        for epoch in range(self.net_config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.net_config.epochs}")
            epoch_start = time.perf_counter()
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            loader = DataLoader(dataset, batch_size=self.net_config.batch_size, shuffle=True)
            t = tqdm(loader, desc='Training Net')
            for batch_number, (boards, target_pis, target_vs) in enumerate(t):
                if self.net_config.cuda:
                    boards, target_pis, target_vs = (boards.cuda(),
                                                   target_pis.cuda(),
                                                   target_vs.cuda())

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                if metrics:
                    metrics.log_training(
                        generation=generation,
                        epoch=epoch,
                        batch_number=batch_number,
                        pi_loss=l_pi.item(),
                        v_loss=l_v.item(),
                        total_loss=total_loss.item(),
                        avg_pi_loss=pi_losses.avg,
                        avg_v_loss=v_losses.avg,
                    )

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.perf_counter() - epoch_start
            if metrics:
                metrics.log_training_throughput(
                    generation=generation,
                    epoch=epoch,
                    num_examples=len(examples),
                    epoch_time_s=epoch_time,
                )

    def predict(self, board: IBoard) -> tuple[np.ndarray, float]:
        """Make a prediction for a given board state.

        Args:
            board: Board object (canonical, i.e. player 1 perspective).
        """
        tensor = torch.tensor(board.as_multi_channel(1), dtype=torch.float32)
        if self.net_config.cuda:
            tensor = tensor.contiguous().cuda()
        tensor = tensor.unsqueeze(0)  # (C, H, W) → (1, C, H, W)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(tensor)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    @staticmethod
    def loss_pi(targets: Tensor, outputs: Tensor) -> Tensor:
        """Calculate the policy loss (KL divergence)."""
        return F.kl_div(outputs, targets, reduction='batchmean')

    @staticmethod
    def loss_v(targets: Tensor, outputs: Tensor) -> Tensor:
        """Calculate the value loss."""
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, filename: str) -> None:
        """Save the neural network state to a checkpoint file."""
        folder = self.config.net_directory
        filepath = folder / filename

        if not folder.exists():
            logger.info(f"Checkpoint Directory does not exist! Making directory {folder}")
            folder.mkdir(exist_ok=True, parents=True)
        else:
            logger.info("Checkpoint Directory exists!")

        checkpoint = {
            'state_dict': self.nnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str) -> None:
        """Load a neural network state from a checkpoint file."""
        folder = self.config.net_directory
        filepath = folder / filename

        if not filepath.exists():
            logger.error(f"No model in path {filepath}")
            raise FileNotFoundError(f"No model in path {filepath}")

        map_location = None if self.net_config.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
