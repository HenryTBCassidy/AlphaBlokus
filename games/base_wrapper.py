from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from core.config import RunConfig
from core.interfaces import IBoard, IGame, INeuralNetWrapper
from core.sparse_policy import as_dense
from core.storage import EvalSet, MetricsCollector


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
        eval_set: EvalSet | None = None,
    ) -> None:
        """Train the neural network using provided examples.

        Args:
            examples: Training examples produced from self-play.
            generation: Current training generation (for logging).
            metrics: Optional metrics collector for parquet/W&B logging.
            eval_set: Optional frozen held-out positions. When provided, three
                AlphaZero-style diagnostics are computed on the network alone
                (no MCTS) at the end of every training epoch and logged via
                ``metrics.log_training_entropy``, ``log_policy_accuracy``, and
                ``log_value_calibration``: policy entropy (confidence),
                top-1 / top-5 accuracy against MCTS targets, and a value-head
                reliability diagram.
        """
        if not examples:
            logger.warning("No training examples provided, skipping training.")
            return

        boards_np, raw_pis, vs_np = zip(*examples)
        # Policies are stored sparse (indices, values) to keep replay-buffer RAM
        # small; normalise to the dense vector the loss expects (accepts already-
        # dense too — e.g. resume-loaded or hand-built test examples).
        action_size = self.game.get_action_size()
        pis_np = [as_dense(p, action_size) for p in raw_pis]

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

            # Per-epoch held-out diagnostics on the frozen eval set.
            if eval_set is not None and len(eval_set) > 0 and metrics is not None:
                diagnostics = self._compute_eval_set_diagnostics(eval_set)
                metrics.log_training_entropy(
                    generation=generation,
                    epoch=epoch,
                    mean_entropy=diagnostics["entropy_mean"],
                    std_entropy=diagnostics["entropy_std"],
                    eval_set_size=len(eval_set),
                )
                metrics.log_policy_accuracy(
                    generation=generation,
                    epoch=epoch,
                    top1_accuracy=diagnostics["top1"],
                    top5_accuracy=diagnostics["top5"],
                    eval_set_size=len(eval_set),
                )
                metrics.log_value_calibration(
                    generation=generation,
                    epoch=epoch,
                    bucket_centers=diagnostics["calib_centers"],
                    bucket_means=diagnostics["calib_means"],
                    bucket_counts=diagnostics["calib_counts"],
                )

    def _compute_eval_set_diagnostics(self, eval_set: EvalSet) -> dict:
        """Forward-pass the network over the eval set and compute three
        AlphaZero-style diagnostics in one shot:

        - ``entropy_mean`` / ``entropy_std``: per-position policy entropy.
        - ``top1`` / ``top5``: fraction of positions where the network's
          argmax / top-5 actions include the MCTS target's argmax.
        - Value calibration: 10 reliability buckets over predicted v ∈ [-1, 1]
          mapping to mean(actual outcome) per bucket. Returned as three
          aligned arrays (centers, means, counts).
        """
        self.nnet.eval()
        per_position_entropies: list[float] = []
        top1_hits = 0
        top5_hits = 0
        predicted_values: list[float] = []

        # Treat any action with non-zero target probability as "credit-worthy".
        # For MCTS-target eval sets this collapses to the single argmax (since
        # the MCTS visit distribution is rarely exactly uniform). For minimax
        # eval sets the target is uniform over all optimal actions, so the
        # net is credited if it picks any of them — the right behaviour when
        # several moves are equally optimal.
        target_supports = eval_set.target_policies > 0
        target_values = eval_set.target_values

        with torch.no_grad():
            for chunk_start in range(0, len(eval_set), self.net_config.batch_size):
                end = chunk_start + self.net_config.batch_size
                boards_chunk = eval_set.boards[chunk_start:end]
                tensor = torch.tensor(boards_chunk, dtype=torch.float32)
                if self.net_config.cuda:
                    tensor = tensor.cuda()
                log_pi, v = self.nnet(tensor)
                pi = torch.exp(log_pi)

                # Entropy per row.
                entropies = -(pi * log_pi).sum(dim=1).cpu().numpy()
                per_position_entropies.extend(entropies.tolist())

                # Top-1 / Top-5 agreement vs the target's *support set* (set of
                # actions with non-zero target probability — the argmax for an
                # MCTS target, or all optimal actions for a minimax target).
                k = min(5, log_pi.shape[1])
                topk = log_pi.topk(k, dim=1).indices.cpu().numpy()
                chunk_supports = target_supports[chunk_start:end]
                # top1: did the net's top action hit any optimal target?
                top1_hits += int(chunk_supports[np.arange(len(topk)), topk[:, 0]].sum())
                # top5: did any of the net's top-5 actions hit an optimal target?
                row_idx = np.arange(len(topk))[:, None]
                top5_hits += int(chunk_supports[row_idx, topk].any(axis=1).sum())

                # Value predictions, flattened to 1-D for calibration.
                predicted_values.extend(v.view(-1).cpu().numpy().tolist())

        ent_arr = np.asarray(per_position_entropies, dtype=float)
        n = len(eval_set)

        # Reliability binning of predicted v ∈ [-1, 1] into 10 buckets.
        pred_v = np.asarray(predicted_values, dtype=float)
        bucket_edges = np.linspace(-1.0, 1.0, 11)
        bucket_idx = np.clip(
            np.digitize(pred_v, bucket_edges) - 1, 0, len(bucket_edges) - 2,
        )
        bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2.0
        bucket_means = np.full(10, np.nan, dtype=float)
        bucket_counts = np.zeros(10, dtype=int)
        for b in range(10):
            mask = bucket_idx == b
            bucket_counts[b] = int(mask.sum())
            if bucket_counts[b] > 0:
                bucket_means[b] = float(target_values[mask].mean())

        return {
            "entropy_mean": float(ent_arr.mean()),
            "entropy_std": float(ent_arr.std()),
            "top1": top1_hits / n,
            "top5": top5_hits / n,
            "calib_centers": bucket_centers,
            "calib_means": bucket_means,
            "calib_counts": bucket_counts,
        }

    def _inference_autocast(self):
        """fp16 autocast context for the forward pass, or a no-op.

        Active only when ``fp16_inference`` is set *and* we're on CUDA — autocast
        with ``device_type="cuda"`` requires a GPU. On CPU (or when disabled) this
        is ``nullcontext``, so the forward runs exactly as before.
        """
        if self.net_config.fp16_inference and self.net_config.cuda:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def predict_encoded(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the network on a pre-encoded batch of board planes.

        The single source of truth for the inference forward pass: both
        :meth:`predict` / :meth:`predict_batch` (which encode boards first) and
        the cross-worker inference server (which receives already-encoded
        planes over shared memory) route through here, so they are guaranteed
        bit-identical.

        Args:
            planes: ``(N, C, H, W)`` float32 — ``N`` boards already encoded via
                ``board.as_multi_channel(1)`` and stacked.

        Returns:
            ``(policies, values)`` — ``(N, A)`` softmaxed policy array and
            ``(N,)`` value array, both float32 on the CPU.
        """
        tensor = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32))
        if self.net_config.cuda:
            tensor = tensor.cuda()
        self.nnet.eval()
        with torch.no_grad(), self._inference_autocast():
            log_pi, v = self.nnet(tensor)

        # .float() casts back from fp16 (if autocast was active) so downstream
        # code always sees float32; a no-op when inference ran in float32.
        policies = torch.exp(log_pi).float().data.cpu().numpy()
        values = v.view(-1).float().data.cpu().numpy()
        return policies, values

    def predict(self, board: IBoard) -> tuple[np.ndarray, float]:
        """Make a prediction for a given board state.

        Args:
            board: Board object (canonical, i.e. player 1 perspective).
        """
        policies, values = self.predict_encoded(board.as_multi_channel(1)[np.newaxis, ...])
        return policies[0], values[0]

    def predict_batch(self, boards: Sequence[IBoard]) -> tuple[list[np.ndarray], list[float]]:
        """Run the network on N boards in a single forward pass.

        Equivalent to ``[self.predict(b) for b in boards]`` but executes the
        forward pass once with batch dimension ``len(boards)``. Used by
        batched MCTS inference. See :meth:`INeuralNetWrapper.predict_batch`.

        Args:
            boards: Board objects in canonical form (player 1 perspective).
        """
        arrs = np.stack([board.as_multi_channel(1) for board in boards])
        policies, values = self.predict_encoded(arrs)
        return [policies[i] for i in range(len(arrs))], [float(x) for x in values]

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
