from __future__ import annotations

import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from alphablokus.interfaces import IBoard, IGame, INeuralNetWrapper
from alphablokus.storage.sparse_policy import as_dense

# Ceiling on the eval set the per-epoch diagnostics accept. The eval set holds
# DENSE boards and policies — fine at the pinned ~200 positions (~14 MB for
# Blokus's (n, 17837) targets) but it would OOM if someone scaled it toward
# buffer size, so the bound is enforced, not assumed (oom-hardening O9).
MAX_EVAL_SET_POSITIONS = 2_000

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from alphablokus.config import RunConfig
    from alphablokus.selfplay.episode import ProcessedExample
    from alphablokus.storage.metrics import EvalSet, MetricsCollector


def count_parameters(net: nn.Module) -> int:
    """Total trainable parameter count — the net-size number quoted in docs/reports."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def resolve_dataloader_context(context_name: str) -> mp.context.BaseContext:
    """Resolve the multiprocessing context for the training DataLoader's workers.

    The default **fork** start method is unsafe here: self-play (JAX) and
    training (torch) share one process, so JAX's live threads are present when
    the loader forks its workers — forking a multithreaded process is what
    killed the pin-memory thread mid-run (``blokus_cloud_60`` gen 59).
    ``forkserver``/``spawn`` create workers from a clean process instead. An
    unavailable method (e.g. ``forkserver`` on a platform that lacks it) falls
    back to ``spawn``, which is always available.

    Args:
        context_name: Requested start method — "forkserver", "spawn", or "fork".

    Returns:
        The resolved multiprocessing context.
    """
    try:
        return mp.get_context(context_name)
    except ValueError:
        logger.warning(
            "DataLoader multiprocessing context {!r} is unavailable here; falling back to 'spawn'.",
            context_name,
        )
        return mp.get_context("spawn")


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
        return f"{self.avg:.2e}"

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


class _LazyPolicyDataset(Dataset):
    """Training dataset that materialises boards and policies one item at a time.

    Boards are held in their **compact** stored form (``IBoard.to_compact`` —
    e.g. the 196-byte int8 placement board for Blokus) and re-encoded to the
    dense ``(C, N, N)`` network input lazily in ``__getitem__`` via the game's
    ``encode_compact``; policies are densified per item; only the (tiny) values
    are held as one tensor. This matters because holding the dense encoding for
    every position OOM-killed large buffers (run2: ~12 GB at gen 15). Storing
    compact boards and encoding per item means only one ``DataLoader`` batch of
    boards (and one of policies, ~71 KB each) is ever dense at once.

    ``encode_fn`` is passed in (``game.encode_compact``) so this stays
    game-agnostic — it never imports a game-specific symbol.
    """

    def __init__(
        self,
        boards: Sequence[np.ndarray],
        raw_pis: Sequence,
        values: Sequence[float],
        action_size: int,
        encode_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        # Reference the buffer's compact board arrays rather than stacking a copy.
        self._boards = list(boards)
        self._values = torch.from_numpy(np.asarray(values, dtype=np.float32))
        self._raw_pis = list(raw_pis)
        self._action_size = action_size
        self._encode_fn = encode_fn

    def __len__(self) -> int:
        return len(self._raw_pis)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        # Re-encode the compact board to dense planes here — one item at a time,
        # so the dense form only ever exists for the current DataLoader batch.
        # ascontiguousarray is a no-op when encode_fn already returns contiguous
        # float32 (the common case).
        board = torch.from_numpy(
            np.ascontiguousarray(self._encode_fn(self._boards[idx]), dtype=np.float32),
        )
        pi = torch.from_numpy(as_dense(self._raw_pis[idx], self._action_size))
        return board, pi, self._values[idx]


class _LossWindow:
    """On-device loss accumulator for the training hot loop.

    Sums detached per-batch losses on the training device so the loop runs
    free of forced host syncs; a single ``.item()`` trio happens per flush
    (every ``log_every_batches`` batches). Window size 1 reproduces the
    original per-batch behaviour exactly.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self.batches = 0
        self.examples = 0
        self._pi = torch.zeros((), device=device)
        self._v = torch.zeros((), device=device)
        self._total = torch.zeros((), device=device)

    def add(self, l_pi: Tensor, l_v: Tensor, total: Tensor, num_examples: int) -> None:
        """Accumulate one batch's losses (detached — no graph retention)."""
        self._pi += l_pi.detach().float()
        self._v += l_v.detach().float()
        self._total += total.detach().float()
        self.batches += 1
        self.examples += num_examples

    def drain(self) -> tuple[float, float, float, int] | None:
        """Sync + reset: ``(mean_pi, mean_v, mean_total, examples)``, or None if empty."""
        if self.batches == 0:
            return None
        result = (
            self._pi.item() / self.batches,
            self._v.item() / self.batches,
            self._total.item() / self.batches,
            self.examples,
        )
        self.batches = 0
        self.examples = 0
        self._pi = torch.zeros((), device=self._device)
        self._v = torch.zeros((), device=self._device)
        self._total = torch.zeros((), device=self._device)
        return result


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
        # torch stubs type attribute access on Module as Tensor | Module;
        # these are plain ints on our net classes.
        self.board_rows: int = cast("int", self.nnet.board_rows)
        self.board_cols: int = cast("int", self.nnet.board_cols)

        self._device = self._resolve_device()
        self.nnet.to(self._device)
        self._apply_cuda_perf_settings()
        # Forward passes (train + inference) go through this alias; it is the
        # torch.compile-wrapped view of ``self.nnet`` when ``perf.compile`` is
        # set, else ``self.nnet`` itself. ``self.nnet`` stays the module of
        # record — state_dicts keep their normal (un-prefixed) keys, so
        # checkpoints are interchangeable between compiled and eager runs.
        self._forward_net: nn.Module = self._maybe_compile()

        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.net_config.learning_rate)
        self.scheduler: LRScheduler | None = self._create_scheduler()

    def _maybe_compile(self) -> nn.Module:
        """torch.compile the net when ``perf.compile`` is set, falling back to eager.

        Guarded twice: a failure in ``torch.compile`` itself logs and returns the
        eager module, and ``dynamo.suppress_errors`` turns any *runtime*
        compilation failure (which only surfaces at the first forward) into a
        per-graph eager fallback instead of a crashed (paid) run.
        """
        if not self.net_config.perf.compile:
            return self.nnet
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            compiled = torch.compile(self.nnet)
        except Exception as err:
            logger.warning("torch.compile failed ({}); continuing with the eager net.", err)
            return self.nnet
        logger.info("torch.compile enabled for the net's forward pass")
        return cast("nn.Module", compiled)

    def _apply_cuda_perf_settings(self) -> None:
        """One-time opt-in CUDA perf toggles (``net_config.perf``); inert on CPU/MPS.

        Guarded on the resolved device so a config with these knobs set is still
        safe to load on the Mac — the defaults-off contract only has to hold in
        one direction (defaults never change behaviour; explicit knobs never
        break CPU runs).
        """
        perf = self.net_config.perf
        if self._device.type != "cuda":
            return
        if perf.tf32:
            # TF32 matmul/conv: ~2x tensor-core throughput on Ampere+ at
            # negligible precision cost for a policy/value ResNet.
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True
        if perf.cudnn_benchmark:
            # Safe here: conv shapes are fixed (C×14×14 boards, fixed batch).
            torch.backends.cudnn.benchmark = True
        if perf.channels_last:
            # torch's Module.to stubs miss the memory_format-only overload.
            self.nnet.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

    def _resolve_device(self) -> torch.device:
        """Pick the compute device.

        ``net_config.cuda`` selects CUDA (the box). Otherwise CPU — *unless*
        ``ALPHABLOKUS_MPS=1`` is set and Apple's MPS (Metal) backend is available,
        in which case use MPS (batched inference is far faster than CPU on Apple
        Silicon). MPS is opt-in and scoped to eval: training/tests stay on CPU on
        the Mac (the MPS training path is unvalidated), so default behaviour is
        unchanged. ``PYTORCH_ENABLE_MPS_FALLBACK=1`` lets any unimplemented op fall
        back to CPU rather than erroring.
        """
        if self.net_config.cuda and torch.cuda.is_available():
            return torch.device("cuda")
        if os.environ.get("ALPHABLOKUS_MPS") == "1" and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return torch.device("mps")
        return torch.device("cpu")

    def _create_scheduler(self) -> LRScheduler | None:
        """Create LR scheduler based on config. Returns None if no schedule configured."""
        match self.net_config.lr_scheduler:
            case "cosine":
                total_epochs = self.config.num_generations * self.net_config.epochs
                # ``eta_min`` floors the anneal. Default 0.0 reproduces the
                # original behaviour; a non-zero floor stops the LR reaching ~0
                # and killing late-run training (blokus_cloud_60 §3).
                if self.net_config.lr_eta_min == 0.0 and self.config.num_generations > 1:
                    logger.warning(
                        "Cosine LR schedule has lr_eta_min=0.0 over {} generations: the learning "
                        "rate anneals to ~0 by the run's end, which strangled the last quarter of "
                        "blokus_cloud_60. Set net_config.lr_eta_min (e.g. 1e-4) to floor it.",
                        self.config.num_generations,
                    )
                return CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=self.net_config.lr_eta_min)
            case None:
                return None
            case unknown:
                raise ValueError(f"Unknown lr_scheduler: {unknown!r}")

    @abstractmethod
    def _create_network(self) -> nn.Module:
        """Create and return the game-specific neural network."""
        ...

    @staticmethod
    def _flush_loss_window(
        window: _LossWindow,
        pi_losses: AverageMeter,
        v_losses: AverageMeter,
        progress: tqdm,
        metrics: MetricsCollector | None,
        generation: int,
        epoch: int,
        batch_number: int,
    ) -> None:
        """Drain the on-device loss window into the meters/metrics/progress bar.

        With the default window of 1 this runs every batch and reproduces the
        original per-batch ``.item()`` + ``log_training`` behaviour exactly;
        larger windows log the window mean under the window's last batch number
        (same metrics schema, fewer rows and fewer forced GPU syncs).
        """
        drained = window.drain()
        if drained is None:
            return
        mean_pi, mean_v, mean_total, num_examples = drained
        pi_losses.update(mean_pi, num_examples)
        v_losses.update(mean_v, num_examples)
        progress.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
        if metrics:
            metrics.log_training(
                generation=generation,
                epoch=epoch,
                batch_number=batch_number,
                pi_loss=mean_pi,
                v_loss=mean_v,
                total_loss=mean_total,
            )

    def train(
        self,
        examples: list[ProcessedExample],
        generation: int,
        metrics: MetricsCollector | None = None,
        eval_set: EvalSet | None = None,
    ) -> None:
        """Train with ``epochs`` full, shuffled passes over the whole buffer.

        Every position in ``examples`` (the entire rolling replay buffer) is
        trained on exactly ``net_config.epochs`` times per generation — we use
        all the data we have rather than sampling a subset (the project is
        game-limited). Reuse over a position's life is the emergent quantity
        ``epochs × (replay_buffer_games / num_eps)``, logged by the Coach.

        Args:
            examples: The whole replay buffer flattened to (board, policy, value)
                tuples.
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

        boards_np, raw_pis, vs_np = zip(*examples, strict=True)
        action_size = self.game.get_action_size()

        # Validate training data at the interface boundary. Boards are stored
        # **compact** (``to_compact()``) and re-encoded lazily, so check the
        # shape of the *encoded* sample (what the network actually sees), not the
        # compact array. Policies are stored sparse (indices, values) to keep
        # replay-buffer RAM small; densify just this one sample for the check —
        # the rest are densified per mini-batch inside the dataset below, never
        # the whole buffer at once (that dense spike OOM'd large buffers).
        encode_fn = self.game.encode_compact
        sample_board = encode_fn(boards_np[0])
        sample_pi = as_dense(raw_pis[0], action_size)
        sample_v = vs_np[0]
        expected_board_shape = (sample_board.shape[0], self.board_rows, self.board_cols)
        assert sample_board.shape == expected_board_shape, (
            f"Encoded board shape {sample_board.shape} != expected {expected_board_shape}"
        )
        assert abs(sample_pi.sum() - 1.0) < 0.01, f"Policy vector sums to {sample_pi.sum()}, expected ~1.0"
        assert -1.0 <= sample_v <= 1.0, f"Value {sample_v} outside [-1, 1]"
        dataset = _LazyPolicyDataset(boards_np, raw_pis, vs_np, action_size, encode_fn)

        # Opt-in training-perf knobs (net_config.perf). Everything defaults to
        # off = the original fp32, in-process, per-batch-sync loop; CUDA-only
        # knobs are inert on CPU so perf-enabled configs still load on the Mac.
        perf = self.net_config.perf
        on_cuda = self._device.type == "cuda"
        use_autocast = on_cuda and perf.autocast_dtype != "off"
        autocast_dtype = torch.bfloat16 if perf.autocast_dtype == "bf16" else torch.float16

        def train_autocast() -> AbstractContextManager:
            """Autocast context for the training forward+loss, or a no-op off CUDA."""
            if use_autocast:
                return torch.autocast(device_type="cuda", dtype=autocast_dtype)
            return nullcontext()

        # Loss scaling is only needed for fp16 (bf16 has fp32-like range).
        scaler = torch.amp.GradScaler("cuda", enabled=use_autocast and perf.autocast_dtype == "fp16")
        pin_memory = perf.pin_memory and on_cuda
        # non_blocking copies only overlap with compute from pinned buffers.
        non_blocking = pin_memory
        worker_kwargs: dict[str, Any] = {}
        if perf.dataloader_workers > 0:
            # Worker processes take over the per-item hot path — densifying the
            # full-action-space policy and encoding the compact board — which
            # otherwise runs single-threaded on the main process and starves a
            # fast GPU.
            worker_kwargs = {
                "num_workers": perf.dataloader_workers,
                "persistent_workers": perf.persistent_workers,
                "prefetch_factor": perf.prefetch_factor,
            }
            if perf.dataloader_context != "fork":
                # Never fork workers from this (JAX-loaded, multithreaded)
                # process — that deadlocked/killed the pin-memory thread at
                # gen 59 of blokus_cloud_60. forkserver/spawn start workers from
                # a clean process. See docs/plans/archive/harden-long-runs.md H1.
                worker_kwargs["multiprocessing_context"] = resolve_dataloader_context(perf.dataloader_context)
        # Built once so persistent workers survive across epochs; iterating a
        # shuffle=True loader reshuffles each epoch exactly as the old
        # loader-per-epoch construction did.
        loader = DataLoader(
            dataset,
            batch_size=self.net_config.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            **worker_kwargs,
        )
        # Metric-sync cadence: 1 = the original per-batch .item() behaviour;
        # N > 1 accumulates losses on-device and forces a sync only every N
        # batches, logging the window mean (same metrics schema, fewer rows).
        log_window = max(perf.log_every_batches, 1)

        # Full-pass training: every position in the buffer is trained on exactly
        # ``epochs`` times this generation (use all the data). Reuse over a
        # position's life is the emergent ``epochs × (B / num_eps)``, logged by
        # the Coach — not a sampler knob.
        for epoch in range(self.net_config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.net_config.epochs}")
            epoch_start = time.perf_counter()
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            window = _LossWindow(self._device)
            t = tqdm(loader, desc="Training Net")
            for batch_number, (boards, target_pis, target_vs) in enumerate(t):
                boards = boards.to(self._device, non_blocking=non_blocking)
                target_pis = target_pis.to(self._device, non_blocking=non_blocking)
                target_vs = target_vs.to(self._device, non_blocking=non_blocking)
                if on_cuda and perf.channels_last:
                    boards = boards.to(memory_format=torch.channels_last)

                with train_autocast():
                    out_pi, out_v = self._forward_net(boards)
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v

                window.add(l_pi, l_v, total_loss, boards.size(0))
                if window.batches >= log_window:
                    self._flush_loss_window(window, pi_losses, v_losses, t, metrics, generation, epoch, batch_number)

                self.optimizer.zero_grad()
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    self.optimizer.step()

            # Partial window at epoch end (only reachable when log_window > 1).
            self._flush_loss_window(window, pi_losses, v_losses, t, metrics, generation, epoch, len(loader) - 1)

            # Record the LR this epoch actually trained at — read *before* the
            # scheduler step, which is what makes the schedule reviewable (L2).
            if metrics is not None:
                metrics.log_learning_rate(
                    generation=generation,
                    epoch=epoch,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                )

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
                mcts_agreement = self._compute_mcts_agreement(eval_set)
                metrics.log_policy_accuracy(
                    generation=generation,
                    epoch=epoch,
                    top1_accuracy=diagnostics["top1"],
                    top5_accuracy=diagnostics["top5"],
                    eval_set_size=len(eval_set),
                    mcts_top1_accuracy=mcts_agreement[0] if mcts_agreement is not None else None,
                    mcts_top5_accuracy=mcts_agreement[1] if mcts_agreement is not None else None,
                )
                metrics.log_value_calibration(
                    generation=generation,
                    epoch=epoch,
                    bucket_centers=diagnostics["calib_centers"],
                    bucket_means=diagnostics["calib_means"],
                    bucket_counts=diagnostics["calib_counts"],
                )

    def _compute_eval_set_diagnostics(self, eval_set: EvalSet) -> dict[str, Any]:
        """Forward-pass the network over the eval set and compute three
        AlphaZero-style diagnostics in one shot:

        - ``entropy_mean`` / ``entropy_std``: per-position policy entropy.
        - ``top1`` / ``top5``: fraction of positions where the network's
          argmax / top-5 actions include the MCTS target's argmax.
        - Value calibration: 10 reliability buckets over predicted v ∈ [-1, 1]
          mapping to mean(actual outcome) per bucket. Returned as three
          aligned arrays (centers, means, counts).
        """
        assert len(eval_set) <= MAX_EVAL_SET_POSITIONS, (
            f"eval set of {len(eval_set)} positions exceeds MAX_EVAL_SET_POSITIONS="
            f"{MAX_EVAL_SET_POSITIONS} — it is held dense (boards + full-action-space "
            "policies), so keep it a small pinned sample, never buffer-scale."
        )
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
                tensor = tensor.to(self._device)
                log_pi, v = self._forward_net(tensor)
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
            np.digitize(pred_v, bucket_edges) - 1,
            0,
            len(bucket_edges) - 2,
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

    def _compute_mcts_agreement(self, eval_set: EvalSet) -> tuple[float, float] | None:
        """Agreement of the raw net policy with the net's *own* MCTS on the eval set.

        For each frozen eval position, rebuild a playable board
        (:meth:`IGame.board_from_compact`), run the current net's MCTS from it,
        and compare the search's chosen action (argmax visit count) against the
        raw network policy's top action(s), restricted to legal moves:

        - ``top1``: fraction of positions where the raw policy's best legal move
          is the move search settled on.
        - ``top5``: fraction where search's move is in the raw policy's top-5
          legal moves.

        This is the *net-vs-its-own-search* gap — "is the raw policy keeping up
        with search?" — which should hold or rise as training works, unlike the
        frozen-gen-1 agreement that decays once the net surpasses gen-1's search.
        Returns ``None`` when the eval set predates compact-board persistence
        (nothing to rebuild from). Uses the python PUCT search regardless of the
        configured ``search_policy`` (that setting is jax-backend only).
        """
        if eval_set.compact_boards is None or len(eval_set) == 0:
            return None

        from alphablokus.search.mcts import MCTS

        self.nnet.eval()
        top1_hits = 0
        top5_hits = 0
        n = len(eval_set)
        for compact in eval_set.compact_boards:
            board = self.game.board_from_compact(compact)
            # A fresh tree per position so searches never share state; ``self`` is
            # the current-net predictor MCTS evaluates leaves with.
            mcts = MCTS(self.game, self, self.config.mcts_config)
            probs = mcts.get_action_prob(board, temp=0.0, add_root_noise=False)
            mcts_action = int(np.argmax(probs))

            policy, _ = self.predict(board)
            valids = self.game.valid_move_masking(board, 1)
            # Restrict the raw-policy ranking to legal moves so an illegal high-
            # prior action can't crowd the top-k the search action is compared to.
            masked = np.where(valids > 0, policy, -np.inf)
            k = min(5, int(np.count_nonzero(valids)))
            net_top = np.argsort(masked)[::-1][:k]
            if len(net_top) > 0 and int(net_top[0]) == mcts_action:
                top1_hits += 1
            if mcts_action in net_top:
                top5_hits += 1

        return top1_hits / n, top5_hits / n

    def _inference_autocast(self) -> AbstractContextManager:
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
        tensor = tensor.to(self._device)
        self.nnet.eval()
        with torch.no_grad(), self._inference_autocast():
            log_pi, v = self._forward_net(tensor)

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
        return F.kl_div(outputs, targets, reduction="batchmean")

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
            "state_dict": self.nnet.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str, *, restore_lr_schedule: bool = True) -> None:
        """Load a neural network state from a checkpoint file.

        Args:
            filename: Checkpoint file under ``config.net_directory``.
            restore_lr_schedule: When True (the ``--resume`` case), restore the
                saved scheduler position — a resumed run must continue the exact
                schedule it was on. When False (the arena reject-reload case),
                the schedule clock must *not* rewind: the LR advances once per
                generation regardless of accept/reject, so the pre-training
                weights and Adam moments are reverted (the gate's job) but the
                scheduler keeps its current position and the just-restored
                optimizer LR is re-synced to it. No-op for a scheduler-less run
                (constant LR), which then reverts fully — bit-for-bit as before.
        """
        folder = self.config.net_directory
        filepath = folder / filename

        if not filepath.exists():
            logger.error(f"No model in path {filepath}")
            raise FileNotFoundError(f"No model in path {filepath}")

        map_location = None if self.net_config.cuda else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is None:
            return
        if restore_lr_schedule:
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            return
        # Reject-reload: keep the scheduler's clock. The optimizer restore above
        # brought back the pre-step LR, so re-sync each param group to the
        # scheduler's current LR (the value the next generation should train at).
        for group, lr in zip(self.optimizer.param_groups, self.scheduler.get_last_lr(), strict=True):
            group["lr"] = lr
