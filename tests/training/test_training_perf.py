"""Opt-in training-perf knobs (``NetConfig.perf``) preserve training results.

The contract (docs/plans/cloud-scale-training.md C2/C3): defaults are today's
behaviour, CUDA-only knobs are inert on CPU, and the knobs that *are* active on
CPU (DataLoader workers, windowed metric logging) change how the loop is fed
and logged — never the training math. All tests assert bit-identical final
weights against the defaults path at the same seed.
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from alphablokus.config import TrainingPerfConfig
from alphablokus.games.base_wrapper import _LazyPolicyDataset, resolve_dataloader_context
from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
from tests.conftest import RecordingMetrics

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.tictactoe.game import TicTacToeGame


def _buffer(action_size: int, n: int) -> list:
    examples = []
    for i in range(n):
        board = np.zeros((3, 3), dtype=np.int8)
        board.flat[i % 9] = 1
        policy = np.full(action_size, 1.0 / action_size, dtype=np.float64)
        examples.append((board, policy, float((-1) ** i)))
    return examples


def _train_with_perf(
    ttt_game: TicTacToeGame,
    config: RunConfig,
    perf: TrainingPerfConfig,
    seed: int = 123,
) -> tuple[dict[str, torch.Tensor], RecordingMetrics]:
    """Seeded wrapper init + train with the given perf knobs → (weights, metrics)."""
    run_config = replace(config, net_config=replace(config.net_config, epochs=2, perf=perf))
    torch.manual_seed(seed)
    nnet = NNetWrapper(ttt_game, run_config)
    metrics = RecordingMetrics()
    torch.manual_seed(seed + 1)  # shuffle RNG
    nnet.train(_buffer(ttt_game.get_action_size(), 40), generation=0, metrics=metrics)
    state = {k: v.detach().clone() for k, v in nnet.nnet.state_dict().items()}
    return state, metrics


def _assert_identical_weights(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"weight {key} diverged from the defaults path"


def test_windowed_logging_trains_identically_with_fewer_rows(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """log_every_batches only changes logging cadence, never the training math."""
    baseline_weights, baseline_metrics = _train_with_perf(ttt_game, test_config, TrainingPerfConfig())
    windowed_weights, windowed_metrics = _train_with_perf(
        ttt_game, test_config, TrainingPerfConfig(log_every_batches=4)
    )

    _assert_identical_weights(baseline_weights, windowed_weights)
    assert len(windowed_metrics.rows) < len(baseline_metrics.rows)
    assert np.isfinite([row["pi_loss"] for row in windowed_metrics.rows]).all()
    assert np.isfinite([row["total_loss"] for row in windowed_metrics.rows]).all()


def test_windowed_logging_flushes_partial_final_window(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """Every batch is represented: window rows cover ceil(batches/window) per epoch."""
    _, metrics = _train_with_perf(ttt_game, test_config, TrainingPerfConfig(log_every_batches=4))
    batches_per_epoch = -(-40 // test_config.net_config.batch_size)
    rows_per_epoch = -(-batches_per_epoch // 4)
    assert len(metrics.rows) == rows_per_epoch * 2  # epochs=2 in _train_with_perf


def test_cuda_only_knobs_are_inert_on_cpu(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """A cloud config's CUDA knobs must not change (or break) a CPU run."""
    baseline_weights, _ = _train_with_perf(ttt_game, test_config, TrainingPerfConfig())
    cuda_knobs = TrainingPerfConfig(
        autocast_dtype="bf16",
        tf32=True,
        cudnn_benchmark=True,
        channels_last=True,
    )
    knobs_weights, _ = _train_with_perf(ttt_game, test_config, cuda_knobs)
    _assert_identical_weights(baseline_weights, knobs_weights)


def test_dataloader_workers_train_identically(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """Worker processes change who does the encoding, not what gets trained.

    The shuffle permutation and the loader's per-epoch base seed are both drawn
    from the main process's torch RNG regardless of worker count, so final
    weights are bit-identical to the in-process path at the same seed.
    (``persistent_workers`` is excluded here: reusing the iterator across epochs
    legitimately skips the per-epoch base-seed draw, shifting the RNG stream —
    an accepted property of that opt-in knob, covered by the test below.)
    """
    baseline_weights, baseline_metrics = _train_with_perf(ttt_game, test_config, TrainingPerfConfig())
    workers_weights, workers_metrics = _train_with_perf(
        ttt_game,
        test_config,
        TrainingPerfConfig(dataloader_workers=2, prefetch_factor=2),
    )
    _assert_identical_weights(baseline_weights, workers_weights)
    assert len(workers_metrics.rows) == len(baseline_metrics.rows)


def test_compile_trains_and_checkpoints_stay_interchangeable(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """perf.compile keeps checkpoints eager-compatible (and falls back safely).

    Checkpoints are saved from the module of record (``self.nnet``), never the
    compiled wrapper, so state-dict keys carry no ``_orig_mod.`` prefix and an
    eager wrapper can load them. On hosts where inductor can't build (e.g. CPU
    without a toolchain) dynamo's suppress_errors degrades to eager — either
    way this must train, predict, and round-trip a checkpoint.
    """
    run_config = replace(
        test_config,
        net_config=replace(test_config.net_config, perf=TrainingPerfConfig(compile=True)),
    )
    torch.manual_seed(0)
    nnet = NNetWrapper(ttt_game, run_config)
    nnet.train(_buffer(ttt_game.get_action_size(), 12), generation=0)
    pi, v = nnet.predict(ttt_game.initialise_board())
    assert pi.shape == (ttt_game.get_action_size(),)
    assert -1.0 <= float(v) <= 1.0

    nnet.save_checkpoint(filename="compiled.pth.tar")
    checkpoint = torch.load(run_config.net_directory / "compiled.pth.tar", map_location="cpu")
    assert not any(key.startswith("_orig_mod.") for key in checkpoint["state_dict"])

    eager = NNetWrapper(ttt_game, test_config)  # defaults: no compile
    eager.load_checkpoint(filename="compiled.pth.tar")
    for key, value in eager.nnet.state_dict().items():
        assert torch.equal(value, nnet.nnet.state_dict()[key])


def test_persistent_workers_train_to_completion(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """persistent_workers keeps the pool alive across epochs and still logs every batch."""
    _, metrics = _train_with_perf(
        ttt_game,
        test_config,
        TrainingPerfConfig(dataloader_workers=2, persistent_workers=True, prefetch_factor=2),
    )
    batches_per_epoch = -(-40 // test_config.net_config.batch_size)
    assert len(metrics.rows) == batches_per_epoch * 2  # epochs=2 in _train_with_perf
    assert np.isfinite([row["total_loss"] for row in metrics.rows]).all()


# ── H1: non-fork DataLoader workers (docs/plans/archive/harden-long-runs.md) ──


def test_resolve_dataloader_context_returns_requested_method() -> None:
    """The requested start method is honoured when the platform supports it."""
    for method in ("forkserver", "spawn"):
        assert resolve_dataloader_context(method).get_start_method() == method


def test_resolve_dataloader_context_falls_back_to_spawn() -> None:
    """An unavailable start method degrades to spawn rather than raising."""
    assert resolve_dataloader_context("not-a-real-method").get_start_method() == "spawn"


def test_lazy_dataset_encode_fn_pickles_without_dragging_the_game(blokus_game: BlokusDuoGame) -> None:
    """forkserver/spawn workers pickle the dataset — it must not drag the whole game.

    ``encode_compact`` is a ``@staticmethod``, so ``game.encode_compact`` pickles
    as a bare function reference. A bound method would instead serialise the
    entire game, including its multi-MB optimised move generator, to every
    worker on every respawn.
    """
    blokus_game.enable_optimised_movegen()  # the production path — loads the heavy F2 state
    game_blob = pickle.dumps(blokus_game)
    encode_blob = pickle.dumps(blokus_game.encode_compact)
    # The full game (with the F2 generator) is multi-MB; the encode_fn alone is tiny.
    assert len(encode_blob) < 1_000
    assert len(encode_blob) < len(game_blob)


def test_lazy_dataset_round_trips_through_pickle(blokus_game: BlokusDuoGame) -> None:
    """The dataset a forkserver worker receives pickles and still encodes correctly."""
    blokus_game.enable_optimised_movegen()
    action_size = blokus_game.get_action_size()
    compact_boards = [np.zeros((14, 14), dtype=np.int8) for _ in range(4)]
    raw_pis = [np.full(action_size, 1.0 / action_size, dtype=np.float32) for _ in range(4)]
    values = [0.0, 1.0, -1.0, 1e-4]
    dataset = _LazyPolicyDataset(compact_boards, raw_pis, values, action_size, blokus_game.encode_compact)

    restored = pickle.loads(pickle.dumps(dataset))
    assert len(restored) == len(dataset)
    board, pi, value = restored[1]
    expected_board = blokus_game.encode_compact(compact_boards[1])
    assert board.shape == expected_board.shape
    assert pi.shape == (action_size,)
    assert float(value) == 1.0


def test_default_dataloader_context_is_non_fork(ttt_game: TicTacToeGame, test_config: RunConfig) -> None:
    """The default spawn context trains identically to the in-process path.

    ``dataloader_context`` defaults to "spawn" (P0/S4 — forkserver deadlocked the
    memmap DataLoader at v3 gen-4), so a worker-backed run no longer forks the
    JAX-loaded parent. The shuffle permutation and per-epoch base seed still come
    from the main process, so weights stay bit-identical.
    """
    assert TrainingPerfConfig().dataloader_context == "spawn"
    baseline_weights, _ = _train_with_perf(ttt_game, test_config, TrainingPerfConfig())
    workers_weights, _ = _train_with_perf(
        ttt_game,
        test_config,
        TrainingPerfConfig(dataloader_workers=2, prefetch_factor=2, dataloader_context="spawn"),
    )
    _assert_identical_weights(baseline_weights, workers_weights)
