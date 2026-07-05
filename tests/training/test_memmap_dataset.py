"""Memmap-backed training dataset (``training/memmap_dataset.py``).

The worker-multiplication OOM fix (docs/plans/fix-training-oom.md M2): the
DataLoader must not pickle a full copy of the buffer to each worker. These tests
pin the two properties that matter — the dataset round-trips the buffer exactly
(so training stays bit-identical to the in-RAM path) and the pickle a worker
receives is tiny and constant, independent of buffer size (so workers share the
memmap page cache instead of each holding the buffer).
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import torch

from alphablokus.games.base_wrapper import _LazyPolicyDataset
from alphablokus.storage.sparse_policy import sparsify
from alphablokus.training.memmap_dataset import MemmapPolicyDataset

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.tictactoe.game import TicTacToeGame


def _synthetic_buffer(action_size: int, num_positions: int, seed: int = 0) -> list:
    """(compact 14x14 int8 board, sparse policy, value) triples, sparse like self-play."""
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(num_positions):
        board = rng.integers(-21, 22, (14, 14)).astype(np.int8)
        nnz = int(rng.integers(5, 40))
        dense = np.zeros(action_size, dtype=np.float32)
        idx = rng.choice(action_size, size=nnz, replace=False)
        weights = rng.random(nnz).astype(np.float32)
        dense[idx] = weights / weights.sum()
        examples.append((board, sparsify(dense), float((-1) ** i * rng.random())))
    return examples


def test_memmap_dataset_matches_in_ram_dataset(blokus_game: BlokusDuoGame, tmp_path: Path) -> None:
    """Every item equals the in-RAM ``_LazyPolicyDataset`` item — training stays identical."""
    action_size = blokus_game.get_action_size()
    examples = _synthetic_buffer(action_size, 32)
    boards_np, raw_pis, vs_np = zip(*examples, strict=True)

    in_ram = _LazyPolicyDataset(list(boards_np), list(raw_pis), list(vs_np), action_size, blokus_game.encode_compact)
    memmap = MemmapPolicyDataset.build(examples, action_size, blokus_game.encode_compact, tmp_path / "mm")

    assert len(memmap) == len(in_ram)
    for i in range(len(examples)):
        board_a, pi_a, value_a = in_ram[i]
        board_b, pi_b, value_b = memmap[i]
        assert torch.equal(board_a, board_b), f"board {i} diverged"
        assert torch.equal(pi_a, pi_b), f"policy {i} diverged"
        assert torch.equal(value_a, value_b), f"value {i} diverged"


def test_memmap_dataset_pickle_is_tiny_and_buffer_independent(blokus_game: BlokusDuoGame, tmp_path: Path) -> None:
    """A worker receives paths + a header, never the buffer — so pickle is small and flat.

    This is the whole point of the fix: the in-RAM dataset pickles linearly in
    buffer size (M1), which is what OOM'd 8 workers at 60k games. The memmap
    dataset's pickle must be small and *not grow* with the buffer.
    """
    action_size = blokus_game.get_action_size()
    small = MemmapPolicyDataset.build(
        _synthetic_buffer(action_size, 16), action_size, blokus_game.encode_compact, tmp_path / "small"
    )
    large = MemmapPolicyDataset.build(
        _synthetic_buffer(action_size, 512), action_size, blokus_game.encode_compact, tmp_path / "large"
    )
    small_blob = pickle.dumps(small)
    large_blob = pickle.dumps(large)
    assert len(small_blob) < 2_000, f"memmap dataset pickle should be tiny, got {len(small_blob)} bytes"
    # 32x the positions must not meaningfully grow the pickle (paths + ints only).
    assert len(large_blob) < len(small_blob) + 200


def test_memmap_dataset_round_trips_through_pickle(blokus_game: BlokusDuoGame, tmp_path: Path) -> None:
    """The dataset a worker unpickles reopens its own memmap and reads correctly."""
    action_size = blokus_game.get_action_size()
    examples = _synthetic_buffer(action_size, 8)
    dataset = MemmapPolicyDataset.build(examples, action_size, blokus_game.encode_compact, tmp_path / "mm")

    restored = pickle.loads(pickle.dumps(dataset))
    assert len(restored) == len(dataset)
    board, pi, value = restored[3]
    expected_board = blokus_game.encode_compact(examples[3][0])
    assert board.shape == expected_board.shape
    assert np.allclose(board.numpy(), expected_board)
    assert pi.shape == (action_size,)


def test_memmap_dataset_works_for_tictactoe_shape(ttt_game: TicTacToeGame, tmp_path: Path) -> None:
    """Game-agnostic: a 3x3 TicTacToe compact board round-trips too."""
    action_size = ttt_game.get_action_size()
    examples = []
    for i in range(10):
        board = np.zeros((3, 3), dtype=np.int8)
        board.flat[i % 9] = 1
        dense = np.full(action_size, 1.0 / action_size, dtype=np.float32)
        examples.append((board, sparsify(dense), float((-1) ** i)))
    dataset = MemmapPolicyDataset.build(examples, action_size, ttt_game.encode_compact, tmp_path / "ttt")
    board, pi, value = dataset[0]
    assert pi.shape == (action_size,)
    assert board.shape == ttt_game.encode_compact(examples[0][0]).shape
