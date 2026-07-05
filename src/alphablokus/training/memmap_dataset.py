"""On-disk (memmap-backed) training dataset for the worker-multiplication OOM fix.

The in-RAM ``_LazyPolicyDataset`` (``games/base_wrapper.py``) holds references
to every position in the replay buffer, so a ``forkserver``/``spawn`` DataLoader
**pickles a full copy of the whole buffer to every worker** — N workers ⇒ N
copies of an ~18 GB buffer ⇒ OOM at the buffer-fill generation (confirmed in
``docs/plans/fix-training-oom.md`` M1: the pickled dataset is exactly linear in
buffer size).

This dataset instead spills the buffer to a small set of flat memmap files once
per generation and hands workers only the *paths* (plus a tiny shape/offset
header). Each worker opens its own read-only memmap of the same files, so the
position data lives in the OS page cache **shared across all workers** rather
than duplicated in each worker's heap. The pickled dataset is then constant-size
(a few hundred bytes) regardless of buffer size.

Layout (CSR-style, so variable-length sparse policies pack without padding):

- ``boards.dat``      — ``(N, *board_shape)`` int8 compact boards
- ``values.dat``      — ``(N,)`` float32 game outcomes
- ``pi_offsets.dat``  — ``(N + 1,)`` int64 prefix sums into the policy arrays
- ``pi_indices.dat``  — ``(total_nnz,)`` int32 concatenated sparse-policy indices
- ``pi_values.dat``   — ``(total_nnz,)`` float32 concatenated sparse-policy values

``__getitem__`` re-encodes the compact board to dense planes and densifies the
sparse policy exactly as ``_LazyPolicyDataset`` does, so training is
bit-identical to the in-RAM path (same tensors, same DataLoader shuffle order).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from alphablokus.storage.sparse_policy import densify, sparsify

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from alphablokus.selfplay.episode import ProcessedExample

# Positions written to the memmap per chunk. Bounds the transient RAM the
# stack/concatenate build step uses (one chunk of dense-ish arrays at a time)
# rather than materialising the whole buffer contiguously in one shot.
_BUILD_CHUNK_POSITIONS = 100_000

_BOARDS_FILE = "boards.dat"
_VALUES_FILE = "values.dat"
_OFFSETS_FILE = "pi_offsets.dat"
_INDICES_FILE = "pi_indices.dat"
_PI_VALUES_FILE = "pi_values.dat"


class MemmapPolicyDataset(Dataset):
    """Memmap-backed replay-buffer dataset — workers share the page cache.

    Construct via :meth:`build`, which writes the flat memmap files and returns a
    ready dataset. The instance holds only paths + a shape/offset header, so a
    DataLoader worker receives a tiny pickle and opens its own read-only memmap
    view of the shared files (lazily, per process).

    ``encode_fn`` (``game.encode_compact``) is passed in so this stays
    game-agnostic — it never imports a game-specific symbol.
    """

    def __init__(
        self,
        directory: Path,
        num_positions: int,
        board_shape: tuple[int, ...],
        board_dtype: str,
        total_nnz: int,
        action_size: int,
        encode_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self._directory = Path(directory)
        self._num_positions = num_positions
        self._board_shape = tuple(board_shape)
        self._board_dtype = board_dtype
        self._total_nnz = total_nnz
        self._action_size = action_size
        self._encode_fn = encode_fn
        # Memmap handles are opened lazily and per process (see _ensure_open), so
        # they are always None across a pickle boundary — a worker maps its own.
        self._boards: np.memmap | None = None
        self._values: np.memmap | None = None
        self._offsets: np.memmap | None = None
        self._indices: np.memmap | None = None
        self._pi_values: np.memmap | None = None

    @classmethod
    def build(
        cls,
        examples: Sequence[ProcessedExample],
        action_size: int,
        encode_fn: Callable[[np.ndarray], np.ndarray],
        directory: Path,
    ) -> MemmapPolicyDataset:
        """Write the buffer to flat memmap files and return a dataset over them.

        Args:
            examples: The whole flattened replay buffer as ``(compact_board,
                (indices, values), value)`` triples (sparse policies).
            action_size: Dense action-space length the sparse policies index into.
            encode_fn: ``game.encode_compact`` — compact board → dense planes.
            directory: Destination directory for the memmap files (created;
                any existing files at these names are overwritten).

        Returns:
            A ready :class:`MemmapPolicyDataset`.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        num_positions = len(examples)
        board_shape = tuple(examples[0][0].shape)
        board_dtype = str(examples[0][0].dtype)

        # Prefix-sum the per-position nonzero counts to lay out the CSR policy
        # arrays; offsets[i]:offsets[i+1] slices position i's (indices, values).
        # Policies are normally sparse ``(indices, values)`` tuples (self-play's
        # output); dense arrays (hand-built fixtures) are accepted too, matching
        # ``storage.sparse_policy.as_dense``.
        offsets = np.zeros(num_positions + 1, dtype=np.int64)
        for i, (_board, policy, _value) in enumerate(examples):
            offsets[i + 1] = offsets[i] + len(cls._sparse_arrays(policy)[0])
        total_nnz = int(offsets[-1])

        boards_mm = np.memmap(
            directory / _BOARDS_FILE, mode="w+", dtype=np.dtype(board_dtype), shape=(num_positions, *board_shape)
        )
        values_mm = np.memmap(directory / _VALUES_FILE, mode="w+", dtype=np.float32, shape=(num_positions,))
        indices_mm = np.memmap(directory / _INDICES_FILE, mode="w+", dtype=np.int32, shape=(max(total_nnz, 1),))
        pi_values_mm = np.memmap(directory / _PI_VALUES_FILE, mode="w+", dtype=np.float32, shape=(max(total_nnz, 1),))
        offsets_mm = np.memmap(directory / _OFFSETS_FILE, mode="w+", dtype=np.int64, shape=(num_positions + 1,))
        offsets_mm[:] = offsets

        # Fill in position-chunks so the stack/concatenate transient is bounded
        # to one chunk, never the whole (multi-GB) buffer at once.
        for start in range(0, num_positions, _BUILD_CHUNK_POSITIONS):
            end = min(start + _BUILD_CHUNK_POSITIONS, num_positions)
            chunk = examples[start:end]
            boards_mm[start:end] = np.stack([board for board, _pi, _value in chunk])
            values_mm[start:end] = np.asarray([value for _board, _pi, value in chunk], dtype=np.float32)
            nnz_start = int(offsets[start])
            nnz_end = int(offsets[end])
            if nnz_end > nnz_start:
                sparse = [cls._sparse_arrays(policy) for _board, policy, _value in chunk]
                indices_mm[nnz_start:nnz_end] = np.concatenate([indices for indices, _values in sparse])
                pi_values_mm[nnz_start:nnz_end] = np.concatenate([values for _indices, values in sparse])

        for handle in (boards_mm, values_mm, indices_mm, pi_values_mm, offsets_mm):
            handle.flush()

        return cls(directory, num_positions, board_shape, board_dtype, total_nnz, action_size, encode_fn)

    @staticmethod
    def _sparse_arrays(policy: object) -> tuple[np.ndarray, np.ndarray]:
        """Normalise a stored policy to ``(int32 indices, float32 values)``.

        Sparse ``(indices, values)`` tuples (self-play's output) pass through;
        dense arrays (hand-built fixtures) are sparsified — lossless, since only
        the exact zeros are dropped. Mirrors ``storage.sparse_policy.as_dense``'s
        both-forms tolerance on the write side.
        """
        if isinstance(policy, tuple):
            indices, values = policy
            return np.asarray(indices, dtype=np.int32), np.asarray(values, dtype=np.float32)
        return sparsify(np.asarray(policy, dtype=np.float32))

    def __len__(self) -> int:
        return self._num_positions

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        self._ensure_open()
        assert self._boards is not None
        assert self._offsets is not None
        assert self._indices is not None
        assert self._pi_values is not None
        assert self._values is not None

        # np.array() copies the tiny compact board out of the mmap before
        # encoding, so the dense encode never keeps a mapped view alive.
        board = torch.from_numpy(
            np.ascontiguousarray(self._encode_fn(np.array(self._boards[idx])), dtype=np.float32),
        )
        low = int(self._offsets[idx])
        high = int(self._offsets[idx + 1])
        indices = np.asarray(self._indices[low:high], dtype=np.int32)
        values = np.asarray(self._pi_values[low:high], dtype=np.float32)
        pi = torch.from_numpy(densify(indices, values, self._action_size))
        value = torch.tensor(float(self._values[idx]), dtype=torch.float32)
        return board, pi, value

    def _ensure_open(self) -> None:
        """Open the memmaps in the current process, once (lazy, per worker)."""
        if self._boards is not None:
            return
        self._boards = np.memmap(
            self._directory / _BOARDS_FILE,
            mode="r",
            dtype=np.dtype(self._board_dtype),
            shape=(self._num_positions, *self._board_shape),
        )
        self._values = np.memmap(
            self._directory / _VALUES_FILE, mode="r", dtype=np.float32, shape=(self._num_positions,)
        )
        self._offsets = np.memmap(
            self._directory / _OFFSETS_FILE, mode="r", dtype=np.int64, shape=(self._num_positions + 1,)
        )
        self._indices = np.memmap(
            self._directory / _INDICES_FILE, mode="r", dtype=np.int32, shape=(max(self._total_nnz, 1),)
        )
        self._pi_values = np.memmap(
            self._directory / _PI_VALUES_FILE, mode="r", dtype=np.float32, shape=(max(self._total_nnz, 1),)
        )

    def __getstate__(self) -> dict[str, Any]:
        # Never pickle the mmap handles (that would try to serialise mapped
        # data); workers reopen them lazily from the paths in the header.
        state = self.__dict__.copy()
        for key in ("_boards", "_values", "_offsets", "_indices", "_pi_values"):
            state[key] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
