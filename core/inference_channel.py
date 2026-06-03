"""Shared-memory transport for the cross-worker inference server (F5 / C3).

Connects N self-play / arena / Elo worker processes to a single inference-server
process over shared memory, so leaf evaluations are coalesced across *all*
workers into large GPU batches (recovering the ~70% idle GPU that per-worker F3
batching leaves on the table).

**Why this is correct-by-construction.** Each worker is *synchronous*: it writes
its batch of leaves, signals the server, then **blocks** until its result is
ready before it can ever submit again. So each worker has at most one request
in flight, and gets one fixed shared-memory slot — no dynamic allocation, no
slot-contention, no per-slot locking. The server is single-threaded, so its
bookkeeping (how many of a worker's leaves remain) needs no locking either.

Layout (all in shared memory, indexed ``[worker_id]``):
- ``req_boards``   ``(N, K, C, H, W)`` float32 — each worker's up-to-K leaf planes
- ``req_counts``   ``(N,)``           int32   — how many leaves are valid this request
- ``resp_policies````(N, K, A)``       float32 — per-leaf policy out
- ``resp_values``  ``(N, K)``          float32 — per-leaf value out

Signalling: a ``Queue`` carries ``worker_id``s from workers to the server; a
per-worker ``Event`` tells each worker its result is ready; a shared ``Event``
requests shutdown. These are passed to children as process args (works under
both ``fork`` and ``spawn``); the big arrays are attached by name.

See ``docs/plans/cross-worker-inference-server.md`` (C3).
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from core.interfaces import IBoard

_BOARDS = "boards"
_COUNTS = "counts"
_POLICIES = "policies"
_VALUES = "values"


@dataclass(frozen=True)
class ChannelSpec:
    """Fixed dimensions of a channel. Picklable so it crosses to children."""

    num_workers: int
    max_leaves: int  # K: max leaves per worker request (= mcts_batch_size)
    board_shape: tuple[int, ...]  # (C, H, W) of the encoded board
    action_size: int  # policy vector length (17,837 for Blokus Duo)


@dataclass(frozen=True)
class ChannelHandles:
    """Everything a child process needs to attach to an existing channel.

    Picklable: shared-memory blocks are referenced by name; the synchronisation
    primitives are mp objects passed as process args.
    """

    spec: ChannelSpec
    shm_names: dict[str, str]
    submit_queue: mp.Queue
    result_events: tuple  # tuple[mp.Event, ...], one per worker
    stop_event: object  # mp.Event


def _array_specs(spec: ChannelSpec) -> dict[str, tuple[tuple[int, ...], type]]:
    n, k = spec.num_workers, spec.max_leaves
    return {
        _BOARDS: ((n, k, *spec.board_shape), np.float32),
        _COUNTS: ((n,), np.int32),
        _POLICIES: ((n, k, spec.action_size), np.float32),
        _VALUES: ((n, k), np.float32),
    }


class SharedInferenceChannel:
    """Owns (or attaches to) the shared arrays + signalling for one channel.

    Create once in the parent with :meth:`create`, pass :meth:`handles` to the
    server process and to each worker (via the pool initialiser), and
    :meth:`attach` in each child. Worker code calls :meth:`submit`; the server
    consumes via :class:`SharedMemoryRequestSource`.
    """

    def __init__(
        self,
        spec: ChannelSpec,
        shms: dict[str, shared_memory.SharedMemory],
        arrays: dict[str, NDArray],
        submit_queue: mp.Queue,
        result_events: tuple,
        stop_event: object,
        *,
        owner: bool,
    ) -> None:
        self.spec = spec
        self._shms = shms
        self.req_boards: NDArray[np.float32] = arrays[_BOARDS]
        self.req_counts: NDArray[np.int32] = arrays[_COUNTS]
        self.resp_policies: NDArray[np.float32] = arrays[_POLICIES]
        self.resp_values: NDArray[np.float32] = arrays[_VALUES]
        self.submit_queue = submit_queue
        self.result_events = result_events
        self.stop_event = stop_event
        self._owner = owner

    @classmethod
    def create(cls, spec: ChannelSpec, ctx: object = None) -> SharedInferenceChannel:
        """Allocate a fresh channel in the parent process.

        Args:
            spec: Channel dimensions.
            ctx: The multiprocessing context whose ``Queue`` / ``Event`` to use.
                **Must be the same context the workers and server are spawned
                with** — a lock created in one context cannot be shared with a
                process in another (e.g. ``fork`` lock + ``spawn`` process raises
                at runtime). Defaults to the active context.
        """
        if ctx is None:
            ctx = mp.get_context()
        shms: dict[str, shared_memory.SharedMemory] = {}
        arrays: dict[str, NDArray] = {}
        for name, (shape, dtype) in _array_specs(spec).items():
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            shms[name] = shm
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            arr.fill(0)
            arrays[name] = arr
        submit_queue: mp.Queue = ctx.Queue()
        result_events = tuple(ctx.Event() for _ in range(spec.num_workers))
        stop_event = ctx.Event()
        return cls(spec, shms, arrays, submit_queue, result_events, stop_event, owner=True)

    @classmethod
    def attach(cls, handles: ChannelHandles) -> SharedInferenceChannel:
        """Attach to an existing channel from inside a child process."""
        shms: dict[str, shared_memory.SharedMemory] = {}
        arrays: dict[str, NDArray] = {}
        for name, (shape, dtype) in _array_specs(handles.spec).items():
            shm = shared_memory.SharedMemory(name=handles.shm_names[name])
            shms[name] = shm
            arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return cls(
            handles.spec,
            shms,
            arrays,
            handles.submit_queue,
            handles.result_events,
            handles.stop_event,
            owner=False,
        )

    def handles(self) -> ChannelHandles:
        """Picklable handle bundle for children."""
        return ChannelHandles(
            spec=self.spec,
            shm_names={name: shm.name for name, shm in self._shms.items()},
            submit_queue=self.submit_queue,
            result_events=self.result_events,
            stop_event=self.stop_event,
        )

    # -- worker side ----------------------------------------------------------

    def submit(self, worker_id: int, boards: NDArray[np.float32]) -> tuple[list[NDArray], list[float]]:
        """Submit one batch of encoded leaf boards and block for the result.

        Args:
            worker_id: This worker's fixed slot index (0..num_workers-1).
            boards: ``(k, C, H, W)`` float32, ``1 <= k <= max_leaves``.

        Returns:
            ``(policies, values)`` — ``k`` policy arrays and ``k`` floats, in the
            same order as ``boards``.
        """
        k = boards.shape[0]
        if not 1 <= k <= self.spec.max_leaves:
            raise ValueError(f"batch of {k} leaves outside [1, {self.spec.max_leaves}]")
        self.req_boards[worker_id, :k] = boards
        self.req_counts[worker_id] = k
        event = self.result_events[worker_id]
        event.clear()
        self.submit_queue.put(worker_id)
        event.wait()
        policies = [self.resp_policies[worker_id, i].copy() for i in range(k)]
        values = [float(self.resp_values[worker_id, i]) for i in range(k)]
        return policies, values

    # -- lifecycle ------------------------------------------------------------

    def request_stop(self) -> None:
        """Signal the server to shut down."""
        self.stop_event.set()

    def close(self) -> None:
        """Detach this process's view of the shared memory."""
        for shm in self._shms.values():
            shm.close()

    def unlink(self) -> None:
        """Free the shared memory (parent/owner only — call once, at shutdown)."""
        if not self._owner:
            return
        for shm in self._shms.values():
            shm.close()
            shm.unlink()


class _SharedPendingRequest:
    """One leaf evaluation, backed by a worker's shared-memory slot.

    Implements the :class:`~core.inference_server.PendingRequest` protocol.
    """

    __slots__ = ("_source", "_worker_id", "_leaf_idx")

    def __init__(self, source: SharedMemoryRequestSource, worker_id: int, leaf_idx: int) -> None:
        self._source = source
        self._worker_id = worker_id
        self._leaf_idx = leaf_idx

    @property
    def board(self) -> NDArray[np.float32]:
        return self._source.channel.req_boards[self._worker_id, self._leaf_idx]

    def set_result(self, policy: NDArray[np.float32], value: float) -> None:
        ch = self._source.channel
        ch.resp_policies[self._worker_id, self._leaf_idx] = policy
        ch.resp_values[self._worker_id, self._leaf_idx] = value
        self._source.on_leaf_done(self._worker_id)


class SharedMemoryRequestSource:
    """Adapts a :class:`SharedInferenceChannel` to the server's RequestSource.

    ``poll`` drains submitted ``worker_id``s and expands each into per-leaf
    requests; the server batches them across workers. When all of a worker's
    leaves have been written back, the worker's result event is set so it
    unblocks. The server is single-threaded, so ``_remaining`` needs no lock.
    """

    def __init__(self, channel: SharedInferenceChannel, poll_first_timeout_s: float = 0.05) -> None:
        self.channel = channel
        self._remaining: dict[int, int] = {}
        self._poll_first_timeout_s = poll_first_timeout_s

    def poll(self, timeout_s: float) -> list[_SharedPendingRequest]:
        worker_ids: list[int] = []
        # Block briefly for the first arrival, then drain the rest without waiting.
        try:
            worker_ids.append(self.channel.submit_queue.get(timeout=max(timeout_s, 0.0)))
        except _queue.Empty:
            return []
        while True:
            try:
                worker_ids.append(self.channel.submit_queue.get_nowait())
            except _queue.Empty:
                break
        requests: list[_SharedPendingRequest] = []
        for wid in worker_ids:
            k = int(self.channel.req_counts[wid])
            self._remaining[wid] = k
            requests.extend(_SharedPendingRequest(self, wid, i) for i in range(k))
        return requests

    def on_leaf_done(self, worker_id: int) -> None:
        self._remaining[worker_id] -= 1
        if self._remaining[worker_id] == 0:
            del self._remaining[worker_id]
            self.channel.result_events[worker_id].set()

    def stop_requested(self) -> bool:
        return bool(self.channel.stop_event.is_set())


class InferenceClientNet:
    """Worker-side stand-in for the network that routes inference to the server.

    Implements only the inference surface MCTS uses (``predict`` /
    ``predict_batch``) with the same signatures and return shapes as
    :class:`~games.base_wrapper.BaseNeuralNetWrapper`, so it drops into
    ``MCTS(game, nnet, ...)`` unchanged. Board encoding happens here (worker
    side, cheap CPU); only the stacked planes cross the shared-memory boundary.

    Workers that use the client never train, so the heavier wrapper methods
    (train / save / load) are intentionally absent.
    """

    def __init__(self, channel: SharedInferenceChannel, worker_id: int) -> None:
        self._channel = channel
        self._worker_id = worker_id

    def predict_batch(self, boards: Sequence[IBoard]) -> tuple[list[NDArray], list[float]]:
        planes = np.stack([board.as_multi_channel(1) for board in boards]).astype(np.float32, copy=False)
        return self._channel.submit(self._worker_id, planes)

    def predict(self, board: IBoard) -> tuple[NDArray, float]:
        policies, values = self._channel.submit(
            self._worker_id, board.as_multi_channel(1)[np.newaxis, ...].astype(np.float32, copy=False)
        )
        return policies[0], values[0]
