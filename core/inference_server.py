"""Cross-worker batched inference server.

The transport-agnostic *core* of the inference server: it accumulates leaf
evaluation requests arriving from many self-play / arena / Elo workers and
flushes them to the network in single large batches, recovering the GPU
under-utilisation (~30% during self-play) that per-worker batching leaves on
the table.

This module deliberately knows nothing about *how* requests arrive or *how*
results are delivered (that is the transport layer — shared memory in
production, an in-memory queue in tests) and nothing about *how* inference runs
(that is an injected ``predict_fn`` wrapping the GPU net). Keeping those
concerns out makes the batching logic unit-testable on CPU with no IPC and no
CUDA.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

# A predict function maps a stacked batch of encoded boards (N, C, H, W) to
# (policies (N, A), values (N,)). In production this wraps the GPU forward pass;
# in tests it is a deterministic, per-row-independent fake. Per-row independence
# is what guarantees a board's result is unaffected by its batch-mates — the
# property that lets server-on stay bit-identical to server-off.
PredictFn = Callable[["NDArray[np.float32]"], "tuple[NDArray[np.float32], NDArray[np.float32]]"]


@dataclass(frozen=True)
class FlushPolicy:
    """When the server flushes its accumulated batch to the network.

    The batch flushes as soon as **either** condition holds, whichever first:

    - it contains ``max_batch`` requests, or
    - ``max_wait_s`` has elapsed since the batch's first request arrived.

    This size-OR-timeout rule is self-correcting: under load the batch fills to
    ``max_batch`` before the timer fires (maximising GPU efficiency); under light
    load the timer bounds per-request latency. There is no single fragile
    threshold — both bounds are safe defaults rather than a tuned knife-edge.

    Attributes:
        max_batch: Maximum positions per GPU forward pass. Set near the net's
            throughput-saturation point.
        max_wait_s: Maximum time the first queued request waits before the batch
            is flushed even if it is not yet full.
    """

    max_batch: int
    max_wait_s: float

    def __post_init__(self) -> None:
        if self.max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {self.max_batch}")
        if self.max_wait_s < 0:
            raise ValueError(f"max_wait_s must be >= 0, got {self.max_wait_s}")

    def should_flush(self, n_pending: int, wait_elapsed_s: float) -> bool:
        """Whether a batch of ``n_pending`` requests should flush now.

        Args:
            n_pending: Requests currently accumulated.
            wait_elapsed_s: Seconds since the first request in this batch arrived.
        """
        if n_pending <= 0:
            return False
        return n_pending >= self.max_batch or wait_elapsed_s >= self.max_wait_s


@runtime_checkable
class PendingRequest(Protocol):
    """One in-flight evaluation request the server must fulfil.

    Provided by the transport layer. The server needs only the encoded board
    planes and a way to hand the result back to the originating worker.
    """

    @property
    def board(self) -> NDArray[np.float32]:
        """Encoded board planes for one position, shape ``(C, H, W)``."""
        ...

    def set_result(self, policy: NDArray[np.float32], value: float) -> None:
        """Deliver this request's policy ``(A,)`` and scalar value to the worker."""
        ...


@runtime_checkable
class RequestSource(Protocol):
    """Where the server pulls pending requests from (transport-agnostic)."""

    def poll(self, timeout_s: float) -> list[PendingRequest]:
        """Return any requests available within ``timeout_s`` (possibly empty)."""
        ...

    def stop_requested(self) -> bool:
        """Whether the server has been asked to shut down."""
        ...


class InferenceServer:
    """Accumulate-flush-route loop over a :class:`RequestSource`.

    Owns no network and no transport: it pulls requests from ``source``, batches
    them per ``policy``, evaluates each batch via ``predict_fn``, and routes each
    result back through the request's ``set_result``. Run it with
    :meth:`serve_forever` on its own thread/process.
    """

    def __init__(self, predict_fn: PredictFn, source: RequestSource, policy: FlushPolicy) -> None:
        self._predict = predict_fn
        self._source = source
        self._policy = policy
        self._batches_served = 0
        self._requests_served = 0
        # Requests pulled from the source but not yet flushed. Persists across
        # batches so an over-full poll (more than ``max_batch`` arriving at once)
        # is split into correctly-sized batches rather than one oversized one.
        self._buffer: list[PendingRequest] = []
        self._batch_started: float = 0.0

    @property
    def batches_served(self) -> int:
        """Number of GPU batches flushed so far (for stats / tests)."""
        return self._batches_served

    @property
    def requests_served(self) -> int:
        """Number of individual requests fulfilled so far (for stats / tests)."""
        return self._requests_served

    def serve_forever(self) -> None:
        """Run accumulate-flush cycles until the source requests a stop."""
        logger.info(
            "InferenceServer starting (max_batch={}, max_wait_ms={:.1f})",
            self._policy.max_batch,
            self._policy.max_wait_s * 1000.0,
        )
        while not self._source.stop_requested():
            self._serve_one_batch()
        logger.info(
            "InferenceServer stopped after {} batch(es) / {} request(s)",
            self._batches_served,
            self._requests_served,
        )

    def _serve_one_batch(self) -> None:
        """Accumulate up to ``max_batch`` requests under the flush policy, then
        evaluate + route exactly that many; any surplus stays buffered for the
        next call so batches are never larger than ``max_batch``."""
        # Step 1: ensure at least one request, blocking up to max_wait_s so an
        # idle server doesn't spin. If nothing arrives, return to re-check stop.
        if not self._buffer:
            arrived = self._source.poll(self._policy.max_wait_s)
            if not arrived:
                return
            self._buffer.extend(arrived)
            self._batch_started = time.perf_counter()

        # Step 2: top up until the batch is full or the timer expires.
        while not self._source.stop_requested():
            wait_elapsed = time.perf_counter() - self._batch_started
            if self._policy.should_flush(len(self._buffer), wait_elapsed):
                break
            remaining = max(0.0, self._policy.max_wait_s - wait_elapsed)
            self._buffer.extend(self._source.poll(min(remaining, 0.0005)))

        # Step 3: flush at most max_batch; keep the remainder for the next call,
        # restarting its wait timer from now.
        batch = self._buffer[: self._policy.max_batch]
        self._buffer = self._buffer[self._policy.max_batch :]
        if self._buffer:
            self._batch_started = time.perf_counter()
        if batch:
            self._flush(batch)

    def _flush(self, pending: list[PendingRequest]) -> None:
        """Evaluate ``pending`` in a single batch and route results back.

        Results are routed positionally: ``predict_fn`` must preserve row order,
        so ``policies[i]`` / ``values[i]`` belong to ``pending[i]``.
        """
        boards = np.stack([req.board for req in pending]).astype(np.float32, copy=False)
        policies, values = self._predict(boards)
        if len(policies) != len(pending) or len(values) != len(pending):
            raise RuntimeError(
                f"predict_fn returned {len(policies)} policies / {len(values)} values "
                f"for a batch of {len(pending)} requests"
            )
        for i, req in enumerate(pending):
            req.set_result(policies[i], float(values[i]))
        self._batches_served += 1
        self._requests_served += len(pending)
