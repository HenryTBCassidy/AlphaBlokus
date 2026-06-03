"""Unit tests for the transport-agnostic inference-server core (F5 / C2).

These exercise the batching logic in isolation: an in-memory request source
stands in for the shared-memory transport (C3), and a deterministic,
per-row-independent fake stands in for the GPU forward pass. No CUDA, no IPC.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from core.inference_server import FlushPolicy, InferenceServer

# Small fixed board shape for the fakes — (channels, height, width).
_C, _H, _W = 3, 4, 4
_A = 7  # action-space size for the fake policy vectors


def _fake_predict(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, per-row-independent stand-in for the network.

    Each board maps to a policy vector filled with its own sum and a value equal
    to its own mean — so a board's output depends only on that board, never on
    its batch-mates (the property the real server relies on).
    """
    n = boards.shape[0]
    flat = boards.reshape(n, -1)
    sums = flat.sum(axis=1).astype(np.float32)
    policies = np.repeat(sums[:, None], _A, axis=1).astype(np.float32)
    values = flat.mean(axis=1).astype(np.float32)
    return policies, values


class _Request:
    """One in-memory evaluation request with an event-signalled result."""

    def __init__(self, board: np.ndarray) -> None:
        self._board = board
        self.policy: np.ndarray | None = None
        self.value: float | None = None
        self._done = threading.Event()

    @property
    def board(self) -> np.ndarray:
        return self._board

    def set_result(self, policy: np.ndarray, value: float) -> None:
        self.policy = policy
        self.value = value
        self._done.set()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)


class _InMemorySource:
    """Queue-backed request source; ``poll`` drains everything available."""

    def __init__(self) -> None:
        self._q: queue.Queue[_Request] = queue.Queue()
        self._stop = threading.Event()

    def submit(self, req: _Request) -> None:
        self._q.put(req)

    def poll(self, timeout_s: float) -> list[_Request]:
        out: list[_Request] = []
        try:
            out.append(self._q.get(timeout=max(timeout_s, 0.0)))
        except queue.Empty:
            return out
        while True:  # drain the rest without blocking
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()


def _run_server_in_thread(server: InferenceServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def test_routing_and_per_row_equivalence() -> None:
    """Every request gets exactly predict_fn applied to its own board."""
    rng = np.random.default_rng(0)
    source = _InMemorySource()
    server = InferenceServer(_fake_predict, source, FlushPolicy(max_batch=8, max_wait_s=0.02))
    thread = _run_server_in_thread(server)

    reqs = [_Request(rng.standard_normal((_C, _H, _W)).astype(np.float32)) for _ in range(20)]
    for r in reqs:
        source.submit(r)

    for r in reqs:
        assert r.wait(2.0), "request was not fulfilled in time"

    for r in reqs:
        exp_pol, exp_val = _fake_predict(r.board[None, ...])
        np.testing.assert_array_equal(r.policy, exp_pol[0])
        assert r.value == float(exp_val[0])

    source.stop()
    thread.join(timeout=2.0)
    assert server.requests_served == 20


def test_batches_never_exceed_max_batch() -> None:
    """A burst larger than max_batch is split into capped batches, not one."""
    source = _InMemorySource()
    server = InferenceServer(_fake_predict, source, FlushPolicy(max_batch=4, max_wait_s=0.02))
    thread = _run_server_in_thread(server)

    reqs = [_Request(np.full((_C, _H, _W), i, dtype=np.float32)) for i in range(10)]
    for r in reqs:
        source.submit(r)

    for r in reqs:
        assert r.wait(2.0)

    source.stop()
    thread.join(timeout=2.0)
    # 10 requests, max_batch 4 -> ceil(10/4) = 3 batches (4, 4, 2).
    assert server.requests_served == 10
    assert server.batches_served == 3


def test_timeout_flush_without_full_batch() -> None:
    """A partial batch flushes on the timer rather than waiting to fill."""
    source = _InMemorySource()
    server = InferenceServer(_fake_predict, source, FlushPolicy(max_batch=100, max_wait_s=0.05))
    thread = _run_server_in_thread(server)

    reqs = [_Request(np.full((_C, _H, _W), i, dtype=np.float32)) for i in range(3)]
    start = time.perf_counter()
    for r in reqs:
        source.submit(r)

    for r in reqs:
        assert r.wait(2.0)
    elapsed = time.perf_counter() - start

    # Flushed well before a full batch of 100 could ever arrive, on the timer.
    assert server.requests_served == 3
    assert server.batches_served == 1
    assert elapsed < 1.0

    source.stop()
    thread.join(timeout=2.0)


def test_flush_policy_should_flush_rules() -> None:
    """The size-OR-timeout decision logic in isolation."""
    policy = FlushPolicy(max_batch=4, max_wait_s=0.1)
    assert not policy.should_flush(0, 999.0)  # nothing pending
    assert not policy.should_flush(2, 0.0)  # neither bound met
    assert policy.should_flush(4, 0.0)  # size bound
    assert policy.should_flush(1, 0.1)  # timeout bound
