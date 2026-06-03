"""Multi-process round-trip tests for the shared-memory channel (F5 / C3).

Spawns a real inference-server process and real worker processes that talk over
shared memory, verifying results route back correctly and per-row-independently
when leaves are coalesced *across* workers. CPU + a fake predict_fn (no CUDA).
"""

from __future__ import annotations

import multiprocessing as mp
import threading

import numpy as np
import pytest

from core.inference_channel import (
    ChannelHandles,
    ChannelSpec,
    InferenceClientNet,
    SharedInferenceChannel,
    SharedMemoryRequestSource,
)
from core.inference_server import FlushPolicy, InferenceServer

_C, _H, _W = 3, 4, 4
_A = 7


def _fake_predict(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, per-row-independent stand-in for the GPU net."""
    n = boards.shape[0]
    flat = boards.reshape(n, -1)
    sums = flat.sum(axis=1).astype(np.float32)
    policies = np.repeat(sums[:, None], _A, axis=1).astype(np.float32)
    values = flat.mean(axis=1).astype(np.float32)
    return policies, values


def _run_server(handles: ChannelHandles, max_batch: int, max_wait_s: float) -> None:
    channel = SharedInferenceChannel.attach(handles)
    source = SharedMemoryRequestSource(channel)
    InferenceServer(_fake_predict, source, FlushPolicy(max_batch, max_wait_s)).serve_forever()
    channel.close()


def _run_worker(handles: ChannelHandles, worker_id: int, boards: np.ndarray, out: mp.Queue) -> None:
    channel = SharedInferenceChannel.attach(handles)
    policies, values = channel.submit(worker_id, boards)
    out.put((worker_id, [np.asarray(p) for p in policies], list(values)))
    channel.close()


def test_multiprocess_cross_worker_round_trip() -> None:
    num_workers = 4
    spec = ChannelSpec(num_workers=num_workers, max_leaves=8, board_shape=(_C, _H, _W), action_size=_A)
    channel = SharedInferenceChannel.create(spec)
    handles = channel.handles()

    # max_batch large enough to coalesce all workers' leaves into one GPU call.
    server = mp.Process(target=_run_server, args=(handles, 64, 0.01), daemon=True)
    server.start()

    rng = np.random.default_rng(0)
    # Each worker submits a different number of leaves (k = worker_id + 2).
    boards_by_worker = {
        wid: rng.standard_normal((wid + 2, _C, _H, _W)).astype(np.float32) for wid in range(num_workers)
    }
    out: mp.Queue = mp.Queue()
    workers = [
        mp.Process(target=_run_worker, args=(handles, wid, boards_by_worker[wid], out), daemon=True)
        for wid in range(num_workers)
    ]
    try:
        for w in workers:
            w.start()

        results: dict[int, tuple[list[np.ndarray], list[float]]] = {}
        for _ in range(num_workers):
            wid, policies, values = out.get(timeout=20)
            results[wid] = (policies, values)

        for w in workers:
            w.join(timeout=10)
        channel.request_stop()
        server.join(timeout=10)

        # Each leaf's result must equal predict_fn applied to that leaf alone,
        # regardless of which workers it was batched with.
        for wid, boards in boards_by_worker.items():
            exp_pol, exp_val = _fake_predict(boards)
            policies, values = results[wid]
            assert len(policies) == len(boards)
            for i in range(len(boards)):
                np.testing.assert_array_equal(policies[i], exp_pol[i])
                assert values[i] == pytest.approx(float(exp_val[i]))
    finally:
        channel.request_stop()
        for w in workers:
            if w.is_alive():
                w.terminate()
        if server.is_alive():
            server.terminate()
        channel.unlink()


class _FakeBoard:
    """Minimal board exposing only what InferenceClientNet needs to encode."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def as_multi_channel(self, current_player: int) -> np.ndarray:  # noqa: ARG002 - signature parity
        return self._arr


def test_client_net_predict_and_predict_batch() -> None:
    """InferenceClientNet.predict / predict_batch route correctly via the channel."""
    spec = ChannelSpec(num_workers=1, max_leaves=8, board_shape=(_C, _H, _W), action_size=_A)
    channel = SharedInferenceChannel.create(spec)
    source = SharedMemoryRequestSource(channel)
    server = InferenceServer(_fake_predict, source, FlushPolicy(max_batch=8, max_wait_s=0.01))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        rng = np.random.default_rng(1)
        client = InferenceClientNet(channel, worker_id=0)
        boards = [_FakeBoard(rng.standard_normal((_C, _H, _W)).astype(np.float32)) for _ in range(5)]

        policies, values = client.predict_batch(boards)
        assert len(policies) == 5 and len(values) == 5
        for i, b in enumerate(boards):
            exp_pol, exp_val = _fake_predict(b.as_multi_channel(1)[np.newaxis, ...])
            np.testing.assert_array_equal(policies[i], exp_pol[0])
            assert values[i] == pytest.approx(float(exp_val[0]))

        # Single-board predict mirrors predict_batch([board]).
        pol, val = client.predict(boards[0])
        np.testing.assert_array_equal(pol, policies[0])
        assert val == pytest.approx(values[0])
    finally:
        channel.request_stop()
        thread.join(timeout=5)
        channel.unlink()
