"""Validation + integration check for the cross-worker inference server, on real GPU.

Run on the PC (CUDA) from the repo root:

    uv run python scripts/validate_inference_server.py

Two checks:

1. **Per-board equivalence.** A set of boards is evaluated (a) directly via
   ``nnet.predict_encoded`` in one batch and (b) through the inference server,
   submitted in several concurrent worker-sized groups so the server *coalesces*
   them into a single large GPU batch. If the per-row outputs match, the server
   is correct AND the net's forward pass is batch-composition-invariant (so
   server-on stays equivalent to server-off). Any difference here is GPU/cuDNN
   fp16 batch-size numerics, not our code — reported, not hidden.

2. **Integration check.** ``run_self_play_episodes_parallel`` actually runs with
   ``inference_server=True`` on real GPU, proving the spawned server + shared-
   memory channel + client wiring all work end to end.
"""

from __future__ import annotations

import dataclasses as dc
import threading

import numpy as np

from core.config import load_args
from core.game_factory import instantiate_game_and_network
from core.inference_channel import ChannelSpec, SharedInferenceChannel, SharedMemoryRequestSource
from core.inference_server import FlushPolicy, InferenceServer
from core.parallel_self_play import run_self_play_episodes_parallel

_CONFIG = "run_configurations/blokus_quicktest.json"


def equivalence_check() -> None:
    cfg = load_args(_CONFIG)
    game, nnet = instantiate_game_and_network(cfg)
    board_shape = tuple(game.initialise_board().as_multi_channel(1).shape)
    action_size = game.get_action_size()

    num_workers, per_worker = 4, 12
    total = num_workers * per_worker
    rng = np.random.default_rng(0)
    boards = rng.standard_normal((total, *board_shape)).astype(np.float32)

    # (a) direct: one batch of all boards
    pol_direct, val_direct = nnet.predict_encoded(boards)

    # (b) via server: submit num_workers concurrent groups; server coalesces them
    spec = ChannelSpec(num_workers=num_workers, max_leaves=per_worker, board_shape=board_shape, action_size=action_size)
    channel = SharedInferenceChannel.create(spec)
    source = SharedMemoryRequestSource(channel)
    server = InferenceServer(nnet.predict_encoded, source, FlushPolicy(max_batch=total, max_wait_s=0.05))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    results: dict[int, tuple[list, list]] = {}

    def _submit(wid: int) -> None:
        group = boards[wid * per_worker : (wid + 1) * per_worker]
        results[wid] = channel.submit(wid, group)

    try:
        submitters = [threading.Thread(target=_submit, args=(w,)) for w in range(num_workers)]
        for t in submitters:
            t.start()
        for t in submitters:
            t.join(timeout=30)

        pol_server = np.stack([p for w in range(num_workers) for p in results[w][0]])
        val_server = np.array([v for w in range(num_workers) for v in results[w][1]], dtype=np.float32)

        max_pi = float(np.max(np.abs(pol_direct - pol_server)))
        max_v = float(np.max(np.abs(val_direct - val_server)))
        print(f"[equivalence] compared {total} boards across {num_workers} coalesced groups")
        print(f"[equivalence] max |policy diff| = {max_pi:.3e}   max |value diff| = {max_v:.3e}")
        if max_pi == 0.0 and max_v == 0.0:
            print("[equivalence] VERDICT: BIT-IDENTICAL — server-on == server-off, games stay reproducible")
        elif max_pi < 1e-2 and max_v < 1e-2:
            print("[equivalence] VERDICT: numerically equivalent (fp16/cuDNN batch-size numerics; server is correct)")
        else:
            print("[equivalence] VERDICT: DIVERGENT — investigate (possible bug)")
    finally:
        channel.request_stop()
        thread.join(timeout=5)
        channel.unlink()


def smoke_check() -> None:
    cfg = load_args(_CONFIG)
    cfg = dc.replace(
        cfg,
        run_name="validate_f5_smoke",
        num_eps=4,
        inference_server=True,
        mcts_config=dc.replace(cfg.mcts_config, num_mcts_sims=40),
    )
    _game, nnet = instantiate_game_and_network(cfg)
    ckpt = "validate_f5_init.pth.tar"
    nnet.save_checkpoint(filename=ckpt)
    examples, stats = run_self_play_episodes_parallel(cfg, generation=0, checkpoint_path=ckpt, num_workers=2)
    n_examples = sum(len(e) for e in examples)
    print(f"[smoke] server-mode self-play completed: {len(examples)} episodes, {n_examples} training examples")
    assert len(examples) == 4 and n_examples > 0, "server-mode self-play produced no examples"
    print("[smoke] VERDICT: server-mode self-play runs end-to-end on GPU")


if __name__ == "__main__":
    print("=== F5 per-board equivalence ===")
    equivalence_check()
    print("\n=== F5 integration smoke (server-mode self-play) ===")
    smoke_check()
    print("\nDONE")
