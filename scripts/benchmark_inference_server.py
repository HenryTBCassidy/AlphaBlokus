"""Benchmark self-play wall-clock + GPU utilisation, server-off vs server-on.

Run on the PC (CUDA), repo root:  uv run python -m scripts.benchmark_inference_server

Times ``run_self_play_episodes_parallel`` at production settings (8 workers, 300
sims, K=16) with the cross-worker inference server off (per-worker batching) then
on (cross-worker batching), and reports the self-play s/game speedup plus the GPU
utilisation each path achieves.
"""

from __future__ import annotations

import dataclasses as dc
import subprocess
import threading
import time

import numpy as np

from core.config import load_args
from core.game_factory import instantiate_game_and_network
from core.parallel_self_play import run_self_play_episodes_parallel

_CONFIG = "run_configurations/blokus_quicktest.json"
_NUM_WORKERS = 8
_NUM_EPS = 24
# The off-vs-on speedup ratio is independent of sim count (both paths use the
# same), so a lighter search keeps the benchmark fast.
_NUM_SIMS = 150

# Each phase runs as its own process (mode "off" / "on") so a single CUDA-
# initialised parent never spawns two pools back-to-back — that combination
# was destabilising WSL on this box. Result s/game is written to a file the
# orchestrating shell aggregates.


class _GpuSampler:
    """Background nvidia-smi sampler (GPU utilisation %)."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self.samples: list[int] = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                self.samples.append(int(out.stdout.strip().split("\n")[0]))
            except (ValueError, subprocess.SubprocessError):
                pass
            time.sleep(1.0)

    def __enter__(self) -> _GpuSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def summary(self) -> str:
        if not self.samples:
            return "n/a"
        return f"mean={np.mean(self.samples):.0f}%, max={np.max(self.samples)}% (n={len(self.samples)})"


def _time_run(cfg: object, checkpoint: str, label: str) -> float:
    with _GpuSampler() as gpu:
        start = time.perf_counter()
        examples, _stats = run_self_play_episodes_parallel(
            cfg, generation=0, checkpoint_path=checkpoint, num_workers=_NUM_WORKERS
        )
        elapsed = time.perf_counter() - start
    per_game = elapsed / cfg.num_eps  # type: ignore[attr-defined]
    n_ex = sum(len(e) for e in examples)
    print(f"[{label}] {cfg.num_eps} games in {elapsed:.1f}s = {per_game:.2f}s/game "  # type: ignore[attr-defined]
          f"| {n_ex} examples | GPU {gpu.summary()}")
    return per_game


def main(mode: str) -> None:
    cfg = load_args(_CONFIG)
    cfg = dc.replace(
        cfg,
        run_name="bench_f5",
        num_eps=_NUM_EPS,
        num_generations=1,
        mcts_config=dc.replace(cfg.mcts_config, num_mcts_sims=_NUM_SIMS),
    )
    checkpoint = "bench_f5_init.pth.tar"
    if mode == "off":
        # The "off" phase builds and saves the shared checkpoint both phases load.
        _game, nnet = instantiate_game_and_network(cfg)
        nnet.save_checkpoint(filename=checkpoint)

    print(f"=== F5 benchmark [{mode}]: {_NUM_WORKERS} workers, {cfg.mcts_config.num_mcts_sims} sims, "
          f"K={cfg.mcts_config.mcts_batch_size}, {_NUM_EPS} games ===")
    cfg = dc.replace(cfg, inference_server=(mode == "on"), server_max_wait_ms=1.0)
    per_game = _time_run(cfg, checkpoint, f"server-{mode.upper()}")
    with open(f"temp/bench_{mode}.txt", "w") as fh:
        fh.write(f"{per_game:.4f}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "off")
