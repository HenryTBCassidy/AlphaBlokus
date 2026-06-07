

# Lean Self-Play Workers — CPU workers + central GPU (S1)

Implements **S1** from the [parallelism options decision doc](../research/parallelism-options.md): make the game-playing pool workers **lightweight (CPU-only, no per-worker CUDA context)** so we can run ~20 of them on the 20-core PC, with the GPU used by training (and optionally a single central inference process). 

**Why:** each worker today loads its own ~ 2.5 GB PyTorch+CUDA stack (the net is 1.5 MB), so 8 workers ≈ 20 GB — which **caps the worker count at 8 despite 20 cores and OOM-crashed a run at the 28 GB cap** (at gen 4). Self-play is **~86% of the training cycle and CPU-bound** ([profiling report](../research/profiling-report.md)), so the lever is *more concurrent workers* — which this unlocks. Removing the duplication also removes the crash.

**Target outcome:** ~20 lean workers using all cores, RAM flat well under the cap, the OOM gone, and self-play throughput up vs the 8-worker baseline — i.e. a crash-proof run that trains faster.

**Two supported inference strategies, kept as config flags** (per the design directive — the best choice shifts with net size + hardware, so we maintain *both* and flip the flag, never deleting one):

- **(a) CPU-in-worker** — `worker_cuda: false`, no server: each worker runs the net on its own CPU. Simplest, zero IPC.
- **(b) central GPU server** — `inference_server: true`: workers are netless and batch their leaf evaluations to one GPU process.

L3 benchmarks them to pick the sensible **default** for the *current* small 64f×4b net on the PC — but neither is removed. Profile prior: inference is ~33% and transfer-bound and the net is tiny → **(a) likely wins now**; a much bigger net (or a beefier GPU) later likely flips the default to **(b)**. L3 decides the default on numbers, not faith.

---

## Checklist


| #   | Item                                                                                                                                                                                                                                      | Effort     | Priority | Done |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | ---- |
| L1  | **Worker nets on CPU** — pool workers (self-play/arena/elo) build their net on CPU regardless of the global `cuda` flag; training stays on GPU in the main process                                                                        | 0.5 day    | High     |      |
| L2  | **Raise worker count** toward core count (e.g. 16 on the 20-core PC) now that workers are ~0.5 GB                                                                                                                                         | 15 min     | High     |      |
| L3  | **Benchmark to pick the default** — s/game + total + peak RAM for: baseline 8 GPU workers · ~16–20 CPU-in-worker · ~16–20 CPU + central GPU server. Choose the **default**; both paths kept either way                                    | 0.5 day    | High     |      |
| L4  | **Maintain BOTH inference paths (flag-selectable)** — (a) `worker_cuda` CPU-in-worker (pin intra-op threads/worker) **and** (b) `inference_server` central GPU; ensure both work + are tested. Optional: fork+CoW for the move-gen tables | 0.5–2 days | High     |      |
| L5  | **Correctness gate** — parallel determinism preserved (workers=1 ≡ N at a fixed device/seed); MCTS/game logic unchanged; full suite green                                                                                                 | 0.5 day    | High     |      |
| L6  | **Memory + OOM benchmark** — full config at ~20 workers, run **past gen 3–4** (the crash point): RAM flat, all cores used, OOM gone, throughput vs the 8-worker baseline                                                                  | 1 hr + run | High     |      |
| L7  | **Launch the real 15-gen run** — results + final verification                                                                                                                                                                             | run        | High     |      |


---

## L1. Worker nets on CPU

Today every pool worker builds its net from `config.net_config.cuda` (`config.py:54`) inside `_worker_init_self_play` (`parallel_self_play.py:130`) and the two-net arena/elo inits — so all 8 workers create a CUDA context. The change: **the game-playing workers always build their net on CPU**, while the **main process keeps the GPU** for the training step (`base_wrapper.train`). Cleanest seam: add a `worker_device`/`self_play_cuda` notion (default CPU) and have the worker inits force `cuda=False` when constructing/loading the worker net, leaving `net_config.cuda` (the training device) untouched. Net is 1.5 MB — loading it to CPU is instant; no CUDA context → ~0.5 GB/worker instead of ~2.5 GB.

## L2. Raise worker count

Bump `num_parallel_workers` (config.py:185; configs currently 8) toward the core count — start at **16** on the 20-core PC (leave a few cores for the main process + OS). Now safe because workers are cheap (16 × ~0.5 GB ≈ 8 GB + the GPU training net, comfortably under 28 GB).

## L3. Benchmark the inference path *(decision gate)*

Use the existing `scripts/benchmark_phases.py` / `scripts/profile_self_play.py` harness to measure **self-play s/game, total wall, and peak RSS** for three configs:

1. **Baseline:** 8 GPU workers (today).
2. **(a) CPU-in-worker:** ~16 CPU-only workers, each running inference on its own CPU net.
3. **(b) Central GPU server:** ~16 CPU-only workers + one GPU process serving batched inference (F5; gate relaxed — see L4).

Decision: pick the **fastest config that fits memory**. Expected (to confirm): (a) likely wins — CPU inference of the tiny net is cheap, ~2× more workers beats the per-call slowdown, and it has zero IPC; (b) only wins if central batching materially beats CPU inference, which F5's prior null-result makes doubtful. Record the numbers + the choice here.

## L4. Maintain both inference paths (flag-selectable)

Both strategies stay in the codebase, selected by config — neither is deleted (design directive: the optimal choice shifts with net size + hardware).

- **(a) CPU-in-worker** (`worker_cuda: false`): set `torch.set_num_threads(1)` per worker so N workers don't oversubscribe the cores. *Optional extra memory win:* on Linux/WSL, `fork` the CPU-only workers so the read-only move-gen tables + interpreter share via copy-on-write (CUDA can't fork, but these workers never touch it).
- **(b) central GPU server** (`inference_server: true`): the existing F5 server (`inference_server.py`/`inference_channel.py`) owns the GPU net; workers are netless `InferenceClientNet`s. Already wired; the server uses the (GPU) training device, workers build no net. Validate end-to-end.

L3's numbers set the recommended **default** (likely (a) for the current 64f×4b net), but both are kept and tested so we flip the flag when the net grows or the hardware changes — a bigger net / beefier GPU favours (b).

## L5. Correctness gate

Switching self-play inference from GPU to CPU changes floats slightly, so games won't be bit-identical to old GPU runs — that's expected and fine. What must hold: **(i)** parallel determinism preserved — `tests/test_core/test_parallel_self_play.py` (workers=1 ≡ workers=N at the same device/seed) still passes; **(ii)** the MCTS + move-gen + game logic is unchanged (only the net's device moved); **(iii)** the full suite is green. Add a check that a 1-worker CPU run and an N-worker CPU run produce identical examples at a fixed seed.

## L6. Memory + OOM benchmark *(the proof)*

Restore the full config at ~16–20 workers and run **past generation 3–4** (where it OOM-crashed). Confirm on W&B: `proc.memory.rssMB` stays flat (≈ workers × ~0.5 GB + the GPU training net, well under 28 GB), `system.memory_percent` never approaches 100, all cores are busy during self-play, and self-play s/game improves vs the 8-worker baseline. This is the evidence the OOM is gone *and* throughput is up — measured, not asserted.

## L7. Launch the real 15-gen run

With L6 green, launch the full 15-gen × 1000-game run (the click-script). It both verifies the fix at scale and delivers the training results the whole effort has been blocked on.

---

## Sub-plans

- **[MCTS memory reduction](mcts-memory-reduction.md)** — benchmarking found the per-worker **MCTS tree (~2 GB)** is the real cap on how many workers fit, and it's the one cost no OS trick can share (private + mutated → CoW/shared-memory can't touch it). Shrinking it in code is **OS-agnostic** and is the biggest remaining lever for "use all the cores." Profile-first (M1–M2) then design the cuts (M3+).

## Notes / dependencies

- **Needs the PC reachable** (Tailscale on the PC is currently down) for L3, L6, L7 — they run on the GPU box.
- **Arena/Elo workers** get the same CPU treatment as self-play (they run the same MCTS play-loop via the pool); only the main-process training net stays on GPU.
- **Not a free-threading project.** This is plain multiprocessing with cheaper workers — zero new dependencies, works on Python 3.12 today. The free-threaded "S6" upgrade is deliberately deferred until a CUDA + free-threaded PyTorch wheel ships (see the decision doc) — it adds complexity for no net gain while the GPU must live in a separate process.
- **Scales forward:** the "thin workers + central inference" seam is exactly what a future AWS/Ray (S5) setup would put a network boundary at — so S1 is also the right first step toward the farm case.

