# Cross-Worker Inference Server — Still Slower at the Big Net (2026-06-29)

Re-test of the cross-worker batched-inference server (the old "F5") at the
**bignet** production config, after the Numba/array CPU work (N1–N5) pushed GPU
inference to ~44% of single-worker self-play. Companion to
[linux-throughput-rebaseline.md](linux-throughput-rebaseline.md) (which rejected
the server at the **small** net) and [numba-hot-path-results.md](numba-hot-path-results.md).

## The question

The original "server not worth it" verdict was measured at **64f×4b / 300 sims**,
where inference was ~10% and the GPU sat idle — so cross-worker batching had no
GPU-efficiency to capture and its IPC overhead was pure loss. After N1–N5, at
**128f×8b / 800 sims**, inference is ~44% single-worker and the GPU is the
bottleneck at 16 workers (the single-worker 2.33×/sim collapses to 1.26×
production precisely because of GPU contention). Intuitively that should be when
pooling all workers' leaves into one big GPU batch finally pays off. Does it?

## Method

`scripts/bench_parallel.py`, 16 workers, bignet config, 80 episodes, pool/spawn
overhead included. One seeded checkpoint reused across both modes → **identical
games** (both ran exactly 2,265 moves / 887,130 sims), so this is an
apples-to-apples comparison; only the inference routing differs.

- **all-GPU** (`worker_cuda: true`, `inference_server: false`): each worker holds
  its own net on the GPU and does its own batch-16 `predict_batch`.
- **server** (`worker_cuda: false`, `inference_server: true`, `server_max_batch: 0`
  → auto 16×16 = 256, `server_max_wait_ms: 5`): one process owns the GPU net;
  workers ship leaves over shared memory; the server batches them and routes
  results back. Config: `run_configurations/blokus_run2_bignet_server.json`.

## Result

| Mode (16 workers, identical games) | games/s | sims/s | vs all-GPU |
|------------------------------------|---------|--------|------------|
| **all-GPU** | **0.947** | 10,499 | 1.00× |
| cross-worker server | **0.686** | 7,606 | **0.72× (28% slower)** |

**The server is decisively slower — even at the big net.** The small-net verdict
holds across both regimes.

## Why — the corrected mental model

The tempting reasoning is "one 256-board batch beats sixteen 16-board batches."
That ignores what all-GPU mode already gets **for free: concurrency.** The 16
independent CUDA contexts naturally overlap — while worker A's kernel runs,
worker B's host→device copy happens and worker C runs CPU tree-search. Sixteen
overlapping pipelines keep the GPU busy without any explicit batching.

The server **centralises** inference into a **single serialized process** and
adds, per leaf-batch:
- an **IPC round-trip** (encode → shared-memory channel → server → GPU → results
  back → worker), and
- a **synchronization point** (workers block while the server assembles a batch,
  up to `server_max_wait_ms`).

That destroys the cross-worker overlap and introduces a single chokepoint. The
larger 256-board forward is more GPU-efficient in isolation, but it doesn't
recover the lost concurrency plus the IPC/sync cost. Net: slower. The effect
scales with net size too, so the bigger net doesn't flip it.

## Implication

The cheap lever — the **existing IPC server** — is a dead end, now confirmed in
*both* the small-net and big-net regimes. The real GPU-side win needs
**in-process** batched inference: many games advanced in lockstep within one
process, sharing memory, batched onto the GPU with **no IPC and no per-worker
serialization** (vectorised-MCTS / native-engine architecture). The server was
the IPC-shortcut version of that idea, and IPC is exactly what kills it.

**Do not** reopen the cross-worker server for either net size. Point the next
performance effort at the in-process rewrite.
