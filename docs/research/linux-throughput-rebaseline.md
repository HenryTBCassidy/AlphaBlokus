# Linux Throughput Re-baseline — Findings (2026-06-16)

Durable write-up of the self-play throughput investigation done on native Linux,
the companion to [self-play-speed-investigation.md](self-play-speed-investigation.md)
and the evidence behind **P3** in
[../plans/self-play-throughput.md](../plans/self-play-throughput.md). Everything here
was measured on the box (RTX 3060 Ti 8 GB; i5-13600KF 14C/20T; 32 GB) at commit
`6cfa88e`, 150-episode self-play, production net (64f×4b, 300 sims), random-init
weights, fixed `seed=42`. Timed via `run_self_play_episodes_parallel` (the exact
path `core.coach` uses), spawn/pool overhead included.

> **Why this doc exists:** the original roadmap evidence was Windows-era and turned
> out misleading. The numbers below are controlled within-session Linux measurements
> — trust these over any Windows comparison.

---

## The three worker modes (what they actually do)

All run **N worker processes** playing self-play; they differ only in *where each
worker runs the per-leaf neural-net inference* MCTS needs.

| Mode | Flags | Inference runs… |
|------|-------|-----------------|
| **all-GPU** | `worker_cuda=true`, `inference_server=false` | each worker loads **its own net on the GPU** (N independent CUDA contexts, ~220 MiB each); tree search on its CPU core |
| **cpu-only** | `worker_cuda=false`, `inference_server=false` | each worker builds its net **on CPU**, inference on CPU; GPU idle |
| **server** | `inference_server=true` | **one** central process holds a single GPU net; workers ship leaf-eval requests over IPC, it batches them onto the GPU (old "F5") |

Code: `_worker_net_config` / `_server_enabled` in `core/parallel_self_play.py`.

## R1 — worker-config sweep

| N | all-GPU | cpu-only | server |
|----|---------|----------|--------|
| 8 | 1.18 | — | — |
| 10 | 1.33 | — | — |
| 12 | 1.39 | 0.66 | — |
| 14 | 1.56 | — | — |
| 16 | **1.59** | 0.73 | 1.20 |
| 18 | 1.61 | — | — |
| 20 | 1.63 | 0.76 | 1.34 |
| 24 | — | — | 1.45 |

(games/s). Peak VRAM all-GPU: 3.97 GB @ N=18, 4.38 GB @ N=20.

**Conclusions:** all-GPU wins decisively; VRAM is *not* the cap (the "8-worker cap"
was a Windows ghost); the real limit is the **20 CPU threads** — throughput flattens
at N≈16. **Production default: ~16 GPU workers** (leaves cores for main/OS). Hybrid
(mixed GPU+CPU workers) is unnecessary — every worker is CPU-core-bound regardless of
inference device, so there's no idle resource for CPU workers to fill.

## Within-worker batching (`mcts_batch_size` = K) — still a huge win

Sweep at all-GPU N=16:

| K | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|----|----|
| games/s | **0.30** | 1.00 | 1.34 | 1.59 | 1.65 |

K=16 is **~5.2× faster than K=1**. K=32 only +4% over K=16 (and adds virtual-loss
search distortion), so **K=16 is the right default**, maybe nudge to 32.

**Why batching helps — and why my "GPU is cheap so it won't matter" guess was wrong.**
The mistake was conflating *"GPU compute is cheap"* (true — tiny net) with *"getting
data to/from the GPU is cheap"* (false). With K=1 and 16 worker processes, each worker
fires a **separate tiny GPU inference per leaf** — 16 processes hammering one GPU with
a flood of one-board calls. Each call costs a fixed round-trip (Python→torch dispatch,
CPU→GPU copy, kernel launch, GPU→CPU copy), and the driver serializes them across
processes. The GPU sits idle between tiny serialized round-trips; **overhead ×
call-count**, not FLOPs, is the cost. K=16 collects 16 leaves per worker (via virtual
loss) into one `predict_batch` → 16× fewer round-trips → overhead amortized → 5×.

So batching matters *more* in our overhead/CPU-bound regime, not less. It's not about
saturating GPU compute; it's about slashing the number of expensive round-trips.

## Cross-worker batching (the inference server) — still not worth it

The server batches across *different* workers. But within-worker K-batching already
cut each worker's GPU calls 16×, and the GPU compute was never the bottleneck — so
bigger batches don't speed anything up. Meanwhile the server **replaces each worker's
direct GPU round-trip with an IPC round-trip** (serialize boards → channel → server →
GPU → results back) plus a synchronization point (workers wait for the server to
assemble a batch). Trading a GPU round-trip for an IPC round-trip + sync nets out
neutral-to-worse — hence server (1.20–1.45) < all-GPU (1.59), matching F5's old 0.99×.

**Net:** keep within-worker K-batching (essential); don't use the cross-worker server.
The two were always distinct — within-worker batching was a win before (F3) and still
is; the cross-worker server never won.

## Verification runs

**Reproducibility (all-GPU, pinned, K=16):** gpu8 measured 1.194 / 1.180 / 1.204
across three separate sessions (~2% spread); gpu16 1.594 / 1.563. The measurement is
stable; the monotonic sweep curve is not noise.

**Thread-pinning isolation (`torch.set_num_threads(1)`, `6a9ac17`):** disabling it
changed all-GPU little — gpu8 1.20→1.13, gpu16 1.56→1.60 (≈ noise). So thread-pinning
is **not** a big lever for GPU workers (their CPU time is MCTS, not torch ops; it
matters far more for cpu-only mode). It was **not** the hidden confound behind the
Windows→Linux gap.

## The Windows→Linux gap — honest status

The original "Windows gpu8 = 0.756" is an **uncontrolled historical anchor**: between
it and the Linux numbers, four hot-path commits landed (thread-pin, sparsify,
forkserver, P2). We've now **ruled out** the code-change confounds — P2 is ~0%,
thread-pin is ~0–6% for GPU workers, sparsify is bit-identical/memory-only. That
leaves the Windows→Linux all-GPU gap (~0.76 → ~1.18, ~1.55×) as either:

1. **Genuine OS/driver** — consumer GeForce on Windows is stuck on the **WDDM** driver
   model (higher per-kernel-launch latency); the K-sweep proves GPU round-trip cost is
   the dominant factor, and WDDM makes every round-trip pricier. Plus Windows is
   spawn-only with heavier IPC. A real 1.3–1.6× is plausible.
2. **Windows measurement was warmup-inflated** — the roadmap flagged the Windows runs
   as "warmup-heavy / muddied."

We can't separate these without re-measuring on Windows (the box now boots Linux), so
**we don't cite a Windows→Linux multiplier as fact.** The actionable Linux numbers
above stand on their own — controlled, reproducible, same-code, same-session.

## What's robust vs. open

**Robust (controlled, reproducible):** all-GPU ≫ cpu-only > server; ~16 GPU workers is
the sweet spot; cores (not VRAM) are the cap; within-worker K-batching is ~5× and
essential; thread-pin is minor for GPU workers.

**Open:** the exact Windows→Linux mechanism (won't chase — box is Linux now); the deep
per-function time profile (py-spy/Scalene) and memory-over-generations check (R2/R3)
remain if we want to hunt further per-worker speedups.
