# Numba Hot-Path — Speedup Tracker

Running before/after for the [numba-hot-path](../plans/archive/numba-hot-path.md) plan (archived,
complete). Companion to the plan and the
[MCTS subplan](../plans/archive/numba-hot-path-mcts-subplan.md).

**Method.** `scripts/profile_self_play.py --mode timing --games 2 --seed 42`, single-process
(no worker pool — clean per-game attribution, no IPC/contention noise). Box `gpu-linux`
(RTX 3060 Ti, i5-13600KF), CUDA torch 2.10+cu128, numpy 2.4.3, config
`blokus_run2_bignet.json` (128f×8b, 800-cap branching sims, K=16). "Before" = the kernel
disabled (`ALPHABLOKUS_NUMBA_MOVEGEN=0`, i.e. today's pure-Python F2 path); "after" = enabled.
Same machine, same torch, same seed → the **ratio** is the trustworthy number (per-game wall
is noisy; averages shown).

## Single-worker speed (the tracker)

Same-session run on 2026-06-29/30 (checkpoints measured back-to-back on the box; bignet
config, seed 42, 2 games each).

> **Methodology caveat (found 2026-06-30).** `profile_self_play.py` seeds numpy but **not
> torch**, so each process gets a different random-init net → different games → different
> move/sim counts run-to-run (baseline 55 moves / 21.1k sims, N5 60 / 21.5k, etc.). That makes
> raw **wall/game** noisy across checkpoints. The confound-free metric is **wall per sim**
> (normalises out game length + sim count); it's the authoritative attribution below. Raw
> wall/game is kept as the "feel" number but read the µs/sim column for code-speed gains.
> A proper fix — seed the net in the profiler for reproducible games — is logged as a harness
> to-do; until then, µs/sim is the honest comparison.

| Stage | wall/game | total sims | **µs/sim** | speedup (µs/sim) | step | inference share |
|-------|-----------|-----------|------------|------------------|------|-----------------|
| **Baseline** (`bench/baseline`, pre-Numba) | 8.01 s | 21,102 | **759** | 1.00× | — | 18.9% |
| **+ N2** (`bench/n2`, Numba move-gen) | 5.18 s | 22,788 | **455** | **1.67×** | 1.67× | 32.1% |
| **+ N3/N4.1** (`bench/n3-n4.1`, per-node arrays) | 3.78 s | 22,732 | **332** | **2.29×** | 1.37× | 43.6% |
| **+ N5** (incremental side-danger) | 3.50 s | 21,516 | **325** | **2.33×** | **1.02×** | 44.2% |

**Total: 2.33× single-worker per sim.** The big wins are N2 (move-gen kernel, 1.67×) and N3
(select-gather removal, 1.37×). **N5 is marginal — ~1.02× (~2%)** — exactly the ~10%-slice /
≤1.1×-ceiling prediction (side-danger was ~1.4% of the wall; N5 removes roughly half). It's
correct and bit-identical, just small.

The inference share climbing 18.9% → 44.2% is the headline: the bottleneck has **crossed over
from CPU search to GPU inference**. The CPU well is essentially dry — N5's marginal return
confirms further CPU micro-opt isn't worth it. The next real lever is the GPU side
(batched-inference re-architecture), not more of this — and the cheap IPC version of that
(the cross-worker server) is ruled out at this config too: see
[inference-server-bignet.md](inference-server-bignet.md). It must be **in-process** batching.

Earlier cross-session readings (2026-06-23) were 8.13 s / 4.93 s wall/game for baseline / N2 —
consistent with the same-session run; the box is stable.

## 16-worker production throughput (the number that matters)

Single-worker per-sim speed is *not* the production figure. Production self-play runs
**16 all-GPU workers** (`run_self_play_episodes_parallel`, the exact path Coach uses). Measured
via `scripts/bench_parallel.py` (one seeded checkpoint reused across runs → **identical games**:
both ran exactly 2,265 moves / 887,130 sims, so this is a pristine apples-to-apples comparison),
bignet config, 80 episodes, pool/spawn overhead included:

| Stage | games/s | sims/s | speedup |
|-------|---------|--------|---------|
| Baseline (pre-Numba) | 0.751 | 8,331 | 1.00× |
| **N5 (full N1–N5 stack)** | **0.947** | **10,499** | **1.26×** |

**Production speedup is 1.26× — far below the 2.33× single-worker per-sim.** Why: at 16 workers
the box is no longer CPU-bound — 16 processes contend for the one GPU, and at the bignet config
inference is ~44% of the single-worker wall. So a 2.3× faster CPU search only buys 1.26× more
games/s, because the **GPU is now the bottleneck**. This is the CPU→GPU crossover the plan
predicted, quantified at scale: the CPU-side well is dry, and the **next real lever is the GPU
side** (cross-game batched inference / native engine), not more CPU micro-opt.

It is still a real win — **+26% games/s** at production scale (the per-sim gains also matter more
as the net/sim count grow, and they free CPU for the eventual GPU re-architecture). The identical
move/sim counts across baseline and N5 are also an **end-to-end bit-identical proof** of the whole
N1–N5 stack across 16 workers.

## Benchmark checkpoints (git refs)

Each body of work is captured as a commit + lightweight tag so its speed gain can be
attributed later — check out the ref, rsync to the box, re-run the timing benchmark (same
config/seed). Tags are local to the `worktree-numba-hot-path` branch (not pushed).

| Stage | Tag | Commit | wall/game | Benchmarked? |
|-------|-----|--------|-----------|--------------|
| Baseline (pre-Numba) | `bench/baseline` | `a17ea94` | 759 µs/sim | ✅ 2026-06-29 |
| N1+N2 (Numba move-gen) | `bench/n2` | `6f4e642` | 455 µs/sim | ✅ 2026-06-29 |
| N3+N4.1 (per-node arrays) | `bench/n3-n4.1` | `5342f15` | 332 µs/sim | ✅ 2026-06-29 |
| N4.2+N5 (shim removal + incremental danger) | *(tag on commit)* | *(uncommitted)* | 325 µs/sim | ✅ 2026-06-30 |

**To benchmark a checkpoint later** (from this worktree):

```bash
git checkout <tag>                       # e.g. bench/n3-n4.1  (detached HEAD)
rsync -az --delete --exclude .venv --exclude .git --exclude temp --exclude __pycache__ \
      ./ gpu-linux:/tmp/numba-bench/      # carries the untracked bench config too
ssh gpu-linux 'export PATH=$HOME/.local/bin:$PATH; cd ~/AlphaBlokus && uv pip install -q numba'
ssh gpu-linux 'cd /tmp/numba-bench && PYTHONPATH=/tmp/numba-bench \
   ALPHABLOKUS_NUMBA_MOVEGEN=1 ~/AlphaBlokus/.venv/bin/python \
   scripts/profile_self_play.py --config run_configurations/blokus_run2_bignet.json \
   --mode timing --games 2 --seed 42'
git checkout worktree-numba-hot-path      # return
```

The benchmark config (`run_configurations/blokus_run2_bignet.json`) is kept as an **untracked
file in the worktree** so it survives `git checkout` and rsyncs to the box with every checkpoint
(the box's own copy had drifted — missing fields — so don't `cp` from `~/AlphaBlokus`). The
baseline checkpoint has no Numba code (pure-Python F2 + dict MCTS), so the flag is a no-op there;
the N2/N3 checkpoints toggle the kernel with `ALPHABLOKUS_NUMBA_MOVEGEN` (1 = on). Behaviour is
identical across checkpoints at a fixed seed, so the only thing that changes row-to-row is the
wall-clock — exactly the attribution we want.

Single-worker wall/game is what we track step-to-step; the production figure is the
16-worker games/s (measured in N6). Because self-play is CPU-bound, the single-worker ratio
carries over to games/s until the GPU becomes the bottleneck — which is what the rising
inference share is tracking toward.

## What N2 changed

The Numba move-gen kernel (`_fill_mask_kernel` / `_has_any_move_kernel`) crushed the ~40%
move-gen slice. Micro-benchmark of the slice itself (`valid_move_masking`, mid-game position,
Mac CPU): **244 µs → 10.3 µs (~24×)**. At the whole-game level that 24× on a 40% slice yields
the 1.65× observed (Amdahl: move-gen ~40% → 1/(0.6 + 0.4/24) ≈ 1.62×, matching).

The phase split confirms the mechanism: NN inference is unchanged in absolute terms (~3.1–3.3 s
for 2 games) but its **share rose 19% → 33%** as the CPU wall shrank. The remaining ~65%
"other search" is now the `_select_action` dict-gather (~32% of the original wall, the N3
target) plus backprop/transition.

## Bit-identical confirmation

N2 is exact (integer/boolean kernel): `test_movegen_equivalence` (5,000 stratified positions),
full `tests/test_blokusduo/` (160 passed), and `test_mcts`/`test_parallel_self_play`/
`test_batched_inference` (32 passed) all green at a fixed seed.

## N1 environment notes (box)

`numba==0.65.1` / `llvmlite==0.47.0` install clean into the box's existing CUDA-torch venv
(numpy 2.4.3, torch 2.10+cu128); `import torch` + `import numba` coexist; `@njit(cache=True)`
persists compiled code to `__pycache__` so the 16 per-generation self-play workers load the
cache rather than recompiling. (Benchmark run reused the box venv from a scratch copy of the
code; the box's own working repo was not modified.)
