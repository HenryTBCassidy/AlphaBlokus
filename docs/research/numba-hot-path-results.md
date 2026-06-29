# Numba Hot-Path — Speedup Tracker

Running before/after for the [numba-hot-path](../plans/numba-hot-path.md) plan. Updated as
each step lands so we can watch the self-play speedup climb. Companion to the plan and the
[MCTS subplan](../plans/numba-hot-path-mcts-subplan.md).

**Method.** `scripts/profile_self_play.py --mode timing --games 2 --seed 42`, single-process
(no worker pool — clean per-game attribution, no IPC/contention noise). Box `gpu-linux`
(RTX 3060 Ti, i5-13600KF), CUDA torch 2.10+cu128, numpy 2.4.3, config
`blokus_run2_bignet.json` (128f×8b, 800-cap branching sims, K=16). "Before" = the kernel
disabled (`ALPHABLOKUS_NUMBA_MOVEGEN=0`, i.e. today's pure-Python F2 path); "after" = enabled.
Same machine, same torch, same seed → the **ratio** is the trustworthy number (per-game wall
is noisy; averages shown).

## Single-worker wall/game (the tracker)

| Stage | wall/game | speedup vs baseline | inference share | notes |
|-------|-----------|---------------------|-----------------|-------|
| **Baseline** (pre-Numba, prod F2 path) | **8.13 s** | 1.00× | 19% | 2026-06-23, box |
| **+ N2** Numba move-gen kernel | **4.93 s** | **1.65×** | 33% | 2026-06-23, box |
| + N3 kill select gather | *(bench pending)* | *(proj ~2.5×)* | *(proj ~45%)* | code done + bit-identical (254 tests); box unreachable 2026-06-25 |
| + N5 board transition (optional) | *(pending)* | — | — | — |

> **N3 benchmark owed.** N3 (per-node array storage, select gather removed) is code-complete
> and bit-identical, but the box was unreachable on 2026-06-25/26 (off-LAN; devtunnel host
> service down; Tailscale Netskope-blocked). Re-run the timing benchmark once access is
> restored — same command, same config — to fill the N3 row. Expected: the ~65% "other
> search" from the N2 run drops sharply as the ~32%-of-original select gather disappears.

## Benchmark checkpoints (git refs)

Each body of work is captured as a commit + lightweight tag so its speed gain can be
attributed later — check out the ref, rsync to the box, re-run the timing benchmark (same
config/seed). Tags are local to the `worktree-numba-hot-path` branch (not pushed).

| Stage | Tag | Commit | wall/game | Benchmarked? |
|-------|-----|--------|-----------|--------------|
| Baseline (pre-Numba) | `bench/baseline` | `a17ea94` | 8.13 s | ✅ 2026-06-23 |
| N1+N2 (Numba move-gen) | `bench/n2` | `6f4e642` | 4.93 s | ✅ 2026-06-23 |
| N3+N4.1 (per-node arrays) | `bench/n3-n4.1` | *(this commit)* | — | ⏳ box access |

**To benchmark a checkpoint later** (from this worktree):

```bash
git checkout <tag>                       # e.g. bench/n3-n4.1  (detached HEAD)
rsync -az --delete --exclude .venv --exclude .git --exclude temp --exclude __pycache__ \
      ./ gpu-linux:/tmp/numba-bench/      # or via gpu-anywhere once the tunnel is up
ssh gpu-linux 'cd ~/AlphaBlokus && uv pip install numba; \
   cp run_configurations/blokus_run2_bignet.json /tmp/numba-bench/run_configurations/'
ssh gpu-linux 'cd /tmp/numba-bench && PYTHONPATH=/tmp/numba-bench \
   ALPHABLOKUS_NUMBA_MOVEGEN=1 ~/AlphaBlokus/.venv/bin/python \
   scripts/profile_self_play.py --config run_configurations/blokus_run2_bignet.json \
   --mode timing --games 2 --seed 42'
git checkout worktree-numba-hot-path      # return
```

The baseline checkpoint has no Numba code (pure-Python F2 + dict MCTS), so it needs no flag;
the N2/N3 checkpoints toggle the kernel with `ALPHABLOKUS_NUMBA_MOVEGEN` (1 = on). Behaviour
is identical across checkpoints at a fixed seed, so the only thing that changes row-to-row is
the wall-clock — exactly the attribution we want.

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
