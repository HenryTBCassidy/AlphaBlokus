# Training Time Estimates

Estimated wall-clock time for Blokus Duo self-play training under various configurations. Originally derived from Mac CPU profiling (March 2026); updated 2026-05-18 with **measured** RTX 3060 Ti numbers from `scripts/mcts_profiling.py` on the home PC.

**Key bottleneck:** Move generation, which is pure-Python and CPU-bound. Across all profiled runs, move generation is **65-72% of MCTS search time**, neural net inference is **17-27%**, and everything else (UCB selection, game-ended checks) is the remainder. **The GPU only accelerates the smaller portion** — total wall clock is bound by Python-side move gen on the CPU.

## Surprising finding: the home PC is slower than the Mac for self-play

The earlier estimates in this doc projected 3080 Ti speedups assuming inference was the bottleneck. That premise was wrong. Measured Blokus self-play on the **small profiling net** (5 games, 50 sims/move):

| Machine | ms/sim | ms/move | Inference fraction | Move-gen fraction |
|---------|--------|---------|--------------------|-------------------|
| Mac (Apple Silicon, CPU inference) | 2.96 | 74.0 | 26% | 67% |
| Home PC (RTX 3060 Ti, CUDA inference) | **4.95** | **247.4** | 17% | **72%** |

The PC is **~1.7× slower per simulation** than the Mac. The GPU does shave inference time (smaller % of total), but the PC's CPU is enough slower at move generation that the net effect is a slowdown. WSL2 syscall overhead and a slower CPU vs the Mac M-series likely both contribute.

**Implication:** the GPU PC only pays off when nets get large enough that inference dominates — which happens at Serious/Paper-scale net sizes (256+ filters, 10+ residual blocks). For Minimal/Moderate configs, the Mac is faster.

---

## Hardware

| Machine | GPU | Measured ms/sim (small profiling net) | Inference fraction | Notes |
|---------|-----|---------------------------------------|--------------------|-------|
| MacBook (M-series) | CPU only | 2.96 | 26% | Faster overall for small/medium nets — single-thread perf wins |
| Personal PC | RTX 3060 Ti | **4.95** | 17% | GPU under-utilised; CPU is the bottleneck for current configs |

Inference cost for **larger** nets has not been directly measured at full training sim counts; the projections below assume inference scales roughly with parameter count (rough, not measured).

---

## Configurations

### Minimal — verify the pipeline works

| Parameter | Value |
|-----------|-------|
| Net | 128 filters, 5 residual blocks (8.6M params) |
| MCTS sims | 100 |
| Self-play games/gen | 10 |
| Arena games | 20 |
| Generations | 50 |

| Hardware | Time/move | Time/game | Total | Source |
|----------|-----------|-----------|-------|--------|
| MacBook CPU | ~430ms | ~13s | **~5 hours** | March 2026 measurement |
| RTX 3060 Ti | ~540ms | ~16s | **~6.5 hours** | Projected from measured 4.95 ms/sim + ~1.5× inflation for the bigger net's heavier inference |

### Moderate — net learns something

| Parameter | Value |
|-----------|-------|
| Net | 128 filters, 5 residual blocks (8.6M params) |
| MCTS sims | 200 |
| Self-play games/gen | 25 |
| Arena games | 40 |
| Generations | 100 |

| Hardware | Time/move | Time/game | Total | Source |
|----------|-----------|-----------|-------|--------|
| MacBook CPU | ~861ms | ~26s | **~1.9 days** | March 2026 measurement |
| RTX 3060 Ti | ~1080ms | ~32s | **~2.4 days** | Projected — PC roughly tracks Mac at moderate sim counts |

### Serious — aim to beat Pentobi

| Parameter | Value |
|-----------|-------|
| Net | 256 filters, 10 residual blocks (19M params) |
| MCTS sims | 400 |
| Self-play games/gen | 50 |
| Arena games | 40 |
| Generations | 200 |

| Hardware | Time/move | Time/game | Total | Source |
|----------|-----------|-----------|-------|--------|
| MacBook CPU | ~3.9s | ~116s | **~24 days** | March 2026 measurement |
| RTX 3060 Ti | ~3.0s | ~90s | **~18 days** | Projected — bigger net (19M params) means inference is finally a meaningful slice, GPU helps |

### Paper-scale — full AlphaZero equivalent

| Parameter | Value |
|-----------|-------|
| Net | 256 filters, 19 residual blocks (30M params) |
| MCTS sims | 800 |
| Self-play games/gen | 100 |
| Arena games | 400 |
| Generations | 200 |

| Hardware | Time/move | Time/game | Total | Source |
|----------|-----------|-----------|-------|--------|
| MacBook CPU | ~13s | ~392s | **~1.2 years** | March 2026 measurement |
| RTX 3060 Ti | ~8s | ~240s | **~165 days** | Projected — GPU pays off most here (30M params), but still not feasible on this hardware |

The home PC genuinely helps at Paper-scale because inference becomes a meaningful fraction of total time. But "165 days" is still not feasible on a single 3060 Ti — for Paper-scale we'd want either parallel self-play, a real datacenter GPU, or scope reduction.

---

## AlphaZero Paper Reference

For comparison, the original AlphaZero paper (Silver et al., 2018) used:

- **Hardware:** 5,000 TPUs for self-play + 16 TPUs for training
- **Network:** 256 filters, 19 residual blocks
- **MCTS:** 800 sims/move
- **Training:** 700,000 steps at batch size 4,096
- **Arena:** 400 games per evaluation
- **Result:** Chess mastered in 9 hours, Shogi in 12 hours, Go in 13 days

The massive parallelism (5,000 TPUs generating games simultaneously) is what makes this tractable. Our single-machine setup generates games sequentially.

---

## Impact of Move Generation Optimisation

Move generation is 65-72% of MCTS search time on both Mac and PC. The current implementation is a naive Python loop. Potential speedups and their impact on the "Serious" config (RTX 3060 Ti, ~18 days baseline):

| Move gen speed | Serious total | Speedup vs baseline |
|----------------|---------------|---------------------|
| 3.6ms (current measured PC) | ~18 days | baseline |
| 1.5ms (2× faster) | ~9 days | ~2× |
| 0.5ms (6× faster) | ~4 days | ~4.5× |
| 0.1ms (30× faster, e.g. bitboard) | ~2 days | ~9× |

Optimisation approaches (deferred to `docs/plans/move-gen-further-optimisation.md`): pre-computed piece corners, numpy vectorisation, bitboard representation. These are the highest-leverage performance work because the PC GPU is currently bottlenecked on Python.

---

## Assumptions

- 30 moves per game (from random game analysis, 100 games)
- 80% of MCTS simulations expand a new leaf (from profiling data)
- Training time is negligible compared to self-play (~0.1% of total)
- No symmetry augmentation yet (would improve training quality, not speed)
- Single-threaded self-play (no parallel game generation)
- GPU inference estimates are rough (5-8x CPU speedup for this model size)

*Originally compiled March 2026 (Mac CPU only, PC numbers estimated). Updated 2026-05-18 with measured 3060 Ti baseline (5 games, 50 sims/move, small profiling net) from `temp/mcts_profiling/pc_3060ti/`. Re-run `scripts/mcts_profiling.py` after move-gen optimisations or net-size changes to refresh these numbers.*
