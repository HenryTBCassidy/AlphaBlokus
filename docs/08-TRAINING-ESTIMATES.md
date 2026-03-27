# Training Time Estimates

Estimated wall-clock time for Blokus Duo self-play training under various configurations. Based on profiling data from `scripts/mcts_profiling.py` (March 2026).

**Key bottleneck:** Move generation at ~2.9ms per leaf expansion (pure Python, CPU-bound, unaffected by GPU). This dominates self-play time at all scales.

---

## Hardware

| Machine | GPU | Inference speed (small/medium/large net) |
|---------|-----|------------------------------------------|
| MacBook (M-series) | CPU only | 2.1 / 8.5 / 16.4 ms |
| Personal PC | RTX 3080 Ti | ~0.3 / ~1.0 / ~2.0 ms (estimated) |

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

| Hardware | Time/move | Time/game | Total |
|----------|-----------|-----------|-------|
| MacBook CPU | 430ms | 13s | **5 hours** |
| RTX 3080 Ti | 279ms | 8s | **3.5 hours** |

### Moderate — net learns something

| Parameter | Value |
|-----------|-------|
| Net | 128 filters, 5 residual blocks (8.6M params) |
| MCTS sims | 200 |
| Self-play games/gen | 25 |
| Arena games | 40 |
| Generations | 100 |

| Hardware | Time/move | Time/game | Total |
|----------|-----------|-----------|-------|
| MacBook CPU | 861ms | 26s | **1.9 days** |
| RTX 3080 Ti | 559ms | 17s | **1.3 days** |

### Serious — aim to beat Pentobi

| Parameter | Value |
|-----------|-------|
| Net | 256 filters, 10 residual blocks (19M params) |
| MCTS sims | 400 |
| Self-play games/gen | 50 |
| Arena games | 40 |
| Generations | 200 |

| Hardware | Time/move | Time/game | Total |
|----------|-----------|-----------|-------|
| MacBook CPU | 3.9s | 116s | **24 days** |
| RTX 3080 Ti | 1.4s | 41s | **8.5 days** |

### Paper-scale — full AlphaZero equivalent

| Parameter | Value |
|-----------|-------|
| Net | 256 filters, 19 residual blocks (30M params) |
| MCTS sims | 800 |
| Self-play games/gen | 100 |
| Arena games | 400 |
| Generations | 200 |

| Hardware | Time/move | Time/game | Total |
|----------|-----------|-----------|-------|
| MacBook CPU | 13s | 392s | **1.2 years** |
| RTX 3080 Ti | 3.4s | 101s | **117 days** |

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

Move generation accounts for ~70% of search time. The current implementation is a naive Python loop. Potential speedups and their impact on the "Serious" config (RTX 3080 Ti):

| Move gen speed | Serious total | Speedup |
|----------------|---------------|---------|
| 2.9ms (current) | 8.5 days | baseline |
| 1.5ms (2x faster) | 5 days | 1.7x |
| 0.5ms (6x faster) | 2.5 days | 3.4x |
| 0.1ms (30x faster, e.g. bitboard) | 1.5 days | 5.7x |

Optimisation approaches (from plan M6): pre-computed piece corners, numpy vectorisation, bitboard representation.

---

## Assumptions

- 30 moves per game (from random game analysis, 100 games)
- 80% of MCTS simulations expand a new leaf (from profiling data)
- Training time is negligible compared to self-play (~0.1% of total)
- No symmetry augmentation yet (would improve training quality, not speed)
- Single-threaded self-play (no parallel game generation)
- GPU inference estimates are rough (5-8x CPU speedup for this model size)

*Last updated: March 2026. Re-run `scripts/mcts_profiling.py` after optimisations to update these numbers.*
