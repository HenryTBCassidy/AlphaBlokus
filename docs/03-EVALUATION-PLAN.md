# AlphaBlokus — Evaluation Plan

## Overview

Evaluation for AlphaBlokus operates at three levels:

1. **Training diagnostics** — Is the network actually learning? Loss curves, policy accuracy, value calibration
2. **Generational strength tracking** — Is each new generation stronger? Elo estimation from arena results
3. **External benchmarking** — Can we beat Pentobi? Systematic evaluation against the strongest open-source Blokus AI

The AlphaZero paper (Silver et al., 2017) demonstrated that self-play training produces a monotonically increasing Elo curve. Replicating this curve is the primary signal that the system is working correctly.

---

## 1. Training Diagnostics

### Loss Monitoring

Every training generation produces per-batch metrics logged to pickle files and consolidated to Parquet:

| Metric | Formula | Target Behaviour |
|--------|---------|-----------------|
| Policy loss (L_π) | `-Σ π_target · log(π_pred) / N` | Decreasing over epochs; decreasing across generations |
| Value loss (L_v) | `Σ(v_target - v_pred)² / N` | Decreasing over epochs; converging toward ~0.5-0.7 |
| Total loss | `L_π + L_v` | Decreasing overall |

**What to watch for:**

- **Policy loss not decreasing across generations:** The network isn't learning move selection from MCTS targets. Check: MCTS simulation count too low, training data too noisy, learning rate too high
- **Value loss stuck near 1.0:** The network can't predict game outcomes. At random play, MSE between a random v ∈ [-1,1] and actual outcomes {-1,+1} has expected value ~1.33. Getting below 1.0 means the network is doing better than random
- **Value loss below 0.3:** Strong signal — the network is reliably predicting who wins from board positions
- **Loss spikes between generations:** New self-play data distribution shifts. Normal in early training, should stabilise
- **Policy loss decreasing but value loss stuck:** The network is memorising MCTS patterns without understanding position quality. Increase MCTS simulations to provide better value targets

### Policy Accuracy

Beyond loss, track whether the network's top-k predicted moves match MCTS's preferred moves:

| Metric | Description | Target |
|--------|-------------|--------|
| Top-1 match | Network's best move = MCTS's best move | >50% by generation 20 |
| Top-5 match | MCTS's best move in network's top 5 | >80% by generation 20 |
| Entropy of π | `H(π) = -Σ π·log(π)` | Decreasing over generations (more confident) |

High policy accuracy with low MCTS simulations means the network has internalised good move selection — the goal of the entire training process.

### Value Calibration

The value head should be well-calibrated: when it predicts v=0.7, the current player should win ~85% of the time (mapping from [-1,1] to [0,1]).

| Metric | Description | Target |
|--------|-------------|--------|
| Value-outcome correlation | Pearson r between v_pred and actual outcome | >0.5 by gen 20, >0.7 by gen 50 |
| Calibration curve | Binned predicted v vs actual win rate | Monotonically increasing, close to diagonal |
| Mean absolute error | Average \|v_pred - v_actual\| | <0.5 |

### Training Timing

Each phase of the training loop should be timed to identify bottlenecks:

| Phase | Metric | Expected Bottleneck |
|-------|--------|-------------------|
| Self-play (per game) | seconds/game | MCTS simulations × neural net inference |
| Self-play (per generation) | minutes/generation | num_eps × seconds/game |
| Neural net training | seconds/epoch | Forward/backward pass, data loading |
| Arena evaluation | minutes/evaluation | num_arena_matches × seconds/game |
| Total generation | minutes/generation | Sum of all phases |

The existing framework saves timing data to `{run_directory}/Timings/`. Key optimisation targets:

1. **MCTS simulation speed** — currently the bottleneck. Iterates all 17,837 actions per simulation
2. **Neural net inference latency** — called once per MCTS simulation. Batching would amortise this
3. **Self-play parallelism** — games are independent and could run in parallel (not yet implemented)

---

## 2. Generational Strength Tracking

### Arena Win Rates

After each training generation, the new network plays the previous best in an arena:

```
New network vs Old network
  - num_arena_matches games (e.g., 40)
  - Alternating sides (White/Black)
  - Full MCTS for both players
  - Accept new if win_rate > update_threshold (e.g., 55%)
```

Track per generation:
- Win/loss/draw counts
- Win rate with confidence intervals
- Acceptance/rejection decision
- Number of consecutive acceptances/rejections

**Warning signs:**
- Many consecutive rejections → training may have stalled, consider adjusting learning rate or MCTS parameters
- 100% win rate for new network → generations too far apart in strength, reduce training epochs
- ~50% win rate → network not improving meaningfully per generation

### Elo Rating System

Compute an Elo rating for each generation to produce the characteristic AlphaZero training curve:

```
After each arena match between generation i and generation j:

E_i = 1 / (1 + 10^((R_j - R_i) / 400))    # Expected score for i
S_i = actual result (1=win, 0.5=draw, 0=loss)
R_i_new = R_i + K × (S_i - E_i)            # Update rating
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Initial Elo | 1000 | Arbitrary baseline for generation 0 |
| K-factor | 32 | Standard for provisional ratings |
| Draws | 0.5 | Standard Elo draw handling |

**Expected trajectory:**
- Generations 1-10: Rapid Elo gain (learning basic moves)
- Generations 10-30: Steady improvement (learning piece interactions)
- Generations 30-50+: Diminishing returns (fine-tuning positional understanding)

### Game Length Analysis

As the network improves, game characteristics should change:

| Metric | Random Play | Weak Network | Strong Network |
|--------|-------------|-------------|----------------|
| Average game length | Short (few pieces placed) | Medium | Long (more pieces placed) |
| Pieces placed per game | Low | Medium | High (approaching 42 = all pieces) |
| Score differential | High variance | Medium | Lower (closer games) |
| Pass frequency | High (can't find moves) | Medium | Low (efficient piece usage) |

### Visualisation

Generate an interactive HTML report (using Plotly) per training run:

1. **Elo curve** — Elo rating vs generation number. The money plot
2. **Loss curves** — π_loss and v_loss vs generation, with per-epoch detail
3. **Arena results** — Stacked bar chart of W/L/D per generation
4. **Timing breakdown** — Stacked area chart of time spent in each phase
5. **Game statistics** — Average game length, pieces placed, score differential over time
6. **Policy entropy** — Average entropy of π over generations (should decrease)

The reporting infrastructure already exists in the framework (`{run_directory}/Reporting/`).

---

## 3. External Benchmarking (Pentobi)

### Why Pentobi?

Pentobi is the strongest open-source Blokus AI:
- Uses MCTS with RAVE (Rapid Action Value Estimation)
- No neural network — pure tree search with hand-crafted heuristics
- 9 configurable difficulty levels
- Supports GTP (Go Text Protocol) for automated play
- Available at: https://pentobi.sourceforge.io/

### Benchmark Protocol

```
For each difficulty level d ∈ {1, 2, ..., 9}:
    For each evaluation round:
        Play 100 games:
            50 as White (AlphaBlokus starts)
            50 as Black (Pentobi starts)

        Record:
            - Win/loss/draw counts by side
            - Score differentials
            - Average game length
            - Average pieces placed

    Report:
        - Overall win rate with 95% CI
        - Win rate by side
        - Score distribution
```

### Difficulty Ladder

Approach Pentobi evaluation as a ladder — don't skip ahead:

| Level | Pentobi Strength | Target Win Rate | Estimated Elo |
|-------|-----------------|----------------|---------------|
| 1 | Beginner | >90% | ~800 |
| 2 | Easy | >85% | ~1000 |
| 3 | Below Average | >75% | ~1200 |
| 4 | Average | >65% | ~1400 |
| 5 | Above Average | >60% | ~1600 |
| 6 | Strong | >55% | ~1800 |
| 7 | Very Strong | >50% | ~2000 |
| 8 | Expert | >45% | ~2200 |
| 9 | Maximum | Win majority of 100 | ~2400+ |

**The ultimate goal: beat Pentobi level 9 in a majority of 100 games.**

### GTP Integration

Pentobi supports GTP, enabling automated game play:

```
# Start Pentobi in GTP mode
pentobi-gtp --level 5

# GTP commands
boardsize 14              # Blokus Duo board
clear_board               # Reset
play b <move>             # Opponent plays
genmove w                 # Ask Pentobi for a move
showboard                 # Display current state
```

Integration approach:
1. Subprocess management — spawn `pentobi-gtp` as a child process
2. Command/response protocol — send commands, parse responses
3. Move translation — convert between AlphaBlokus action format and GTP move notation
4. Game loop — alternate between AlphaBlokus MCTS and Pentobi GTP

### Statistical Significance

For 100-game evaluations, the 95% confidence interval for win rate:

```
CI = p ± 1.96 × sqrt(p(1-p)/n)

At p=0.55 (55% win rate), n=100:
CI = 0.55 ± 0.097 = [0.453, 0.647]

At p=0.60 (60% win rate), n=100:
CI = 0.60 ± 0.096 = [0.504, 0.696]
```

A 55% win rate over 100 games is **not** statistically significant at the 95% level (CI includes 50%). To be confident:
- 60% win rate over 100 games → significant
- 55% win rate would need ~400 games for significance
- Consider increasing to 200+ games for borderline results

### Elo Estimation vs Pentobi

Once we have win rates against multiple Pentobi levels, estimate our network's absolute Elo:

```
For each Pentobi level L with estimated Elo R_L and observed win rate p:
    R_network = R_L - 400 × log10((1 - p) / p)

Average across levels for final estimate.
```

---

## Evaluation Matrix

The full evaluation produces a matrix:

```
                    Generation
                 1   5   10  20  50  100
Metric          ┌───┬───┬───┬───┬───┬───┐
Elo             │   │   │   │   │   │   │
Policy loss     │   │   │   │   │   │   │
Value loss      │   │   │   │   │   │   │
Top-1 accuracy  │   │   │   │   │   │   │
Game length     │   │   │   │   │   │   │
Pentobi L1      │   │   │   │   │   │   │
Pentobi L3      │   │   │   │   │   │   │
Pentobi L5      │   │   │   │   │   │   │
Pentobi L7      │   │   │   │   │   │   │
Pentobi L9      │   │   │   │   │   │   │
                └───┴───┴───┴───┴───┴───┘
```

Evaluate against Pentobi periodically during training (e.g., every 10-20 generations) to track absolute strength improvement alongside the internal Elo curve.

---

## AlphaZero Paper Benchmarks (for Reference)

The original AlphaZero paper provides useful reference points for expected training dynamics:

| Observation | AlphaZero (Chess) | AlphaBlokus (Expected) |
|-------------|-------------------|----------------------|
| Elo at generation 0 | ~800 | ~500 (random play) |
| Elo at generation 100 | ~2500 | Unknown |
| Training games to converge | ~44M | Much fewer (simpler game) |
| Loss convergence | ~200k training steps | Faster (smaller state space than Go/Chess) |
| Policy accuracy plateau | ~55% top-1 | Higher expected (fewer "equally good" moves) |

Key differences from AlphaZero's original setup:
- **No parallel MCTS** — single-threaded self-play is much slower
- **Smaller network** — AlphaZero Chess used 20 residual blocks, 256 channels. We start with 4 blocks, 512 channels
- **Smaller action space than Go** (17,837 vs 362 for Chess vs 362 for Go) — but many actions are geometrically invalid, so effective branching factor may be comparable
- **No transposition tables** — current MCTS rebuilds from scratch each move

---

## Success Criteria

### Phase 2 (Game Logic) — "It Works"
- [ ] Self-play produces valid games that terminate correctly
- [ ] Policy loss decreases over the first 5 generations
- [ ] Value loss drops below 1.0 within 10 generations
- [ ] Arena accepts at least one new network in the first 10 generations

### Phase 3 (Training) — "It Learns"
- [ ] Monotonically increasing Elo curve (with noise)
- [ ] Beats Pentobi level 1 within 20 generations
- [ ] Beats Pentobi level 5 within 50 generations
- [ ] Training throughput > 10 games/hour

### Phase 4 (Benchmarking) — "It's Strong"
- [ ] Beats Pentobi level 7 in majority of 100 games
- [ ] Beats Pentobi level 9 in majority of 100 games (stretch goal)
- [ ] Clean Elo curve in final report
- [ ] ECE < 0.05 for value predictions (calibration)
