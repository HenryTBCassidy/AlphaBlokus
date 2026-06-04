# AlphaBlokus — Evaluation Plan

## Overview

Evaluation for AlphaBlokus operates at two levels:

1. **Training diagnostics** — Is the network actually learning? Loss curves, policy accuracy, value calibration, Elo vs a frozen baseline, a minimax oracle (TTT), a symmetry diagnostic, and arena results. Cheap signals computed every generation. **These are implemented.**
2. **Pentobi benchmarking** — How strong is the network in absolute terms? Win rates against Pentobi's 9 difficulty levels, composite scores, and the headline "Pentobi Level" metric. Expensive but meaningful, run periodically. **This is a plan — the Pentobi adapter is not built yet (see [06-INTERFACES.md](06-INTERFACES.md)).**

AlphaZero (Silver et al., 2017) used Elo from self-play arena results as a continuous training health monitor, but grounded its headline strength claims in external matches against Stockfish and Elmo. We follow the same philosophy: internal Elo (vs a frozen gen-0 baseline) tracks training progress; Pentobi results will measure real strength once the adapter exists.

---

## 1. Training Diagnostics

These metrics are computed every generation and are cheap — they use data the training loop already produces. Their purpose is answering "is training working?" not "how strong is the model?"

### Loss Monitoring

Every training generation produces per-batch metrics logged to hive-partitioned Parquet (and mirrored to W&B when configured):

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

### Arena Results

After each training generation, the new network plays the previous best in an arena:

```
New network vs Old network
  - num_arena_matches games (e.g., 40)
  - Half the games each colour (controls for first-move advantage)
  - Full MCTS for both players, temperature 0
  - Accept new if score ≥ update_threshold, where
        score = (wins + 0.5·draws) / (wins + losses + draws)   (e.g., 0.55)
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

### Elo vs a frozen gen-0 baseline

The headline "is it actually getting stronger?" curve. At run start the random-init network is **frozen** and saved as `elo_baseline.pth.tar`. Every generation (when `elo_games_per_gen > 0`, default 50) the *current* network plays that fixed anchor head-to-head, noise-free, and the result is converted to an Elo difference:

```
score_rate = (wins + 0.5·draws) / games            # clamped to [0.001, 0.999]
elo_diff   = 400 · log10(score_rate / (1 − score_rate))
elo_rating = baseline_rating + elo_diff             # baseline_rating default 400
```

The crucial property is that the opponent **never moves** — unlike chaining Elo between consecutive generations, every generation is measured against the *same* fixed reference, so the curve is directly comparable across the whole run and isn't distorted by a drifting baseline.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Baseline (anchor) rating | 400 (`elo_baseline_rating`) | Display anchor for the random net. Above 0 so a gen that briefly learns something worse-than-random can dip without going negative; low enough to read as "weak". A converged model has room to climb to ~800–1200+ |
| Games per generation | 50 (`elo_games_per_gen`) | Cheap; 0 disables Elo entirely |
| Draws | 0.5 | Standard Elo draw handling, baked into `score_rate` |

**Important caveat:** this is a *relative* measure with an arbitrary anchor. The numbers are not comparable to chess Elo, Pentobi Elo, or any external system — an Elo of 1200 here means only "≈800 points stronger than the random gen-0 net." Its job is tracking progress; if the curve flattens for many generations, something is wrong.

**Expected trajectory:**
- Generations 1-10: Rapid Elo gain (learning basic moves)
- Generations 10-30: Steady improvement (learning piece interactions)
- Generations 30-50+: Diminishing returns (fine-tuning positional understanding)

### Minimax oracle (Tic-Tac-Toe only)

For the validation game we have a *perfect* reference. When `game == "tictactoe"` and `minimax_games_per_gen > 0`, the current network plays a perfect-play minimax opponent each generation. Because TTT is a forced draw under optimal play, the target is **draw-rate → 1.0 with loss-rate → 0** — the signal that the network has internalised optimal play. The minimax solver also supplies the *oracle eval set* (target policy = uniform over all game-theoretically optimal actions, target value = the true minimax value), which is what the per-epoch top-1/5 accuracy plots are scored against for TTT.

### Symmetry diagnostic

A trained network should be *equivariant* under the game's symmetry group: mirroring the board should mirror its policy. Every generation (when `symmetry_diagnostic_positions > 0`, default 100) we take a fixed, seeded set of reference positions and, for each non-identity symmetry, compute the KL divergence between the network's policy on the symmetric board and the symmetric image of its policy on the original. **Zero is perfect.** A rising or persistently-high KL means the network has baked in directional biases that augmentation should be averaging out. The reference set is the same every generation, so the per-gen trend is comparable.

### Game Length Analysis

As the network improves, game characteristics should change:

| Metric | Random Play | Weak Network | Strong Network |
|--------|-------------|-------------|----------------|
| Average game length | Short (few pieces placed) | Medium | Long (more pieces placed) |
| Pieces placed per game | Low | Medium | High (approaching 42 = all pieces) |
| Score differential | High variance | Medium | Lower (closer games) |
| Pass frequency | High (can't find moves) | Medium | Low (efficient piece usage) |

### Training Timing

Each phase of the training loop should be timed to identify bottlenecks:

| Phase | Metric | Expected Bottleneck |
|-------|--------|-------------------|
| Self-play (per game) | seconds/game | MCTS simulations × neural net inference |
| Self-play (per generation) | minutes/generation | num_eps × seconds/game |
| Neural net training | seconds/epoch | Forward/backward pass, data loading |
| Arena evaluation | minutes/evaluation | num_arena_matches × seconds/game |
| Total generation | minutes/generation | Sum of all phases |

The framework saves timing data to `{run_directory}/Timings/` and per-episode MCTS profiling to `{run_directory}/SelfPlayProfiling/`. The major per-game cost levers — all **implemented** — are:

1. **Precomputed move generation** — the Pentobi-style table generator (PUCT already iterates only legal actions, never the full 17,837 space)
2. **Batched neural-net inference** — leaves are collected under virtual loss and evaluated in one GPU call per batch, plus optional fp16 on CUDA
3. **Parallel self-play** — independent games run across worker processes (`num_parallel_workers`)

See [02-ALGORITHMS.md](02-ALGORITHMS.md) for how each works and [08-TRAINING-ESTIMATES.md](08-TRAINING-ESTIMATES.md) for measured cost splits.

---

## 2. Pentobi Benchmarking

This is the primary evaluation — the section that answers "how good is the model?" in absolute terms.

### Why Pentobi?

Pentobi is the strongest open-source Blokus AI:
- Uses MCTS with RAVE (Rapid Action Value Estimation)
- No neural network — pure tree search with hand-crafted heuristics
- 9 configurable difficulty levels
- Supports GTP (Go Text Protocol) for automated play
- Available at: https://pentobi.sourceforge.io/

There is no established competitive Blokus Duo rating system (unlike chess with its FIDE Elo). Pentobi's 9 difficulty levels are the best available external strength anchor for the game.

### Benchmark Protocol

```
For each difficulty level d ∈ {1, 2, ..., 9}:
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

Run Pentobi benchmarks every 10-20 generations during training to track absolute strength improvement over time.

### Headline Metrics

Four metrics summarise Pentobi benchmark results. Together they give a complete picture at a glance.

#### Pentobi Level (the headline number)

**Highest Pentobi level beaten at >50% win rate over 100 games.**

Integer from 0 to 9. When someone asks "how strong is your model?", you say "it beats Pentobi level 7." This is the Blokus equivalent of "AlphaZero reached superhuman Elo" — except grounded in an actual opponent rather than an arbitrary number.

The >50% threshold is correct for benchmarking (the model wins more than it loses). This is distinct from the 55% acceptance threshold used during training, where a margin of safety avoids accepting noise. For borderline cases, see the statistical significance section below.

#### Pentobi Score (equal-weight composite)

**Total wins across all levels / total games across all levels.**

Range: 0.0 to 1.0. This treats a win at level 1 with the same weight as a win at level 9. The key property: a *loss* at level 1 is implicitly terrible (because level 1 is easy), while a loss at level 9 is expected. The metric naturally captures this asymmetry without any weighting.

```
Example: 100 games per level, results = [100, 100, 100, 80, 60, 30, 10, 2, 0] wins

Pentobi Score = (100+100+100+80+60+30+10+2+0) / 900 = 0.536
```

This number increases monotonically with improvement and requires no arbitrary weight choices.

#### Pentobi Weighted Score (difficulty-weighted composite)

**Σ(level × wins_at_level) / Σ(level × games_at_level)**

Range: 0.0 to 1.0. Uses the level number (1-9) as a natural weight, emphasising performance against harder opponents. This differentiates two models that both have a Pentobi Score of 0.5 but where one dominates levels 1-4 while the other spreads wins across all levels.

```
Same example: wins = [100, 100, 100, 80, 60, 30, 10, 2, 0]

Numerator   = 1(100) + 2(100) + 3(100) + 4(80) + 5(60) + 6(30) + 7(10) + 8(2) + 9(0) = 1486
Denominator = 1(100) + 2(100) + 3(100) + 4(100) + 5(100) + 6(100) + 7(100) + 8(100) + 9(100) = 4500

Pentobi Weighted Score = 1486 / 4500 = 0.330
```

#### Win-Rate Profile (per-level detail)

**Win rate at each Pentobi level individually, as a 9-element vector.**

```
Example: [1.00, 1.00, 1.00, 0.80, 0.60, 0.30, 0.10, 0.02, 0.00]
```

This is the raw input to the other metrics and is valuable on its own — it tells you exactly where the model's strength frontier lies.

### Difficulty Ladder

Approach Pentobi evaluation as a ladder. The target win rates below represent approximate milestones — the >50% threshold in "Pentobi Level" is the formal definition of "beaten."

| Level | Pentobi Strength | Milestone Win Rate |
|-------|-----------------|-------------------|
| 1 | Beginner | >90% |
| 2 | Easy | >85% |
| 3 | Below Average | >75% |
| 4 | Average | >65% |
| 5 | Above Average | >60% |
| 6 | Strong | >55% |
| 7 | Very Strong | >50% |
| 8 | Expert | >50% |
| 9 | Maximum | >50% |

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

A 55% win rate over 100 games is **not** statistically significant at the 95% level (the CI includes 50%). To be confident:
- 60% win rate over 100 games → significant (CI lower bound > 50%)
- 55% win rate would need ~400 games for significance
- Consider increasing to 200+ games for borderline results

For the "Pentobi Level" metric, if the win rate is between 50-60% over 100 games, run additional games to confirm. A clear >60% needs no further validation.

---

## 3. Visualisation

Generate an interactive HTML report (using Plotly) per training run. The reporting infrastructure already exists in the framework (`{run_directory}/Reporting/`).

### Pentobi Heatmap (the money plot)

A heatmap showing win rates against Pentobi across training:
- **X-axis:** Generation number (sampled every 10-20 generations)
- **Y-axis:** Pentobi levels 1-9
- **Cell colour:** Win rate (white = 0%, dark green = 100%)

This is the single most information-dense visualisation. You immediately see the "frontier" of which levels the model can beat expanding rightward and upward over training time. Every cell is meaningful — "generation 40 beats level 5 at 72%."

Annotate each cell with the win rate percentage for readability.

### Pentobi Composite Scores (line chart)

Overlay three lines on a single chart, all against generation number:
- **Pentobi Level** (integer 0-9, stepped line, right y-axis)
- **Pentobi Score** (0.0-1.0, solid line, left y-axis)
- **Pentobi Weighted Score** (0.0-1.0, dashed line, left y-axis)

### Training Diagnostics (secondary plots)

1. **Elo curve** — Elo vs the frozen gen-0 baseline, per generation. A sanity check that training is progressing between Pentobi evaluations
2. **Loss curves** — π_loss and v_loss vs generation, with per-epoch detail on hover
3. **Arena results** — Stacked bar chart of W/L/D per generation
4. **Timing breakdown** — Stacked area chart of time spent in each phase
5. **Game statistics** — Average game length, pieces placed, score differential over time
6. **Policy entropy** — Average entropy of π over generations (should decrease)

---

## AlphaZero Paper Benchmarks (for Reference)

The original AlphaZero paper provides useful reference points for expected training dynamics:

| Observation | AlphaZero (Chess) | AlphaBlokus (Expected) |
|-------------|-------------------|----------------------|
| Training games to converge | ~44M | Much fewer (simpler game) |
| Loss convergence | ~200k training steps | Faster (smaller state space than Go/Chess) |
| Policy accuracy plateau | ~55% top-1 | Higher expected (fewer "equally good" moves) |
| Elo-vs-baseline trajectory | Monotonically increasing | Same expected |

Key differences from AlphaZero's original setup:
- **Far less compute** — AlphaZero used thousands of TPUs; we use a single consumer GPU, so far fewer games and simulations per move
- **Smaller network** — AlphaZero Chess used 20 residual blocks, 256 channels. We use ~4 blocks at production (64 filters), configurable up
- **Larger action space** (17,837 vs 4,672 for Chess) — but only 13,729 placements ever fit on the board and only a few hundred are legal at any position, so the effective branching factor is comparable
- **Self-play parallelism is across processes, not a batched MCTS server** — independent games run on worker processes; within a search, leaves are batched via virtual loss. Transpositions are handled implicitly (the tree is keyed by board state and reused across a game's moves)
- **Different external benchmark** — AlphaZero had Stockfish (with known FIDE Elo). We have Pentobi (no established rating system, but 9 calibrated difficulty levels)

---

## Success Criteria

### Phase 2 (Game Logic) — "It Works"
- [ ] Self-play produces valid games that terminate correctly
- [ ] Policy loss decreases over the first 5 generations
- [ ] Value loss drops below 1.0 within 10 generations
- [ ] Arena accepts at least one new network in the first 10 generations

### Phase 3 (Training) — "It Learns"
- [ ] Monotonically increasing Elo-vs-baseline curve (with noise)
- [ ] Pentobi Level >= 1 within 20 generations
- [ ] Pentobi Level >= 5 within 50 generations
- [ ] Training throughput > 10 games/hour

### Phase 4 (Benchmarking) — "It's Strong"
- [ ] Pentobi Level >= 7
- [ ] Pentobi Level = 9 (stretch goal)
- [ ] Pentobi Score > 0.7
- [ ] Clean Pentobi heatmap showing progressive frontier expansion
- [ ] ECE < 0.05 for value predictions (calibration)
