# AlphaBlokus — Evaluation Plan

## Overview

Evaluation for AlphaBlokus operates at two levels:

1. **Training diagnostics** — Is the network actually learning? Loss curves, policy accuracy, value calibration, rolling arena-derived Elo, a minimax oracle (TTT), a symmetry diagnostic, and arena results. Cheap signals computed every generation. **These are implemented.**
2. **Pentobi benchmarking** — How strong is the network in absolute terms? Win rates against Pentobi's 9 difficulty levels, composite scores, and the headline "Pentobi Level" metric. Expensive but meaningful, run periodically. **This is a plan — the Pentobi adapter is not built yet (see [06-INTERFACES.md](06-INTERFACES.md)).**

AlphaZero (Silver et al., 2017) used Elo from self-play arena results as a continuous training health monitor, but grounded its headline strength claims in external matches against Stockfish and Elmo. We follow the same philosophy: internal Elo (live rolling arena-derived + end-of-run pooled BayesElo) tracks training progress; Pentobi results will measure real strength once the adapter exists.

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
  - num_arena_matches games (e.g., 100)
  - Half the games each colour (controls for first-move advantage)
  - Full MCTS for both players, temperature 0
  - Accept per the gate mode (below); score = (wins + 0.5·draws) / (wins + losses + draws)
```

Track per generation:
- Win/loss/draw counts
- **White-win / Black-win counts** (per-colour split — see the colour-pinning note)
- Win rate with confidence intervals
- Acceptance/rejection decision
- Number of consecutive acceptances/rejections

**Warning signs:**
- Many consecutive rejections → training may have stalled, consider adjusting learning rate or MCTS parameters
- 100% win rate for new network → generations too far apart in strength, reduce training epochs
- ~50% win rate → network not improving meaningfully per generation
- **Exact-0.500 scores or sub-binomial score variance, or a white-win rate ≫ 50%** → the gate is *colour-pinned* (see below). The report raises an automatic red-flag banner for these.

#### Paired colour-swapped arena (`paired_arena`)

Alternating colours across independent games does **not** actually control for the first-move advantage when one colour dominates: in Blokus Duo ~96% of decisive deterministic games are won by White (the first mover), so between near-equal nets each side banks ~50 near-guaranteed points and the score is pinned to ~0.50 ± a few Black upsets — no candidate that is merely *somewhat* better can clear a 0.55 (or even 0.52) threshold. This froze `blokus_search_harder` at 0/17 accepted (docs/research/plateau-investigation.md §2 B8).

With `paired_arena: true`, the gate instead plays `num_arena_matches / 2` **pairs**. Each pair samples one opening prefix (`arena_opening_moves` plies from the incumbent at `arena_opening_temp`), then plays it out **twice — colours swapped — replaying the identical prefix**. The first-mover advantage cancels *within* the pair, so the score measures net-strength differential (does the candidate win as Black from openings where the incumbent lost as Black?) rather than a colour coin-flip. Scoring uses **rule (a), paired win-differential**: each pair contributes `candidate_wins − incumbent_wins ∈ {−2..+2}`; aggregated linearly and mapped to `[0,1]` this is algebraically the ordinary `(wins + 0.5·draws)/total` over the paired games — the resolution gain is the *variance reduction* of shared-opening pairing, not different arithmetic. The pool tournament (Tier 2) uses the same paired construction when `paired_arena` is on. Default `false` preserves the unpaired path.

#### Gate mode (`gate_mode`)

The acceptance policy is config-selectable:

- **`threshold`** (default): accept iff `score ≥ update_threshold` (AlphaGo-Zero, e.g. 0.55). Kept as the default so existing configs are unchanged.
- **`regression_guard`**: accept **unless clearly worse** — reject only if `score < guard_floor` (default 0.48); otherwise adopt. The 0.55 gate was the *direct* cause of the stationary loop (every DeepMind successor dropped it); with a rolling buffer a mediocre accepted net self-corrects within a few generations, whereas a frozen incumbent never does. Only trustworthy alongside `paired_arena` (colour-cancelled score).
- **`always`**: AlphaZero-style, always adopt the candidate; the pool tournament + `accepted_*.pth.tar` checkpoints remain the offline strength record.

### Elo: a two-tier scheme

Strength is tracked at two tiers — a cheap live signal streamed every generation, and a rigorous curve computed once at end-of-run. This mirrors the philosophy constraint: per-generation metrics stream live; anything needing multiple generations runs in the end-of-run step.

**Tier 1 — rolling arena-derived Elo (live, per-generation).** The accept/reject arena *already* plays the candidate against the current incumbent, and on acceptance the candidate *becomes* the incumbent — so the incumbent is a rolling benchmark. Each generation the candidate's Elo is derived from that same arena score (zero extra games):

```
score_rate    = (wins + 0.5·draws) / games          # clamped to [0.001, 0.999]
elo_delta     = 400 · log10(score_rate / (1 − score_rate))   # vs the incumbent
candidate_elo = incumbent_elo + elo_delta
```

The starting net is anchored at `elo_baseline_rating` (400). On acceptance the benchmark rolls forward to the candidate; a rejected generation still logs its provisional candidate Elo but leaves the benchmark untouched (so the next candidate is measured against the same incumbent). Because each generation is rated against an opponent of *comparable* strength — not a fixed weak anchor — the curve **never saturates** at the ±1200 clamp the way the retired frozen-gen-0 metric did. Its cost is that it's a *chained* estimate: drift accumulates and the high-score steps are noisy on ~100 games. That's exactly what the end-of-run pooled fit corrects. On `--resume` the benchmark is reconstructed from the last *accepted* net's logged Elo, so the chain continues seamlessly. Reuses `num_arena_matches` as its sample size, so very low arena counts make it noisier (100 is comfortable, ≤40 is coarse).

> **Retired:** the old per-generation "Elo vs a frozen gen-0 net" eval (`elo_games_per_gen`) was removed. It saturated once the net ≫ gen-0 (bimodal noise) and cost extra games each generation. The gen-0 net is *still* frozen to `elo_baseline.pth.tar` at run start — but only as the pool tournament's anchor (Tier 2), no longer played per generation.

**Cross-run comparability.** The anchor (Elo = `elo_baseline_rating`) is run-specific, so `Nets/elo_anchor.json` records what it is: `scratch` (random-init) or `warm_start` (a donor net, with the weights' SHA-256). To splice this run's rolling curve onto another run's, match the donor hash to a checkpoint whose pooled Elo is known.

**Tier 2 — pool BayesElo tournament (the canonical strength curve).**

This is how DeepMind actually measured strength: not against one fixed anchor, but from games *among a pool* of checkpoints, with one consistent rating per player fit by **BayesElo** (a Bradley–Terry maximum-likelihood fit). Because the comparison is relative to *nearby* checkpoints rather than a fixed weak anchor, the curve keeps rising until genuine convergence — it never saturates.

It runs **automatically at end-of-run** when `TournamentConfig.run_at_end` is set (enabled in cloud/production configs), so the report ships with the rigorous curve. It can also be run post-hoc on any finished run's saved checkpoints (no retraining):

```
uv run python -m scripts.tournament_elo --config <run.json>
uv run python -m scripts.tournament_elo --config <run.json> --dry-run   # schedule + game count only
```

The tool enumerates `Nets/accepted_<N>.pth.tar` (plus the gen-0 `elo_baseline.pth.tar` anchor), plays a **sparse but connected** round-robin (`TournamentConfig.back_ref_offsets`, exponentially spaced so the comparison graph stays connected at O(K·log K) pairings, not O(K²)) at a deliberately low `TournamentConfig.num_mcts_sims` (ranking is robust to weak play — keeps a full run to ~30–60 min), fits BayesElo (`evaluation/rating.py`), and writes `Tournament/tournament_ratings.parquet` + `tournament_raw.json`. The report renders the rising pool-Elo curve alongside the live rolling-Elo chart. The gen-0 checkpoint is pinned at `anchor_rating` so the scale is comparable within a run; cross-run comparability still needs a shared external anchor (e.g. Pentobi). Full methodology and the DeepMind lineage: [`research/pool-elo-methodology.md`](research/pool-elo-methodology.md).

### Minimax oracle (Tic-Tac-Toe only)

For the validation game we have a *perfect* reference. When `game == "tictactoe"` and `minimax_games_per_gen > 0`, the current network plays a perfect-play minimax opponent each generation. Because TTT is a forced draw under optimal play, the target is **draw-rate → 1.0 with loss-rate → 0** — the signal that the network has internalised optimal play. The minimax solver also supplies the *oracle eval set* (target policy = uniform over all game-theoretically optimal actions, target value = the true minimax value), which is what the per-epoch top-1/5 accuracy plots are scored against for TTT.

### Symmetry diagnostic

A trained network should be *equivariant* under the game's symmetry group: mirroring the board should mirror its policy. Every generation (when `symmetry_diagnostic_positions > 0`, default 100) we take a fixed, seeded set of reference positions and, for each non-identity symmetry, compute the KL divergence between the network's policy on the symmetric board and the symmetric image of its policy on the original. **Zero is perfect.** A rising or persistently-high KL means the network has baked in directional biases that augmentation should be averaging out. The reference set is the same every generation, so the per-gen trend is comparable.

### Policy–Value Consistency (PVC)

The policy and value heads are trained on the same self-play but with different targets, so they can drift apart — v3 plateaued externally while its policy kept improving internally, and we had no signal that *decomposed* the two heads. PVC gives that decomposition: does the policy agree with a one-ply lookahead through the value head?

For each frozen eval position `s` (current player to move) we take the top-K (default 8) legal moves by policy probability. For candidate move `a` leading to child `s'`, the one-ply value is the negamax `Q₁(a) = −V(s')` (after our move it's the opponent's turn, so their value negates to ours); terminal children use the true game result (mover's perspective) instead of `V`. Two agreement measures are aggregated over the eval set each generation:

- **`pvc_argmax_match`** (0–1) — fraction of positions where the policy's best candidate is also the `Q₁`-best candidate.
- **`pvc_spearman`** (−1 to 1) — mean Spearman rank correlation between `π` and `Q₁` across the candidates. Positions with fewer than two legal moves are excluded (rank correlation is undefined), so a sparse late-game board can't poison the mean.

**Read it as a trend, not a target.** Perfect agreement is *not* expected and disagreement is often *correct*: the policy is trained on the MCTS *visit* distribution, which reflects **multi-ply** search, while `Q₁` is only **one ply** of value. A move that looks weak one-ply but is best after deep search — the policy rightly likes it, `Q₁` doesn't. So a healthy net rises early (both heads improving and becoming consistent) then **plateaus below 100%** — the residual ≈ how much deeper the policy sees than one-ply value. The red flag is a **late drop or a persistently low level**: the value head lagging (it can't evaluate the states the policy leads to) or the policy chasing lines the value head doesn't support (the v3 decoupling). Diagnostic only — it does not change training, search, or the loss.

**Value-symmetry MAE** (optional companion) — `mean|V(s) − V(reflect(s))|` over the eval set. The value of a position is invariant under the game's symmetry group, so this should hug 0; a rising value means the *value* head isn't respecting the symmetry (the policy head's equivariance is already tracked by the symmetry-KL diagnostic above). Rendered on the PVC chart's secondary axis.

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

### Running the benchmark

```bash
uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --level 5 --games 100
uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --sweep --games 100
uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --levels 1-3 --games 40
uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --sweep --games 100 --workers 4
```

**Parallelism (`--workers`).** The benchmark is *not* inference-bound — the GPU sits near-idle
while Pentobi's CPU search (which grows sharply with level) and the per-move GTP round-trip
dominate. `--workers N` splits the requested games across `N` worker processes, each with its
own net + its own `pentobi-gtp` engine, then aggregates the results. Games are split into even
per-worker chunks (so each keeps `Arena`'s half-white/half-black colour swap) with disjoint
Pentobi seeds (no two workers replay the same games), and one pool serves *all* levels at once
so fast low-level chunks free their worker to pick up slow level-8/9 chunks. Expect a
near-linear speedup up to the VRAM/core ceiling (measured ~2.9× at 4 CPU-net workers on the Mac;
a full 1–9 ladder drops from ~45–70 min to well under 15 min).

- `--workers 1` reproduces the serial path bit-for-bit. The default (when `--workers` is unset)
  is `num_parallel_workers` from the config if it opts into parallelism, else 4.
- Workers use the `spawn` start method — forking a process that has imported Torch/CUDA (or JAX)
  deadlocks. Each worker is a fresh interpreter that rebuilds its own game/net/engine from the
  config path, so nothing GPU-touching crosses the process boundary.
- Each GPU worker needs its own CUDA context (~0.6–1.5 GB), so on the 8 GB 3060 Ti expect ~4
  workers before CUDA OOM — lower `--workers`, or pass `--cpu-net` to run the net on CPU and
  scale past the VRAM cap (slower per move, but the win is parallelism). The ceiling scales up
  with VRAM on a bigger card.

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

1. **Elo curves** — the live rolling arena-derived Elo per generation, plus the end-of-run pooled BayesElo curve. A sanity check that training is progressing between Pentobi evaluations
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
| Elo trajectory | Monotonically increasing | Same expected (rolling + pooled Elo) |

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
- [ ] Monotonically increasing rolling-Elo curve (with noise); rising pooled-Elo curve
- [ ] Pentobi Level >= 1 within 20 generations
- [ ] Pentobi Level >= 5 within 50 generations
- [ ] Training throughput > 10 games/hour

### Phase 4 (Benchmarking) — "It's Strong"
- [ ] Pentobi Level >= 7
- [ ] Pentobi Level = 9 (stretch goal)
- [ ] Pentobi Score > 0.7
- [ ] Clean Pentobi heatmap showing progressive frontier expansion
- [ ] ECE < 0.05 for value predictions (calibration)
