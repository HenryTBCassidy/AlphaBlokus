# AlphaBlokus — Background & Motivation

## Mission

Train a neural network through self-play reinforcement learning to master **Blokus Duo**, using DeepMind's AlphaZero algorithm. The trained agent should beat **Pentobi** (the strongest open-source Blokus AI) in a majority of 100 games.

**One network, zero human knowledge, superhuman Blokus.**

---

## Why Blokus?

Blokus Duo is an interesting target for AlphaZero because:

1. **Combinatorial complexity:** The early-game branching factor (~ 400–600 legal moves) rivals Go and dwarfs Chess (~30 legal moves per position). The network outputs over a fixed **17,837-dimensional** action space, of which **13,729** are placements that ever fit on the board (the rest of the 14×14×91 grid hangs off the edge) — see [Action Space Scale](#1-action-space-scale)
2. **Spatial reasoning:** Piece placement requires geometric intuition — rotations, reflections, corner-touching constraints
3. **Nobody has done it:** As of 2026, no strong AlphaZero-style Blokus agent exists. The few attempts either abandoned neural networks (too slow) or have zero published results
4. **Clear benchmark:** Pentobi provides a well-established opponent with configurable difficulty levels (1-9)

---

## Competitive Landscape

### Overview

Blokus AI is a small but active space. The dominant approach is hand-crafted MCTS (Pentobi), with a handful of reinforcement learning attempts that have either abandoned neural networks or produced no published results. As of 2026, **no strong AlphaZero-style Blokus agent exists with published training curves or benchmark results.**

This is the gap AlphaBlokus aims to fill.

---

### The Benchmark: Pentobi

#### What It Is

Pentobi is the strongest open-source Blokus AI, maintained by Markus Enzenberger since 2011.

- **Repository:** [github.com/enz/pentobi](https://github.com/enz/pentobi)
- **Algorithm:** MCTS with RAVE (Rapid Action Value Estimation) + opening book
- **No neural network** — pure tree search with hand-crafted heuristics
- **9 difficulty levels** — from beginner to expert
- **GTP support** — Go Text Protocol enables automated play
- **Supports all Blokus variants** — Duo, Classic (4-player), Trigon, Junior, Nexos, Callisto
- **Written in C++** — fast move generation and simulation

#### Why It's Strong

Pentobi's strength comes from three pillars:

1. **Optimised move generation:** The board implementation is specifically designed for fast legal move enumeration — critical for high MCTS playouts per second
2. **RAVE heuristic:** Rapid Action Value Estimation lets MCTS share value estimates across moves that appear in different parts of the tree. A move that works well in one line of play biases the search toward trying it in others. This is particularly effective for Blokus where piece placement patterns transfer across positions
3. **Opening book:** Pre-computed strong openings avoid wasting search time in the early game where the branching factor is enormous

#### Why It's Beatable

Pentobi has fundamental limitations that a neural approach can exploit:

| Limitation | AlphaZero Advantage |
|-----------|-------------------|
| No positional evaluation function — relies purely on random playouts | Neural value head provides instant position assessment |
| RAVE assumes move values transfer across positions — not always true | Neural policy head learns position-specific move probabilities |
| Fixed search depth/time per move | Learned evaluation effectively gives infinite-depth positional understanding |
| Opening book is hand-curated, finite | Neural network generalises opening principles from self-play |
| No learning between games | Each generation of self-play makes the agent stronger |

---

### Existing Blokus RL Projects

#### KubiakJakub01/Blokus-RL

- **Repository:** [github.com/KubiakJakub01/Blokus-RL](https://github.com/KubiakJakub01/Blokus-RL)
- **Approach:** PPO and AlphaZero implementations for Blokus using OpenAI Gym interface
- **Status:** Active development
- **Key observations:**
  - Implements both PPO (policy gradient) and AlphaZero approaches
  - Uses Gymnasium environment interface — good software engineering
  - No published benchmark results against Pentobi or any external opponent
  - 4-player Blokus focus (not Duo) — significantly harder RL problem due to non-zero-sum dynamics

**What AlphaBlokus does differently:**
- Focuses on 2-player Duo (clean zero-sum game, compatible with AlphaZero's framework)
- Game-agnostic framework validated on Tic-Tac-Toe first
- Comprehensive evaluation plan with Pentobi benchmarking

#### DerekGloudemans/Blokus-Reinforcement-Learning

- **Repository:** [github.com/DerekGloudemans/Blokus-Reinforcement-Learning](https://github.com/DerekGloudemans/Blokus-Reinforcement-Learning)
- **Approach:** Game interface with heuristics, search strategies, and learning agents
- **Status:** Stale (last updated 2019)
- **Key observations:**
  - Implements random, greedy, and minimax strategies
  - Deep RL was planned but marked as future work ("when I get some free time!")
  - Good Blokus game engine implementation in Python
  - No neural network training results

**What AlphaBlokus does differently:**
- Actually implements the neural network + MCTS pipeline
- ResNet architecture designed for the Blokus action space

#### roger-creus/blokus-ai

- **Repository:** [github.com/roger-creus/blokus-ai](https://github.com/roger-creus/blokus-ai)
- **Approach:** Blokus as a Gymnasium environment for RL agent training
- **Status:** Recent activity
- **Key observations:**
  - Clean environment implementation following Gymnasium conventions
  - Designed as a training environment rather than a complete agent
  - No trained agent results published

**What AlphaBlokus does differently:**
- Full AlphaZero pipeline (self-play → training → arena) rather than just an environment
- Board representation designed specifically for neural network input (44×14×14 multi-channel encoding)

#### mknapper1/Machine-Learning-Blokus

- **Repository:** [github.com/mknapper1/Machine-Learning-Blokus](https://github.com/mknapper1/Machine-Learning-Blokus)
- **Approach:** Random, Greedy, and Minimax strategies
- **Status:** Academic project, no longer maintained
- **Key observations:**
  - Jupyter notebook-based implementation
  - Compares algorithmic strategies but no ML/RL training
  - Good reference for Blokus game logic in Python

#### JunhaoWang/blokus

- **Repository:** [github.com/JunhaoWang/blokus](https://github.com/JunhaoWang/blokus)
- **Approach:** Generate gameplay data and train classifier using neural networks and RNNs
- **Status:** Stale
- **Key observations:**
  - Interesting approach of using supervised learning on gameplay data
  - Memory-based recurrent neural networks for sequence modeling
  - No published results or trained models

---

### Academic Work

#### FPGA-Based Blokus Duo Solver (2015)

- **Paper:** "Highly scalable, shared-memory, Monte-Carlo tree search based Blokus Duo Solver on FPGA"
- **Approach:** Hardware-accelerated MCTS on FPGA for massive parallelism
- **Key result:** Can always beat Pentobi level 1 through brute-force search speed
- **Relevance:** Demonstrates that raw search speed can compensate for lack of evaluation function. AlphaZero's neural evaluation should achieve the same effect more efficiently in software

#### AlphaZero-Inspired General Board Game Learning (2022)

- **Paper:** "AlphaZero-Inspired General Board Game Learning and Playing"
- **Approach:** Adapting AlphaZero to arbitrary board games
- **Key insight:** The framework generalises well, but action space encoding and board representation are game-specific challenges
- **Relevance:** Validates our game-agnostic `IGame`/`INeuralNetWrapper` protocol approach

#### AlphaViT (2024)

- **Paper:** "AlphaViT: A flexible game-playing AI"
- **Approach:** Vision Transformer (ViT) replacing ResNet in the AlphaZero framework
- **Key insight:** ViT can handle variable board sizes with a single model
- **Relevance:** Interesting architectural alternative, but Blokus's fixed 14×14 board doesn't benefit from variable-size handling. ResNet remains simpler and proven

#### Warm-Start AlphaZero (2020)

- **Paper:** [arxiv.org/abs/2004.12357](https://arxiv.org/abs/2004.12357)
- **Approach:** Initialising AlphaZero with knowledge from related games or previous training runs
- **Key insight:** Warm-starting can significantly reduce training time for new games
- **Relevance:** Potential future optimisation — could warm-start Blokus training from the Tic-Tac-Toe model (though transfer between such different games is questionable)

---

### The AlphaZero Reference Implementation Landscape

AlphaBlokus is built on concepts from existing AlphaZero implementations:

| Project | Language | Games | Notable Feature |
|---------|----------|-------|----------------|
| [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | Python/PyTorch | Othello, Connect4, TTT | Clean, educational, game-agnostic |
| [alpha_zero (michaelnny)](https://github.com/michaelnny/alpha_zero) | Python/PyTorch | Go, Gomoku | Full-featured, well-documented |
| [AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl) | Julia | Connect4, others | Fast, generic framework |
| [minigo](https://github.com/tensorflow/minigo) | Python/TF | Go | Google-backed, production quality |
| [blanyal/alpha-zero](https://github.com/blanyal/alpha-zero) | Python/Keras | Othello, Connect4, TTT | Based on both AlphaGo Zero and AlphaZero papers |

AlphaBlokus draws most directly from `alpha-zero-general`'s architecture (the `IGame` protocol pattern) while targeting a game that none of these projects have attempted.

---

### Why Neural Blokus Is Hard

Several factors make Blokus a challenging target for AlphaZero, explaining why no strong results exist:

#### 1. Action Space Scale

| Game | Legal moves (typical) | Network output dim |
|------|----------------------|-------------------|
| Tic-Tac-Toe | ~5 | 10 |
| Connect 4 | ~5 | 7 |
| Chess | ~30 | 4,672 |
| Go (19×19) | ~250 | 362 |
| **Blokus Duo** | **~400–600 (early), → handful (late)** | **17,837** |

The network outputs and masks over a fixed **17,837**-dimensional policy — 3.8× the output space of Chess and 49× that of Go. That number is the clean cartesian product `14×14 grid × 91 piece-orientations + 1 pass` (see [Key Decisions Log](#key-decisions-log)). Only **13,729** of those placements ever fit entirely on the board; the rest extend off the edge and are permanently illegal. Of the on-board placements, only a few hundred are legal at any single position, and the network must learn to concentrate probability on them.

The *legal* branching factor is highest in the opening and collapses as the board fills. Both players' first move is a fixed set of **414** placements covering their starting square. Averaged over 5 random-play games (every second ply shown; data from `scripts/move_count_analysis.py`):

```
turn   mean legal moves
  0  ████████████████████ 414   ← fixed opening (covers start square)
  2  ███████████████████████ 468
  4  ███████████████████████ 466
  6  ████████████████████████ 480   ← peak ~500
  8  ██████████████████████ 451
 10  ███████████████████ 379
 12  ███████████████ 309
 14  ███████████ 219
 16  ██████ 131
 18  ████ 86
 20  ██ 43
 22  █ 20
 24  ▏ 7
 26  ▏ 3
 28  ▏ 2
 30    0   ← board full / no legal placements
```

A full game runs ~30–34 plies. The early/mid game is where both the branching factor and the subtlety of the corner-touch / no-edge constraints are greatest — which is why move generation (not inference) was historically the per-game bottleneck, and why the optimised generator targets exactly that regime (see [02-ALGORITHMS.md](02-ALGORITHMS.md)).

#### 2. Geometric Complexity

Blokus moves are inherently spatial — piece rotations, reflections, corner-touching constraints. This requires the network to develop geometric reasoning that goes beyond simple pattern recognition.

#### 3. Move Generation Speed

In Chess/Go, legal move generation is fast. In Blokus, checking whether each of 21 pieces in up to 8 orientations can be legally placed at each board position is computationally expensive. This makes MCTS simulations slower, reducing the number of simulations achievable per move.

#### 4. Sparse Rewards

A Blokus game can last 20+ moves per player before the outcome is clear. The reward signal (win/loss) comes only at the end. The value network must learn to evaluate positions with very little direct feedback about intermediate board quality.

#### 5. Asymmetric Starting Positions

Unlike Chess (symmetric) or Go (mostly symmetric), Blokus Duo has players starting in opposite corners (positions (4,4) and (9,9)). The first player has a structural advantage that the network must learn to account for.

---

### AlphaBlokus Differentiation

| Feature | Pentobi | Existing RL Projects | AlphaBlokus |
|---------|---------|---------------------|-------------|
| Neural evaluation | No | Planned/incomplete | Yes (ResNet) |
| Self-play training | No | Partial | Full pipeline |
| MCTS integration | Yes (RAVE) | Varies | Yes (PUCT) |
| Validated on simpler game first | N/A | No | Yes (Tic-Tac-Toe) |
| Published training curves | N/A | No | Planned |
| Pentobi benchmark | N/A (is Pentobi) | No | Systematic (levels 1-9) |
| Game-agnostic framework | No | Some | Yes (`IGame` protocol) |
| Symmetry augmentation | N/A | Unknown | Yes (2×, main-diagonal reflection) |
| Configurable network depth | N/A | Varies | Yes (1-8 ResNet blocks) |
| Board encoding with per-piece spatial planes | N/A | Unknown | Yes (44×14×14) |

#### The Core Bet

AlphaBlokus bets that a neural network trained through self-play can develop positional understanding that outperforms Pentobi's raw search + RAVE heuristic. The AlphaZero paper demonstrated this exact dynamic in Chess (neural evaluation + modest search > deep search + hand-crafted evaluation). The question is whether this transfers to Blokus's unique geometric and combinatorial structure.

---

### Key References

| Paper | Year | Relevance |
|-------|------|-----------|
| Silver et al., "Mastering the game of Go without human knowledge" | 2017 | AlphaGo Zero — foundation algorithm |
| Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" | 2018 | AlphaZero — generalised approach we implement |
| Gelly & Silver, "Monte-Carlo tree search and rapid action value estimation in computer Go" | 2011 | RAVE — Pentobi's core technique |
| He et al., "Deep Residual Learning for Image Recognition" | 2015 | ResNet architecture we use |
| Rosin, "Multi-Armed Bandits with Episode Context" | 2011 | PUCT formula used in MCTS |

---

## Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Validation game | Tic-Tac-Toe | Solved game, fast training, easy to verify correctness |
| Target game | Blokus Duo (not 4-player) | Two-player variant simplifies to zero-sum game, compatible with AlphaZero's framework |
| Framework design | Game-agnostic protocols | `IGame` and `INeuralNetWrapper` interfaces make adding new games easy without touching core MCTS/Coach/Arena |
| Blokus network | ResNet (not plain CNN) | Residual connections help with deeper networks needed for Blokus complexity. Configurable depth (1-8 blocks) allows experimentation |
| Board encoding | 44×14×14 (per-piece spatial planes) | 21 binary planes per player (one per piece, 1s where that piece sits on the board) + 2 aggregate planes. Follows AlphaZero convention. Piece inventory is implicit: an all-zero plane means the piece hasn't been played |
| Action space | 14×14×91 + 1 = 17,837 (network output); 13,729 ever-legal placements | The network outputs over the full cartesian product (grid positions × piece-orientation combos + pass). Most are permanently illegal: a piece anchored near an edge extends off the board, so only 13,729 placements ever fit. We keep the full 17,837-wide head anyway — it's a clean encoding and the move mask zeros out the impossible ones. Same approach as AlphaZero for Chess/Go |
| Piece orientations | Symmetry-reduced basis | Each piece stores only unique orientations (1-8) after removing rotational/reflective duplicates. Reduces 21×8=168 to 91 |
| Coordinate system | Board coords (bottom-left origin) + array indices (top-left) | Board coords match standard Blokus notation. CoordinateIndexDecoder handles conversion |
| Training data storage | Parquet (per-generation self-play + hive-partitioned metrics) | Self-play examples and all metrics are written as Parquet — one self-play file per generation (board/policy arrays byte-serialised with shape metadata in the schema), metrics hive-partitioned by generation. Efficient to write during training and to load back for analysis/reporting |
| Reporting | HTML with Plotly | Interactive charts for loss curves, arena results, timing — viewable in browser |
| Benchmark target | Pentobi | Strongest open-source Blokus AI. Uses MCTS + RAVE but no neural networks. Clear, reproducible benchmark |

---

## Stretch Goal: 4-Player Blokus

**Prerequisite:** Beat Pentobi level 9 in Blokus Duo.

If the Duo agent is strong enough to beat Pentobi, the natural next target is **Classic 4-player Blokus** on a 20×20 board. This is a fundamentally harder problem:

| Property | Blokus Duo | 4-Player Blokus |
|----------|-----------|----------------|
| Board size | 14×14 | 20×20 |
| Players | 2 | 4 |
| Zero-sum | Yes | No (coalitions possible) |
| AlphaZero-compatible | Directly | Requires modifications |
| Action space | 17,837 | ~36,000+ (larger board) |
| Game length | ~25-30 moves/player | ~20-25 moves/player |

**Key challenges:**

1. **Not zero-sum.** AlphaZero assumes two-player zero-sum games where one player's gain is the other's loss. With 4 players, implicit alliances form — two players might gang up on the leader. The value head would need to predict relative standings rather than binary win/loss
2. **Larger state space.** 20×20 board with 4 sets of 21 pieces means the board encoding and action space both grow significantly
3. **Self-play dynamics.** Training 4 agents through self-play raises questions about equilibrium — do the agents converge to a stable strategy or cycle?
4. **Evaluation is harder.** No clean "majority of 100 games" win criterion when there are 4 players

**Possible approaches:**

- **Treat as 2v2:** If two agents are paired, it reduces back to a zero-sum team game
- **Population training:** Train a population of agents that play each other, selecting the strongest
- **Reward shaping:** Use score differential (total squares placed) as a continuous reward signal instead of binary win/loss
- **Transfer learning:** Warm-start the 4-player network from the trained Duo weights

This is a genuine research-level problem — the existing literature on multi-player AlphaZero is thin. But that's what makes it interesting.

---

## Open Questions

- **Regularisation:** The loss function has no explicit regularisation term (no weight decay / L2 penalty). May improve generalisation, especially at larger network sizes
- **cpuct tuning:** The PUCT exploration constant is a fixed per-run config value. It may need tuning for Blokus's larger action space rather than carrying TTT's setting across
- **Resigning / early termination:** Self-play games are always played to a terminal position. A value-threshold resignation rule (as in AlphaZero) could cut wasted compute on decided games

> Resolved since this doc was first written: batched MCTS inference (collect K leaves per step under virtual loss, one GPU call), parallel self-play across worker processes, the precomputed-table move generator, and the 44-channel per-piece board encoding are all implemented. See [02-ALGORITHMS.md](02-ALGORITHMS.md) and [03-NEURAL-NETWORKS.md](03-NEURAL-NETWORKS.md).
