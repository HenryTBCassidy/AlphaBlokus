# AlphaBlokus — Project Overview

## Mission

Train a neural network through self-play reinforcement learning to master **Blokus Duo**, using DeepMind's AlphaZero algorithm. The trained agent should beat **Pentobi** (the strongest open-source Blokus AI) in a majority of 100 games.

**One network, zero human knowledge, superhuman Blokus.**

---

## Why Blokus?

Blokus Duo is an interesting target for AlphaZero because:

1. **Combinatorial complexity:** The action space (17,837 possible moves) dwarfs Chess (~30 legal moves per position) and approaches Go-level branching factor in the early game
2. **Spatial reasoning:** Piece placement requires geometric intuition — rotations, reflections, corner-touching constraints
3. **Nobody has done it:** As of 2026, no strong AlphaZero-style Blokus agent exists. The few attempts either abandoned neural networks (too slow) or have zero published results
4. **Clear benchmark:** Pentobi provides a well-established opponent with configurable difficulty levels (1-9)

---

## Phased Approach

### Phase 1: Tic-Tac-Toe Validation ✅ Complete

Build the full AlphaZero pipeline on a solved game to verify correctness before tackling Blokus.

| Task | Status | Notes |
|------|--------|-------|
| Core framework (MCTS, Coach, Arena) | ✅ Done | Game-agnostic via `IGame` and `INeuralNetWrapper` protocols |
| Tic-Tac-Toe game logic | ✅ Done | Board, legal moves, win detection, symmetries |
| 4-layer CNN | ✅ Done | Conv → FC → policy + value heads |
| Full training pipeline | ✅ Done | Self-play → training → arena → HTML reporting |
| Competitive with perfect play | ✅ Done | Draws/beats optimal tic-tac-toe |

**Deliverables:** Working end-to-end AlphaZero that plays competitive tic-tac-toe. Validates MCTS, training loop, arena evaluation, and reporting.

### Phase 2: Blokus Duo Implementation 🔧 In Progress

Implement Blokus Duo game logic and connect it to the existing framework.

| Task | Status | Notes |
|------|--------|-------|
| Board representation (14x14) | ✅ Done | NumPy array, coordinate conversion |
| Piece system (21 pieces, 91 orientations) | ✅ Done | PieceManager, BidirectionalDict, symmetry reduction |
| Placement validation | ✅ Done | Corner-touching, side-avoiding, bounds checking |
| Initial move caching | ✅ Done | All valid first moves pre-computed for both players |
| Placement point cache | ✅ Done | Tracks valid corners after each insertion |
| ResNet architecture | ✅ Done | Configurable depth, compiles and runs forward pass |
| Move generation algorithm | ❌ Blocked | `valid_moves()` returns empty — core logic unimplemented |
| Game termination detection | ❌ Blocked | `game_ended()` raises NotImplementedError |
| Valid move masking | ❌ Blocked | `valid_move_masking()` raises NotImplementedError |
| Symmetry augmentation | ❌ Blocked | `get_symmetries()` raises NotImplementedError |

**Estimated remaining effort (with Claude assistance):** 8-15 hours. Move generation is the hard part — the algorithm itself is conceptually straightforward (for each piece, for each orientation, try each cached placement point) but getting the edge cases right (bounds checking, overlap detection, connectivity validation) and making it fast enough for MCTS will take most of the time. The other three blocked methods are mechanical once `valid_moves()` works.

**Without Claude assistance:** 20-35 hours. Move generation has fiddly coordinate system interactions (board coords vs array indices) and the placement point cache logic needs careful integration. Debugging spatial bugs without an AI pair-programmer would be significantly slower.

### Phase 3: Training & Optimisation 📋 Not Started

Run self-play training and iterate on performance.

| Task | Hours (est.) | Notes |
|------|-------------|-------|
| Initial self-play training | 3-5 | Get the loop running, debug issues |
| Learning rate scheduler | 2-3 | Warm-up then cosine/step decay |
| MCTS optimisation | 5-8 | Valid-move-only iteration, caching |
| Parallel MCTS | 5-8 | Batched neural network inference |
| Training tuning | 5-10 | cpuct, MCTS sims, epochs, architecture |

### Phase 4: Benchmarking 📋 Not Started

Systematic evaluation against Pentobi.

| Task | Hours (est.) | Notes |
|------|-------------|-------|
| Pentobi integration | 3-5 | Automate games via GTP protocol |
| Difficulty ladder | 5-8 | Beat levels 1-9 sequentially |
| 100-game evaluation | 2-3 | Final benchmark at each difficulty |
| Elo estimation | 2-3 | Track strength across generations |

---

## Total Effort Estimate

All estimates assume Claude assistance. Solo estimates would be roughly 2-2.5× these figures.

| Phase | Hours (Claude-assisted) | Status |
|-------|------------------------|--------|
| Phase 1: Tic-Tac-Toe | ~30 | ✅ Complete |
| Phase 2: Blokus Duo game logic | 8-15 | 🔧 In progress (~60% done) |
| Phase 3: Training & optimisation | 15-25 | 📋 Not started |
| Phase 4: Benchmarking | 8-14 | 📋 Not started |
| **Remaining** | **~30-55** | |

**Where the real risk is:** Phase 2 is engineering — it's tedious but predictable. Phase 3 is where things can go sideways. Self-play RL training can fail to converge, get stuck in loops, or produce agents that exploit bugs in the game logic rather than learning real strategy. Debugging a training run that "isn't learning" is qualitatively harder than debugging a function that produces wrong output. Budget extra time for Phase 3 — the estimates above assume things go reasonably well.

---

## Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Validation game | Tic-Tac-Toe | Solved game, fast training, easy to verify correctness |
| Target game | Blokus Duo (not 4-player) | Two-player variant simplifies to zero-sum game, compatible with AlphaZero's framework |
| Framework design | Game-agnostic protocols | `IGame` and `INeuralNetWrapper` interfaces make adding new games easy without touching core MCTS/Coach/Arena |
| Blokus network | ResNet (not plain CNN) | Residual connections help with deeper networks needed for Blokus complexity. Configurable depth (1-8 blocks) allows experimentation |
| Board encoding | 14x18 (board + piece tracking) | 14x14 game board plus 2-column encoded regions per player tracking which pieces have been played. Gives the network information about remaining pieces |
| Action space | 14×14×91 + 1 = 17,837 | Grid positions × piece-orientation combos + pass. Large but finite — same approach as AlphaZero for Chess/Go |
| Piece orientations | Symmetry-reduced basis | Each piece stores only unique orientations (1-8) after removing rotational/reflective duplicates. Reduces 21×8=168 to 91 |
| Coordinate system | Board coords (bottom-left origin) + array indices (top-left) | Board coords match standard Blokus notation. CoordinateIndexDecoder handles conversion |
| Training data storage | Pickle per generation → Parquet consolidation | Pickle for fast writes during training, Parquet for efficient analysis afterwards |
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

## Open Questions / Future Work

- **Parallel MCTS:** Batching neural network inference across multiple MCTS simulations could dramatically speed up self-play. Root parallelisation vs leaf parallelisation trade-offs need investigation
- **MCTS node reuse:** Currently trees are rebuilt from scratch each game. Reusing subtrees between moves would save computation
- **Action space iteration:** MCTS currently iterates ALL 17,837 actions to find the best move. Must optimise to only iterate valid moves
- **Board representation alternatives:** Current 14x18 encoding may not be optimal. Could experiment with separate channels for each player's pieces, remaining pieces, valid placement points
- **Connect 4 as debugging step:** If Blokus move generation proves too difficult, Connect 4 could serve as an intermediate complexity game
- **Regularisation:** Current loss function has no explicit regularisation term. Weight decay or L2 penalty may improve generalisation
- **cpuct tuning:** The PUCT exploration constant is fixed at 1.0. May need to be tuned for Blokus's larger action space
