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

**Estimated remaining effort:** 15-25 hours (move generation is the hard part)

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

| Phase | Hours | Status |
|-------|-------|--------|
| Phase 1: Tic-Tac-Toe | ~30 | ✅ Complete |
| Phase 2: Blokus Duo game logic | 15-25 | 🔧 In progress (~60% done) |
| Phase 3: Training & optimisation | 20-34 | 📋 Not started |
| Phase 4: Benchmarking | 12-19 | 📋 Not started |
| **Remaining** | **~50-80** | |

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

## Open Questions / Future Work

- **Parallel MCTS:** Batching neural network inference across multiple MCTS simulations could dramatically speed up self-play. Root parallelisation vs leaf parallelisation trade-offs need investigation
- **MCTS node reuse:** Currently trees are rebuilt from scratch each game. Reusing subtrees between moves would save computation
- **Action space iteration:** MCTS currently iterates ALL 17,837 actions to find the best move. Must optimise to only iterate valid moves
- **Board representation alternatives:** Current 14x18 encoding may not be optimal. Could experiment with separate channels for each player's pieces, remaining pieces, valid placement points
- **Connect 4 as debugging step:** If Blokus move generation proves too difficult, Connect 4 could serve as an intermediate complexity game
- **4-player Blokus:** After solving Duo, extending to 4-player Blokus would require significant changes (no longer zero-sum, coalition dynamics)
- **Regularisation:** Current loss function has no explicit regularisation term. Weight decay or L2 penalty may improve generalisation
- **cpuct tuning:** The PUCT exploration constant is fixed at 1.0. May need to be tuned for Blokus's larger action space
