# AlphaBlokus — Handoff Context

> This document provides context for future Claude instances (or other AI assistants) continuing work on this project. Read this first, then skim the other docs as needed.

---

## What Is This Project?

AlphaBlokus trains a neural network through self-play reinforcement learning to master **Blokus Duo**, using DeepMind's AlphaZero algorithm. The trained agent should beat **Pentobi** (the strongest open-source Blokus AI) in a majority of 100 games.

**One network, zero human knowledge, superhuman Blokus.**

---

## Who Is Henry?

- **Background:** Data scientist at Transak (trading and pricing). Previously quantitative researcher at G-Research (London quant fund)
- **Technical skills:** Strong Python, software engineering, statistics. Has built an AlphaZero tic-tac-toe implementation (this project) and a mini Shakespeare GPT (character-level transformer)
- **ML experience:** Conceptual understanding is solid. Hands-on production ML training experience is limited — this is his first major ML project beyond tutorials
- **Tools:** Unlimited Claude and Codex access. Uses AI assistance heavily for implementation
- **Communication style:** Results-oriented. Dislikes overcomplication and tutorial-mode explanations. Will push back if you're adding unnecessary complexity. Prefers "do the thing that works best" over "let's try everything for learning"
- **Game knowledge:** Understands Blokus from a player's perspective but isn't a competitive player

---

## Current State

### What's Done

| Component | Status | Notes |
|-----------|--------|-------|
| Core framework (MCTS, Coach, Arena) | ✅ Done | Game-agnostic via `IGame` and `INeuralNetWrapper` protocols |
| Tic-Tac-Toe game logic | ✅ Done | Board, legal moves, win detection, 8 symmetries |
| Tic-Tac-Toe CNN | ✅ Done | 4-layer CNN, verified competitive with perfect play |
| Full training pipeline | ✅ Done | Self-play → training → arena → HTML reporting |
| Blokus board representation | ✅ Done | 14×14 board, 14×18 net input with piece tracking |
| Piece system | ✅ Done | 21 pieces, 91 basis orientations, PieceManager, BidirectionalDict |
| Placement validation | ✅ Done | Corner-touching, side-avoiding, bounds checking |
| Initial move caching | ✅ Done | All valid first moves pre-computed for both players |
| Placement point cache | ✅ Done | Tracks valid corners after each piece insertion |
| ResNet architecture | ✅ Done | Configurable depth (1-8 blocks), compiles and runs forward pass |
| Project documentation | ✅ Done | This docs/ folder |

### What's Blocked

| Component | Status | Blocking Issue |
|-----------|--------|---------------|
| Move generation (`valid_moves()`) | ❌ Blocked | Core logic unimplemented — returns empty list |
| Game termination (`game_ended()`) | ❌ Blocked | Raises `NotImplementedError` |
| Valid move masking (`valid_move_masking()`) | ❌ Blocked | Raises `NotImplementedError` |
| Symmetry augmentation (`get_symmetries()`) | ❌ Blocked | Raises `NotImplementedError` |

### What Hasn't Been Started

- Training on Blokus (blocked by move generation)
- MCTS optimisation (valid-move-only iteration, caching)
- Parallel MCTS / batched inference
- Pentobi integration and benchmarking
- Learning rate scheduling

### Immediate Next Steps

1. **Implement `valid_moves()`** — This is the critical path. The placement point cache infrastructure exists but `populate_moves_for_new_placement()` has TODO stubs. The algorithm needs to: for each remaining piece, for each basis orientation, check if it can be legally placed at each cached placement point
2. **Implement `game_ended()`** — Game ends when both players have no legal moves (both `valid_moves()` return empty) OR a player has placed all pieces
3. **Implement `valid_move_masking()`** — Convert `valid_moves()` output (list of Actions) into a binary mask over the 17,837-dimensional action space
4. **Implement `get_symmetries()`** — Blokus Duo board has 4 symmetries: identity, 180° rotation, vertical flip + 90° rotation, vertical flip + 270° rotation. Must transform both board AND policy vector
5. **Run first self-play training** — Even a few generations will validate the pipeline end-to-end
6. **Optimise MCTS** — The current MCTS iterates all 17,837 actions per simulation. Must restrict to valid moves only

---

## Architecture Quick Reference

### File Layout

```
AlphaBlokus/
├── core/                          # Game-agnostic framework
│   ├── mcts.py                    # Monte Carlo Tree Search (PUCT)
│   ├── coach.py                   # Training loop orchestrator
│   ├── arena.py                   # Network vs network evaluation
│   ├── config.py                  # RunConfig, MCTSConfig, NetConfig
│   └── interfaces.py              # IGame, INeuralNetWrapper protocols
├── blokusduo/                     # Blokus Duo implementation
│   ├── game.py                    # BlokusDuoGame, BlokusDuoBoard, Player, Action
│   ├── pieces.py                  # Piece, PieceManager, BidirectionalDict
│   ├── pieces.json                # 21 piece definitions with orientations
│   └── neuralnets/
│       ├── net.py                 # AlphaBlokusDuo ResNet
│       └── wrapper.py            # Training, prediction, checkpointing
├── tictactoe/                     # Tic-Tac-Toe (Phase 1 validation)
│   ├── board.py                   # TicTacToeBoard
│   ├── game.py                    # TicTacToeGame (complete IGame impl)
│   └── neuralnets/
│       ├── net.py                 # AlphaTicTacToe CNN
│       └── wrapper.py            # Training, prediction, checkpointing
├── run_configurations/            # JSON configs for training runs
│   ├── test_run.json              # Quick validation config
│   └── full_run.json              # Full training config
├── eval.ipynb                     # Development scratchpad
├── utils.py                       # AverageMeter and helpers
├── docs/                          # Project documentation (you are here)
├── README.md                      # SEO-optimised project README
└── PROJECT-STATE.md               # Detailed technical state summary
```

### Key Numbers

| Property | Value |
|----------|-------|
| Board size | 14 × 14 |
| Net input size | 14 × 18 (board + piece tracking) |
| Pieces per player | 21 |
| Total piece-orientation combos | 91 (symmetry-reduced from 168) |
| Action space | 14 × 14 × 91 + 1 = 17,837 |
| ResNet channels | 512 (configurable) |
| ResNet blocks | 4 (configurable, 1-8) |
| Approximate parameters | ~28M (512ch, 4 blocks) |

### Training Loop (from Coach)

```
for generation in range(num_generations):
    1. Self-play: Play num_eps games using MCTS + current network
       → Collect training examples: [(board, π, v), ...]

    2. Train: Update network weights on accumulated examples
       → Policy loss (cross-entropy) + Value loss (MSE)

    3. Arena: New network vs old network
       → If new wins > update_threshold: accept, else reject

    4. Log: Timing, losses, arena results → Pickle/Parquet
```

---

## Key Architecture Decisions (and Why)

### Why ResNet over plain CNN?

Residual connections prevent gradient degradation in deeper networks. Blokus needs more capacity than Tic-Tac-Toe's 4-layer CNN. The AlphaZero paper uses ResNet — we follow the proven recipe. Configurable depth (1-8 blocks) lets us experiment without code changes.

### Why 14×18 board encoding (not 14×14)?

The network needs to know which pieces each player has already played. The simplest approach: append a 14×2 encoded region per player to the board array. When piece #k is played, slot k in the region is set to the player's value. Cost: 4 extra columns. Benefit: the network can reason about remaining pieces.

### Why 91 piece-orientations (not 168)?

Each piece has up to 8 orientations (4 rotations × 2 flips), but many are duplicates due to symmetry. The 1-square piece has only 1 unique orientation. A symmetric 2×2 square has only 1. Asymmetric pentominoes have all 8. After symmetry reduction: 91 unique orientations. This keeps the action space as small as possible while being complete.

### Why game-agnostic framework?

The `IGame` and `INeuralNetWrapper` protocols let us add new games without touching MCTS, Coach, or Arena. Tic-Tac-Toe validated the framework before we invested time in Blokus game logic. Could extend to Connect 4, Othello, or other games later.

### Why no parallel MCTS yet?

Premature optimisation. Get the pipeline working correctly first, then optimise. The current bottleneck is implementing move generation, not MCTS speed. Parallelism is documented in `docs/01-ALGORITHMS.md` for when we're ready.

### Why canonical form = player × board?

Multiplying the board by the current player's value means "my pieces" are always +1 and "opponent's pieces" are always -1. The network sees the same representation regardless of which colour it's playing. Simple, effective, standard in two-player zero-sum games.

---

## Gotchas and Lessons Learned

1. **Move generation is the hard part.** The board, pieces, placement validation, and neural network are all done. The blocker is efficiently generating legal moves from the placement point cache. This was the predicted bottleneck in the project plan and it proved correct.

2. **The action space is huge.** 17,837 actions means the policy head's final FC layer alone is ~9M parameters. MCTS iterating over all actions (even illegal ones) is O(17,837) per simulation. Must optimise to only iterate valid moves.

3. **`string_representation` uses `board.tobytes()`.** MCTS hashes board states using the raw numpy byte representation. This works but produces long keys. A Zobrist hash would be more efficient — see `docs/01-ALGORITHMS.md`.

4. **Piece-orientation IDs have gaps.** The `BidirectionalDict` assigns IDs sequentially but skips IDs when moving to the next piece (the populate_lookup loop increments `i` one extra time between pieces). Be careful when mapping between action indices and piece-orientation IDs.

5. **No optimizer state in checkpoints.** The wrapper saves only `state_dict`, not the Adam optimizer's momentum buffers. This means training "restarts" each generation. Acceptable for now, but saving optimizer state would give smoother training.

6. **`calc_conv2d_output` exists but is only used once.** The Blokus ResNet uses it to compute the flattened size for the FC layers. The Tic-Tac-Toe net hardcodes the computation. If you change kernel sizes or padding, use this helper.

7. **Coordinate systems are confusing.** Board coordinates use bottom-left origin (matching standard Blokus notation). Array indices use top-left origin (matching numpy). `CoordinateIndexDecoder` handles conversion. Always be explicit about which system you're in.

8. **`eval.ipynb` has move generation design notes.** Check the notebook for Henry's thinking about the move generation algorithm before implementing.

---

## Things NOT to Do

- **Don't rewrite the framework.** MCTS, Coach, Arena work correctly — they're validated on Tic-Tac-Toe. Focus on completing the Blokus game logic
- **Don't over-engineer move generation.** Get it working correctly first, optimise second. A correct but slow `valid_moves()` is infinitely better than a fast but buggy one
- **Don't add 4-player Blokus support until Duo beats Pentobi level 9.** It's a stretch goal, not a current priority. 4-player would require fundamental changes (coalition dynamics, non-zero-sum) — see `docs/00-PROJECT-OVERVIEW.md` for details
- **Don't switch to Transformer/Mamba/ViT.** ResNet is proven for AlphaZero. Novel architectures are research risk we don't need
- **Don't implement learning rate scheduling before the first successful training run.** Get the basics working, then refine
- **Don't build a web UI.** The HTML reporting that already exists is sufficient for analysis
- **Don't pad options for educational value.** If one approach clearly works, recommend it. Henry will push back on unnecessary alternatives
- **Don't forget to read `PROJECT-STATE.md`.** It has detailed technical context complementing this handoff doc

---

## Useful Files for Context

| File | Purpose | When to Read |
|------|---------|-------------|
| `PROJECT-STATE.md` | Detailed technical state dump | First, alongside this doc |
| `docs/00-PROJECT-OVERVIEW.md` | Mission, phases, decisions log | Understanding the big picture |
| `docs/01-ALGORITHMS.md` | MCTS, self-play, caching strategies | Before working on MCTS optimisation |
| `docs/02-NEURAL-NETWORKS.md` | ResNet architecture, loss functions | Before touching neural network code |
| `docs/03-EVALUATION-PLAN.md` | Metrics, Pentobi benchmarking, success criteria | Before training or evaluation work |
| `docs/04-COMPETITIVE-LANDSCAPE.md` | Existing Blokus AI projects | Understanding what's been tried |
| `eval.ipynb` | Move generation design scratchpad | Before implementing `valid_moves()` |
| `run_configurations/full_run.json` | Production training config | Before running training |
| `blokusduo/pieces.json` | All 21 piece definitions | Before working on piece-related logic |
