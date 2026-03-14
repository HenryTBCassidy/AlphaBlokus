# AlphaBlokus - Project State

> Last updated: 2026-03-14
> Author: Henry Cassidy (GitHub: HenryTBCassidy)

## What Is This Project?

AlphaBlokus is an implementation of the AlphaZero algorithm applied to the board game **Blokus Duo** (a two-player variant of Blokus played on a 14x14 board). The goal is to train a neural network through self-play to reach superhuman performance, benchmarked against **Pentobi** (the strongest open-source Blokus AI, which uses MCTS + RAVE heuristics but no neural networks).

**Success criterion:** Beat Pentobi in >50% of 100 games.

The project uses a phased approach: first implement and validate on **Tic-Tac-Toe** (solved game, easy to verify correctness), then scale to **Blokus Duo**.

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Core AlphaZero framework | **Complete** | MCTS, Coach, Arena, config, logging, reporting |
| Tic-Tac-Toe implementation | **Complete** | Network plays competitively with perfect play |
| Blokus Duo game logic | **Partial** | Board, pieces, placement validation work; move generation incomplete |
| Blokus Duo neural net | **Complete** | ResNet architecture defined and compiles |
| Blokus Duo integration | **Blocked** | Cannot run self-play until move generation is complete |
| Model loading/resuming | **Broken** | `load_train_examples()` raises `NotImplementedError` |

### What Works Right Now
- Full tic-tac-toe training pipeline end-to-end (self-play -> training -> arena -> reporting)
- IBoard protocol with immutable boards for both games
- 44-channel per-piece spatial encoding for BlokusDuo neural net input (`board.as_multi_channel`)
- 2-channel player-split encoding for TicTacToe neural net input
- State key on board objects (196-byte signed placement board for Blokus, flat board bytes for TicTacToe)
- Blokus board representation, piece insertion, placement validation, coordinate conversion
- Initial move caching (all valid first moves for both players are pre-computed)
- Placement point cache updating (tracks valid placement corners after each piece insertion)
- Neural network for Blokus compiles and runs forward pass
- HTML reporting with Plotly (loss curves, arena win rates, timing statistics)

### What's Broken / Incomplete
1. **`BlokusDuoBoard.valid_moves()`** returns empty list - move generation logic is unimplemented
2. **`BlokusDuoGame.valid_move_masking()`** raises `NotImplementedError` - can't mask net output
3. **`BlokusDuoGame.get_symmetries()`** raises `NotImplementedError` - no data augmentation
4. **`BlokusDuoBoard.game_ended()`** raises `NotImplementedError`
5. **Move generation from placement cache** — `BlokusDuoBoard._update_placement_points()` maintains the cache, but move generation from it is still TODO
6. **Model loading** in `main.py` raises `NotImplementedError`
7. **`main.py` uses TicTacToeGame** but imports `blokusduo.neuralnets.wrapper.NNetWrapper` - mismatch

## Architecture Overview

```
AlphaBlokus/
├── main.py                    # Entry point - loads config, creates Coach, runs learn()
├── reporting.py               # HTML report generation (Plotly: loss, arena, timings)
├── utils.py                   # AverageMeter, load_args(), setup_logging()
├── logging_config.json        # Rotating file handler + stderr, queue handler pattern
├── todo.txt                   # Original TODO list
├── eval.ipynb                 # Scratch notebook (piece testing, net viz, gameplay, plotting)
├── run_configurations/
│   ├── test_run.json          # Quick test config (2 gens, 10 eps, 2 MCTS sims)
│   └── full_run.json          # Production config (50 gens, 100 eps, 200 MCTS sims)
├── core/                      # Game-agnostic AlphaZero framework
│   ├── config.py              # RunConfig, MCTSConfig, NetConfig dataclasses
│   ├── interfaces.py          # IGame and INeuralNetWrapper Protocol classes
│   ├── mcts.py                # MCTS with PUCT formula
│   ├── coach.py               # Training orchestrator (self-play -> train -> arena)
│   └── arena.py               # Agent evaluation (alternating start positions)
├── games/
│   └── base_wrapper.py        # Base neural net wrapper shared across games
├── tictactoe/                 # Reference implementation (COMPLETE)
│   ├── board.py               # Board class with IBoard protocol (legal moves, win detection)
│   ├── game.py                # IGame implementation
│   └── neuralnets/
│       ├── net.py             # 4-layer CNN (Conv -> FC -> policy + value heads)
│       └── wrapper.py         # Training, prediction, checkpointing
└── blokusduo/                 # Target implementation (IN PROGRESS)
    ├── board.py               # BlokusDuoBoard (IBoard), Action, CoordinateIndexDecoder
    ├── game.py                # IGame implementation
    ├── pieces.py              # Piece definitions, PieceManager, BidirectionalDict
    ├── pieces.json            # 21 pieces with 91 total basis orientations
    └── neuralnets/
        ├── net.py             # ResNet (configurable blocks, conv -> res -> policy + value)
        └── wrapper.py         # Training, prediction, checkpointing
```

## Core Training Loop

```
┌─────────────────────────────────────────────────────────┐
│                    Coach.learn()                         │
│                                                         │
│  For each generation:                                   │
│                                                         │
│  1. SELF-PLAY                                           │
│     - Create fresh MCTS tree per game                   │
│     - Play num_eps games using MCTS + current net       │
│     - Temperature=1 for first temp_threshold moves      │
│     - Temperature≈0 after (exploitation)                │
│     - Augment with symmetries                           │
│     - Store (board, policy, value) tuples               │
│                                                         │
│  2. TRAINING                                            │
│     - Save current net as "temp" checkpoint             │
│     - Combine examples from recent generations          │
│     - Training window grows linearly 5 -> max_lookback  │
│     - Train net for N epochs on shuffled examples       │
│     - Loss = policy_loss + value_loss                   │
│                                                         │
│  3. ARENA EVALUATION                                    │
│     - Pit new net vs old net (both via MCTS, temp=0)    │
│     - Play num_arena_matches games, alternating starts   │
│     - If new_wins/(new_wins+old_wins) >= threshold:     │
│       → ACCEPT new net, save as "best"                  │
│     - Else:                                             │
│       → REJECT, revert to old net                       │
│                                                         │
│  After all generations:                                 │
│     - Write timing data                                 │
│     - Collect training data to parquet                  │
│     - Generate HTML report                              │
└─────────────────────────────────────────────────────────┘
```

## Key Technical Details

### MCTS Implementation (`core/mcts.py`)
- Uses 6 dictionaries keyed by string state representation:
  - `Qsa`: Q-values for (state, action) pairs
  - `Nsa`: Visit counts for (state, action) pairs
  - `Ns`: Total visit counts for states
  - `Ps`: Neural net policy predictions for states
  - `Es`: Game-ended status cache
  - `Vs`: Valid moves mask cache
- PUCT formula: `u = Q(s,a) + cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
- **Known inefficiency:** Iterates over ALL actions to find best move. For Blokus (action space = 17,837), this needs to be optimised to only iterate valid moves.
- Tree is rebuilt from scratch each self-play game (no reuse between games)

### Blokus Board Representation
- **Game board:** `_piece_placement_board` — int8 14x14 array where +piece_id = White, -piece_id = Black, 0 = empty. `as_2d` property derives the +1/-1/0 view via `np.sign()`.
- **Net input:** 44x14x14 float32 tensor (44 channels: 21 per-piece planes per player + 2 aggregate occupancy planes), built by `board.as_multi_channel(player)`
- **Action space:** 14 x 14 x 91 + 1 = 17,837 (grid positions x piece-orientation combos + pass)
- **91 piece-orientation combos:** 21 pieces, each with 1-8 basis orientations (accounting for rotational/reflective symmetry), totalling 91 unique placements

### Piece System (`blokusduo/pieces.py`)
- 21 Blokus Duo pieces defined in `pieces.json` with fill values and basis orientations
- `PieceManager` maintains a `BidirectionalDict` mapping (piece_id, orientation) <-> numeric ID
- 8 possible orientations: Identity, Rot90, Rot180, Rot270, Flip, Flip90, Flip180, Flip270
- Each piece only stores its "basis orientations" (the minimal unique set after symmetry reduction)

### Blokus Rules Implemented
- Pieces must touch at least one corner of a friendly piece
- Pieces must NOT touch any side of a friendly piece
- First move must cover the designated starting square (White: (9,9), Black: (4,4))
- Scoring: +15 for placing all pieces, +5 bonus if last piece was the 1-piece, negative for remaining piece squares
- Coordinate system: Board coordinates (origin bottom-left, y-up) vs array indices (origin top-left, row-down)

### Neural Network Architectures

**Tic-Tac-Toe (`tictactoe/neuralnets/net.py`):**
- 2 input channels (player-split encoding) -> 4 Conv2d layers (3x3 kernel) -> Flatten -> FC(1024) -> FC(512) -> split to policy head + value head
- No residual connections
- Dropout regularisation

**Blokus Duo (`blokusduo/neuralnets/net.py`):**
- Input: 44 x 14 x 14 (44-channel per-piece spatial planes)
- Initial Conv2d(44 -> num_filters, 3x3, padding=1, bias=False) + BatchNorm + ReLU
- N configurable `ResNetBlock`s (each = Conv+BN+ReLU -> Conv+BN -> residual add -> ReLU)
- **Policy head:** Conv2d(channels -> 2, 1x1) + BN + ReLU -> Flatten -> Linear(2×196 -> 17837)
- **Value head:** Conv2d(channels -> 1, 1x1) + BN + ReLU -> Flatten -> Linear(196 -> num_filters) -> ReLU -> Linear(num_filters -> 1) -> Tanh
- Output: log_softmax policy vector (17,837 dims) + value scalar (-1 to 1)

### Loss Functions
- **Policy loss:** `-sum(target * output) / batch_size` (cross-entropy with log-softmax output)
- **Value loss:** `sum((target - output)^2) / batch_size` (MSE)
- **Total loss:** policy_loss + value_loss (no weighting, no regularisation term yet)

### Training Data Storage
- **Self-play history:** Pickle files per generation (`self_play_history_{gen}.pickle`)
- **Training metrics:** Pickle files per generation, consolidated to parquet at end
- **Arena results:** Individual parquet files per generation
- **Timing data:** Single parquet file for all generations
- **Model checkpoints:** PyTorch `.pth.tar` files (best, temp, accepted/rejected per generation)

### Configuration (`run_configurations/`)

| Parameter | Test Run | Full Run |
|-----------|----------|----------|
| num_generations | 2 | 50 |
| num_eps (games per gen) | 10 | 100 |
| num_mcts_sims | 2 | 200 |
| num_arena_matches | 2 | 100 |
| batch_size | 10 | 32 |
| epochs | 1 | 5 |
| num_filters | 512 | 512 |
| num_residual_blocks | 1 | 8 |
| max_generations_lookback | 1 | 25 |
| cpuct | 1 | 1 |
| learning_rate | 0.001 | 0.001 |
| dropout | 0.3 | 0.3 |
| update_threshold | 0.55 | 0.55 |

## Dependencies
- Python 3.12+
- PyTorch (neural networks)
- NumPy (board representation, array operations)
- Pandas (data logging, parquet I/O)
- Plotly / plotly_express (HTML report charts)
- tqdm (progress bars)
- dataclass_wizard (JSON -> dataclass config loading)
- nptyping (type hints for numpy arrays, used in pieces.py)
- coloredlogs (coloured terminal logging)

## Critical Path to Working Blokus Self-Play

To get the Blokus training loop running, these must be implemented **in order**:

1. **`BlokusDuoBoard.valid_moves()`** - Aggregate valid moves from all placement points (placement cache maintained by `_update_placement_points()`, but move generation from it is still TODO)
2. **`BlokusDuoBoard.game_ended()`** - Check if neither player has legal moves
3. **`BlokusDuoGame.valid_move_masking()`** - Convert valid move list to binary action-space mask
4. **`BlokusDuoGame.get_symmetries()`** - Generate symmetric board+policy pairs for data augmentation
5. **Fix `main.py`** - Switch from TicTacToeGame to BlokusDuoGame
6. **Fix model loading** - Implement checkpoint resume functionality

## Competitive Landscape

As of 2026-03, **nobody has completed a strong AlphaZero-style Blokus agent**:
- **Pentobi** is the strongest Blokus AI (MCTS + RAVE, no neural nets, 7800+ commits)
- **nikohass/rust-socha2021** tried CNN + MCTS, explicitly abandoned neural nets (inference >100ms, too slow)
- **KubiakJakub01/Blokus-RL** is the most active attempt (209 commits, AlphaZero + PPO) but has zero published results
- **shindavid/AlphaZeroArcade** has Blokus only in test infrastructure
- This project is one of the few serious attempts at AlphaZero for Blokus

## Existing TODO List Highlights (from `todo.txt`)

Key items that remain relevant:
- Learning rate scheduler (warm-up then decay)
- Parallel MCTS for faster self-play
- ~~Board representation conversion~~ (done — `board.as_multi_channel` builds 44-channel tensor)
- Move masking for net output
- ResNet option for tic-tac-toe (benchmarking)
- Regularisation term on loss function
- Research: cpuct tuning, MCTS sims vs game progress, epoch count
- Presenting: Elo tracking, game length, win rate vs Pentobi, draw rate

## eval.ipynb Summary

The notebook is a development scratchpad containing:
- Piece insertion testing and board visualisation
- Move generation algorithm design notes (extensive markdown cells with caching strategies)
- Neural net architecture inspection (parameter counts, torchviz graphs)
- Data loading and plotting experiments
- Human vs computer tic-tac-toe gameplay
- Window size function visualisation
- Training data collection testing

The move generation algorithm notes in the notebook are particularly valuable - they outline a caching strategy for efficient legal move computation that hasn't been fully implemented yet.
