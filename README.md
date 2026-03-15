# AlphaBlokus

**An AlphaZero implementation for Blokus Duo — training a neural network through self-play to master the board game Blokus.**

> Status: **In Progress** — Tic-Tac-Toe pipeline complete, Blokus Duo game logic under development

---

## What is AlphaBlokus?

AlphaBlokus applies DeepMind's [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) algorithm to **Blokus Duo**, a two-player territorial board game played on a 14x14 grid. The system learns to play Blokus entirely through self-play reinforcement learning — no human game data, no handcrafted heuristics.

The goal: **beat [Pentobi](https://pentobi.sourceforge.io/)** (the strongest open-source Blokus AI, which uses MCTS + RAVE heuristics) in a majority of 100 games.

### How It Works

1. **Self-play:** A neural network plays games against itself using Monte Carlo Tree Search (MCTS) to generate training data
2. **Training:** The network learns to predict move probabilities and game outcomes from self-play positions
3. **Evaluation:** New network versions are pitted against previous versions in an arena — only improvements are kept
4. **Iteration:** Repeat for many generations until the network converges to strong play

This is the same approach that achieved superhuman performance in Chess, Shogi, and Go — applied here to the combinatorially complex domain of Blokus.

---

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Core AlphaZero framework | **Complete** | MCTS, Coach, Arena, config, logging, reporting |
| Tic-Tac-Toe implementation | **Complete** | Network plays competitively with perfect play |
| Blokus Duo game logic | **Partial** | Board, pieces, placement validation work; move generation incomplete |
| Blokus Duo neural net | **Complete** | ResNet architecture defined and compiles |
| Blokus Duo integration | **Blocked** | Cannot run self-play until move generation is complete |

### What Works

- Full tic-tac-toe training pipeline end-to-end (self-play → training → arena → reporting)
- IBoard protocol with immutable boards for both games
- 44-channel per-piece spatial encoding for BlokusDuo neural net input (`board.as_multi_channel`)
- 2-channel player-split encoding for TicTacToe neural net input
- State key on board objects (196-byte signed int8 placement board for Blokus, flat board bytes for TicTacToe)
- Blokus board representation, piece insertion, placement validation, coordinate conversion
- Initial move caching (all valid first moves for both players pre-computed)
- Placement point cache (tracks valid placement corners after each piece insertion)
- Neural network for Blokus compiles and runs forward pass (verified by tests)
- HTML reporting with Plotly (loss curves, arena win rates, timing statistics)

### What's Blocked

1. **`BlokusDuoBoard.valid_moves()`** — move generation logic unimplemented (returns empty)
2. **`BlokusDuoGame.valid_move_masking()`** — raises `NotImplementedError`
3. **`BlokusDuoGame.get_symmetries()`** — raises `NotImplementedError`
4. **`BlokusDuoBoard.game_ended()`** — raises `NotImplementedError`
5. **Model loading** in `main.py` — raises `NotImplementedError`

### Critical Path

To get Blokus self-play running, these must be implemented **in order**:

1. `BlokusDuoBoard.valid_moves()` — aggregate valid moves from placement point cache
2. `BlokusDuoBoard.game_ended()` — check if neither player has legal moves
3. `BlokusDuoGame.valid_move_masking()` — convert valid moves to binary action-space mask
4. `BlokusDuoGame.get_symmetries()` — generate symmetric board+policy pairs for data augmentation
5. Fix `main.py` — switch from TicTacToeGame to BlokusDuoGame
6. Fix model loading — implement checkpoint resume

---

## Features

- **Game-agnostic AlphaZero framework** — modular `IGame`, `IBoard`, and `INeuralNetWrapper` interfaces make it easy to add new games
- **Complete Tic-Tac-Toe implementation** — validates the full training pipeline end-to-end (self-play, training, arena evaluation)
- **ResNet architecture** — configurable depth and width with residual blocks, batch normalisation, and dual policy/value heads
- **MCTS with PUCT** — Upper Confidence bounds applied to Trees for balanced exploration and exploitation
- **HTML training reports** — interactive Plotly dashboards showing loss curves, arena win rates, and timing statistics
- **Dynamic training window** — linearly growing lookback over self-play history to balance recency and diversity
- **Temperature scheduling** — exploration early in games, exploitation in later moves

---

## Architecture

```
AlphaBlokus/
├── core/                       # Game-agnostic AlphaZero framework
│   ├── mcts.py                 # Monte Carlo Tree Search with PUCT
│   ├── coach.py                # Training orchestrator (self-play → train → arena)
│   ├── arena.py                # Model evaluation with alternating start positions
│   ├── config.py               # Configuration dataclasses
│   └── interfaces.py           # IGame, IBoard, and INeuralNetWrapper protocols
├── games/
│   ├── base_wrapper.py         # Shared neural net wrapper (train, predict, checkpoint)
│   ├── tictactoe/              # Reference implementation (complete)
│   │   ├── board.py            # Immutable board (IBoard) — 2-channel encoding
│   │   ├── game.py             # Tic-Tac-Toe game logic (IGame)
│   │   └── neuralnets/         # 4-layer CNN (Conv → FC → policy + value)
│   └── blokusduo/              # Target implementation (in progress)
│       ├── board.py            # Immutable board (IBoard) — 44-channel encoding
│       ├── game.py             # Blokus Duo game logic (IGame)
│       ├── pieces.py           # 21 pieces, 91 orientations, symmetry reduction
│       ├── pieces.json         # Piece definitions with basis orientations
│       └── neuralnets/         # ResNet (configurable blocks → policy + value)
├── docs/                       # Project documentation
│   ├── guides/                 # Conventions, process, AI assistant context
│   └── plans/                  # Active checklists and archived plans
├── notebooks/                  # Jupyter notebooks (eval.ipynb)
├── run_configurations/         # JSON configs for test and production runs
├── reporting.py                # HTML report generation with Plotly
└── main.py                     # Entry point
```

---

## Technical Details

### Neural Network

| Component | Tic-Tac-Toe | Blokus Duo |
|-----------|-------------|------------|
| Architecture | 4-layer CNN | ResNet (configurable depth) |
| Input | 2 x 3 x 3 (2-channel) | 44 x 14 x 14 (per-piece spatial planes) |
| Policy output | 10 actions | 17,837 actions |
| Value output | Scalar ∈ [-1, 1] | Scalar ∈ [-1, 1] |
| Residual blocks | None | 1–8 (configurable) |

The Blokus action space of **17,837** comes from 14 × 14 grid positions × 91 piece-orientation combinations + 1 pass action. The 91 orientations represent the 21 Blokus pieces, each with 1–8 unique basis orientations after symmetry reduction (rotations and reflections).

### MCTS

- PUCT formula: `u = Q(s,a) + cpuct · P(s,a) · √N(s) / (1 + N(s,a))`
- Configurable simulation count (2 for testing, 200+ for production)
- Temperature-based move selection: `τ=1` for exploration, `τ→0` for exploitation

### Training Loop

```
For each generation:
  1. Self-play N games using MCTS + current network
  2. Augment positions with symmetries
  3. Train network on recent self-play history
  4. Evaluate new network vs previous best in arena
  5. Accept/reject based on win rate threshold
```

---

## Blokus Duo Rules

Blokus Duo is played on a 14x14 board where two players take turns placing polyomino pieces:

- Each player has **21 pieces** (1 monomino through 5 pentominoes)
- Pieces must touch at least one **corner** of a friendly piece
- Pieces must **not** touch any **side** of a friendly piece
- First move must cover the designated starting square
- The game ends when neither player can place a piece
- Score = squares placed (with bonuses for placing all pieces)

---

## Getting Started

### Prerequisites

- Python 3.12+
- PyTorch
- NumPy, Pandas, Plotly, tqdm

### Installation

```bash
git clone https://github.com/HenryTBCassidy/AlphaBlokus.git
cd AlphaBlokus
uv sync
```

### Running Tic-Tac-Toe Training

```bash
python main.py --config run_configurations/test_run.json
```

---

## Documentation

Detailed documentation lives in the [`docs/`](docs/) folder.

### Reference Docs (`docs/`)

| Document | Description |
|----------|-------------|
| [01 — Background](docs/01-BACKGROUND.md) | Why Blokus, competitive landscape, key decisions |
| [02 — Algorithms](docs/02-ALGORITHMS.md) | MCTS implementation, self-play, arena evaluation, caching strategies |
| [03 — Neural Networks](docs/03-NEURAL-NETWORKS.md) | ResNet and CNN architectures, board encoding, loss functions |
| [04 — Blokus Duo](docs/04-BLOKUS-DUO.md) | Game rules, 21 pieces, 91 orientations, symmetry analysis |
| [05 — Evaluation](docs/05-EVALUATION.md) | Training diagnostics, Pentobi benchmarking, headline metrics |
| [06 — Interfaces](docs/06-INTERFACES.md) | Pentobi GTP adapter, human-playable UI, translation layer |
| [07 — Data Storage](docs/07-DATA-STORAGE.md) | Parquet format, metrics tables, checkpoints |

### Guides (`docs/guides/`)

| Document | Description |
|----------|-------------|
| [Style Guide](docs/guides/STYLE-GUIDE.md) | Code conventions, naming, type system, testing |
| [Plan Format](docs/guides/PLAN-FORMAT.md) | How to write implementation plans |
| [AI Context](docs/guides/AI-CONTEXT.md) | Context for Claude / AI assistants |

### Plans (`docs/plans/`)

| Document | Description |
|----------|-------------|
| [Bug Fixes](docs/plans/bug-fixes.md) | Interface mismatches, MCTS performance, board bugs, training pipeline fixes |
| [Multi-Channel Board Encoding](docs/plans/archive/multi-channel-board-encoding.md) | IBoard protocol, immutable boards, 44-channel BlokusDuo / 2-channel TicTacToe (complete) |
| [Structural Refactor](docs/plans/archive/structural-refactor.md) | Project structure, logging, package management, naming, testing strategy (complete) |

---

## Roadmap

- [x] Core AlphaZero framework (MCTS, Coach, Arena)
- [x] Tic-Tac-Toe implementation and validation
- [x] Blokus Duo board representation and piece system
- [x] Blokus Duo neural network architecture
- [x] IBoard protocol with immutable boards and multi-channel neural net encoding
- [ ] Architecture review fixes (MCTS optimisation, interface contracts, piece ID mapping)
- [ ] Blokus Duo move generation with caching
- [ ] End-to-end Blokus Duo self-play training
- [ ] Learning rate scheduling
- [ ] Parallel MCTS for faster self-play
- [ ] Benchmark against Pentobi at increasing difficulty levels

---

## Inspiration and References

This project builds on the ideas and code from:

- Silver, D. et al. — [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961) (AlphaGo, 2016)
- Silver, D. et al. — [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero, 2017)
- Silver, D. et al. — [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404) (AlphaZero, 2018)
- Schrittwieser, J. et al. — [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://www.nature.com/articles/s41586-020-03051-4) (MuZero, 2020)
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) by Surag Nair — Game-agnostic AlphaZero framework
- [alpha_zero](https://github.com/michaelnny/alpha_zero) by Michael Nny — Clean AlphaZero implementation
- [Pentobi](https://pentobi.sourceforge.io/) — Open-source Blokus AI and benchmark target

---

## License

TBD
