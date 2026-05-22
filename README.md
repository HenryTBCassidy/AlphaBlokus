# AlphaBlokus

**An AlphaZero implementation for Blokus Duo — training a neural network through self-play to master the board game Blokus.**

> Status: **In Progress** — Core framework, full reporting stack, and Tic-Tac-Toe end-to-end training all validated on GPU. Blokus Duo game logic mostly complete; final game-logic gap (`get_symmetries()`) is the next algorithmic step before Blokus training.

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
| Core AlphaZero framework | **Complete** | MCTS, Coach, Arena, config, logging |
| Tic-Tac-Toe implementation | **Complete** | Network reaches near-perfect play (~98% top-1 agreement with minimax oracle) within 1-2 generations |
| Blokus Duo game logic | **Mostly complete** | Board, pieces, placement validation, move generation, game-end detection all working. `get_symmetries()` is the remaining gap. |
| Blokus Duo neural net | **Complete** | ResNet architecture defined and compiles |
| Reporting stack | **Complete** | Interactive HTML report (Plotly + arena replays) + W&B dashboard with 7-section workspace built programmatically |
| Remote GPU training | **Validated** | TTT runs end-to-end on home PC (RTX 3060 Ti) via Tailscale + WSL2 + systemd-user; 10-gen run completes in ~4 min |
| End-to-end Blokus training | **Pending** | Blocked on `get_symmetries()` |

### What Works

**Training pipeline**
- Full Tic-Tac-Toe training pipeline end-to-end: self-play → training → arena → strength evaluation
- Score-based acceptance criterion (chess-style: `(wins + 0.5·draws) / total ≥ threshold`) — handles forced-draw games cleanly
- Reproducible runs via single `seed` config field propagating to numpy / torch / CUDA
- Frozen gen-0 baseline anchoring Elo measurements
- Minimax-oracle eval set for TTT (target policy = uniform over optimal actions; target value = true minimax value ∈ {-1, 0, +1})

**Game logic**
- IBoard protocol with immutable boards for both games
- 44-channel per-piece spatial encoding for BlokusDuo neural net input (`board.as_multi_channel`)
- 2-channel player-split encoding for TicTacToe neural net input
- State key on board objects (196-byte signed int8 placement board for Blokus, flat board bytes for TicTacToe)
- Blokus board representation, piece insertion, placement validation, coordinate conversion
- Initial move caching + placement point cache for fast move generation
- BlokusDuo move generation, masking, and game-end detection (merged via PR #6)
- Neural network for Blokus compiles and runs forward pass (verified by tests)

**Reporting**
- Interactive HTML report with Plotly: loss curves, Elo curve, policy entropy, top-1/top-5 agreement vs minimax, value-head reliability diagram, arena replay viewer with player-tinted MCTS policy boards
- W&B dashboard with 7-section workspace built via `wandb-workspaces` SDK (`scripts/setup_wandb_workspace.py`)
- Headline metrics: `arena/acceptance_rate`, `minimax/win_rate`, `elo/rating`, `progress/eta_seconds`
- Per-batch and per-episode metrics correctly wired to custom step_metrics (no W&B auto-step sawtooths)
- `play_ttt.py`, `arena_run.py`, `replay.py` — CLI tools for ad-hoc evaluation and game inspection

### Remaining Game Logic

1. **`BlokusDuoGame.get_symmetries()`** — raises `NotImplementedError`. Needed for data augmentation by symmetric board+policy pairs. Only 1 non-trivial symmetry (180° rotation) for Blokus Duo, unlike TTT's 8.
2. **Model checkpoint resume** in `main.py` — currently raises when `load_model: true`.

### Next Up

Game logic and Blokus training:

1. Implement `BlokusDuoGame.get_symmetries()` (the last algorithmic gap).
2. Add a position-symmetry diagnostic to the post-run report (compares raw network policy on a pair of symmetric Blokus positions to catch directional biases).
3. Tiny Blokus Duo smoke run on the PC to validate the pipeline end-to-end with the new game.
4. Design the first real Blokus training run — net size, sim count, generation count budget.
5. Eventually: benchmark against Pentobi at increasing difficulty levels.

---

## Features

- **Game-agnostic AlphaZero framework** — modular `IGame`, `IBoard`, and `INeuralNetWrapper` interfaces make it easy to add new games
- **Complete Tic-Tac-Toe implementation** — validates the full training pipeline end-to-end (self-play, training, arena, strength evaluation)
- **ResNet architecture** — configurable depth and width with residual blocks, batch normalisation, and dual policy/value heads
- **MCTS with PUCT** — Upper Confidence bounds applied to Trees for balanced exploration and exploitation
- **Reproducible runs** — single `seed` config field seeds numpy, torch, and CUDA RNGs at training start
- **Score-based acceptance** — chess-style acceptance criterion that handles forced-draw games correctly
- **Per-game oracle benchmarks** — TTT uses minimax as a perfect-play reference; framework supports adding game-specific oracles
- **Live + offline reporting** — W&B dashboard for streaming metrics during a run, plus a self-contained HTML report with interactive arena replays for post-mortem analysis
- **Frozen baseline Elo tracking** — random-init network anchored at run start, used as the fixed strength reference across generations
- **Temperature scheduling** — exploration early in games, exploitation in later moves
- **Dynamic training window** — linearly growing lookback over self-play history to balance recency and diversity

---

## Architecture

```
AlphaBlokus/
├── core/                       # Game-agnostic AlphaZero framework
│   ├── mcts.py                 # Monte Carlo Tree Search with PUCT
│   ├── coach.py                # Training orchestrator (self-play → train → arena → strength eval)
│   ├── arena.py                # Model evaluation with alternating start positions + replay recording
│   ├── players.py              # Player abstraction: RandomPlayer, NetworkPlayer, ...
│   ├── storage.py              # Metrics collector — parquet writes + W&B mirroring
│   ├── config.py               # Configuration dataclasses (RunConfig, MCTSConfig, NetConfig, WandbConfig)
│   └── interfaces.py           # IGame, IBoard, and INeuralNetWrapper protocols
├── games/
│   ├── base_wrapper.py         # Shared neural net wrapper (train, predict, checkpoint)
│   ├── tictactoe/              # Reference implementation (complete)
│   │   ├── board.py            # Immutable board (IBoard) — 2-channel encoding
│   │   ├── game.py             # Tic-Tac-Toe game logic (IGame)
│   │   ├── minimax.py          # Perfect-play minimax oracle (negamax + memoisation)
│   │   └── neuralnets/         # 4-layer CNN (Conv → FC → policy + value)
│   └── blokusduo/              # Target implementation (in progress)
│       ├── board.py            # Immutable board (IBoard) — 44-channel encoding
│       ├── game.py             # Blokus Duo game logic (IGame)
│       ├── pieces.py           # 21 pieces, 91 orientations, symmetry reduction
│       ├── pieces.json         # Piece definitions with basis orientations
│       └── neuralnets/         # ResNet (configurable blocks → policy + value)
├── reporting/                  # HTML report rendering
│   ├── training.py             # Top-level report builder (Plotly + embedded JSON)
│   ├── display.py              # IBoardRenderer protocol + get_renderer factory
│   ├── display_tictactoe.py    # TTT board + policy HTML rendering
│   └── display_blokusduo.py    # Blokus board + top-K move HTML rendering
├── scripts/                    # Ad-hoc CLI tools
│   ├── arena_run.py            # Head-to-head between any two players
│   ├── play_ttt.py             # Human vs trained model on the terminal
│   ├── replay.py               # Step-through a recorded arena game
│   └── setup_wandb_workspace.py  # Build the W&B dashboard layout via SDK
├── docs/                       # Project documentation
│   ├── guides/                 # Conventions, process, remote training runbook
│   └── plans/                  # Active checklists and archived plans
├── notebooks/                  # Jupyter notebooks (eval.ipynb)
├── run_configurations/         # JSON configs for test and production runs
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

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- (Optional, for GPU training) NVIDIA GPU with CUDA + a recent PyTorch wheel

### Installation

```bash
git clone https://github.com/HenryTBCassidy/AlphaBlokus.git
cd AlphaBlokus
uv sync
```

### Running Tic-Tac-Toe Training

```bash
# Quick smoke test (CPU, no W&B, finishes in seconds)
uv run python main.py --config run_configurations/test_run.json

# Full Mac demo run (CPU, with W&B, ~10 min)
uv run python main.py --config run_configurations/ttt_mac_demo.json

# Strong PC run (CUDA, 25 generations, ~40 min on RTX 3060 Ti)
uv run python main.py --config run_configurations/ttt_pc_strong.json
```

After a run completes, the interactive HTML report lives at `temp/<run_name>/Reporting/report.html`.

### Running on a remote GPU

See [`docs/guides/REMOTE-TRAINING.md`](docs/guides/REMOTE-TRAINING.md) for the Tailscale + WSL2 + systemd-user workflow used for unattended training on a home PC.

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
| [07 — Data Storage](docs/07-DATA-STORAGE.md) | Parquet format, metrics tables, W&B integration, dashboard layout |
| [08 — Training Estimates](docs/08-TRAINING-ESTIMATES.md) | Wall-clock estimates for different configs and hardware |
| [09 — Compute Options](docs/09-COMPUTE-OPTIONS.md) | Local + cloud hardware comparison, hourly rates, estimated cost per training run, phasing recommendation |

### Guides (`docs/guides/`)

| Document | Description |
|----------|-------------|
| [Style Guide](docs/guides/STYLE-GUIDE.md) | Code conventions, naming, type system, testing |
| [Plan Format](docs/guides/PLAN-FORMAT.md) | How to write implementation plans |
| [Remote Training](docs/guides/REMOTE-TRAINING.md) | Tailscale + WSL2 + systemd-user runbook for unattended GPU training |
| [AI Context](docs/guides/AI-CONTEXT.md) | Context for Claude / AI assistants |

### Plans (`docs/plans/`)

Active plans live at the top of `docs/plans/`; completed plans are archived under `docs/plans/archive/` as a historical record of what has shipped. Recent highlights:

| Document | Description |
|----------|-------------|
| [Move generation further optimisation](docs/plans/move-gen-further-optimisation.md) | Deferred work on Cython / bitboard rewrites of valid move generation |
| [archive/Reporting overhaul](docs/plans/archive/reporting-overhaul.md) | The full reporting + W&B + arena-replay + Elo + minimax push that brought the project to its current state |
| [archive/Valid move algorithm](docs/plans/archive/blokus-valid-move-algorithm.md) | Blokus Duo move generation |
| [archive/Remote training setup](docs/plans/archive/remote-training-setup.md) | Original Tailscale + WSL2 + SSH setup work |

---

## Roadmap

- [x] Core AlphaZero framework (MCTS, Coach, Arena)
- [x] Tic-Tac-Toe implementation and validation
- [x] Blokus Duo board representation and piece system
- [x] Blokus Duo neural network architecture
- [x] IBoard protocol with immutable boards and multi-channel neural net encoding
- [x] Architecture review fixes (MCTS optimisation, interface contracts, piece ID mapping)
- [x] Blokus Duo move generation with caching
- [x] W&B integration alongside HTML reporting + programmatic workspace setup
- [x] End-to-end TicTacToe training on home GPU (validated)
- [x] Reproducibility (global seed) + score-based acceptance + minimax-oracle eval set
- [x] Interactive arena replay viewer in HTML report
- [x] Elo vs frozen baseline + per-game oracle benchmarks (minimax for TTT)
- [ ] `get_symmetries()` for BlokusDuo
- [ ] End-to-end Blokus Duo self-play training
- [ ] Learning rate scheduling
- [ ] Parallel MCTS for faster self-play
- [ ] Parallel arena / Elo / minimax evaluation across CPU cores
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
