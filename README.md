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

## Features

- **Game-agnostic AlphaZero framework** — modular `IGame` and `INeuralNetWrapper` interfaces make it easy to add new games
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
│   └── interfaces.py           # IGame and INeuralNetWrapper protocols
├── tictactoe/                  # Reference implementation (complete)
│   ├── game.py                 # Tic-Tac-Toe game logic
│   └── neuralnets/             # 4-layer CNN (Conv → FC → policy + value)
├── blokusduo/                  # Target implementation (in progress)
│   ├── game.py                 # Blokus Duo game logic with piece placement
│   ├── pieces.py               # 21 pieces, 91 orientations, symmetry reduction
│   ├── pieces.json             # Piece definitions with basis orientations
│   └── neuralnets/             # ResNet (configurable blocks → policy + value)
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
| Input | 3x3 board | 14x18 (board + piece encoding) |
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
pip install -r requirements.txt  # if available
```

### Running Tic-Tac-Toe Training

```bash
python main.py --config run_configurations/test_run.json
```

---

## Documentation

Detailed documentation lives in the [`docs/`](docs/) folder:

| Document | Description |
|----------|-------------|
| [00 — Project Overview](docs/00-PROJECT-OVERVIEW.md) | Mission, phased approach, key decisions, effort estimates |
| [01 — Algorithms](docs/01-ALGORITHMS.md) | MCTS implementation, self-play, arena evaluation, caching strategies |
| [02 — Neural Networks](docs/02-NEURAL-NETWORKS.md) | ResNet and CNN architectures, board encoding, loss functions |
| [03 — Evaluation Plan](docs/03-EVALUATION-PLAN.md) | Training diagnostics, Elo tracking, Pentobi benchmarking |
| [04 — Competitive Landscape](docs/04-COMPETITIVE-LANDSCAPE.md) | Pentobi, existing Blokus RL projects, why neural Blokus is hard |
| [05 — Handoff](docs/05-HANDOFF.md) | Context for contributors, current state, next steps, gotchas |
| [06 — Pre-Flight Fixes](docs/06-PREFLIGHT-FIXES.md) | Bugs and performance issues to fix before Blokus training |

---

## Roadmap

- [x] Core AlphaZero framework (MCTS, Coach, Arena)
- [x] Tic-Tac-Toe implementation and validation
- [x] Blokus Duo board representation and piece system
- [x] Blokus Duo neural network architecture
- [ ] Pre-flight fixes (MCTS optimisation, loss function, piece ID mapping)
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
