# AlphaBlokus

**An AlphaZero implementation for Blokus Duo — training a neural network through self-play to master the board game Blokus, with the goal of beating Pentobi (the strongest open-source Blokus AI).**

> **Status — In progress.** The full AlphaZero pipeline is built and validated end-to-end on Tic-Tac-Toe (GPU). The Blokus Duo game logic, neural net, reporting, and a ~14× self-play speedup stack are all complete. The project is now at the threshold of its **first scaled Blokus training run** — currently working through self-play memory pressure and unattended-run reliability before letting a long run loose on the home GPU.

---

## What is AlphaBlokus?

AlphaBlokus applies DeepMind's [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) algorithm to **Blokus Duo**, a two-player territorial game on a 14×14 grid where each player places 21 polyomino pieces. The system learns entirely through self-play — no human games, no handcrafted heuristics.

The target: **beat [Pentobi](https://pentobi.sourceforge.io/)** (MCTS + RAVE, no neural net) in a majority of 100 games.

**The loop** — each *generation*:

1. **Self-play:** the current network plays games against itself, guided by Monte Carlo Tree Search (MCTS), producing training positions.
2. **Training:** the network learns to predict MCTS move probabilities (policy) and game outcomes (value).
3. **Arena:** the new network plays the previous best; it's kept only if it scores above a threshold.
4. **Strength eval:** Elo vs a frozen gen-0 baseline (+ a perfect-play minimax oracle and a symmetry diagnostic for Tic-Tac-Toe), logged every generation.

This is the same recipe that reached superhuman play in Chess, Shogi, and Go — applied to the combinatorially awkward, geometric world of Blokus.

---

## Project status

The work has moved through roughly six phases. The first five are done; the project is in the sixth.

### ✅ Phase 1 — Core framework + Tic-Tac-Toe (complete)
Game-agnostic AlphaZero framework (`IGame` / `IBoard` / `INeuralNetWrapper` protocols), MCTS with PUCT, the Coach training loop, Arena evaluation, Parquet storage, and HTML reporting. Validated end-to-end on Tic-Tac-Toe, which reaches near-perfect play within a couple of generations.

### ✅ Phase 2 — Blokus Duo game logic (complete)
Immutable 44-channel board, the 21-piece / 91-orientation system, move generation, legal-move masking, game-end/scoring, and `get_symmetries` (order-2: identity + main-diagonal reflection). Coordinate/action codecs map the 17,837-wide action space (of which 13,729 placements ever fit on the board).

### ✅ Phase 3 — Training infrastructure & reporting (complete)
Remote GPU training over Tailscale + WSL2 to a home RTX 3060 Ti; Weights & Biases dashboard alongside a self-contained interactive HTML report (loss curves, Elo, policy/value diagnostics, arena replay viewer). Reproducible runs via a single global seed; chess-style score-based acceptance; Elo vs a frozen gen-0 baseline; minimax oracle and per-generation symmetry diagnostic.

### ✅ Phase 4 — Full-cycle performance optimisation (complete, ~14× vs serial)
A stack of optimisations to make Blokus training practical at scale (see [`docs/plans/archive/full-cycle-optimisation.md`](docs/plans/archive/full-cycle-optimisation.md)):

| | Optimisation | Result |
|---|---|---|
| **F1** | Parallel self-play / arena / Elo across worker processes | **5.7×** at 8 workers |
| **F2** | Pentobi-style precomputed move-generation tables | **9×** per-call; move-gen share 52% → 9% |
| **F3** | Batched MCTS inference with virtual loss | **1.86×** at batch size 16 (~14× cumulative vs serial) |
| **F4** | Fully-convolutional policy head | **21.6×** fewer params, 19× smaller checkpoints |
| **F5** | Cross-worker inference server | Built & bit-identical, but **no speedup → not adopted** (F4 shrank the net off the GPU bottleneck) |

### ✅ Phase 5 — Pre-run prep (complete)
Dirichlet root-exploration noise (self-play only), fp16 inference on CUDA, and float32 policy storage to roughly halve the self-play buffer footprint.

### 🔧 Phase 6 — First scaled Blokus run (in progress)
The optimisation stack makes ~1,000 self-play games/generation cost what 80 used to. Standing up that first real run (config in `run_configurations/blokus_scaled.json`) surfaced two blockers being worked now:

- **Self-play memory blow-up — fixed.** The dense 17,837-float policy target × 8 workers used to exhaust 24 GB RAM and OOM mid-run; **fixed** by sparse policy storage + per-batch densification ([`docs/plans/archive/self-play-memory-fix.md`](docs/plans/archive/self-play-memory-fix.md)).
- **Unattended-run reliability.** WSL2 tears down the distro when no foreground session is attached; long runs currently need a held-open SSH session (see [`docs/guides/REMOTE-TRAINING.md`](docs/guides/REMOTE-TRAINING.md)).

### ⏭ What's next
1. Land the memory fix and launch the first **mid-to-long Blokus training run** on the home PC GPU, watching the Elo-vs-gen-0 curve.
2. Continue per-run optimisation and tuning (more MCTS sims, replay window, net size) as the data warrants.
3. Build the **Pentobi GTP adapter** ([`docs/06-INTERFACES.md`](docs/06-INTERFACES.md), not yet built) and **benchmark against Pentobi levels 1–9**.
4. If the agent is weaker than hoped, **scale up** — bigger network, more games, moving from the single home GPU to rented cloud GPUs (the cost/throughput ladder is in [`docs/09-COMPUTE-OPTIONS.md`](docs/09-COMPUTE-OPTIONS.md)).

---

## Repository layout

```
AlphaBlokus/
├── core/                       # Game-agnostic AlphaZero framework
│   ├── interfaces.py           # IGame / IBoard / INeuralNetWrapper protocols
│   ├── config.py               # RunConfig / MCTSConfig / NetConfig / WandbConfig
│   ├── mcts.py                 # MCTS with PUCT, Dirichlet noise, batched inference + virtual loss
│   ├── coach.py                # Training orchestrator (self-play → train → arena → strength eval)
│   ├── self_play.py            # Single-source-of-truth self-play episode loop
│   ├── parallel_self_play.py   # F1 worker pool (self-play / arena / Elo)
│   ├── arena.py                # Head-to-head evaluation + replay recording
│   ├── acceptance.py           # Score-based accept/reject rule (single source of truth)
│   ├── players.py              # RandomPlayer, NetworkPlayer, ...
│   ├── symmetry_diagnostic.py  # Per-generation policy-equivariance KL diagnostic
│   ├── storage.py              # Parquet I/O + W&B mirroring (MetricsCollector, SelfPlayStore)
│   └── game_factory.py         # (game, nnet) construction from a RunConfig
├── games/
│   ├── base_wrapper.py         # Shared NN wrapper (train / predict / predict_batch / checkpoint)
│   ├── tictactoe/              # Reference implementation (complete) + minimax oracle
│   └── blokusduo/              # Target game
│       ├── board.py            # Immutable board, 44-channel encoding, Action/coordinate codecs
│       ├── game.py             # Rules engine, move generation, get_symmetries
│       ├── pieces.py           # 21 pieces, 91 orientations, OrientationCodec
│       ├── movegen_tables.py   # F2 precomputed placement + lookup tables
│       ├── movegen_runtime.py  # F2 runtime generator (Pentobi-style)
│       └── neuralnets/         # ResNet trunk + conv/FC policy head + value head
├── reporting/                  # Interactive HTML report (Plotly + arena replays)
├── scripts/                    # CLI tools: arena_run, play_ttt, replay, benchmarks, profiling
├── run_configurations/         # JSON run configs (TTT + Blokus, test → scaled)
├── tests/                      # Unit + integration tests (core, blokusduo, tictactoe)
├── docs/                       # Reference docs, guides, and plans (see below)
└── main.py                     # Entry point
```

---

## Getting started

### Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- (Optional, for GPU training) an NVIDIA GPU with CUDA + a recent PyTorch wheel

### Install & test
```bash
git clone https://github.com/HenryTBCassidy/AlphaBlokus.git
cd AlphaBlokus
uv sync
uv run pytest                 # full suite
uv run pytest -m "not slow"   # skip integration tests
```

### Run training
```bash
# Quick pipeline check (CPU, no W&B, seconds)
uv run python main.py --config run_configurations/test_run.json

# Tic-Tac-Toe demo on a Mac (CPU, ~10 min)
uv run python main.py --config run_configurations/ttt_mac_demo.json

# Strong Tic-Tac-Toe run on the home GPU (CUDA)
uv run python main.py --config run_configurations/ttt_pc_strong.json

# Small Blokus run
uv run python main.py --config run_configurations/blokus_3gen.json
```

On the Mac set `"cuda": false`; on the home PC set `"cuda": true`. After a run, the interactive report is at `temp/<run_name>/Reporting/report.html`. Regenerate it from existing data without retraining via `--report-only`. See [`docs/guides/REMOTE-TRAINING.md`](docs/guides/REMOTE-TRAINING.md) for the home-GPU workflow.

---

## How it works (in one screen)

- **Board encoding:** Blokus uses a 44-channel tensor (21 per-piece planes per player + 2 aggregate planes); piece inventory is implicit (all-zero plane = unplayed). The board's ground truth is a 196-byte signed `int8` placement grid.
- **MCTS:** dictionary-keyed tree (by board state, so transpositions and within-game subtree reuse are free), PUCT selection over *legal* moves only, optional Dirichlet root noise in self-play, and batched leaf evaluation under virtual loss for GPU efficiency.
- **Move generation:** a readable reference generator plus a Pentobi-style precomputed-table fast path (proven bit-identical), ~9× faster per call.
- **Network:** a configurable-depth ResNet trunk with a fully-convolutional policy head (default) and a scalar value head; KL-divergence policy loss + MSE value loss.
- **Acceptance:** chess-style `(wins + ½·draws) / games ≥ threshold`, handling forced-draw regimes cleanly.

Algorithms are documented in depth in [`docs/02-ALGORITHMS.md`](docs/02-ALGORITHMS.md); architectures in [`docs/03-NEURAL-NETWORKS.md`](docs/03-NEURAL-NETWORKS.md).

---

## Blokus Duo rules (brief)

Two players, 14×14 board, 21 polyomino pieces each (sizes 1–5). The first move must cover a designated starting square ((4,4) for White, (9,9) for Black, per Pentobi's convention). Every later piece must touch a **corner** of a friendly piece and may **not** share an **edge** with one (edges with the opponent are fine). The game ends when neither player can place; score = squares placed, with bonuses for placing all pieces (and finishing on the monomino). Full reference and the piece catalog: [`docs/04-BLOKUS-DUO.md`](docs/04-BLOKUS-DUO.md).

---

## Documentation

### Reference (`docs/`)
| Doc | Description |
|-----|-------------|
| [01 — Background](docs/01-BACKGROUND.md) | Why Blokus, competitive landscape, key decisions |
| [02 — Algorithms](docs/02-ALGORITHMS.md) | MCTS, self-play, arena, move generation, what makes it fast |
| [03 — Neural Networks](docs/03-NEURAL-NETWORKS.md) | ResNet + conv policy head, board encoding, losses |
| [04 — Blokus Duo](docs/04-BLOKUS-DUO.md) | Rules, all 21 pieces, 91 orientations, symmetry |
| [05 — Evaluation](docs/05-EVALUATION.md) | Training diagnostics, Elo, planned Pentobi benchmarking |
| [06 — Interfaces](docs/06-INTERFACES.md) | Pentobi GTP adapter, UI, translation layer (scoping — not built yet) |
| [07 — Data Storage](docs/07-DATA-STORAGE.md) | Parquet schemas, metrics tables, W&B integration |
| [08 — Training Estimates](docs/08-TRAINING-ESTIMATES.md) | Measured + projected wall-clock per config |
| [09 — Compute Options](docs/09-COMPUTE-OPTIONS.md) | Local + cloud hardware, cost per run, phasing |

### Guides (`docs/guides/`)
`STYLE-GUIDE.md` (code conventions), `PLAN-FORMAT.md` (how plans are written), `REMOTE-TRAINING.md` (home-GPU runbook), `AI-CONTEXT.md` (extended context for AI assistants).

### Plans (`docs/plans/`)
Top-level plans are in-flight; `docs/plans/archive/` is the historical record of shipped work — including the full optimisation stack ([`full-cycle-optimisation.md`](docs/plans/archive/full-cycle-optimisation.md)) and the training-step memory fix ([`self-play-memory-fix.md`](docs/plans/archive/self-play-memory-fix.md)). The current active plan is the **profiling investigation** ([`profiling-investigation.md`](docs/plans/profiling-investigation.md)). Candidate-but-uncommitted ideas live in [`docs/IDEAS.md`](docs/IDEAS.md).

---

## Roadmap

- [x] Core AlphaZero framework (MCTS, Coach, Arena) + Tic-Tac-Toe validation
- [x] Blokus Duo board, piece system, move generation, and `get_symmetries`
- [x] Blokus Duo neural network (ResNet + convolutional policy head)
- [x] W&B + interactive HTML reporting, Elo vs frozen baseline, minimax oracle, symmetry diagnostic
- [x] Reproducibility (global seed) + score-based acceptance
- [x] Remote GPU training (Tailscale + WSL2)
- [x] Performance stack F1–F4 (~14× vs serial); F5 evaluated and shelved
- [x] Dirichlet noise, fp16 inference, float32 policy storage
- [ ] Sparse policy storage (unblock the scaled run's memory ceiling)
- [ ] First mid-to-long Blokus self-play training run on the home GPU
- [ ] Pentobi GTP adapter + benchmark across difficulty levels 1–9
- [ ] Scale up (bigger net / more games / cloud GPUs) if needed to beat Pentobi 9

---

## Inspiration & references

- Silver, D. et al. — [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero, 2017)
- Silver, D. et al. — [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404) (AlphaZero, 2018)
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) by Surag Nair — the game-agnostic framework pattern this draws from
- [alpha_zero](https://github.com/michaelnny/alpha_zero) by Michael Nny — clean AlphaZero reference
- [Pentobi](https://pentobi.sourceforge.io/) by Markus Enzenberger — the benchmark target, and the source of the precomputed move-generation design (F2)

---

## License

TBD
