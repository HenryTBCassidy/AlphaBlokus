# AlphaBlokus

[![CI](https://github.com/HenryTBCassidy/AlphaBlokus/actions/workflows/ci.yml/badge.svg)](https://github.com/HenryTBCassidy/AlphaBlokus/actions/workflows/ci.yml)

**An AlphaZero implementation for Blokus Duo — training a neural network through self-play to master the board game Blokus, with the goal of beating Pentobi (the strongest open-source Blokus AI).**

> **Status — training at scale.** The full AlphaZero pipeline is built, optimised (~14× vs serial on the python engine), and running scaled Blokus training on a home GPU under native Ubuntu. Self-play generation now also runs **GPU-native in JAX** (game rules as int8 matmuls + mctx Gumbel search) at ~12× the python pipeline's games/s at production net size, validated at strength parity. The Pentobi benchmark harness is built; the strongest run so far wins ~25% of games against Pentobi level 1 — the climb up the nine-level ladder is the current frontier.

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

The work has moved through seven phases; the project is in the eighth.

### ✅ Phase 1 — Core framework + Tic-Tac-Toe (complete)
Game-agnostic AlphaZero framework (`IGame` / `IBoard` / `INeuralNetWrapper` protocols), MCTS with PUCT, the Coach training loop, Arena evaluation, Parquet storage, and HTML reporting. Validated end-to-end on Tic-Tac-Toe, which reaches near-perfect play within a couple of generations.

### ✅ Phase 2 — Blokus Duo game logic (complete)
Immutable 44-channel board, the 21-piece / 91-orientation system, move generation, legal-move masking, game-end/scoring, and `get_symmetries` (order-2: identity + main-diagonal reflection). Coordinate/action codecs map the 17,837-wide action space (of which 13,729 placements ever fit on the board).

### ✅ Phase 3 — Training infrastructure & reporting (complete)
Remote GPU training to a home RTX 3060 Ti (now running native Ubuntu); Weights & Biases dashboard alongside a self-contained interactive HTML report (loss curves, Elo, policy/value diagnostics, arena replay viewer). Reproducible runs via a single global seed; chess-style score-based acceptance; Elo vs a frozen gen-0 baseline; minimax oracle and per-generation symmetry diagnostic.

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

### ✅ Phase 6 — Scaled Blokus runs (complete)
The blockers that stood between the optimisation stack and real training runs all fell:

- **Memory:** boards are stored compact (196 bytes, not the dense 44×14×14 planes) in a rolling game-sized replay buffer, lazily re-encoded at batch time — ~175× less buffer RAM ([`replay-buffer-refactor`](docs/plans/archive/replay-buffer-refactor.md), [`lazy-board-encoding`](docs/plans/archive/lazy-board-encoding.md)).
- **Unattended-run reliability:** the box migrated from Windows/WSL2 to **native Ubuntu** ([`linux-migration`](docs/plans/archive/linux-migration.md)) — detached runs in `tmux` just survive.
- **Crash recovery:** runs are **resumable** (`--resume`) from the last completed generation, reusing the frozen Elo baseline ([`resumable-runs`](docs/plans/archive/resumable-runs.md)).

Multiple scaled training runs have since executed on the home GPU (15–30+ generations, ~1,000 games/generation), with the strongest checkpoint reaching ~25% vs Pentobi level 1.

### ✅ Phase 7 — JAX GPU-native self-play backend (complete)
The single biggest engineering artefact in the repo: Blokus self-play generation reimplemented to run **entirely on the GPU** (`games/blokusduo/jax/`), selected per run via `selfplay_backend: "jax"`:

- **Game rules as int8 matmuls** — placement legality, game-end detection, and legal-move masks computed as batched tensor ops, bit-identical to the python rules engine (parity-gated in CI).
- **mctx search over a top-K compact action space** — the full 17,837-wide action space is untenable inside mctx's dense tree arrays, so each node searches only its top-64 legal actions by prior; validated to track the exact search *better* than the python engine's own batched-inference approximation.
- **Gumbel AlphaZero search** (`search_policy: "gumbel"`, n=64 sims with Sequential Halving) as the production configuration.
- **Inference-only jnp net**, bridged from the torch checkpoint each generation — training stays in torch.

A/B validated over three 10-generation training arms ([`docs/research/jax-pipeline-ab.md`](docs/research/jax-pipeline-ab.md)): the Gumbel arm trains a net **statistically indistinguishable from the python baseline** (53.5% head-to-head, CI includes parity) at **3.5× end-to-end wall-clock even at the most python-friendly config** — rising to **~12× self-play games/s at production net size** (128f×8b). The python engine remains first-class: it drives all arena/Elo/Pentobi evaluation, all of Tic-Tac-Toe, and is the parity oracle the JAX path is validated against.

The Pentobi GTP harness also landed in this phase (`games/blokusduo/pentobi/` + `scripts/pentobi_benchmark.py`): subprocess GTP client, move translation in both directions, and a benchmark runner across difficulty levels.

### 🔧 Phase 8 — Scale up and climb the Pentobi ladder (in progress)
With generation ~12× cheaper, the constraint moves to memory and run length. Two enabling plans landed here (both archived):

- [`oom-hardening.md`](docs/plans/archive/oom-hardening.md) — the on-disk self-play format used to store policies **dense** (~71 KB/position), which OOM-killed a 10k-games/gen overnight run at the save/resume boundaries. Policies are now sparse end-to-end (as they already were in RAM) with streamed parquet I/O, plus guardrails: a startup RAM-budget check and peak-RSS logging at phase transitions. The interim `num_eps ≤ 8000` mitigation is lifted; RAM verification at scale awaits the next box run.
- [`refactor-repo-architecture.md`](docs/plans/archive/refactor-repo-architecture.md) — the repo-wide restructure into the installable `src/alphablokus` package.

### ⏭ What's next
1. Run **long Gumbel-backend training runs** at 10k+ games/generation.
2. **Benchmark against Pentobi levels 1–9** after each run with the built harness; scale net size / games as the ladder demands.
3. If the home GPU tops out, move up the cloud cost/throughput ladder ([`docs/09-COMPUTE-OPTIONS.md`](docs/09-COMPUTE-OPTIONS.md)).

---

## Repository layout

```
AlphaBlokus/
├── src/alphablokus/            # The installable package
│   ├── cli.py                  # Console entry point (`uv run alphablokus`)
│   ├── config.py               # RunConfig / MCTSConfig / NetConfig / WandbConfig
│   ├── interfaces.py           # IBoard / IGame / IPolicyValuePredictor / INeuralNetWrapper / IOracle
│   ├── registry.py             # Composition root — the one module naming concrete games/backends
│   ├── search/                 # MCTS with PUCT, Dirichlet noise, batched inference + virtual loss
│   ├── selfplay/               # Episode loop + backend dispatch (serial / parallel / jax)
│   ├── parallel/               # Worker pool (self-play / arena / Elo) + inference server
│   ├── training/               # Coach loop, replay buffer, eval set, memory diagnostics
│   ├── evaluation/             # Arena, players, acceptance rule, Elo, symmetry diagnostic
│   ├── storage/                # Parquet I/O + W&B mirroring (MetricsCollector, SelfPlayStore)
│   ├── games/
│   │   ├── base_wrapper.py     # Shared NN wrapper (train / predict / predict_batch / checkpoint)
│   │   ├── tictactoe/          # Reference implementation (complete) + minimax oracle
│   │   └── blokusduo/          # Target game
│   │       ├── board.py        # Immutable board, 44-channel encoding
│   │       ├── codec.py        # Action / ActionCodec / CoordinateIndexDecoder
│   │       ├── game.py         # Rules engine, move generation, get_symmetries
│   │       ├── pieces.py       # 21 pieces, 91 orientations, OrientationCodec
│   │       ├── movegen/        # F2 precomputed tables + runtime generator (Pentobi-style)
│   │       ├── pentobi/        # GTP client, Pentobi player, move translation
│   │       ├── jax/            # GPU-native self-play backend (env kernels, mctx search, bridge)
│   │       └── nn/             # ResNet trunk + conv/FC policy head + value head
│   ├── reporting/              # Interactive HTML report (Plotly + arena replays)
│   └── testing/                # Shipped test utilities (position caches)
├── scripts/                    # Operational CLI tools + benchmarks/ + profiling/
├── run_configurations/         # JSON run configs (TTT + Blokus, test → scaled; + archive/)
├── tests/                      # Mirrors src/alphablokus/ one-to-one
├── docs/                       # Reference docs, guides, and plans (see below)
└── .github/workflows/ci.yml    # Lint (ruff check + format), mypy, tests (base + jax)
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
uv sync --extra dev           # + --extra jax (CPU) or --extra jax-cuda (GPU box) for the JAX backend
uv run pytest                 # full suite
uv run pytest -m "not slow"   # skip integration tests
```

### Run training
```bash
# Quick pipeline check (CPU, no W&B, seconds)
uv run alphablokus --config run_configurations/test_run.json

# Tic-Tac-Toe demo on a Mac (CPU, ~10 min)
uv run alphablokus --config run_configurations/ttt_mac_demo.json

# Small Blokus run (python backend)
uv run alphablokus --config run_configurations/blokus_3gen.json

# Production Blokus run on the GPU box (JAX Gumbel self-play backend)
uv run alphablokus --config run_configurations/blokus_jax_gumbel_30.json
```

On the Mac set `"cuda": false`; on the home PC set `"cuda": true`. After a run, the interactive report is at `temp/<run_name>/Reporting/report.html`. Regenerate it from existing data without retraining via `--report-only`; continue a crashed or stopped run from its last completed generation via `--resume`. See [`docs/guides/REMOTE-TRAINING.md`](docs/guides/REMOTE-TRAINING.md) for the home-GPU workflow.

---

## How it works (in one screen)

- **Board encoding:** Blokus uses a 44-channel tensor (21 per-piece planes per player + 2 aggregate planes); piece inventory is implicit (all-zero plane = unplayed). The board's ground truth is a 196-byte signed `int8` placement grid.
- **MCTS:** dictionary-keyed tree (by board state, so transpositions and within-game subtree reuse are free), PUCT selection over *legal* moves only, optional Dirichlet root noise in self-play, and batched leaf evaluation under virtual loss for GPU efficiency.
- **Move generation:** a readable reference generator plus a Pentobi-style precomputed-table fast path (proven bit-identical), ~9× faster per call.
- **JAX self-play backend:** thousands of games stepped in lockstep on the GPU — rules as int8 matmuls, mctx Gumbel search over a top-K compact action space, inference-only jnp net bridged from the torch checkpoint. ~12× the python pipeline's games/s at production net size, at validated strength parity.
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
| [02 — Algorithms](docs/02-ALGORITHMS.md) | MCTS, self-play, arena, move generation, the JAX/Gumbel backend |
| [03 — Neural Networks](docs/03-NEURAL-NETWORKS.md) | ResNet + conv policy head, board encoding, losses |
| [04 — Blokus Duo](docs/04-BLOKUS-DUO.md) | Rules, all 21 pieces, 91 orientations, symmetry |
| [05 — Evaluation](docs/05-EVALUATION.md) | Training diagnostics, Elo, Pentobi benchmarking |
| [06 — Interfaces](docs/06-INTERFACES.md) | Pentobi GTP adapter + translation layer (as built), UI scoping |
| [07 — Data Storage](docs/07-DATA-STORAGE.md) | Parquet schemas, metrics tables, W&B integration |
| [08 — Training Estimates](docs/08-TRAINING-ESTIMATES.md) | Pre-optimisation cost model (superseded; kept for the methodology) |
| [09 — Compute Options](docs/09-COMPUTE-OPTIONS.md) | Local + cloud hardware, cost per run, phasing |

### Guides (`docs/guides/`)
`STYLE-GUIDE.md` (code conventions + project layout), `PLAN-FORMAT.md` (how plans are written), `REMOTE-TRAINING.md` (home-GPU runbook), `AI-CONTEXT.md` (extended context for AI assistants).

### Plans (`docs/plans/`)
Top-level plans are in-flight; `docs/plans/archive/` is the historical record of completed work — the optimisation stack ([`full-cycle-optimisation.md`](docs/plans/archive/full-cycle-optimisation.md)), the JAX pipeline ([`jax-selfplay-pipeline.md`](docs/plans/archive/jax-selfplay-pipeline.md)), the Pentobi harness ([`pentobi-harness.md`](docs/plans/archive/pentobi-harness.md)), the replay-buffer refactor ([`replay-buffer-refactor.md`](docs/plans/archive/replay-buffer-refactor.md)), and ~40 more, most recently [`refactor-repo-architecture.md`](docs/plans/archive/refactor-repo-architecture.md) and [`oom-hardening.md`](docs/plans/archive/oom-hardening.md) (sparse on-disk policies + OOM guardrails). Nothing is currently in flight. Candidate-but-uncommitted ideas live in [`docs/IDEAS.md`](docs/IDEAS.md); deep investigations in `docs/research/`.

---

## Roadmap

- [x] Core AlphaZero framework (MCTS, Coach, Arena) + Tic-Tac-Toe validation
- [x] Blokus Duo board, piece system, move generation, and `get_symmetries`
- [x] Blokus Duo neural network (ResNet + convolutional policy head)
- [x] W&B + interactive HTML reporting, Elo vs frozen baseline, minimax oracle, symmetry diagnostic
- [x] Reproducibility (global seed) + score-based acceptance
- [x] Remote GPU training (home box, now native Ubuntu)
- [x] Performance stack F1–F4 (~14× vs serial); F5 evaluated and shelved
- [x] Dirichlet noise, fp16 inference, float32 policy storage
- [x] Compact-board rolling replay buffer (removes the training-step RAM ceiling)
- [x] Resumable runs + first scaled Blokus training runs on the home GPU
- [x] JAX GPU-native self-play backend + Gumbel search (~12× at production net size)
- [x] Pentobi GTP adapter + benchmark harness
- [x] On-disk sparse policy storage + OOM guardrails ([`oom-hardening`](docs/plans/archive/oom-hardening.md))
- [ ] Long production runs (10k+ games/generation) up the Pentobi ladder
- [ ] Beat Pentobi level 9

---

## Inspiration & references

- Silver, D. et al. — [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero, 2017)
- Silver, D. et al. — [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404) (AlphaZero, 2018)
- Danihelka, I. et al. — [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) (Gumbel MuZero/AlphaZero, 2022) — the low-simulation search the JAX backend uses
- [mctx](https://github.com/google-deepmind/mctx) by DeepMind — JAX-native MCTS the GPU backend searches with
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) by Surag Nair — the game-agnostic framework pattern this draws from
- [alpha_zero](https://github.com/michaelnny/alpha_zero) by Michael Nny — clean AlphaZero reference
- [Pentobi](https://pentobi.sourceforge.io/) by Markus Enzenberger — the benchmark target, and the source of the precomputed move-generation design (F2)

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and build on.
