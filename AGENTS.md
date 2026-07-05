# AlphaBlokus — AGENTS.md

> Canonical agent instructions. `CLAUDE.md` is a symlink to this file — edit here only.

## What is this project?

AlphaZero implementation for Blokus Duo. Self-play reinforcement learning on a 14x14 board with 21 polyomino pieces per player. The goal is to beat Pentobi (strongest open-source Blokus AI) in a majority of 100 games.

**Current state:** The full pipeline is built, optimised, and running scaled Blokus training on the home GPU (native Ubuntu). The code is an installable `src/alphablokus` package with CI (ruff + format check, strict mypy, base + jax test jobs). Scaled runs (15–30+ generations, ~1,000 games/gen) have executed; the Pentobi benchmark harness is built (`games/blokusduo/pentobi/`, `scripts/pentobi_benchmark.py`). See README.md "Project status" for the phase-by-phase story.

**JAX self-play backend (2026-07):** Blokus self-play generation can run GPU-native — rules as int8 matmuls, mctx search over a top-K compact action space, inference-only jnp net bridged from the torch checkpoint each generation (`games/blokusduo/jax/`). Selected per run via `RunConfig.selfplay_backend: "python" | "jax"`, reached through `registry.resolve_jax_selfplay_backend`; production Blokus configs use jax + Gumbel search (`search_policy: "gumbel"`, n=64 — validated at strength parity and ~3.5–12× wall-clock in `docs/research/jax-pipeline-ab.md`). The python path is unchanged, remains the dataclass default, and always drives arena/Elo/Pentobi evaluation. Requires `uv sync --extra jax` (Mac CPU) or `--extra jax-cuda` (box).

## Current focus

In-flight work lives in `docs/plans/` (top-level = in-flight, `archive/` = done). No plan is in-flight right now. Recently landed: [`pool-based-elo.md`](docs/plans/archive/pool-based-elo.md) (2026-07-05) — post-hoc pool BayesElo tournament (`scripts/tournament_elo.py`) that rates a run's checkpoints against each other into a non-saturating strength curve (E8/E9 deferred; methodology in [`docs/research/pool-elo-methodology.md`](docs/research/pool-elo-methodology.md)); [`cloud-scale-training.md`](docs/plans/archive/cloud-scale-training.md) (PR #34) — single rented-GPU runs at a ~£100 budget (uv CUDA container, S3-compatible sync + remote `--resume`, opt-in training-perf knobs, net-size presets, cost-calibration tool, Pentobi ladder in the report; recommendation in [`docs/research/cloud-training-recommendation.md`](docs/research/cloud-training-recommendation.md)); [`oom-hardening.md`](docs/plans/archive/oom-hardening.md) (2026-07-04) — RAM verification at scale is pending the next box run. Candidate ideas queue in `docs/IDEAS.md`.

## Commands

```bash
uv sync --extra dev                              # Install dependencies + tooling (pytest, ruff, mypy)
uv sync --extra jax                              # + JAX self-play backend (CPU; use --extra jax-cuda on the box)
uv run pytest                                    # Run tests
uv run pytest -m "not slow"                      # Skip integration tests
uv run pytest tests/games/blokusduo/             # Blokus tests only
uv run pytest tests/search/ tests/training/      # Framework subsets (tests mirror src/alphablokus/)
uv run alphablokus --config run_configurations/test_run.json   # Run training from some configuration
uv run alphablokus --config <cfg> --report-only  # Re-render the HTML report, no training
uv run alphablokus --config <cfg> --resume       # Continue a crashed/stopped run
uv run ruff check . && uv run ruff format --check src tests scripts   # Lint (matches CI)
uv run mypy                                      # Typecheck (strict: disallow_untyped_defs)
```

Scripts run as modules with the package installed — no `PYTHONPATH` needed: `uv run python scripts/pentobi_benchmark.py`, `uv run python scripts/benchmarks/benchmark.py`.

## Architecture

Game-agnostic framework (`src/alphablokus/`) with pluggable game implementations (`src/alphablokus/games/`).

**Package map** (each subpackage is a pipeline phase or shared machinery):

- `cli.py` / `config.py` / `interfaces.py` / `registry.py` — entry point, frozen-dataclass config, protocols, composition root
- `search/` — MCTS (PUCT, Dirichlet noise, batched inference + virtual loss) + its stats dataclasses
- `selfplay/` — the episode loop (`episode.py`) + backend dispatch serial/parallel/jax (`generate.py`)
- `parallel/` — worker pool (self-play/arena/Elo) + the (unadopted-by-default) inference server
- `training/` — `Coach` generation loop, `ReplayBuffer`, eval set, memory diagnostics
- `evaluation/` — arena, players, acceptance rule, Elo, symmetry diagnostic
- `storage/` — `MetricsCollector` (hive parquet + W&B mirror), `SelfPlayStore`, sparse policy codec
- `games/` — `tictactoe/` (reference impl + minimax oracle), `blokusduo/` (board/codec/game/pieces + `movegen/`, `pentobi/`, `jax/`, `nn/`)
- `reporting/` — interactive HTML report (Plotly charts, arena replay viewer)
- `testing/` — shipped test utilities (position caches)

**Core protocols** (in `alphablokus/interfaces.py`):
- `IBoard` — immutable state snapshot: geometry, encoding (`as_multi_channel`), `state_key`, `canonical`
- `IGame[TBoard]` — rules engine + action space: legal moves, game-over detection, symmetries, board factory (generic over its board type)
- `IPolicyValuePredictor` — the inference surface MCTS actually needs (`predict`, `predict_batch`)
- `INeuralNetWrapper` — extends it with `train`, `save_checkpoint`, `load_checkpoint`
- `IOracle` — perfect-play evaluation hooks (TicTacToe minimax)

**The registry rule:** `registry.py` is the **only** framework module allowed to name concrete game classes or backends (`instantiate_game`, `instantiate_game_and_network`, `resolve_jax_selfplay_backend`, `resolve_oracle`). Everything else depends on the protocols. Adding a game = implement the protocols under `games/<name>/` + register it there; never import `games.*` from framework code.

**Two game implementations:**
- `games/tictactoe/` — complete reference implementation. 2-channel encoding (2x3x3), 10 actions.
- `games/blokusduo/` — the target game, complete. 44-channel encoding (44x14x14), 17,837 actions.

**Key design decisions:**
- Boards are immutable (use `with_piece()`/`with_move()` to get new board)
- 44-channel encoding: 21 binary planes per player (one per piece type showing where it sits) + 2 aggregate planes. Piece inventory is implicit (all-zero plane = unplayed).
- Canonical form via channel reordering in `as_multi_channel(current_player)` — current player's planes always first
- `state_key` = `_piece_placement_board.tobytes()` (196 bytes, signed int8, +piece_id=White, -piece_id=Black)
- Action space: 14x14 grid positions x 91 piece-orientations + 1 pass = 17,837

## Conventions

Follow `docs/guides/STYLE-GUIDE.md` for all code (it also documents the project layout and tooling contract). Key points:
- Full type annotations on every function signature — machine-enforced (`mypy` with `disallow_untyped_defs`, run in CI)
- `ruff` lint + `ruff format` (CI checks both; line length 120, Python 3.11+)
- Google-style docstrings on public classes/methods
- `loguru` for logging (no `print()`), `{}` placeholders
- Frozen dataclasses for config/DTOs
- Protocol interfaces with explicit subclassing
- `from __future__ import annotations` instead of quoted type refs
- No mocks for game logic tests — use real objects
- `time.perf_counter()` for timing, `pathlib.Path` for filesystem

Follow `docs/guides/PLAN-FORMAT.md` when creating implementation plans.

## Gotchas

1. **Move generation is done; don't rewrite it.** Algorithm is documented inline and in `docs/plans/archive/blokus-valid-move-algorithm.md`. Further speedups (Cython, bitboard) were considered and deliberately set aside — candidate work goes through `docs/IDEAS.md`.
2. **Action space is huge (17,837).** MCTS iterates only valid moves (`np.where(valids)[0]`).
3. **Orientation IDs are 0-based (0–90).** `OrientationCodec` in `games/blokusduo/pieces.py` handles `(piece_id, orientation) ↔ int`. `ActionCodec` in `games/blokusduo/codec.py` handles the full `Action ↔ int` (0–17,836) mapping.
4. **Coordinate systems:** Board = bottom-left origin (Blokus notation). Arrays = top-left origin (numpy). `CoordinateIndexDecoder` handles conversion.
5. **Board sizes use class constants.** `BlokusDuoBoard.N = 14`, `Board.N = 3` (TicTacToe). Never hardcode board dimensions as literals.
6. **The jax backend is Blokus-only and inference-only.** `selfplay_backend: "jax"` raises for TicTacToe; training stays in torch (weights bridged per generation). Don't compare internal Elo curves across runs — each is anchored to its own gen-0 net (see `docs/research/jax-pipeline-ab.md` §3.2).
7. **Device selection is a simple `cuda: bool` flag** in `RunConfig.net_config` (`alphablokus/config.py` `NetConfig.cuda`, used in `games/base_wrapper.py`). No MPS auto-detection. On the Mac always set `cuda: false`; on the home PC set `cuda: true`.
8. **`pieces.json` has one accessor.** Resolve it via `alphablokus.games.blokusduo.pieces.default_pieces_path()` — never a repo- or CWD-relative path.

## Documentation

```
docs/
├── 01-BACKGROUND.md         # Why Blokus, competitive landscape, key decisions
├── 02-ALGORITHMS.md         # MCTS, self-play, arena, move generation, jax/Gumbel backend
├── 03-NEURAL-NETWORKS.md    # ResNet, board encoding, loss functions
├── 04-BLOKUS-DUO.md         # Rules, all 21 pieces, 91 orientations
├── 05-EVALUATION.md         # Metrics, Elo, Pentobi benchmarking
├── 06-INTERFACES.md         # Pentobi GTP adapter + translation layer (as built), UI scoping
├── 07-DATA-STORAGE.md       # Parquet format, metrics tables, checkpoints
├── 08-TRAINING-ESTIMATES.md # Pre-optimisation cost model (superseded; kept for methodology)
├── 09-COMPUTE-OPTIONS.md    # Local + cloud hardware, cost per run, phasing
├── IDEAS.md                 # Register of candidate avenues not yet committed (distinct from plans/)
├── guides/
│   ├── STYLE-GUIDE.md       # Code conventions + project layout (ALWAYS reference before writing code)
│   ├── PLAN-FORMAT.md       # How to write implementation plans
│   ├── REMOTE-TRAINING.md   # Runbook for running training on the home box over SSH
│   └── AI-CONTEXT.md        # Extended context, architecture rationale, gotchas
├── research/                # Deep investigations (jax A/B, profiling, Pentobi internals, …)
└── plans/                   # Top-level = in-flight (none currently); archive/ = completed, kept for context
    └── archive/                       # ~40 completed plans — the project's full history, e.g.:
        ├── oom-hardening.md           #   Sparse on-disk policies + OOM guardrails (O1–O9)
        ├── refactor-repo-architecture.md #  src/ package restructure
        ├── full-cycle-optimisation.md #   Master optimisation tracker (F1–F5; ~14× vs serial)
        ├── jax-selfplay-pipeline.md   #   GPU-native self-play backend + Gumbel search
        ├── replay-buffer-refactor.md  #   Rolling game-sized buffer + compact board storage
        ├── resumable-runs.md          #   --resume from the last completed generation
        ├── pentobi-harness.md         #   GTP client + translation + benchmark runner
        ├── linux-migration.md         #   Box moved from Windows/WSL2 to native Ubuntu
        └── ...                        #   (move-gen, batched inference, conv head, reporting, …)
```

## Things NOT to do

- Don't rewrite the core framework (MCTS, Coach, Arena) — it's validated on TicTacToe
- Don't over-engineer move generation — correct first, fast second
- Don't import `games.*` from framework code — `registry.py` is the one composition root
- Don't add 4-player Blokus support yet — Duo must beat Pentobi level 9 first
- Don't switch architectures (no Transformer/ViT) — ResNet is proven for AlphaZero
- Don't pad options for learning — if one approach is clearly best, just recommend it
- Don't add unnecessary complexity, abstractions, or features that weren't asked for
