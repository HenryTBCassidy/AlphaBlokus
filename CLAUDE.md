# AlphaBlokus — CLAUDE.md

## What is this project?

AlphaZero implementation for Blokus Duo. Self-play reinforcement learning on a 14x14 board with 21 polyomino pieces per player. The goal is to beat Pentobi (strongest open-source Blokus AI) in a majority of 100 games.

**Current state:** Core framework complete and validated on Tic-Tac-Toe. Blokus Duo game logic is mostly complete — board, pieces, placement validation, move generation, masking, game-end detection, and neural net all work. Remaining game-logic gap: `BlokusDuoGame.get_symmetries()`. The next *operational* milestone is a TicTacToe training run on the home PC's GPU (W&B integration + GPU smoke test). See README.md "Current Status" for details.

## Commands

```bash
uv sync                                          # Install dependencies
uv run pytest                                    # Run tests
uv run pytest -m "not slow"                      # Skip integration tests
uv run pytest tests/test_blokusduo/              # Blokus tests only
uv run pytest tests/test_core/                   # Core framework tests only
uv run python main.py --config run_configurations/test_run.json   # Run training from some configuration
```

**Linter:** `ruff` (configured in pyproject.toml). Line length 120, Python 3.11+.

## Architecture

Game-agnostic framework (`core/`) with pluggable game implementations (`games/`).

**Core protocols** (in `core/interfaces.py`):
- `IBoard` — immutable state snapshot: geometry, encoding (`as_multi_channel`), `state_key`, `canonical`
- `IGame` — rules engine + action space: legal moves, game-over detection, symmetries, board factory
- `INeuralNetWrapper` — `train`, `predict`, `save_checkpoint`, `load_checkpoint`

**Two game implementations:**
- `games/tictactoe/` — complete reference implementation. 2-channel encoding (2x3x3), 10 actions.
- `games/blokusduo/` — target game, in progress. 44-channel encoding (44x14x14), 17,837 actions.

**Key design decisions:**
- Boards are immutable (use `with_piece()`/`with_move()` to get new board)
- 44-channel encoding: 21 binary planes per player (one per piece type showing where it sits) + 2 aggregate planes. Piece inventory is implicit (all-zero plane = unplayed).
- Canonical form via channel reordering in `as_multi_channel(current_player)` — current player's planes always first
- `state_key` = `_piece_placement_board.tobytes()` (196 bytes, signed int8, +piece_id=White, -piece_id=Black)
- Action space: 14x14 grid positions x 91 piece-orientations + 1 pass = 17,837

## Critical path (what needs to happen next)

1. W&B integration in `MetricsCollector` (additive to existing HTML reports).
2. End-to-end TicTacToe training run on the home PC's RTX 3060 Ti (CUDA smoke test).
3. `BlokusDuoGame.get_symmetries()` — symmetric board+policy pairs for data augmentation.
4. Fix `main.py` checkpoint loading (currently raises when `load_model: true`).
5. Switch `"game": "blokusduo"` in run config once the above are done.

Live plan: `docs/plans/gpu-training-poc.md` (to be created on `feat/wandb-integration`).

## Conventions

Follow `docs/guides/STYLE-GUIDE.md` for all code. Key points:
- Full type annotations on every function signature
- Google-style docstrings on public classes/methods
- `loguru` for logging (no `print()`), `{}` placeholders
- Frozen dataclasses for config/DTOs
- Protocol interfaces with explicit subclassing
- `from __future__ import annotations` instead of quoted type refs
- No mocks for game logic tests — use real objects
- `time.perf_counter()` for timing, `pathlib.Path` for filesystem

Follow `docs/guides/PLAN-FORMAT.md` when creating implementation plans.

## Gotchas

1. **Move generation is done; don't rewrite it.** Algorithm is documented inline and in `docs/plans/archive/blokus-valid-move-algorithm.md`. Further speedups (Cython, bitboard) are intentionally deferred — see `docs/plans/move-gen-further-optimisation.md`.
2. **Action space is huge (17,837).** MCTS iterates only valid moves (`np.where(valids)[0]`).
3. **Orientation IDs are 0-based (0–90).** `OrientationCodec` in `pieces.py` handles `(piece_id, orientation) ↔ int`. `ActionCodec` in `board.py` handles the full `Action ↔ int` (0–17,836) mapping.
4. **Coordinate systems:** Board = bottom-left origin (Blokus notation). Arrays = top-left origin (numpy). `CoordinateIndexDecoder` handles conversion.
5. **`notebooks/eval.ipynb`** has Henry's original design notes on the move generation algorithm — historical reference now that the algorithm is in code.
6. **Board sizes use class constants.** `BlokusDuoBoard.N = 14`, `Board.N = 3` (TicTacToe). Never hardcode board dimensions as literals.
7. **Device selection is a simple `cuda: bool` flag** in `RunConfig.net_config` (`core/config.py:35`, used in `games/base_wrapper.py`). No MPS auto-detection. On the Mac always set `cuda: false`; on the home PC set `cuda: true`.

## Documentation

```
docs/
├── 01-BACKGROUND.md       # Why Blokus, competitive landscape, key decisions
├── 02-ALGORITHMS.md       # MCTS, self-play, arena, caching
├── 03-NEURAL-NETWORKS.md  # ResNet, board encoding, loss functions
├── 04-BLOKUS-DUO.md       # Rules, all 21 pieces, 91 orientations
├── 05-EVALUATION.md       # Metrics, Pentobi benchmarking
├── 06-INTERFACES.md       # Pentobi adapter, UI, translation layer
├── 07-DATA-STORAGE.md     # Parquet format, metrics tables, checkpoints
├── 08-TRAINING-ESTIMATES.md # Wall-clock time estimates for different configs/hardware
├── guides/
│   ├── STYLE-GUIDE.md     # Code conventions (ALWAYS reference before writing code)
│   ├── PLAN-FORMAT.md     # How to write implementation plans
│   ├── REMOTE-TRAINING.md # Runbook for running training on the home PC over SSH
│   └── AI-CONTEXT.md      # Extended context, architecture rationale, gotchas
└── plans/                                # Top-level = in-flight or not-yet-started
    ├── move-gen-further-optimisation.md  # Deferred: Cython, bitboard, caching (post-training)
    ├── training-infrastructure.md        # SSH/GPU/cloud compute options
    └── archive/                          # Completed plans, retained for context
        ├── blokus-valid-move-algorithm.md # Valid move generation
        ├── board-game-separation.md       # Board/Game responsibility split
        ├── bug-fixes.md                   # Bug fixes
        ├── mcts-profiling.md              # MCTS profiling instrumentation
        ├── remote-setup-mac.md            # MacBook-side SSH setup
        ├── remote-setup-windows.md        # Windows-side WSL2 + SSH setup
        ├── remote-training-setup.md       # Original combined remote-training plan
        └── structural-refactor.md         # Structural refactor
```

## Things NOT to do

- Don't rewrite the core framework (MCTS, Coach, Arena) — it's validated on TicTacToe
- Don't over-engineer move generation — correct first, fast second
- Don't add 4-player Blokus support yet — Duo must beat Pentobi level 9 first
- Don't switch architectures (no Transformer/ViT) — ResNet is proven for AlphaZero
- Don't pad options for learning — if one approach is clearly best, just recommend it
- Don't add unnecessary complexity, abstractions, or features that weren't asked for
