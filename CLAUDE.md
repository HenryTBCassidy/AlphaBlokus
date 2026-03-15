# AlphaBlokus — CLAUDE.md

## What is this project?

AlphaZero implementation for Blokus Duo. Self-play reinforcement learning on a 14x14 board with 21 polyomino pieces per player. The goal is to beat Pentobi (strongest open-source Blokus AI) in a majority of 100 games.

**Current state:** Core framework complete and validated on Tic-Tac-Toe. Blokus Duo game logic partially implemented — board, pieces, placement validation, and neural net all work. **Blocked on move generation** (`valid_moves()`). See README.md "Current Status" for details.

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
- `IBoard` — immutable board with `as_2d`, `as_multi_channel(player)`, `state_key`, `canonical(player)`
- `IGame` — game logic: `get_next_state`, `valid_move_masking`, `get_game_ended`, `get_symmetries`
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

1. `BlokusDuoBoard.valid_moves()` — aggregate valid moves from placement point cache
2. `BlokusDuoBoard.game_ended()` — check if neither player has legal moves
3. `BlokusDuoGame.valid_move_masking()` — convert valid moves to binary action-space mask
4. `BlokusDuoGame.get_symmetries()` — symmetric board+policy pairs for data augmentation
5. Switch `main.py` imports from TicTacToe to BlokusDuo once the above are done

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

1. **Move generation is the blocker.** Everything else works. Don't rewrite the framework — focus on completing Blokus game logic.
2. **Action space is huge (17,837).** MCTS iterates only valid moves (`np.where(valids)[0]`).
3. **Orientation IDs are 0-based (0–90).** `OrientationCodec` in `pieces.py` handles `(piece_id, orientation) ↔ int`. `ActionCodec` in `board.py` handles the full `Action ↔ int` (0–17,836) mapping.
4. **Coordinate systems:** Board = bottom-left origin (Blokus notation). Arrays = top-left origin (numpy). `CoordinateIndexDecoder` handles conversion.
5. **`eval.ipynb`** has Henry's design notes on the move generation algorithm — read it before implementing.
6. **Board sizes use class constants.** `BlokusDuoBoard.N = 14`, `Board.N = 3` (TicTacToe). Never hardcode board dimensions as literals.

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
├── guides/
│   ├── STYLE-GUIDE.md     # Code conventions (ALWAYS reference before writing code)
│   ├── PLAN-FORMAT.md     # How to write implementation plans
│   └── AI-CONTEXT.md      # Extended context, architecture rationale, gotchas
└── plans/
    └── archive/
        ├── bug-fixes.md           # Bug fixes (completed)
        └── structural-refactor.md # Structural refactor (completed)
```

## Things NOT to do

- Don't rewrite the core framework (MCTS, Coach, Arena) — it's validated on TicTacToe
- Don't over-engineer move generation — correct first, fast second
- Don't add 4-player Blokus support yet — Duo must beat Pentobi level 9 first
- Don't switch architectures (no Transformer/ViT) — ResNet is proven for AlphaZero
- Don't pad options for learning — if one approach is clearly best, just recommend it
- Don't add unnecessary complexity, abstractions, or features that weren't asked for
