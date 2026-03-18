# AlphaBlokus — Style Guide

Living document. Append as we discover new conventions. All code in this repo should follow these rules. Claude should reference this before reviewing or writing code.

Last updated: 2026-03-14

---

## Philosophy

- **Consistent over clever.** One way to do things, everywhere.
- **Type everything.** If Python can type-check it, annotate it.
- **No boilerplate.** If a framework eliminates ceremony, use it (loguru over stdlib logging, dataclasses over manual `__init__`, etc.).
- **Mathematical where appropriate.** RL has standard notation. Use it when it aids understanding, but not at the cost of readability for non-specialists.
- **Flat is better than nested.** Prefer early returns, guard clauses, and flat loops over deep nesting.
- **Delete dead code.** Don't comment it out, don't leave it behind a flag. Git has history.

---

## Naming

### General rules

| Thing | Convention | Example |
|-------|-----------|---------|
| Classes | PascalCase | `BlokusDuoGame`, `MetricsCollector` |
| Methods / functions | snake_case | `get_next_state`, `valid_move_masking` |
| Variables | snake_case | `current_player`, `piece_orientation` |
| Constants | SCREAMING_SNAKE | `BOARD_SIZE`, `PASS_ACTION_INDEX` |
| Type aliases | PascalCase | `PolicyVector`, `StateStr`, `BoardArray` |
| Files / modules | snake_case | `game.py`, `base_wrapper.py` |
| Directories | snake_case | `run_configurations/`, `neuralnets/` |
| Test files | `test_` prefix | `test_game.py`, `test_mcts.py` |

### Specific conventions

- **Config parameters:** always `config` (not `args`, not `run_config`). Exception: when the type disambiguates (e.g., `mcts_config: MCTSConfig`).
- **Neural networks:** `net` for the PyTorch module, `wrapper` for the NNetWrapper that wraps it.
- **Board dimensions:** `board_rows` and `board_cols` (not `board_x` / `board_y` — `x` is ambiguous).
- **Mathematical RL notation:** Acceptable in algorithm-heavy code (MCTS internals) when directly mapping from paper notation. Must include a comment block at the top explaining the notation. Example:

```python
# MCTS tree state — notation follows AlphaZero paper
# Q(s,a) = mean action-value, N(s,a) = visit count, P(s) = prior policy
self.q_values: dict[tuple[StateStr, int], float] = {}      # Q(s,a)
self.visit_counts: dict[tuple[StateStr, int], int] = {}     # N(s,a)
self.state_visits: dict[StateStr, int] = {}                 # N(s)
self.policy_priors: dict[StateStr, NDArray] = {}            # P(s)
```

- **No single-letter variables** except in comprehensions and trivial loops (`i`, `j`, `x` in `for i in range(n)`).
- **Boolean variables and parameters** should read as assertions: `is_valid`, `has_pieces`, `should_resign`. Not `valid`, `pieces`, `resign`.
- **Acronyms in names:** treat as words. `Mcts` not `MCTS` in compound names (`MctsConfig`). Exception: standalone acronyms are uppercase (`MCTS`, `UCB`).

Wait — actually, the codebase already uses `MCTS` as a class name and `MCTSConfig`. This is the standard Python convention for well-known acronyms. **Keep `MCTS`, `UCB` as-is.** Only lowercase acronyms when they're part of a longer camelCase/PascalCase name and the acronym is ≤3 letters (e.g., `xmlParser` not `XMLParser`). Since `MCTS` is always standalone or the prefix, keep it uppercase.

### What we name things (content, not just casing)

- **Be specific over generic.** `piece_orientation_lookup` not `lookup`. `training_loss` not `loss`.
- **Encode units in names when ambiguous.** `timeout_seconds` not `timeout`. `memory_bytes` not `memory`.
- **Use domain vocabulary.** Blokus has specific terms — use them: `placement`, `corner`, `side_adjacency`, `orientation`, `canonical_form`.
- **Avoid abbreviations** except universally understood ones: `config`, `init`, `num`, `idx`, `dir`, `arg`, `param`, `fn`.

---

## Type System

### Always annotate

Every function signature must have full type annotations — parameters and return type. No exceptions.

```python
# Good
def get_next_state(self, board: IBoard, player: int, action: int) -> tuple[IBoard, int]:

# Bad
def get_next_state(self, board, player, action):
```

### Use modern syntax (Python 3.10+)

```python
# Good
tuple[NDArray, int]
list[TrainingExample]
dict[str, float]
int | None

# Bad
from typing import Tuple, List, Dict, Optional
Tuple[NDArray, int]
Optional[int]
```

### TypeAlias for domain types

Define at the top of each module, after imports:

```python
from typing import TypeAlias

PolicyVector: TypeAlias = NDArray[np.float64]
StateKey: TypeAlias = bytes
Player: TypeAlias = Callable[[IBoard], int]
```

### Interfaces: Protocol with explicit subclassing

Define interfaces as `typing.Protocol`. Implementations **explicitly inherit** from the Protocol — just like `class Foo : IFoo` in C#. This gives us:

- Explicit declaration of intent (you can see what a class implements)
- Static enforcement by type checkers (missing methods caught at class definition)
- Runtime enforcement at instantiation (Protocol uses ABCMeta under the hood)
- Structural subtyping as a fallback (external code with the right shape still matches)

```python
# Define the interface
class IGame(Protocol):
    def get_action_size(self) -> int: ...
    def get_next_state(self, board: IBoard, player: int, action: int) -> tuple[IBoard, int]: ...

# Implement it — explicit inheritance
class TicTacToeGame(IGame):
    def get_action_size(self) -> int:
        return self.n * self.n + 1
    def get_next_state(self, board: Board, player: int, action: int) -> tuple[Board, int]:
        ...
```

When an implementation also needs shared behaviour (e.g. a base class with helpers), inherit from both the Protocol and ABC:

```python
class BaseNNetWrapper(INeuralNetWrapper, ABC):
    ...
```

**Do not** use `@runtime_checkable` — it only checks method existence, not signatures.

**Do not** put `__init__` on Protocols — implementations have their own constructors.

### No quoted type annotations

Never use string-quoted forward references (`board: "IBoard"`). If needed, add `from __future__ import annotations` at the top of the file to make all annotations lazy. Prefer avoiding it entirely by ordering definitions correctly or importing the type.

### Frozen dataclasses for value objects

Configuration, data transfer objects, and immutable domain objects should be `@dataclass(frozen=True)`:

```python
@dataclass(frozen=True)
class RunConfig:
    run_name: str
    num_generations: int
    ...
```

This makes them immutable, hashable, and communicates "this is a value, not a mutable entity."

---

## Documentation

### Docstrings: Google style

```python
def valid_placement(board: NDArray, piece: Piece, row: int, col: int) -> bool:
    """Check if a piece can be placed at the given position.

    Validates boundary constraints, no overlap with existing pieces,
    no side adjacency with same-colour pieces, and at least one
    corner adjacency with same-colour pieces.

    Args:
        board: Current board state, shape (n, n).
        piece: The piece to place, with orientation applied.
        row: Top-left row index for placement.
        col: Top-left column index for placement.

    Returns:
        True if placement is legal, False otherwise.
    """
```

### When to write docstrings

- **Always:** public classes, public methods, module-level functions
- **Skip:** private helpers where the name + type hints are self-documenting
- **Skip:** `__init__` when the class docstring covers construction

### Comments

- Don't state the obvious. `i += 1  # increment i` is noise.
- **Do** explain *why*, not *what*: `# Multiply by player to get canonical form (my pieces = +1, opponent = -1)`
- **Do** flag non-obvious correctness constraints: `# Must check >= 0, not > 0 (row 0 is a valid neighbour)`
- Use `# TODO:` for known work items. Include a ticket/doc reference when possible: `# TODO: see docs/plans/bug-fixes.md §4`

---

## Imports

### Order (enforced by ruff)

1. Standard library
2. Third-party packages
3. Local imports

Blank line between each group.

```python
import copy
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from core.config import RunConfig, MCTSConfig
from core.interfaces import IGame
```

### Rules

- **No `*` imports.** Ever. `from module import *` hides dependencies.
- **No relative imports.** Always use absolute: `from core.config import RunConfig`, not `from .config import RunConfig`.
- **Import modules, not individual names** when you'll use many things from the same module: `import numpy as np` not `from numpy import array, zeros, where, float64`.
- **Exception:** importing specific names from local modules is fine and preferred: `from core.config import RunConfig`.

---

## Code Structure

### One primary class per file

A file should have one main class plus any helpers that only that class uses. If a helper is used by multiple classes, it gets its own file.

**Exception:** multiple classes in one file is fine when they share a single cohesive concern (e.g. `core/storage.py` has `MetricsCollector` and `SelfPlayStore` — both are parquet I/O and neither is large enough to warrant its own module).

### Module layout

Within a file, order things **public-first** ("newspaper" pattern — headlines at the top, details below):

1. Module docstring
2. Imports
3. Constants
4. Type aliases
5. Public classes (public methods first, then private methods within each class)
6. Module-level code (rare — avoid)

**Method ordering within a class:** Public methods come first, then private methods. Within each group, order methods so that **callees appear before callers** — you should never have to scroll past a function to understand what the calling function does. The reader can always look up to find a method's implementation, never down.

Private helpers belong inside the class that uses them (as `@staticmethod` or regular methods), not floating at module level. This keeps them co-located with their only caller and avoids orphaned functions.

### Early returns

```python
# Good
def get_move(self, board: NDArray) -> int:
    if self.is_terminal(board):
        return -1
    valid = self.get_valid_moves(board)
    return self.select_best(valid)

# Bad
def get_move(self, board: NDArray) -> int:
    if not self.is_terminal(board):
        valid = self.get_valid_moves(board)
        return self.select_best(valid)
    else:
        return -1
```

### f-strings for string formatting

Use f-strings everywhere except logging calls where lazy evaluation matters:

```python
# Logging: use % formatting (lazy — string not built if level disabled)
logger.debug("MCTS simulation {} of {}", sim, total)

# Everything else: f-strings
filename = f"checkpoint_gen_{generation:03d}.pth.tar"
```

Note: loguru uses `{}` formatting by default (like `.format()`), not `%` formatting. Use `{}` placeholders with loguru.

---

## Configuration

- **Frozen dataclasses** for all config objects
- **JSON files** in `run_configurations/` for run parameters
- **Native JSON types** — `true`/`false`, not `"True"`/`"False"`
- **Computed properties** for derived values (paths, etc.):

```python
@property
def run_directory(self) -> Path:
    return self.root_directory / self.run_name
```

- **No hardcoded config values in source files.** If it might change between runs, it goes in the config.

---

## Logging

- **loguru** for all application logging (once migrated)
- **No `print()` statements** for output. Use `logger.info()` or `logger.debug()`.
- **Structured data** goes through `MetricsCollector` → parquet, not through log files
- **Log levels:**
  - `DEBUG` — detailed algorithm state (MCTS tree, board dumps)
  - `INFO` — progress updates (generation start/end, game completion)
  - `WARNING` — recoverable issues (fallback policy triggered, low GPU memory)
  - `ERROR` — failures that stop the current operation

---

## Testing

- **pytest** — function-based tests, no test classes
- **Fixtures** in `conftest.py` for shared setup (configs, boards, pieces)
- **`@pytest.mark.parametrize`** for combinatorial tests
- **`@pytest.mark.slow`** for integration tests (>5 seconds)
- **No mocks** for game logic — use real objects
- **Test file naming:** `test_{module}.py` mirroring source structure
- **Assertions:** use plain `assert` with descriptive messages:

```python
assert encoded == expected, f"Encoding {action} should give {expected}, got {encoded}"
```

---

## Git Workflow

### Branch naming

```
{type}/{short-kebab-description}
```

| Type | When |
|------|------|
| `refactor/` | Structural changes that don't change behaviour |
| `fix/` | Bug fixes |
| `feat/` | New features |
| `chore/` | Tooling, CI, docs, config |
| `test/` | Adding or fixing tests |

Examples:
- `refactor/structural-cleanup`
- `fix/mcts-action-iteration`
- `feat/move-generation`
- `chore/uv-setup`

### Commit messages

Imperative mood, ≤72 character subject line. Body explains *why*, not *what* (the diff shows *what*).

```
Fix MCTS to iterate only valid actions instead of full action space

The current loop iterates all 17,837 actions per simulation even though
only ~50 are valid. Using np.where(valids) reduces this to O(valid_moves)
per simulation, giving orders-of-magnitude speedup for Blokus.
```

### PR conventions

- **One concern per PR.** Don't mix refactoring with bug fixes.
- **PR title** matches the commit subject if single-commit, or summarises if multi-commit.
- **Link to doc section** in PR description: "Implements `docs/plans/structural-refactor.md` §2 (project structure)"

---

## Anti-Patterns — What We Don't Do

- ❌ `print()` for output — use logging
- ❌ Magic numbers — extract to named constants
- ❌ Code duplication — extract shared code
- ❌ Mutable global state — pass state explicitly
- ❌ Commented-out code — delete it, git has history
- ❌ `from module import *` — explicit imports only
- ❌ Bare `except:` — always catch specific exceptions
- ❌ `os.path` — use `pathlib.Path`
- ❌ `typing.Tuple/List/Dict/Optional` — use builtins + `X | None`
- ❌ Quoted type annotations (`board: "IBoard"`) — use `from __future__ import annotations` or fix import order
- ❌ Docstrings on obvious one-liners — the type hints are enough
- ❌ Deep nesting — refactor with early returns / guard clauses
- ❌ Abbreviations that aren't universal — `cfg` (use `config`), `mgr` (use `manager`), `proc` (use `process`)

---

## Patterns We Preserve

- ✅ Frozen dataclasses for config and DTOs
- ✅ TypeAlias definitions at the top of files
- ✅ Protocol-based interfaces with explicit subclassing (`class Board(IBoard)`)
- ✅ StrEnum for enumerations
- ✅ Computed properties on config objects
- ✅ Separation of concerns: core/ is game-agnostic
- ✅ Google-style docstrings with Args/Returns
- ✅ `time.perf_counter()` for timing (not `time.time()`)
- ✅ pathlib.Path for all filesystem operations
- ✅ Explicit `__init__.py` re-exports for public APIs
