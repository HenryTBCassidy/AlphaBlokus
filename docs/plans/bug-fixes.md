# AlphaBlokus — Bug Fixes & Correctness

Everything in this doc is about **making things work correctly**. The companion doc ([`structural-refactor.md`](archive/structural-refactor.md)) covers code organisation and infrastructure. Fix bugs here after the structural refactor is done — or before, if you prefer to verify the pipeline works before reorganising.

---

## Checklist

| # | Issue | Topic | Priority | Effort | Done |
|---|-------|-------|----------|--------|------|
| B1 | MCTS full action iteration | [MCTS](#2-mcts-performance-bugs) | Critical | 15 min | |
| B2 | Interface type mismatches (BlokusDuoGame) | [Interfaces](#1-interface-contract-violations) | Critical | 1 hour | |
| B3 | NNetWrapper constructor mismatch | [Interfaces](#1-interface-contract-violations) | Critical | 20 min | |
| B4 | Boundary check off-by-one | [Board Logic](#3-board--game-logic-bugs) | Critical | 15 min | |
| B5 | Piece orientation ID gaps | [Pieces](#4-piece-system-bugs) | Critical | 30 min | |
| B6 | Action encoding/decoding missing | [Missing Impls](#6-missing-implementations) | Critical | 45 min | |
| B7 | main.py game-network mismatch | [Interfaces](#1-interface-contract-violations) | Critical | 10 min | |
| B8 | Hardcoded board sizes | [Board Logic](#3-board--game-logic-bugs) | Critical | 20 min | |
| B9 | Mutable board state in BlokusDuoGame | [Board Logic](#3-board--game-logic-bugs) | Critical | 30 min | |
| B10 | Optimizer reset per epoch | [Training](#5-training-pipeline-bugs) | Important | 15 min | |
| B11 | No optimizer state in checkpoints | [Training](#5-training-pipeline-bugs) | Important | 20 min | |
| B12 | No learning rate schedule | [Training](#5-training-pipeline-bugs) | Important | 1 hour | |
| B13 | Model loading broken | [Missing Impls](#6-missing-implementations) | Important | 45 min | |
| B14 | BidirectionalDict redesign | [Pieces](#4-piece-system-bugs) | Important | 30 min | |
| B15 | Magic number action_size | [Pieces](#4-piece-system-bugs) | Important | 10 min | |
| B16 | MCTS string hashing | [MCTS](#2-mcts-performance-bugs) | Important | 30 min | |
| B17 | Input tensor shape verification | [Minor](#7-minor-correctness-issues) | Important | 30 min | |
| B18 | Policy loss function | [Training](#5-training-pipeline-bugs) | Nice-to-have | 15 min | |
| B19 | Random sampling with replacement | [Training](#5-training-pipeline-bugs) | Nice-to-have | 10 min | |
| B20 | dtype conversion overhead | [Training](#5-training-pipeline-bugs) | Nice-to-have | 10 min | |
| B21 | MCTS tree reuse across games | [MCTS](#2-mcts-performance-bugs) | Nice-to-have | 30 min | |
| B22 | TTT deprecated tostring() | [Minor](#7-minor-correctness-issues) | Nice-to-have | 5 min | ✅ |
| B23 | TTT unnecessary board copies | [Minor](#7-minor-correctness-issues) | Nice-to-have | 15 min | |
| B24 | Input validation | [Missing Impls](#6-missing-implementations) | Nice-to-have | 20 min | |
| B25 | MCTS fallback policy | [Board Logic](#3-board--game-logic-bugs) | Nice-to-have | 5 min | |

**Estimated total: ~9.5 hours** (Critical ~3.5h · Important ~4h · Nice-to-have ~2h)

---

## 1. Interface Contract Violations

The core framework uses `Protocol`-based interfaces (`IGame`, `INeuralNetWrapper`). TicTacToe conforms. BlokusDuo does not.

### Critical: BlokusDuoGame doesn't conform to IGame

| Method | IGame expects | BlokusDuoGame provides | Impact |
|--------|--------------|----------------------|--------|
| `get_next_state(board, player, action)` | `action: int` (action index) | `action: Action` (dataclass) | MCTS passes int → crash |
| `valid_move_masking(board, player)` | Returns `NDArray` (binary mask) | Returns `BoardArray` | Wrong shape, MCTS can't use it |
| `get_game_ended(board, player)` | `board: NDArray` | `board: BlokusDuoBoard` | Type mismatch |
| `get_canonical_form(board, player)` | `board: NDArray` | `board: BlokusDuoBoard` | Type mismatch |
| `string_representation(board)` | Returns `str` | Returns `bytes` via `tobytes()` | Works (bytes are hashable) but wrong type |

**Root cause:** IGame was designed around numpy arrays and integer action indices (works for TicTacToe). BlokusDuo introduced richer types but didn't add the conversion layer.

**Fix:** Build an action encoding/decoding layer (see §6) and ensure all BlokusDuoGame methods accept/return the types IGame specifies. Internally they can use richer types, but the interface boundary must match.

### Critical: NNetWrapper constructor mismatch

`INeuralNetWrapper.__init__` expects `(game: IGame, args: RunConfig)`. The implementations diverge:

| Wrapper | Signature | Problem |
|---------|-----------|---------|
| TicTacToe | `__init__(self, game: Game, args: RunConfig)` | Works but uses concrete `Game` type |
| BlokusDuo | `__init__(self, args: RunConfig)` | Missing `game` parameter entirely |

`Coach.__init__` does `self.pnet = self.nnet.__class__(self.game, run_config)` — crashes with BlokusDuo's wrapper.

**Fix:** Make BlokusDuo wrapper accept `(game, args)` to match the interface.

### Important: main.py game-network mismatch

`main.py` imports TicTacToe game but BlokusDuo neural network:

```python
from tictactoe.game import TicTacToeGame as Game      # line 9
from blokusduo.neuralnets.wrapper import NNetWrapper   # line 10
```

Crashes on first forward pass (wrong board dimensions). Development artifact — needs to be consistent.

**Fix:** Make both imports point to the same game. ~10 minutes.

---

## 2. MCTS Performance Bugs

Three compounding issues that will make Blokus self-play prohibitively slow.

### Critical: Full action space iteration

**File:** `core/mcts.py` line 171

```python
for a in range(self.game.get_action_size()):  # 17,837 iterations
    if valids[a]:
        # ... compute UCB for ~50 valid moves
```

~1,000 sims/move × ~50 moves/game = ~900 million unnecessary iterations per game.

**Fix:**
```python
valid_actions = np.where(valids)[0]  # ~50 indices
for a in valid_actions:
```

**Impact:** Orders-of-magnitude speedup. Zero correctness risk. Highest-value fix in the codebase.

### Important: String-based state hashing

**File:** `core/mcts.py` line 131

`game.string_representation(board)` does `board.tobytes()` — 2,016 bytes per lookup for a 14×18 int64 array. Thousands of lookups per simulation = significant cache pressure.

**Fix options (pick one):**
- `hash(board.tobytes())` — fastest, slight collision risk (fine for transposition table)
- `hashlib.md5(board.tobytes()).hexdigest()` — shorter keys, no collision risk
- Zobrist hash — incremental updates, best for repeated make/unmake

Recommend `hash(board.tobytes())` initially.

### Nice-to-have: Tree rebuilt every game

**File:** `core/coach.py` line 178

New MCTS tree per game. AlphaZero reuses transposition tables across games within a generation. Worth adding once basics work.

---

## 3. Board & Game Logic Bugs

### Important: Boundary check off-by-one

**File:** `blokusduo/game.py` lines 44, 48, 75, 81 (`no_sides` function)

```python
if i - 1 > 0:        # Bug: should be >= 0
    above = board[i - 1, j] == side
if j - 1 > 0:        # Bug: should be >= 0
    left = board[i, j - 1] == side
```

Row 0 and column 0 neighbours are never checked. Pieces at the board edge may have incorrect side-adjacency validation.

**Fix:** `> 0` → `>= 0` in all four boundary checks. ~15 minutes.

### Important: Hardcoded board size

**File:** `blokusduo/game.py` — at least 7 locations use literal `14` or `13` instead of `self.n` / `self.n - 1`. Examples: lines 48, 51, 275, 335.

**Fix:** Replace all literals with `self.n` / `self.n - 1`. Prevents bugs when board size changes or when building `valid_moves()`.

### Important: Mutable board state on game class

**File:** `blokusduo/game.py` line 572

`BlokusDuoGame` stores `self.board` as mutable instance state. The `IGame` interface expects board state to be passed as function arguments, but `get_next_state()` mutates `self.board` as a side effect.

**Impact:** MCTS explores multiple branches from the same position — it needs independent board copies. Mutating shared state will cause incorrect tree exploration.

**Fix:** `get_next_state()` should create and return a new board state without modifying `self.board`. Use `copy.deepcopy()` or implement a `Board.copy()` method.

### Design note: MCTS fallback policy

**File:** `core/mcts.py` line 155

```python
self.Ps[s] = self.Ps[s] + valids  # 0 + valids = valids (works by accident)
```

When all valid moves get zero probability from the network, this "adds" the valid mask to the zero vector. Happens to work because `0 + valids = valids`, but semantically wrong.

**Fix:** `self.Ps[s] = valids.astype(float)` — explicit uniform fallback over valid moves.

---

## 4. Piece System Bugs

### Critical: Orientation ID mapping has gaps

**File:** `blokusduo/pieces.py` line 227

The `BidirectionalDict` population loop increments `i` one extra time between pieces, creating gaps in the ID sequence:

```python
# Current: IDs are 0,1,...,N, [skip], N+2, N+3, ...
# Expected: IDs are 0,1,...,90 (contiguous)
```

**Impact:** Action indices map to wrong piece-orientations. Silently produces illegal moves.

**Fix:** Fix the population loop to assign contiguous IDs. ~30 minutes.

### Important: BidirectionalDict design

**File:** `blokusduo/pieces.py` lines 91-119

Subclasses `dict` and stores both `key→value` and `value→key` in the same dict. `len(lookup)` returns 182 (2×91), iterating gives both directions mixed together.

**Fix:** Replace with a dedicated class with explicit `encode(key)` / `decode(value)` methods. Cleaner API, correct `len()`, no iteration confusion.

### Important: Magic number action_size

**File:** `blokusduo/neuralnets/net.py` line 78

Action size hardcoded as `(14 * 14 * 91) + 1` instead of coming from the game object. There's a TODO acknowledging this.

**Fix:** Pass `game.get_action_size()` to the network constructor.

---

## 5. Training Pipeline Bugs

### Important: Optimizer reset per epoch

**File:** `blokusduo/neuralnets/wrapper.py` line 71

```python
optimizer = optim.Adam(self.nnet.parameters())
```

Fresh Adam optimizer every `train()` call. Momentum and variance estimates lost between generations. Training "restarts" each generation.

**Fix:** Create optimizer once in `__init__()`, reuse across `train()` calls.

### Important: No optimizer state in checkpoints

**Files:** Both wrapper files

Only `model.state_dict()` is saved. Adam's momentum buffers lost on load.

**Fix:**
```python
torch.save({
    'model_state_dict': self.nnet.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
}, filepath)
```

### Important: No learning rate schedule

`core/config.py` has a `learning_rate` field but no schedule. Adam starts at 0.001 for all generations. AlphaZero used step-decay (10× drop at specific steps).

**Fix:** Implement cosine annealing or step decay via `torch.optim.lr_scheduler`. ~1 hour.

### Nice-to-have: Policy loss function suboptimal

**File:** `blokusduo/neuralnets/wrapper.py` lines 137-149

Custom loss `-sum(targets * outputs) / N` where outputs are `log_softmax`. Mathematically correct but misses PyTorch's numerical stability optimizations.

**Fix:** Replace with `F.kl_div(outputs, targets, reduction='batchmean')`.

### Nice-to-have: Random sampling with replacement

**File:** `wrapper.py` line 84

```python
sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
```

Allows duplicate examples in a batch. Minor inefficiency.

**Fix:** Use `torch.utils.data.DataLoader` with proper shuffling (also cleaner code).

### Nice-to-have: dtype conversion overhead

**File:** `wrapper.py` lines 86, 126-128

```python
boards = torch.FloatTensor(np.array(boards).astype(np.float64))
```

Converts to float64 then to float32. Unnecessary double conversion.

**Fix:** `torch.tensor(np.array(boards), dtype=torch.float32)`.

---

## 6. Missing Implementations

### Critical: Action encoding/decoding

No conversion between action indices (integers 0-17,836 from the neural network) and `Action` dataclass instances (piece_id, orientation, x, y) that the game logic uses.

**Needed:**
- `encode(action: Action) → int` — for building policy targets from MCTS
- `decode(index: int) → Action` — for converting network output to playable moves
- `PASS_ACTION_INDEX = 17836` — the pass action constant

The encoding scheme: `index = y * (14 * 91) + x * 91 + orientation_id`, with pass at 17,836.

**Estimated effort:** ~45 minutes.

### Critical: 5 unimplemented BlokusDuoGame methods

| Method | Purpose | Depends on |
|--------|---------|------------|
| `valid_moves()` | Generate all legal moves | Piece system, placement validation |
| `valid_move_masking()` | Binary mask over action space | `valid_moves()` + action encoding |
| `game_ended()` | Detect game over | `valid_moves()` for both players |
| `get_symmetries()` | Data augmentation | Board + policy vector rotation |
| `populate_moves_for_new_placement()` | Corner-based move generation | Placement point cache |

These are Phase 2 work items, listed here for completeness.

### Important: Model loading is broken

**File:** `main.py` lines 32-35

`load_model=True` raises `NotImplementedError`. `load_train_examples()` in `coach.py` references `self.run_config.load_folder_file` which doesn't exist on `RunConfig`. Training cannot resume from checkpoint.

**Fix:** Implement checkpoint loading in main.py and fix `load_train_examples()`. ~45 minutes.

### Nice-to-have: No input validation

No assertions that:
- Training examples have correct shapes
- Policy vectors sum to ~1
- Values are in [-1, 1]
- Board states are valid after moves

Bad data silently causes training divergence.

**Fix:** Add assertions at interface boundaries (wrapper `train()`, MCTS `search()`, etc.).

---

## 7. Minor Correctness Issues

### TicTacToe: deprecated `tostring()`

**File:** `tictactoe/game.py` line 87

`board.tostring()` deprecated since numpy 1.21. Use `tobytes()`.

### TicTacToe: unnecessary board copies

**File:** `tictactoe/game.py` lines 30-53

Every game method creates a `Board()` wrapper. On a 3×3 board this is free, but it establishes a pattern BlokusDuo shouldn't follow.

### Neural net input tensor shape naming

**File:** `blokusduo/neuralnets/net.py` line 143

`x.view(-1, 1, self.board_x, self.board_y)` where `board_x=14` (height) and `board_y=18` (width). `x` usually means width. Verify the actual array layout matches what conv layers expect.

---

---

*Checklist is at the top of this document.*
