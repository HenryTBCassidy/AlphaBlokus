# AlphaBlokus — Architecture Review

A comprehensive review of the codebase covering code structure, interface design, data types, algorithms, performance, storage, naming, and infrastructure gaps. This replaces the earlier "Pre-Flight Fixes" doc — every issue from that doc is absorbed here alongside deeper architectural findings.

The review serves two purposes: (1) a checklist of things to fix before Blokus training, and (2) a reference for understanding why things are the way they are and where they should go.

---

## 1. Interface Contracts & Type Safety

The core framework uses `Protocol`-based interfaces (`IGame` in `core/interfaces.py:12-151`, `INeuralNetWrapper` in `core/interfaces.py:154-243`). This is good design — structural subtyping means game implementations don't need to inherit from a base class, just implement the right methods. The TicTacToe implementation conforms correctly. BlokusDuo does not.

### Critical: BlokusDuoGame doesn't conform to IGame

| Method | IGame expects | BlokusDuoGame provides | Impact |
|--------|--------------|----------------------|--------|
| `get_next_state(board, player, action)` | `action: int` (action index) | `action: Action` (dataclass) | MCTS passes int, will crash |
| `valid_move_masking(board, player)` | Returns `NDArray` (binary mask over all actions) | Returns `BoardArray` | Wrong shape, MCTS can't use it |
| `get_game_ended(board, player)` | `board: NDArray` | `board: BlokusDuoBoard` | Type mismatch |
| `get_canonical_form(board, player)` | `board: NDArray` | `board: BlokusDuoBoard` | Type mismatch |
| `string_representation(board)` | Returns `str` | Returns `bytes` via `tobytes()` | Works (bytes are hashable) but wrong type |

**Root cause:** The IGame interface was designed around numpy arrays and integer action indices (which works for TicTacToe). BlokusDuo introduced richer types (BlokusDuoBoard, Action dataclass) but didn't add the conversion layer between them.

**Fix:** Build an action encoding/decoding layer (see §10) and ensure all BlokusDuoGame methods accept/return the types IGame specifies. Internally they can use richer types, but the interface boundary must match.

### Critical: NNetWrapper constructor mismatch

`INeuralNetWrapper.__init__` expects `(game: IGame, args: RunConfig)`. The implementations diverge:

| Wrapper | Signature | Problem |
|---------|-----------|---------|
| TicTacToe | `__init__(self, game: Game, args: RunConfig)` | Works but uses concrete `Game` type, not `IGame` |
| BlokusDuo | `__init__(self, args: RunConfig)` | Missing `game` parameter entirely |

`Coach.__init__` (line 90) does `self.pnet = self.nnet.__class__(self.game, run_config)` — this will crash with BlokusDuo's wrapper because it doesn't accept a `game` argument.

**Fix:** Make BlokusDuo wrapper accept `(game, args)` to match the interface. Even if it doesn't use the game object, the signature must conform.

### Important: main.py game-network mismatch

`main.py` imports TicTacToe game but BlokusDuo neural network:

```python
from tictactoe.game import TicTacToeGame as Game      # line 9
from blokusduo.neuralnets.wrapper import NNetWrapper   # line 10
```

This will crash on the first forward pass (wrong board dimensions). Likely a development artifact — needs to be made consistent before any training run.

---

## 2. MCTS Performance

Three compounding inefficiencies that will make Blokus self-play prohibitively slow.

### Critical: Full action space iteration

**File:** `core/mcts.py` line 171

```python
for a in range(self.game.get_action_size()):  # 17,837 iterations
    if valids[a]:
        # ... compute UCB for ~50 valid moves
```

With ~1,000 simulations per move and ~50 moves per game, this is ~900 million unnecessary iterations per game. The fix is trivial:

```python
valid_actions = np.where(valids)[0]  # ~50 indices
for a in valid_actions:
    # ... compute UCB
```

**Impact:** Orders-of-magnitude speedup. Zero impact on correctness. This is the single highest-value fix in the codebase.

### Important: String-based state hashing

**File:** `core/mcts.py` line 131

MCTS hashes board states by calling `game.string_representation(board)`, which does `board.tobytes()`. For a 14×18 int64 array, that's 2,016 bytes serialised per state lookup. With thousands of lookups per MCTS simulation, this generates significant cache pressure.

Options:
- Use `hashlib.md5(board.tobytes()).hexdigest()` — shorter keys, same semantics
- Use `hash(board.tobytes())` — fastest, slight collision risk (acceptable for transposition table)
- Use a Zobrist hash — incremental updates when making/unmaking moves

Recommend starting with `hash(board.tobytes())` and revisiting if collisions appear.

### Nice-to-have: Tree rebuilt every game

**File:** `core/coach.py` line 178

```python
self.mcts = MCTS(self.game, self.nnet, self.run_config.mcts_config)
```

A new MCTS tree is created for every game. AlphaZero reuses transposition tables across games within the same generation (since the same network is evaluating positions). This could save significant computation for positions that recur across games.

Not critical for initial training, but worth adding once the basics work.

---

## 3. Board Representation & Encoding

### Current design

| Aspect | Format | Notes |
|--------|--------|-------|
| Game board | `np.ndarray` shape (14, 14), dtype int64 | 1=White, -1=Black, 0=empty |
| NN input | `np.ndarray` shape (14, 18) | Board (14×14) + piece tracking (14×2 per player) |
| Piece tracking | 14×2 region per player | 28 cells allocated for 21 pieces — 7 wasted per player |
| Array indices | Origin top-left, row down, col right | Standard numpy convention |
| Board coords | Origin bottom-left, x right, y up | Standard Blokus notation |
| Conversion | `CoordinateIndexDecoder` class | `row = 13 - y`, `col = x` |

### Important: Boundary check off-by-one bugs

**File:** `blokusduo/game.py` lines 44, 48, 75, 81 (in `no_sides` function)

```python
if i - 1 > 0:        # Bug: should be >= 0
    above = board[i - 1, j] == side
if j - 1 > 0:        # Bug: should be >= 0
    left = board[i, j - 1] == side
```

This means row 0 and column 0 neighbours are never checked. Pieces touching the edge of the board may not have their side-adjacency validated correctly.

**Fix:** Change `> 0` to `>= 0` in all four boundary checks.

### Important: Hardcoded board size

**File:** `blokusduo/game.py` — at least 7 locations use literal `14` or `13` instead of `self.n` or `self.n - 1`. Examples: lines 48, 51, 275, 335.

**Fix:** Replace all literals with `self.n` / `self.n - 1`. No functional change now but prevents a class of bugs when building `valid_moves()` on top of this code.

### Nice-to-have: Piece tracking encoding is wasteful

The NN input uses a 14×2 region per player (28 cells) to track 21 pieces. Each "cell" indicates whether a piece has been played. This wastes 7 cells per player and adds 4 unnecessary columns to every board tensor that flows through the network.

Alternative: use a 21-element binary vector, tiled to match the spatial dimensions. Or a more compact multi-channel encoding (e.g., 14×14×4 with separate channels for: white pieces, black pieces, white remaining, black remaining).

Not blocking — the current encoding works, it's just not optimal. Worth revisiting if training is memory-constrained.

### Design note: Dual coordinate systems

The two coordinate systems (board coords with bottom-left origin, array indices with top-left origin) are correct and well-documented. `CoordinateIndexDecoder` handles conversion. The naming could be clearer — "CoordinateSystemConverter" or "BoardCoordTransform" would be more descriptive — but the logic is sound.

---

## 4. Training Pipeline

### Important: Optimizer reset per epoch

**File:** `blokusduo/neuralnets/wrapper.py` line 71

```python
optimizer = optim.Adam(self.nnet.parameters())
```

This creates a fresh Adam optimizer every time `train()` is called. Adam's adaptive learning rates (momentum and variance estimates) are lost between calls. Since `train()` is called once per generation, Adam restarts from scratch every generation.

**Fix:** Create the optimizer once (in `__init__`) and reuse it across calls to `train()`.

### Important: No optimizer state in checkpoints

**File:** `blokusduo/neuralnets/wrapper.py`, `tictactoe/neuralnets/wrapper.py`

Only `model.state_dict()` is saved. Adam's momentum buffers are lost when loading a checkpoint.

```python
# Current
torch.save(self.nnet.state_dict(), filepath)

# Fix
torch.save({
    'model_state_dict': self.nnet.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
}, filepath)
```

### Important: No learning rate schedule

`core/config.py` has a `learning_rate` field but no schedule. Adam starts at 0.001 for all generations. AlphaZero used a step-decay schedule (dropping by 10× at specific training steps). At minimum, implement cosine annealing or step decay.

### Nice-to-have: Policy loss function

**File:** `blokusduo/neuralnets/wrapper.py` lines 137-149

The loss is `-sum(targets * outputs) / N` where outputs are `log_softmax`. This is mathematically correct (negative log-likelihood under a categorical distribution) but misses numerical stability optimisations that PyTorch's built-in functions handle. Replace with `F.kl_div(outputs, targets, reduction='batchmean')` since outputs are already log-softmax.

### Nice-to-have: Random sampling with replacement

**File:** `wrapper.py` line 84

```python
sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
```

Allows duplicate examples in a batch. Minor inefficiency. Better: `np.random.choice(len(examples), size=batch_size, replace=False)` or use `torch.utils.data.DataLoader` with proper shuffling.

---

## 5. Data Storage & Formats

### Current storage architecture

| Data | Format | Location | Notes |
|------|--------|----------|-------|
| Self-play games | Pickle per generation | `SelfPlayHistory/self_play_history_{gen}.pickle` | Slow, non-portable |
| Training metrics | Pickle → Parquet (two-stage) | `TrainingData/train_{gen}.data` then consolidated | Extra step |
| Arena results | Individual Parquet per generation | `ArenaData/arena_data_{gen}.parquet` | Never consolidated |
| Model checkpoints | PyTorch `.pth.tar` | `Nets/` directory | Missing optimizer state |
| Timing data | Single Parquet file | `Timings/` directory | Appended across generations |
| Reports | HTML with Plotly | `Reporting/` directory | Good |

### Issues

1. **Pickle is slow and fragile.** Python pickle is the slowest serialisation format and breaks across Python versions. For training data (which could be hundreds of MB across generations), this is a real bottleneck. HDF5 or Parquet from the start would be better.

2. **No consolidation.** Arena data accumulates as individual files. No utility to consolidate them for analysis.

3. **No schema versioning.** If the training example format changes (e.g., adding piece tracking), old pickle files silently break.

4. **Two-stage training data.** Self-play writes pickle, wrapper loads pickle and writes Parquet. Could go straight to a more efficient format.

**Recommendation:** For the initial implementation, keep pickle for simplicity but plan a migration to HDF5 for training data once the pipeline is stable. The I/O overhead only matters when self-play generates thousands of games per generation.

---

## 6. Code Organisation & Naming

### Structure (Good)

The directory layout is clean:

```
core/           → Game-agnostic framework (5 files, well-factored)
blokusduo/      → Blokus implementation (game + pieces + neural nets)
tictactoe/      → Reference implementation (complete, working)
```

Config management via frozen dataclasses is excellent — immutable after creation, type-safe, JSON-serialisable.

### Naming issues

**MCTS dictionary names** (`core/mcts.py` lines 60-65):

| Current | Meaning | Better |
|---------|---------|--------|
| `Qsa` | Q-values per (state, action) | `q_values` |
| `Nsa` | Visit counts per (state, action) | `visit_counts` |
| `Ns` | Visit counts per state | `state_visit_counts` |
| `Ps` | Policy priors per state | `policy_priors` |
| `Es` | End-state cache | `game_ended_cache` |
| `Vs` | Valid moves mask | `valid_moves_cache` |

`Vs` is particularly confusing — `V` universally means "value" in RL. Using it for "valid moves" will mislead anyone reading the code.

**Other naming inconsistencies:**

| Issue | Examples |
|-------|---------|
| `config` vs `args` | Coach uses `run_config`, main.py uses `args`, some places use `config` |
| `nnet` vs `net` | Used interchangeably for the neural network |
| `CoordinateIndexDecoder` | Unclear direction — is it coordinate→index or index→coordinate? Better: `CoordinateSystemConverter` |
| `pieces_played_encoded_region` | Overly long. Better: `piece_tracker` or `pieces_played_bitmap` |

**Convention violations:**
- MCTS uses `PascalCase` for dictionary names (`Qsa`, `Nsa`) — should be `snake_case`
- Mixed `Tuple` (typing module) vs `tuple` (builtin) — standardise on builtin for Python 3.10+
- `nptyping.NDArray` in pieces.py is deprecated — use `numpy.typing.NDArray`

### MCTS fallback policy is confusing

**File:** `core/mcts.py` line 155

```python
self.Ps[s] = self.Ps[s] + valids  # 0 + valids = valids (works by accident)
```

When all valid moves get zero probability from the network, this "adds" the valid mask to the zero vector. It works because 0 + valids = valids, but it's semantically wrong. Fix: `self.Ps[s] = valids.astype(float)`.

### Arena player indexing is clever but unreadable

**File:** `core/arena.py` line 105

```python
players = [self.player2, None, self.player1]
current_player = players[cur_player + 1]
```

Maps player IDs {-1, 1} to objects via list offset. Replace with a dict: `players = {1: self.player1, -1: self.player2}`.

### Temperature calculation is obtuse

**File:** `core/coach.py` line 128

```python
temperature = int(move_count < self.run_config.temp_threshold)
```

Implicit bool→int cast. Replace with an explicit if/else for readability.

---

## 7. Piece System

### Piece orientation ID mapping has gaps

**File:** `blokusduo/pieces.py` line 227

The `BidirectionalDict` population loop increments `i` one extra time between pieces, creating gaps in the ID sequence. This means action indices won't map correctly to piece-orientation pairs.

```python
# Current behaviour: IDs are 0,1,...,N, [skip], N+2, N+3, ...
# Expected behaviour: IDs are 0,1,...,90 (contiguous)
```

**Impact:** Wrong pieces placed when decoding actions. Will silently produce illegal moves.

### BidirectionalDict is awkward

**File:** `blokusduo/pieces.py` lines 91-119

Subclasses `dict` and stores both `key→value` and `value→key` in the same dict. This means `len(lookup)` returns 182 (2×91) not 91, and iterating gives both directions.

**Recommendation:** Replace with a dedicated class that stores one mapping and provides explicit `forward(key)` / `reverse(value)` methods.

### Magic number action_size

**File:** `blokusduo/neuralnets/net.py` line 78

Action size is computed from hardcoded constants `(14 * 14 * 91) + 1` instead of being passed from the game object. There's a TODO comment acknowledging this.

**Fix:** Pass `game.get_action_size()` to the network constructor.

---

## 8. Neural Network

### Input tensor shape naming

**File:** `blokusduo/neuralnets/net.py` line 143

The reshape does `x.view(-1, 1, self.board_x, self.board_y)` where `board_x=14` (height) and `board_y=18` (width). This is counterintuitive — `x` usually means width and `y` usually means height. The code works but the naming is misleading.

**Verify:** Confirm the actual array layout matches what the conv layers expect. If the board is stored as (rows, cols) = (14, 18), then `board_x=14` means "rows" which is height. The conv layers will work either way (they're translation-invariant) but transposed boards could confuse the piece-tracking columns.

### Network input dtype conversion

**File:** `wrapper.py` lines 86, 126-128

```python
boards = torch.FloatTensor(np.array(boards).astype(np.float64))
```

Converts to float64 then to float32 (via FloatTensor). Unnecessary — go straight to float32: `torch.tensor(np.array(boards), dtype=torch.float32)`.

---

## 9. TicTacToe Implementation (Reference)

TicTacToe is complete and working but has a few issues worth fixing to establish good patterns:

| Issue | File | Fix |
|-------|------|-----|
| `board.tostring()` deprecated since numpy 1.21 | `tictactoe/game.py:87` | Use `tobytes()` |
| Unnecessary board copies in every game method | `tictactoe/game.py:30-53` | Operate on arrays directly or use views |
| NNetWrapper uses concrete `Game` type, not `IGame` | `tictactoe/neuralnets/wrapper.py:50` | Accept `IGame` parameter |

These don't affect TicTacToe performance (3×3 board copies are free) but establish bad patterns that BlokusDuo shouldn't follow.

---

## 10. Missing Infrastructure

### Critical: Action encoding/decoding

There is no conversion between action indices (integers 0-17,836 that the neural network outputs) and `Action` dataclass instances (piece_id, orientation, x, y) that the game logic uses. This is the bridge between the network and the game, and it doesn't exist.

Needed:
- `encode(action: Action) → int` — for building policy targets from MCTS
- `decode(index: int) → Action` — for converting network output to playable moves
- `pass_action_index = 17836` — the pass action constant

The encoding scheme is implicit in the action space definition: `index = y * (14 * 91) + x * 91 + orientation_id`, with pass at index 17,836. This needs to be made explicit in code.

**Estimated effort:** ~45 minutes.

### Critical: 5 unimplemented BlokusDuoGame methods

These are the Phase 2 work items, listed here for completeness:

| Method | Purpose | Depends on |
|--------|---------|------------|
| `valid_moves()` | Generate all legal moves | Piece system, placement validation |
| `valid_move_masking()` | Binary mask over action space | `valid_moves()` + action encoding |
| `game_ended()` | Detect game over | `valid_moves()` for both players |
| `get_symmetries()` | Data augmentation | Board + policy vector rotation |
| `populate_moves_for_new_placement()` | Corner-based move generation | Placement point cache |

### Important: Model loading is broken

**File:** `main.py` lines 32-35

`load_model=True` raises `NotImplementedError`. `load_train_examples()` in `coach.py` also doesn't work. Training cannot be resumed from a checkpoint.

### Nice-to-have: No input validation

No assertions anywhere that:
- Training examples have correct shapes
- Policy vectors sum to ~1
- Values are in [-1, 1]
- Board states are valid after moves

Bad data would silently cause training to diverge.

### Nice-to-have: No unit tests

No tests exist for any part of the codebase. Priority test targets:
- Piece placement validation (boundary checks, corner touching, side avoidance)
- Action encoding/decoding roundtrip
- MCTS produces valid moves on TicTacToe
- Complete game simulation (start to finish)

---

## 11. Performance Measurement & Profiling

### What exists

The training loop already captures coarse-grained timing:

```
coach.py saves to Timings/:
  - Self-play time (per generation)
  - Training time (per generation)
  - Arena evaluation time (per generation)
  - Total generation time
```

This is useful for tracking macro-level bottlenecks but tells you nothing about *where within* each phase the time is spent.

### What we need to measure

| Category | Metric | Why |
|----------|--------|-----|
| **MCTS** | Time per simulation | The inner loop — how fast is each tree traversal? |
| **MCTS** | Simulations per second | Throughput metric for the core algorithm |
| **MCTS** | Tree size (nodes) at end of search | Memory scaling — does the tree fit in RAM? |
| **MCTS** | Branching factor (average valid moves) | Confirms action space sparsity assumptions |
| **Neural net** | Inference latency (ms per `predict()` call) | Called once per MCTS simulation — this is the hot path |
| **Neural net** | Training throughput (examples/second) | Determines how fast we can iterate |
| **Neural net** | GPU memory usage during training | Are we close to OOM? Room to increase batch size? |
| **Self-play** | Games per hour | The headline throughput number |
| **Self-play** | Average game length (moves) | Tracks whether the agent is learning to play longer |
| **Memory** | Peak RSS during self-play | Does the MCTS tree fit in RAM across a full game? |
| **Memory** | Peak GPU memory during training | Are we leaving headroom or hitting limits? |
| **I/O** | Time to save/load training data | Is pickle serialisation a bottleneck? |

### How to capture it

**Lightweight instrumentation (always on):**
- Wrap MCTS search, neural net inference, and training epochs with `time.perf_counter()` decorators
- Log per-generation summary stats (mean, p50, p95, max) to the existing timing Parquet file
- Add `torch.cuda.max_memory_allocated()` / `torch.mps.current_allocated_memory()` after training

**Deep profiling (on demand):**
- `cProfile` + `snakeviz` for CPU profiling of a single self-play game
- `torch.profiler` for GPU kernel-level analysis of training
- `tracemalloc` for Python memory tracking during MCTS
- `py-spy` for sampling profiler (no code changes needed, good for "where is time going?")

**The profiling report:**

Generate an HTML dashboard (alongside the existing training reports) that surfaces:
- Time breakdown per generation: pie chart of self-play / training / arena / I/O
- MCTS drill-down: inference time vs tree traversal time vs move generation time
- Memory timeline: peak RSS and GPU memory across generations
- Throughput trends: games/hour and examples/second over time
- Bottleneck flags: automatically highlight any metric that's >2× slower than the previous generation

This should be a Plotly report like the existing training reports, generated at the end of each generation. The reporting infrastructure (`reporting.py`) already exists and can be extended.

### Implementation estimate

| Component | Effort | Notes |
|-----------|--------|-------|
| MCTS timing instrumentation | ~30 min | Wrap search loop, log per-simulation stats |
| Neural net timing + memory | ~20 min | Wrap predict/train, query GPU memory |
| Self-play throughput tracking | ~15 min | Games/hour, average game length |
| Extend timing Parquet schema | ~15 min | Add new columns to existing timing data |
| Profiling report (Plotly HTML) | ~1 hour | New report template with the metrics above |
| Deep profiling utilities | ~30 min | cProfile/py-spy wrapper scripts |
| **Total** | **~2.5 hours** | Claude-assisted estimate |

---

## 12. Compute & Training Infrastructure

### Hardware options

| Platform | Chip | Cores | Memory | PyTorch Backend | Best For |
|----------|------|-------|--------|----------------|----------|
| **MacBook Pro M4 Pro** | Apple M4 Pro | 10P+4E CPU, 20 GPU | 24 GB unified | MPS | Development, debugging, small test runs |
| **Custom PC (RTX 3080 Ti)** | NVIDIA GA102 | 10,240 CUDA | 12 GB GDDR6X | CUDA | Serious training, full self-play generations |
| **AWS (e.g. g5.xlarge)** | NVIDIA A10G | 9,216 CUDA | 24 GB GDDR6 | CUDA | Final benchmark runs if PC isn't enough |
| **Snowflake** | TBD | TBD | TBD | CUDA (likely) | Same as AWS but potentially free if approved |

### Tiered approach

**Tier 1: Mac (always available)**
- All development and debugging
- Small test runs (2-5 generations, 10 games/generation, 25 MCTS sims)
- Verify fixes don't break the pipeline
- Profile and identify bottlenecks
- MPS backend has some rough edges with PyTorch but works for basic training

**Tier 2: Custom PC (primary training)**
- Full self-play training runs (50+ generations, 100+ games/generation, 200+ MCTS sims)
- The RTX 3080 Ti is a strong card — 12 GB VRAM should be plenty for the current ResNet (4 blocks, 512 channels)
- CUDA is the most mature PyTorch backend, fewest surprises
- Run overnight for multi-day training

**Tier 3: Cloud (final benchmarking)**
- Only needed if the 3080 Ti's 12 GB VRAM is insufficient or training needs to be faster
- AWS g5.xlarge: ~$1.01/hour (on-demand), ~$0.30/hour (spot). 24 hours = ~$7.50-$24
- A full training run (50 generations × 30 min/gen = ~25 hours) would cost £10-25 on AWS spot
- Well within the £100 budget
- Snowflake: free if approved, but requires building a case (LinkedIn article about the project)

### Open questions (to resolve during Phase 3)

1. **Is the 3080 Ti's 12 GB enough?** The ResNet with 4 blocks and 512 channels is small. But MCTS tree memory + self-play + training might push it. Profiling on the Mac first will give us estimates.

2. **Is training GPU-bound or CPU-bound?** If MCTS spends most of its time in Python (tree traversal, move generation) rather than GPU inference, then a faster GPU won't help — we'd need to optimise the Python code or move MCTS to C++.

3. **How long does a full training run take?** Completely unknown until we run Phase 2. AlphaZero for Chess used 44 million games. Blokus is simpler but our hardware is weaker. Could be 24 hours, could be a week.

4. **Is parallel self-play worth building?** If a single game takes 60 seconds and we need 100 games per generation, that's ~100 minutes per generation without parallelism. With 14 CPU cores on the Mac or 16+ threads on the PC, we could cut this to ~10 minutes. This is a Phase 3 optimisation.

5. **Snowflake — is it worth pursuing?** Depends on the scale of compute needed. If AWS spot keeps us under £30 total, it's not worth the political overhead. If we need multiple days of GPU time, it's worth the conversation.

### Estimated costs

| Scenario | Platform | Duration | Cost |
|----------|----------|----------|------|
| Dev + small tests | Mac | Ongoing | £0 |
| Full training (50 gen) | PC | ~24-48 hours | £0 (electricity) |
| Extended training (100 gen) | PC | ~48-96 hours | £0 (electricity) |
| Final benchmark run | AWS g5.xlarge spot | ~24 hours | ~£8-10 |
| Heavy training | AWS g5.xlarge spot | ~72 hours | ~£25-30 |
| Max scenario | AWS p3.2xlarge spot | ~48 hours | ~£50-70 |

The PC should handle everything except possibly the longest training runs. AWS is the overflow valve, and the budget of £100 gives plenty of headroom.

---

## 13. Summary & Priority Matrix

### Critical (Fix before any Blokus training)

| # | Issue | Section | Effort |
|---|-------|---------|--------|
| 1 | MCTS iterates all 17,837 actions | §2 | 15 min |
| 2 | Interface type mismatches (BlokusDuoGame) | §1 | 1 hour |
| 3 | NNetWrapper constructor mismatch | §1 | 20 min |
| 4 | Boundary check off-by-one bugs | §3 | 15 min |
| 5 | Piece orientation ID gaps | §7 | 30 min |
| 6 | Action encoding/decoding missing | §10 | 45 min |
| 7 | main.py game-network mismatch | §1 | 10 min |
| 8 | Hardcoded board sizes | §3 | 20 min |

### Important (Fix during Phase 2-3)

| # | Issue | Section | Effort |
|---|-------|---------|--------|
| 9 | Optimizer reset per epoch | §4 | 15 min |
| 10 | No optimizer state in checkpoints | §4 | 20 min |
| 11 | No learning rate schedule | §4 | 1 hour |
| 12 | Policy loss function | §4 | 15 min |
| 13 | Model loading broken | §10 | 45 min |
| 14 | string_representation returns bytes | §1 | 10 min |
| 15 | Verify neural net input tensor shape | §8 | 30 min |
| 16 | Profiling instrumentation | §11 | 2.5 hours |

### Nice-to-have (Cleanup when convenient)

| # | Issue | Section | Effort |
|---|-------|---------|--------|
| 17 | MCTS state hashing optimisation | §2 | 30 min |
| 18 | MCTS naming (Qsa → q_values etc.) | §6 | 20 min |
| 19 | Arena player indexing | §6 | 10 min |
| 20 | BidirectionalDict redesign | §7 | 30 min |
| 21 | Magic number action_size | §7 | 10 min |
| 22 | Temperature calculation readability | §6 | 5 min |
| 23 | TTT deprecated tostring() | §9 | 5 min |
| 24 | TTT unnecessary board copies | §9 | 15 min |
| 25 | Training input validation | §10 | 20 min |
| 26 | Random sampling with replacement | §4 | 10 min |
| 27 | Piece tracking encoding waste | §3 | 1 hour |
| 28 | Data storage migration (pickle → HDF5) | §5 | 2 hours |
| 29 | dtype conversion overhead | §8 | 10 min |
| 30 | Unit test suite | §10 | 3-4 hours |

### Estimated total effort

| Priority | Count | Effort |
|----------|-------|--------|
| Critical | 8 issues | ~3 hours |
| Important | 8 issues | ~5.5 hours |
| Nice-to-have | 14 issues | ~8.5 hours |
| **All** | **30 issues** | **~17 hours** |

Critical issues should be tackled as Phase 1.5 before implementing the remaining Blokus game logic. Important issues should be addressed during Phases 2-3. Nice-to-have items are genuine improvements but won't block progress.

---

## Checklist

Use this to track progress:

**Critical:**
- [ ] Fix MCTS action iteration (sparse valid moves only)
- [ ] Fix interface type mismatches in BlokusDuoGame
- [ ] Fix NNetWrapper constructor mismatch
- [ ] Fix boundary check off-by-one bugs
- [ ] Fix piece orientation ID gaps
- [ ] Build action encoding/decoding layer
- [ ] Fix main.py game-network mismatch
- [ ] Replace hardcoded board sizes with `self.n`

**Important:**
- [ ] Fix optimizer reset per epoch
- [ ] Save optimizer state in checkpoints
- [ ] Add learning rate schedule
- [ ] Fix policy loss function
- [ ] Fix model loading
- [ ] Fix string_representation return type
- [ ] Verify neural net input tensor shape
- [ ] Build profiling instrumentation + report

**Nice-to-have:**
- [ ] Optimise MCTS state hashing
- [ ] Rename MCTS dictionaries
- [ ] Clean up arena player indexing
- [ ] Redesign BidirectionalDict
- [ ] Remove magic number action_size
- [ ] Clean up temperature calculation
- [ ] Fix TTT deprecated tostring()
- [ ] Fix TTT unnecessary board copies
- [ ] Add training input validation
- [ ] Fix random sampling with replacement
- [ ] Optimise piece tracking encoding
- [ ] Migrate data storage (pickle → HDF5)
- [ ] Fix dtype conversion overhead
- [ ] Build unit test suite
- [ ] Run TTT training to verify nothing broke
