# AlphaBlokus — Pre-Flight Fixes

> Targeted fixes to apply before implementing the remaining Blokus game logic. These are bugs, performance issues, and correctness problems found during code review. Not a refactor — surgical fixes only.

---

## Critical (Will break Blokus training)

### 1. MCTS iterates all 17,837 actions per simulation

**File:** `core/mcts.py` line 171
**Problem:** The MCTS best-action selection loop does `for a in range(self.game.get_action_size())`. For tic-tac-toe's 10 actions this is invisible. For Blokus, with ~1000 simulations per move and ~50 moves per game, you're doing ~900 million unnecessary iterations per game of self-play.

```python
# Current (line 171)
for a in range(self.game.get_action_size()):
    if valids[a]:
        # ... compute UCB

# Fix
valid_actions = np.where(valids)[0]
for a in valid_actions:
    # ... compute UCB
```

**Impact:** Orders-of-magnitude speedup on Blokus self-play. Zero impact on correctness.

---

### 2. Piece orientation ID mapping has gaps

**File:** `blokusduo/pieces.py` line 227
**Problem:** The `BidirectionalDict` population loop increments `i` one extra time between pieces, creating gaps in the ID sequence. This means action indices won't map correctly to piece-orientation pairs when decoding actions during move generation.

```python
# Current behavior: IDs are 0,1,2,...,N, [skip], N+2, N+3, ...
# Expected behavior: IDs are 0,1,2,...,90 (contiguous)
```

**Impact:** Wrong pieces placed when decoding actions. Will silently produce illegal moves.

---

### 3. `string_representation` returns bytes, not str

**File:** `blokusduo/game.py` line 767
**Problem:** `board.tobytes()` returns `bytes`, not `str`. MCTS uses this as a dictionary key. Python won't error (bytes are hashable) but the type annotation is wrong and mixed key types could cause silent lookup misses.

```python
# Current
return board.tobytes()

# Fix
return board.tobytes().hex()
```

**Impact:** Potential MCTS cache misses causing redundant computation or incorrect tree traversal.

---

### 4. Policy loss function is not proper cross-entropy

**File:** `blokusduo/neuralnets/wrapper.py` lines 137-149, `tictactoe/neuralnets/wrapper.py` equivalent
**Problem:** The loss is `-sum(targets * outputs) / N` where outputs are `log_softmax`. This is technically the negative log-likelihood under a categorical distribution, which is close to cross-entropy but not numerically stable. PyTorch's built-in functions handle edge cases (zero targets, numerical underflow) that this manual implementation doesn't.

```python
# Current
def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]

# Fix — use KL divergence since outputs are already log_softmax
def loss_pi(targets, outputs):
    return F.kl_div(outputs, targets, reduction='batchmean')
```

**Impact:** Training will converge slower and may be less stable. Fixing this before the first Blokus training run avoids wasting GPU hours.

---

### 5. Hardcoded board size `14` scattered across game.py

**File:** `blokusduo/game.py` lines 48, 51, 275, 335, and others
**Problem:** At least 7 places use literal `14` or `13` instead of `self.n` or `self.n - 1`. If the board size ever changes (unlikely but possible), these would silently break. More importantly, it makes the code harder to reason about — you can't tell at a glance whether `14` means "board width" or something else.

```python
# Current
if j + 1 < 14:
if i + 1 < 14:
length_range = (..., min(insertion_index[0] + p_l + 1, 13))

# Fix — use self.n everywhere
if j + 1 < self.n:
if i + 1 < self.n:
length_range = (..., min(insertion_index[0] + p_l + 1, self.n - 1))
```

**Impact:** No functional change now, but prevents a class of subtle bugs when building `valid_moves()` on top of this code.

---

## Important (Will impact training quality or speed)

### 6. MCTS fallback policy adds zeros to valids

**File:** `core/mcts.py` line 155
**Problem:** When all valid moves get masked to zero probability (neural net assigns zero to all legal moves), the fallback does `self.Ps[s] = self.Ps[s] + valids`. Since `Ps[s]` is all zeros at this point, this is equivalent to `self.Ps[s] = valids`, but it's semantically wrong and confusing.

```python
# Current
self.Ps[s] = self.Ps[s] + valids  # 0 + valids = valids (works by accident)

# Fix
self.Ps[s] = valids.astype(float)
```

**Impact:** Works correctly by accident but will confuse anyone reading the code.

---

### 7. Arena player indexing is confusing

**File:** `core/arena.py` line 105
**Problem:** Uses a 3-element list with `None` in the middle to map player IDs {-1, 1} to player objects via `players[cur_player + 1]`. This is clever but unreadable.

```python
# Current
players = [self.player2, None, self.player1]
current_player = players[cur_player + 1]

# Fix
players = {1: self.player1, -1: self.player2}
current_player = players[cur_player]
```

**Impact:** Readability only. But this is in a critical path that will be debugged frequently.

---

### 8. Unnecessary board copies in tic-tac-toe game methods

**File:** `tictactoe/game.py` lines 30-31, 39-40, 52-53
**Problem:** Every call to `get_next_state`, `valid_move_masking`, and `get_game_ended` creates a new `Board` object and copies the numpy array. For a 3×3 board this costs nothing. But it's a bad pattern — the Blokus equivalent would copy a 14×18 array thousands of times per MCTS simulation.

```python
# Current
b = Board(self.n)
b.pieces = np.copy(board)

# Fix — operate on the array directly or use views
```

**Impact:** Sets a bad precedent. The Blokus implementation should not follow this pattern. Worth fixing in TTT to establish the right pattern.

---

### 9. TicTacToe `string_representation` uses deprecated numpy method

**File:** `tictactoe/game.py` line 87
**Problem:** `board.tostring()` is deprecated since numpy 1.21. Will emit warnings and may be removed in future numpy versions.

```python
# Current
return board.tostring()

# Fix
return board.tobytes()
```

**Impact:** Deprecation warnings cluttering training logs.

---

### 10. No optimizer state saved in checkpoints

**File:** `blokusduo/neuralnets/wrapper.py`, `tictactoe/neuralnets/wrapper.py`
**Problem:** Only `model.state_dict()` is saved, not the Adam optimizer's momentum buffers. When training resumes from a checkpoint (after arena evaluation), the optimizer restarts from scratch. Adam's adaptive learning rates are lost, causing a "learning rate reset" every generation.

```python
# Current
torch.save(self.nnet.state_dict(), filepath)

# Fix
torch.save({
    'model_state_dict': self.nnet.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
}, filepath)
```

**Impact:** Training is less smooth. Each generation's first few batches will have suboptimal step sizes until Adam re-estimates its momentum. Not critical for early training but will matter for fine-tuning later.

---

### 11. Neural net input tensor shape may have x/y swapped

**File:** `blokusduo/neuralnets/net.py` line 143
**Problem:** The input reshape does `x.view(-1, 1, self.board_x, self.board_y)` but the comment references `board_y * board_x`. Numpy arrays are row-major (height, width) but the code names suggest (x=width, y=height). If the board representation is 14 rows × 18 columns, this needs to be `view(-1, 1, 14, 18)` which means board_x=14 (height) and board_y=18 (width) — counterintuitive naming.

**Impact:** May cause the conv layers to see transposed boards. Need to verify the actual array layout before fixing.

---

## Nice-to-Have (Cleanup)

### 12. BidirectionalDict mixes forward and reverse entries

**File:** `blokusduo/pieces.py` lines 91-119
**Problem:** `BidirectionalDict` subclasses `dict` and stores both `key→value` and `value→key` in the same dict. This means `len(lookup)` returns 182 (2×91) not 91, and iterating gives both directions. The code works around this but it's confusing.

**Recommendation:** Replace with a dedicated class that stores one mapping internally and provides explicit `forward(key)` / `reverse(value)` methods.

---

### 13. Magic number `action_size = (14 * 14 * 91) + 1` in net.py

**File:** `blokusduo/neuralnets/net.py` line 78
**Problem:** Action size is computed from hardcoded constants instead of being passed from the game object. There's a TODO comment acknowledging this.

**Fix:** Pass `game.get_action_size()` to the network constructor.

---

### 14. Type hint says `Action` but callers pass `int`

**File:** `blokusduo/game.py` line 659
**Problem:** `get_next_state` parameter `action` is typed as `Action` (a dataclass) but the `IGame` protocol and all callers pass an integer action index. Misleading type annotation.

---

### 15. No input validation in training loop

**File:** `blokusduo/neuralnets/wrapper.py` line 63
**Problem:** No assertions that training examples have correct shapes, policy vectors sum to ~1, or values are in [-1, 1]. Bad data would silently cause training to diverge.

**Recommendation:** Add a debug-mode assertion at the start of `train()` that validates the first batch.

---

### 16. Temperature calculation uses implicit bool→int cast

**File:** `core/coach.py` line 128
**Problem:** `temperature = int(move_count < self.run_config.temp_threshold)` — works but is obtuse. Better: explicit if/else.

---

## Checklist

Use this to track progress:

- [ ] Fix MCTS action iteration (sparse valid moves only)
- [ ] Fix piece orientation ID gaps
- [ ] Fix `string_representation` return type
- [ ] Fix policy loss function
- [ ] Replace hardcoded board sizes with `self.n`
- [ ] Fix MCTS fallback policy assignment
- [ ] Clean up arena player indexing
- [ ] Fix TTT unnecessary board copies
- [ ] Fix TTT deprecated `tostring()`
- [ ] Save optimizer state in checkpoints
- [ ] Verify neural net input tensor shape
- [ ] Clean up BidirectionalDict
- [ ] Remove magic number action_size
- [ ] Fix Action type hint
- [ ] Add training input validation
- [ ] Clean up temperature calculation
- [ ] Run tic-tac-toe training to verify nothing broke
