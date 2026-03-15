# AlphaBlokus — Bug Fixes

Execution plan for fixing bugs from the codebase audit. Each step is one reviewable commit, ordered by dependency and execution sequence. The full bug catalogue is in [Appendix A](#appendix-a-bug-catalogue).

Prerequisites: None. This is the first plan on the `fix/bug-fixes` branch.

---

## Checklist

| # | Item | Bugs | Effort | Done |
|---|------|------|--------|------|
| S1 | Replace `BidirectionalDict` with `OrientationCodec`, fix ID gaps | B5, B14 | 30 min | ✅ |
| S2 | Build action space codec (`Action` ↔ `int`) | B6 | 45 min | ✅ |
| S3 | Wire `get_next_state` to accept `int` action index | B2 | 30 min | ✅ |
| S4 | MCTS: iterate only valid moves | B1 | 15 min | ✅ |
| S5 | Replace hardcoded board sizes with `self.N` | B8 | 20 min | ✅ |
| S6 | MCTS: explicit uniform fallback policy | B25 | 5 min | ✅ |
| S7 | Pass `action_size` from game to neural net | B15 | 10 min | |
| S8 | Optimizer lifecycle: create once, save/load state | B10, B11 | 30 min | |
| S9 | Add learning rate schedule | B12 | 1 hour | |
| S10 | Fix main.py: imports, wiring, model loading | B7, B13 | 45 min | |
| S11 | Training code: DataLoader, loss function, dtypes | B18, B19, B20 | 30 min | |
| S12 | Input validation at interface boundaries | B24 | 20 min | |

**Estimated remaining: ~5 hours**

---

## S1. Replace `BidirectionalDict` with `OrientationCodec`, fix ID gaps

**Bugs:** B5, B14

**Done.** Replaced `BidirectionalDict` (which stored both directions in one `dict`, had broken `len()`, and created non-contiguous IDs due to an extra `i += 1` between pieces) with `OrientationCodec`. IDs are now contiguous and 0-based (0–90). `encode(pair) → int` and `decode(id) → pair` provide a clean API. Validated by `test_orientation_ids_contiguous`.

---

## S2. Build action space codec (`Action` ↔ `int`)

**Bugs:** B6

`OrientationCodec` maps `(piece_id, orientation) ↔ int` (0–90). This step builds the layer above it: mapping a full `Action` (piece, orientation, x, y) to a flat action index (0–17,836) and back.

**Encoding scheme:** `index = y * (14 * 91) + x * 91 + orientation_id`, with pass at index 17,836.

**Needed:**
- `encode(action: Action) → int`
- `decode(index: int) → Action`
- `PASS_ACTION_INDEX = 17836`

**Files:** New module or add to `games/blokusduo/board.py` alongside the `Action` dataclass.

---

## S3. Wire `get_next_state` to accept `int` action index

**Bugs:** B2

`IGame.get_next_state` expects `action: int`. `BlokusDuoGame.get_next_state` currently takes `action: Action`. Using the codec from S2, decode the integer at the top of the method and proceed with the `Action` internally.

**Files:** `games/blokusduo/game.py`

---

## S4. MCTS: iterate only valid moves

**Bugs:** B1

**File:** `core/mcts.py` ~line 201

Replace:
```python
for a in range(self.game.get_action_size()):
    if valids[a]:
```

With:
```python
valid_actions = np.where(valids)[0]
for a in valid_actions:
```

One-line change. Orders-of-magnitude speedup for BlokusDuo (17,837 → ~50 iterations).

---

## S5. Replace hardcoded board sizes with `self.N`

**Bugs:** B8

**File:** `games/blokusduo/board.py`

~9 locations use literal `14` or `13`. Replace with `self.N` / `self.N - 1`. Found in `_no_sides()`, `_at_least_one_corner()`, `_update_placement_points()`, `with_piece()`.

---

## S6. MCTS: explicit uniform fallback policy

**Bugs:** B25

**File:** `core/mcts.py` ~line 185

Replace `self.policy_priors[s] = self.policy_priors[s] + valids` (works by accident) with `self.policy_priors[s] = valids.astype(float)`. The subsequent normalisation line already handles dividing by the sum.

---

## S7. Pass `action_size` from game to neural net

**Bugs:** B15

**File:** `games/blokusduo/neuralnets/net.py` ~line 84

Replace hardcoded `(14 * 14 * 91) + 1` with the value from the game object. The net constructor already receives the game — derive `action_size` from it.

---

## S8. Optimizer lifecycle: create once, save/load state

**Bugs:** B10, B11

**File:** `games/base_wrapper.py`

Two tightly coupled changes (fixing one without the other is pointless):

1. Create the Adam optimizer in `__init__()` instead of at the top of every `train()` call.
2. Include `optimizer.state_dict()` in checkpoint save/load so momentum buffers survive across generations.

---

## S9. Add learning rate schedule

**Bugs:** B12

**Files:** `core/config.py`, `games/base_wrapper.py`

Add a scheduler (cosine annealing or step decay) via `torch.optim.lr_scheduler`. Requires a config field for schedule parameters. Step the scheduler per generation or per epoch depending on the strategy.

---

## S10. Fix main.py: imports, wiring, model loading

**Bugs:** B7, B13

**File:** `main.py`

Three issues, all in the same file:
1. Imports TicTacToe game but BlokusDuo neural network — make consistent.
2. Calls `NNetWrapper(args)` with one arg — pass `(game, config)`.
3. `load_model=True` raises `NotImplementedError` — implement checkpoint loading and fix the dead `load_train_examples()` reference.

Do this last among the critical items since it touches the integration point.

---

## S11. Training code: DataLoader, loss function, dtypes

**Bugs:** B18, B19, B20

**File:** `games/base_wrapper.py`

Three small improvements bundled into one commit:
1. Replace `np.random.randint` sampling with `torch.utils.data.DataLoader` (B19).
2. Replace custom policy loss with `F.kl_div(outputs, targets, reduction='batchmean')` (B18).
3. Remove unnecessary `astype(np.float64)` — use `torch.tensor(..., dtype=torch.float32)` directly (B20).

---

## S12. Input validation at interface boundaries

**Bugs:** B24

**File:** `games/base_wrapper.py`

Add assertions in `train()` that:
- Board tensors have the expected shape
- Policy vectors sum to ~1
- Values are in [-1, 1]

Tests already validate these on the output side. This catches bad data on the input side before it silently corrupts the network.

---

## Appendix A: Bug Catalogue

Bugs addressed by this plan. IDs are stable references used in step descriptions and commit messages.

| # | Summary | File | Severity | Step |
|---|---------|------|----------|------|
| B1 | MCTS iterates full action space (17,837) instead of valid moves only | `core/mcts.py` | Critical | S4 |
| B2 | `get_next_state` takes `Action` dataclass, not `int` | `games/blokusduo/game.py` | Critical | S3 |
| B5 | Piece orientation IDs have gaps (extra `i += 1`) | `games/blokusduo/pieces.py` | Critical | S1 ✅ |
| B6 | No `Action` ↔ `int` codec for 17,837 action space | `games/blokusduo/` | Critical | S2 |
| B7 | main.py imports TicTacToe game + BlokusDuo network | `main.py` | Critical | S10 |
| B8 | ~9 hardcoded `14`/`13` literals instead of `self.N` | `games/blokusduo/board.py` | Critical | S5 |
| B10 | Fresh Adam optimizer created every `train()` call | `games/base_wrapper.py` | Important | S8 |
| B11 | Optimizer state not saved in checkpoints | `games/base_wrapper.py` | Important | S8 |
| B12 | No learning rate schedule | `core/config.py` | Important | S9 |
| B13 | Model loading raises `NotImplementedError` | `main.py` | Important | S10 |
| B14 | `BidirectionalDict` broken `len()`, mixed iteration | `games/blokusduo/pieces.py` | Important | S1 ✅ |
| B15 | Action size hardcoded as `(14*14*91)+1` | `games/blokusduo/neuralnets/net.py` | Important | S7 |
| B18 | Custom policy loss misses PyTorch stability optimizations | `games/base_wrapper.py` | Nice-to-have | S11 |
| B19 | Random sampling with replacement in training | `games/base_wrapper.py` | Nice-to-have | S11 |
| B20 | Unnecessary float64→float32 double conversion | `games/base_wrapper.py` | Nice-to-have | S11 |
| B24 | No input validation in training code | `games/base_wrapper.py` | Nice-to-have | S12 |
| B25 | MCTS fallback policy works by accident | `core/mcts.py` | Nice-to-have | S6 |
