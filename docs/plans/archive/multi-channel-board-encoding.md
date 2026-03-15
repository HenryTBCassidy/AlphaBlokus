# Multi-Channel Board Encoding

**Status: Complete.** All steps including Step 18 (documentation updates) are done.

Migrate both games from single-channel board representations to multi-channel
per-player (and per-piece for BlokusDuo) spatial planes. This follows the
AlphaZero convention and gives conv filters richer information at every position.

The rationale for choosing 44-channel per-piece planes (BlokusDuo) and
2-channel player-split (TicTacToe) is covered in the design sections below.
On completion, Step 18 folds this into `docs/reference/02-NEURAL-NETWORKS.md`
and deletes the old `09-BOARD-ENCODING-OPTIONS.md` working doc.

**Prerequisite:** Structural refactor items S1–S11 (especially S11: pytest setup).

---

## Checklist

| Step | Task | Files | Depends on | Done |
|------|------|-------|-----------|------|
| 1 | ~~Create `IBoardEncoder` Protocol~~ | — | — | ✅ (superseded) |
| 2 | Update `BaseNNetWrapper.predict` (channel-agnostic) | `games/base_wrapper.py` | — | ✅ |
| 3 | ~~Create `TicTacToeEncoder`~~ | — | — | ✅ (superseded) |
| 4 | ~~Wire encoder into `TicTacToeGame`~~ | — | — | ✅ (superseded) |
| 5 | ~~Update `get_canonical_form` (encoder)~~ | — | — | ✅ (superseded) |
| 6 | ~~Add decode guards~~ | — | — | ✅ (superseded) |
| 7 | Update TicTacToe net (1→2 input channels) | `games/tictactoe/neuralnets/net.py` | — | ✅ |
| 8 | ~~Update `get_symmetries` (encoder)~~ | — | — | ✅ (superseded) |
| 9 | ~~Update `state_key` (encoder)~~ | — | — | ✅ (superseded) |
| 10 | ~~Validate TicTacToe (encoder)~~ | — | — | ✅ (superseded) |
| **R1** | **Create `IBoard` Protocol** | `core/interfaces.py` | — | ✅ |
| **R2** | **Refactor TicTacToe Board → implements IBoard** | `games/tictactoe/board.py` | R1 | ✅ |
| **R3** | **Update TicTacToeGame to flow Board objects** | `games/tictactoe/game.py` | R2 | ✅ |
| **R4** | **Delete `TicTacToeEncoder`, remove `IBoardEncoder`** | `encoding.py`, `core/interfaces.py` | R3 | ✅ |
| **R5** | **Update TicTacToe tests** | `tests/` | R3 | ✅ |
| **R6** | **Validate TicTacToe end-to-end** | — | R4, R5 | ✅ |
| 11 | Add `piece_placement_board` to Player | `games/blokusduo/board.py` | — | ✅ |
| 12 | Record piece IDs in `insert_piece` | `games/blokusduo/board.py` | 11 | ✅ |
| 13 | Extract BlokusDuoBoard to own file, implement IBoard | `games/blokusduo/board.py` | R1, 12 | ✅ |
| 14 | Wire IBoard into BlokusDuoGame | `games/blokusduo/game.py` | 13 | ✅ |
| 15 | Update BlokusDuo net (1→44 input channels) | `games/blokusduo/neuralnets/net.py` | 14 | ✅ |
| 16 | Cleanup old representation code | `games/blokusduo/game.py` | 14 | ✅ |
| 17 | Tests for both games | `tests/` | R6, 16 | ✅ |
| 18 | Update `02-NEURAL-NETWORKS.md`, delete `09-BOARD-ENCODING-OPTIONS.md` | `docs/reference/` | 17 | ✅ |

Steps 1–10 were originally the encoder/decoder pattern. Steps 2 and 7 still
apply (infra changes). Steps R1–R6 replace the encoder pattern with the IBoard
pattern for TicTacToe. Steps 11–18 implement BlokusDuo using the same pattern.

---

## Design Principle: Board Objects Own Their Representations

A key architectural decision: **the board class itself** knows how to produce
both its flat game-state representation and its multi-channel net representation.
There is no separate encoder/decoder — the board IS the thing that knows both.

Both games use the same `IBoard` Protocol, ensuring a consistent pattern.

```
Board object (implements IBoard)
├── .as_array           → flat H×W game state (±1/0), used by game logic
├── .net_representation(player) → (C, H, W) tensor, built on demand for the net
├── .num_channels       → number of channels in net representation
└── game-specific methods (get_legal_moves, is_win, execute_move, etc.)
```

### Why the board owns this?

- **Single source of truth.** The board holds all game state. It can always
  produce both the flat board (for game logic) and the net tensor (for inference).
- **No lossy encode/decode.** For BlokusDuo, you can go from the rich internal
  state → multi-channel tensor, but you CANNOT go from a flat ±1 board → multi-
  channel tensor (piece identity is lost). The board avoids this problem entirely
  because it holds the rich state and builds the tensor from it.
- **Consistent across games.** TicTacToe and BlokusDuo both implement `IBoard`.
  The rest of the system (MCTS, Coach, Arena) works with boards uniformly.
- **MCTS performance.** Board objects flow through MCTS (cheap to copy/hash via
  `as_array`). The expensive multi-channel tensor is only built at the net
  boundary (`nnet.predict`), which happens ~800 times per move vs ~12,000 calls
  for `get_next_state`.

### What flows where

```
MCTS hot path (~12,000 calls/move):
  get_next_state(board, player, action) → new Board object
  get_canonical_form(board, player)     → Board object (essentially free)
  state_key(board)                      → board.as_array.tobytes()
  get_game_ended(board, player)         → operates on board.as_array
  valid_move_masking(board, player)     → operates on board.as_array

Net boundary (~800 calls/move):
  nnet.predict(board)                   → calls board.net_representation(player)
                                          to get (C, H, W) tensor for inference
```

### IBoard Protocol

```python
class IBoard(Protocol):
    @property
    def as_array(self) -> NDArray:
        """Flat H×W game state board. Used by game logic."""
        ...

    def net_representation(self, current_player: int) -> NDArray:
        """Build (C, H, W) tensor for neural net input.

        Args:
            current_player: Whose turn it is (1 or -1).
                           Determines which player's data goes in which
                           channel group (canonical perspective).

        Returns:
            NDArray of shape (C, H, W) with float32 dtype.
        """
        ...

    @property
    def num_channels(self) -> int:
        """Number of channels in net_representation output."""
        ...
```

`net_representation(current_player)` takes `current_player` because the net
always sees "my pieces" in the first channel group and "their pieces" in the
second. This IS the canonical form — no separate canonical transform needed.

---

## Implementation Order — TicTacToe First

TicTacToe is simpler (2 channels, no piece tracking, small board) so we
implement it first. This validates the IBoard pattern before tackling BlokusDuo.

```
Part A: Shared infra       (IBoard Protocol, wrapper changes)
Part B: TicTacToe refactor (Board implements IBoard, game flows board objects)
Part C: BlokusDuo          (44-channel, same IBoard pattern)
Part D: Documentation      (update 02-NEURAL-NETWORKS.md with full context)
```

---

## Part A: Shared Infrastructure

### Step R1 — Create `IBoard` Protocol, remove `IBoardEncoder` (`core/interfaces.py`)

Replace `IBoardEncoder` Protocol with `IBoard` Protocol:

```python
class IBoard(Protocol):
    @property
    def as_array(self) -> NDArray:
        """Flat H×W game state board (±1/0). Used by game logic."""
        ...

    def net_representation(self, current_player: int) -> NDArray:
        """Build (C, H, W) tensor for neural net input."""
        ...

    @property
    def num_channels(self) -> int:
        """Number of channels in net_representation output."""
        ...
```

Delete the `IBoardEncoder` Protocol — it's superseded.

### Step 2 — Update `BaseNNetWrapper.predict` (already done ✅)

```python
board = board.unsqueeze(0)  # (C, H, W) → (1, C, H, W)
```

This was already changed to be channel-agnostic. The predict method receives
a numpy array (the net representation), not a board object. The call chain is:
MCTS calls `nnet.predict(canonical_board)` → wrapper converts board to tensor
via `board.net_representation(1)` → feeds to net.

**Note:** `BaseNNetWrapper.predict` currently takes an NDArray. It will need to
accept an IBoard and call `board.net_representation(1)` to get the tensor.
Player is always 1 because MCTS always passes canonical boards.

---

## Part B: TicTacToe — Board Objects + 2-Channel Net Representation

### Channel Layout (2 × 3 × 3)

```
Channel 0:  Current player's stones (binary — 1 where they have a piece)
Channel 1:  Opponent's stones (binary — 1 where they have a piece)
```

### Steps

#### Step R2 — Refactor TicTacToe Board (`games/tictactoe/board.py`)

Add `IBoard` implementation to the existing `Board` class:

```python
class Board:
    """TicTacToe board implementing IBoard protocol."""

    def __init__(self, n=3):
        self.n = n
        self.pieces = [None] * self.n
        for i in range(self.n):
            self.pieces[i] = [0] * self.n

    @property
    def as_array(self) -> NDArray:
        return np.array(self.pieces)

    def net_representation(self, current_player: int) -> NDArray:
        board = self.as_array
        rep = np.zeros((2, self.n, self.n), dtype=np.float32)
        rep[0] = (board == current_player)
        rep[1] = (board == -current_player)
        return rep

    @property
    def num_channels(self) -> int:
        return 2

    # ... existing methods (get_legal_moves, has_legal_moves, is_win, execute_move)
```

The encode logic moves from `TicTacToeEncoder` into the board itself.

#### Step R3 — Update `TicTacToeGame` to flow Board objects

Key changes:
- `initialise_board()` returns a `Board` object (not `np.array(b.pieces)`)
- `get_next_state()` creates and returns a new `Board` object
- `get_canonical_form()` returns the board as-is (it's already the board object)
- `valid_move_masking()` operates on `board.as_array` or `board` directly
- `get_game_ended()` operates on `board.as_array` or `board` directly
- `get_symmetries()` receives canonical board, applies transforms
- `state_key()` returns `board.as_array.tobytes()`
- Remove all `ndim == 3` decode guards
- Remove `self.encoder` attribute

```python
class TicTacToeGame(IGame):
    def __init__(self, n=3):
        super().__init__()
        self.n = n

    def initialise_board(self):
        return Board(self.n)

    def get_next_state(self, board, player, action):
        if action == self.n * self.n:
            return board, -player
        b = Board(self.n)
        b.pieces = [row[:] for row in board.pieces]  # copy
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return b, -player

    def get_canonical_form(self, board, player):
        # Board object already holds all state.
        # net_representation(player) handles canonical perspective.
        return board

    def valid_move_masking(self, board, player):
        valids = [0] * self.get_action_size()
        legal_moves = board.get_legal_moves(player)
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legal_moves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def get_game_ended(self, board, player):
        if board.is_win(player):
            return 1
        if board.is_win(-player):
            return -1
        if board.has_legal_moves():
            return 0
        return 1e-4

    def get_symmetries(self, board, pi):
        # board is a Board object. Transform as_array and pi together.
        assert len(pi) == self.n ** 2 + 1
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        board_array = board.as_array
        symmetries = []

        for i in range(1, 5):
            for is_flipped in [True, False]:
                new_array = np.rot90(board_array, i)
                new_pi = np.rot90(pi_board, i)
                if is_flipped:
                    new_array = np.fliplr(new_array)
                    new_pi = np.fliplr(new_pi)
                # Build a Board from the transformed array
                new_board = Board(self.n)
                new_board.pieces = new_array.tolist()
                new_pi_flat = np.append(new_pi.ravel(), pi[-1])
                symmetries.append((new_board, new_pi_flat))
        return symmetries

    def state_key(self, board):
        return board.as_array.tobytes()
```

**Important:** Coach stores `(canonical_board, pi, result)` training examples.
When these are fed to the net, they need to be converted to tensors. This
happens in the training loop — `canonical_board.net_representation(1)` is
called to get the (C, H, W) array that gets batched.

#### Step R4 — Delete `TicTacToeEncoder`, remove `IBoardEncoder`

- Delete `games/tictactoe/encoding.py`
- Remove `IBoardEncoder` from `core/interfaces.py`
- Remove `from games.tictactoe.encoding import TicTacToeEncoder` from `game.py`

#### Step R5 — Update TicTacToe tests

Tests that use raw numpy arrays need updating to work with Board objects:

- `test_initial_board_is_empty`: check `board.as_array` shape and values
- `test_get_next_state`: check `new_board.as_array[0][0]`
- `test_valid_moves_*`: pass Board object (already works if game methods accept it)
- `test_canonical_form_*`: canonical form is now the board itself; test
  `board.net_representation(player)` for the channel layout
- `test_symmetries_*`: symmetries return Board objects
- `test_state_key_*`: works on Board objects
- `test_full_random_game`: minor updates for Board object flow

#### Step R6 — Validate end-to-end

Run all tests. The key integration points:
- MCTS passes Board objects → `state_key`, `get_game_ended`, `valid_move_masking`
  all operate on the board's methods/as_array
- `nnet.predict` receives a Board, calls `net_representation(1)` to get tensor
- Coach training loop converts Board examples to tensors via `net_representation`

---

## Part C: BlokusDuo — 44-Channel Per-Piece Spatial Planes

### Channel Layout (44 × 14 × 14)

```
Channels  0–20:  Current player's 21 pieces (one 14×14 binary plane each)
                 Channel i → 1s where piece (i+1) sits on the board, 0s elsewhere.
                 All-zero if that piece hasn't been played yet.

Channels 21–41:  Opponent's 21 pieces (same layout)

Channel  42:     Aggregate — union of channels 0–20 (all current player's stones)
Channel  43:     Aggregate — union of channels 21–41 (all opponent's stones)
```

"Current player" = the player whose turn it is. The net always sees "my pieces"
in channels 0–20 and "their pieces" in 21–41. This is the canonical form,
handled by `net_representation(current_player)`.

### What a conv filter sees

At any board position, a 3×3 filter reads all 44 channels simultaneously:

- Is there any piece here? (channels 42–43)
- If so, WHICH specific piece? (channels 0–41)
- What pieces neighbour this square and which pieces are they?
- Which pieces haven't been played yet? (their planes are all-zero = still available)

### Sparsity (not a concern)

Each piece occupies 1–5 squares on a 196-cell grid (0.5–2.6% density). Compare
with AlphaZero chess where the "white knights" plane is 3.1% dense and "white
king" is 1.6%. Chess uses 119 planes successfully at comparable sparsity.

The "Representation Matters for Mastering Chess" paper (DeepMind, 2024) found
that adding more sparse planes with richer information improved performance by
~97 Elo. Information content matters more than density.

### Steps

#### Step 11 — Add `piece_placement_board` to Player

Add to `Player.__init__`:

```python
self.piece_placement_board = np.zeros((board_size, board_size), dtype=int)
```

Stores which piece ID (1–21) occupies each square. 0 = empty.

#### Step 12 — Record piece IDs during placement

In `BlokusDuoBoard.insert_piece`, alongside the existing `self.as_array[i, j] = ...`:

```python
if piece_orientation[i, j] != 0:
    player.piece_placement_board[length_idx + i, width_idx + j] = action.piece_id
```

#### Step 13 — Extract BlokusDuoBoard to own file, implement IBoard

Move `BlokusDuoBoard` from `game.py` to `games/blokusduo/board.py`.
Implement `IBoard`:

```python
class BlokusDuoBoard:
    """BlokusDuo board implementing IBoard protocol."""

    @property
    def as_array(self) -> NDArray:
        return self._as_array  # existing 14×14 ±1 board

    def net_representation(self, current_player: int) -> NDArray:
        """Build 44-channel representation."""
        rep = np.zeros((44, self.n, self.n), dtype=np.float32)

        current = self._get_player(current_player)
        opponent = self._get_player(-current_player)

        for piece_id in range(1, 22):
            rep[piece_id - 1] = (current.piece_placement_board == piece_id)
            rep[21 + piece_id - 1] = (opponent.piece_placement_board == piece_id)

        rep[42] = (current.piece_placement_board > 0)
        rep[43] = (opponent.piece_placement_board > 0)
        return rep

    @property
    def num_channels(self) -> int:
        return 44
```

#### Step 14 — Wire IBoard into BlokusDuoGame

Same pattern as TicTacToe:
- `get_canonical_form` returns the board object
- `get_next_state` returns a new/updated board object
- `state_key` uses `board.as_array.tobytes()`
- `nnet.predict` calls `board.net_representation(1)`

#### Step 15 — Update BlokusDuo net (1→44 input channels)

```python
nn.Conv2d(in_channels=44, out_channels=config.num_channels, ...)
```

#### Step 16 — Cleanup

- Remove `Player.pieces_played_encoded_region` (inventory is implicit in
  piece_placement_board — an all-zero plane = "not yet played")
- Remove `BlokusDuoBoard.net_representation` property if it exists separately
- Clean up any old encoding-related code

### Parameter count impact

| Component | Current | New |
|-----------|---------|-----|
| First conv layer | `1 × C × 3 × 3` | `44 × C × 3 × 3` |
| Value head FC | `1 × 252 → C` | `1 × 196 → C` |
| Policy head FC | `2 × 252 → action_size` | `2 × 196 → action_size` |

With C=128 channels: first layer goes from 1,152 to 50,688 parameters. This is
a ~44× increase in the first layer only. All subsequent layers are identical.

---

## Part D: Testing & Documentation

#### Step 17 — Tests for both games

**TicTacToe tests:** Board object flow, `net_representation` channel layout,
symmetries on Board objects, state_key consistency.

**BlokusDuo tests:** `piece_placement_board` tracking, `net_representation`
channel correctness, per-piece planes mutually exclusive, aggregate planes
match union of per-piece planes, canonical perspective swapping.

#### Step 18 — Update `02-NEURAL-NETWORKS.md`, delete `09-BOARD-ENCODING-OPTIONS.md`

- Architecture comparison table (input shapes, parameter counts)
- Input encoding section (describe IBoard pattern for both games)
- Design decisions section (add IBoard rationale)
- Remove references to old single-channel representations

---

## What stays the same

- `self.as_array` (14×14 with ±1/0) — still used by all placement validation
- `ResNetBlock` — unchanged, works on any channel count
- `action_size` — still `14 × 14 × 91 + 1` (BlokusDuo) / `3 × 3 + 1` (TicTacToe)
- `PieceManager`, `pieces.json`, all piece logic — untouched
- `no_sides`, `at_least_one_corner`, `valid_placement` — use `as_array`, unaffected
- MCTS, Coach, Arena — no structural changes (boards flow opaquely)

---

## TODOs (future work, not part of this plan)

- **Copy-and-apply pattern for get_next_state:** Currently MCTS is stateless
  and expects new boards back from get_next_state. For BlokusDuo this means
  copying the board object. Consider optimising with undo/redo or caching.
- **Placement point caches on board:** Store `potential_placement_points` on the
  board object to speed up move generation. Update incrementally on each move.
- **Making BlokusDuo MCTS-playable:** Several game methods are still
  `NotImplementedError`. Implementing those is separate from the encoding work.
