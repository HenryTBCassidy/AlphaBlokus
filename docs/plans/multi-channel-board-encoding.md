# Multi-Channel Board Encoding

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
| 1 | Create `BoardEncoder` ABC | `core/encoding.py` (new) | — | |
| 2 | Update `BaseNNetWrapper.predict` | `games/base_wrapper.py` | — | |
| 3 | Create `TicTacToeEncoder` | `games/tictactoe/encoding.py` (new) | Step 1 | |
| 4 | Wire encoder into `TicTacToeGame` | `games/tictactoe/game.py` | Step 3 | |
| 5 | Update `get_canonical_form` (TTT) | `games/tictactoe/game.py` | Step 4 | |
| 6 | Add decode guards to game methods (TTT) | `games/tictactoe/game.py` | Step 4 | |
| 7 | Update TicTacToe net | `games/tictactoe/neuralnets/net.py` | Step 5 | |
| 8 | Update `get_symmetries` (TTT) | `games/tictactoe/game.py` | Step 5 | |
| 9 | Update `string_representation` (TTT) | `games/tictactoe/game.py` | Step 5 | |
| 10 | Validate TicTacToe end-to-end | — | Steps 2–9 | |
| 11 | Add `piece_placement_board` to Player | `games/blokusduo/game.py` | — | |
| 12 | Record piece IDs in `insert_piece` | `games/blokusduo/game.py` | Step 11 | |
| 13 | Create `BlokusDuoEncoder` | `games/blokusduo/encoding.py` (new) | Step 1 | |
| 14 | Wire encoder + update game methods (BD) | `games/blokusduo/game.py` | Step 13 | |
| 15 | Update BlokusDuo net | `games/blokusduo/neuralnets/net.py` | Step 14 | |
| 16 | Cleanup old representation code (BD) | `games/blokusduo/game.py` | Step 14 | |
| 17 | Encoder/decoder tests | `tests/` | Steps 10, 16 | |
| 18 | Update `02-NEURAL-NETWORKS.md`, delete `09-BOARD-ENCODING-OPTIONS.md` | `docs/reference/` | Step 17 | |

Steps 1–10 (infra + TicTacToe) validate the pattern before Steps 11–16 (BlokusDuo).
Step 17 (tests) runs after both games are done. Step 18 (docs) is always last.

---

## Design Principle: Separate Game State from Net Representation

A key architectural decision: the **game representation** (used by game logic,
move validation, scoring) and the **net representation** (fed to the neural
network) should be cleanly separated with an explicit encoder/decoder boundary.

```
Game state                    Net representation
(simple, used by game logic)  (rich, used by neural net)

BlokusDuo:                    BlokusDuo:
  14×14 array (±1/0)    ──→    44 × 14 × 14 (per-piece spatial planes)
  + piece_placement_board

TicTacToe:                    TicTacToe:
  3×3 array (±1/0)      ──→    2 × 3 × 3 (our-stones / their-stones)
```

### Why separate?

- **Game logic stays simple.** Move validation, scoring, win detection all work
  on the plain ±1 board. No reason to complicate them with multi-channel tensors.
- **Net representation is purpose-built.** The encoder builds exactly the tensor
  the conv layers need — no compromises to serve dual purposes.
- **Canonical form is clean.** Instead of multiplying by ±1 (which doesn't work
  for multi-channel), the encoder simply chooses which player's data goes in
  which channel group based on whose turn it is.
- **Testable.** The encoder/decoder can be unit tested independently: encode a
  known board state, verify the channels are correct, decode back, verify
  round-trip consistency.

### Generalised Encoder/Decoder Architecture

Rather than ad-hoc encoder functions per game, we define a shared abstract base
class with game-specific implementations. This enforces a consistent interface
and makes adding future games straightforward.

```
core/encoding.py              ← Abstract base class (BoardEncoder)
games/tictactoe/encoding.py   ← TicTacToeEncoder
games/blokusduo/encoding.py   ← BlokusDuoEncoder
```

```python
# core/encoding.py
from abc import ABC, abstractmethod
from numpy.typing import NDArray

class BoardEncoder(ABC):
    """Converts between game-state boards and multi-channel net representations."""

    @abstractmethod
    def encode(self, board, current_player: int) -> NDArray:
        """Convert game state → multi-channel net input (canonical form).

        Args:
            board: Game-state board (raw ±1 array or board object).
            current_player: Whose turn it is (1 or -1).

        Returns:
            NDArray of shape (C, H, W) ready for the neural net.
        """
        ...

    @abstractmethod
    def decode(self, encoded_board: NDArray) -> NDArray:
        """Convert multi-channel net input → raw game-state board.

        Inverse of encode(). Primarily used by stateless games (TicTacToe)
        where MCTS passes canonical boards back into game methods that
        expect raw ±1 boards.

        Returns:
            NDArray of the raw game-state board.
        """
        ...

    @property
    @abstractmethod
    def num_channels(self) -> int:
        """Number of channels in the encoded representation."""
        ...

    @property
    @abstractmethod
    def encoded_shape(self) -> tuple[int, int, int]:
        """Full (C, H, W) shape of the encoded representation."""
        ...
```

Each `IGame` implementation holds a reference to its encoder. `get_canonical_form`
delegates to `encoder.encode()`. Game methods that receive canonical boards
(in stateless games) can use `encoder.decode()` to recover the raw board.

---

## Implementation Order — TicTacToe First

TicTacToe is simpler (2 channels, no piece tracking, small board) so we
implement it first. This validates the encoder/decoder pattern and the shared
infrastructure changes before tackling the more complex BlokusDuo encoding.

```
Part A: Shared infra    (BoardEncoder ABC, wrapper changes)
Part B: TicTacToe       (2-channel, validates the pattern)
Part C: BlokusDuo       (44-channel, builds on proven pattern)
Part D: Documentation   (update 02-NEURAL-NETWORKS.md with full context)
```

---

## Part A: Shared Infrastructure

### Step 1 — Create `BoardEncoder` ABC (`core/encoding.py`)

New file with the abstract base class shown above. This defines `encode`,
`decode`, `num_channels`, and `encoded_shape`.

### Step 2 — Update `BaseNNetWrapper.predict` (`games/base_wrapper.py`)

```python
board = torch.FloatTensor(board.astype(np.float64))
if self.net_config.cuda:
    board = board.contiguous().cuda()
board = board.unsqueeze(0)  # (C, H, W) → (1, C, H, W)
```

Replace `board.view(1, self.board_rows, self.board_cols)` with `board.unsqueeze(0)`.
This works for any number of channels — no need to know the channel count.

`train`: No changes needed. `np.array(boards)` on a list of (C, H, W) arrays
gives (batch, C, H, W). The net's `forward` handles the rest.

---

## Part B: TicTacToe — 2-Channel Player-Split Board

TicTacToe doesn't need per-piece planes or inventory tracking (only one "piece
type" and no inventory concept). But it follows the same multi-channel pattern.

### Channel Layout (2 × 3 × 3)

```
Channel 0:  Current player's stones (binary — 1 where they have a piece)
Channel 1:  Opponent's stones (binary — 1 where they have a piece)
```

### Canonical form

**Currently:** `player * board` — multiplies by ±1.

**New:** Build 2 channels from the current player's perspective:

```python
def encode(self, board, current_player):
    rep = np.zeros((2, self.n, self.n), dtype=np.float32)
    rep[0] = (board == current_player)   # My stones
    rep[1] = (board == -current_player)  # Their stones
    return rep
```

### Steps

#### Step 3 — Create `TicTacToeEncoder` (`games/tictactoe/encoding.py`)

```python
class TicTacToeEncoder(BoardEncoder):
    def __init__(self, n: int = 3):
        self.n = n

    def encode(self, board, current_player):
        rep = np.zeros((2, self.n, self.n), dtype=np.float32)
        rep[0] = (board == current_player)
        rep[1] = (board == -current_player)
        return rep

    def decode(self, encoded_board):
        """Recover raw ±1 board from 2-channel encoding.

        Channel 0 was "current player" at encoding time. We reconstruct
        using convention: channel 0 → +1, channel 1 → -1.
        """
        return encoded_board[0].astype(np.float64) - encoded_board[1].astype(np.float64)

    @property
    def num_channels(self) -> int:
        return 2

    @property
    def encoded_shape(self) -> tuple[int, int, int]:
        return (2, self.n, self.n)
```

The decoder is genuinely needed here: TicTacToe is stateless, so MCTS passes
canonical (2, 3, 3) boards to `get_next_state` which expects raw ±1. The decoder
earns its keep.

#### Step 4 — Wire encoder into `TicTacToeGame` (`games/tictactoe/game.py`)

```python
from games.tictactoe.encoding import TicTacToeEncoder

class TicTacToeGame(IGame):
    def __init__(self, n=3):
        super().__init__()
        self.n = n
        self.encoder = TicTacToeEncoder(n)
```

#### Step 5 — Update `get_canonical_form` (`games/tictactoe/game.py`)

```python
def get_canonical_form(self, board, player):
    # Decode if we received an encoded board (from MCTS passing canonical form)
    if board.ndim == 3:
        board = self.encoder.decode(board)
    return self.encoder.encode(board, player)
```

#### Step 6 — Update `get_next_state` with decode guard (`games/tictactoe/game.py`)

```python
def get_next_state(self, board, player, action):
    if action == self.n * self.n:
        return board, -player
    # Decode if we received an encoded board
    if board.ndim == 3:
        board = self.encoder.decode(board)
    b = Board(self.n)
    b.pieces = np.copy(board)
    move = (int(action / self.n), action % self.n)
    b.execute_move(move, player)
    return b.pieces, -player
```

Same decode guard needed in `get_game_ended` and `valid_move_masking`.

#### Step 7 — Update TicTacToe net (`games/tictactoe/neuralnets/net.py`)

```python
# __init__:
self.num_input_channels = 2

# First conv layer:
self.conv1 = nn.Conv2d(2, config.num_channels, 3, stride=1, padding=1)

# forward:
x = x.view(-1, self.num_input_channels, self.board_rows, self.board_cols)
```

FC layer sizes unchanged — the spatial reduction (3×3 → 1×1) is the same.

#### Step 8 — Update `get_symmetries` (`games/tictactoe/game.py`)

Currently rotates/flips a 2D board. With 2 channels, rotate/flip spatial dims only:

```python
for i in range(1, 5):
    for j in [True, False]:
        newB = np.rot90(board, i, axes=(1, 2))  # Rotate spatial dims only
        newPi = np.rot90(pi_board, i)
        if j:
            newB = np.flip(newB, axis=2)  # Flip spatial width only
            newPi = np.fliplr(newPi)
        l += [(newB, list(newPi.ravel()) + [pi[-1]])]
```

#### Step 9 — Update `string_representation` (`games/tictactoe/game.py`)

Currently `board.tostring()` (deprecated). Update to hash the raw board:

```python
def string_representation(self, board):
    if board.ndim == 3:
        board = self.encoder.decode(board)
    return board.tobytes()
```

Uses `tobytes()` (non-deprecated) and always hashes the raw board for compact,
consistent keys regardless of encoding.

#### Step 10 — Validate TicTacToe end-to-end

Run training loop, verify:
- Network trains without errors
- Symmetries produce valid augmented data
- MCTS plays legal moves
- Arena evaluates games correctly

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
in channels 0–20 and "their pieces" in 21–41. This is the canonical form.

### What a conv filter sees

At any board position, a 3×3 filter reads all 44 channels simultaneously:

- Is there any piece here? (channels 42–43)
- If so, WHICH specific piece? (channels 0–41)
- What pieces neighbour this square and which pieces are they?
- Which pieces haven't been played yet? (their planes are all-zero = still available)

### Canonical form

**Currently:** `player * net_representation` — multiplies every cell by ±1.

**New:** Build 44 channels from scratch, choosing whose pieces go in which group:

| Current player | Channels 0–20      | Channels 21–41     |
|----------------|--------------------|--------------------|
| White (1)      | White's pieces     | Black's pieces     |
| Black (-1)     | Black's pieces     | White's pieces     |

No multiplication — just which group goes first.

### Sparsity (not a concern)

Each piece occupies 1–5 squares on a 196-cell grid (0.5–2.6% density). Compare
with AlphaZero chess where the "white knights" plane is 3.1% dense and "white
king" is 1.6%. Chess uses 119 planes successfully at comparable sparsity.

The "Representation Matters for Mastering Chess" paper (DeepMind, 2024) found
that adding more sparse planes with richer information improved performance by
~97 Elo. Information content matters more than density.

### Steps

#### Step 11 — Add `piece_placement_board` to Player (`games/blokusduo/game.py`)

Add to `Player.__init__`:

```python
self.piece_placement_board = np.zeros((board_size, board_size), dtype=int)
```

Stores which piece ID (1–21) occupies each square. 0 = empty.

#### Step 12 — Record piece IDs during placement (`games/blokusduo/game.py`)

In `BlokusDuoBoard.insert_piece`, alongside the existing `self.as_array[i, j] = ...`:

```python
if piece_orientation[i, j] != 0:
    player.piece_placement_board[length_idx + i, width_idx + j] = action.piece_id
```

The existing `self.as_array` (±1 values) stays — it's still used by `valid_placement`,
`no_sides`, `at_least_one_corner`. We're adding a parallel data source.

#### Step 13 — Create `BlokusDuoEncoder` (`games/blokusduo/encoding.py`)

```python
class BlokusDuoEncoder(BoardEncoder):
    def __init__(self, n: int = 14):
        self.n = n

    def encode(self, board, current_player):
        """Build 44-channel representation from BlokusDuoBoard object."""
        representation = np.zeros((44, self.n, self.n), dtype=np.float32)

        current = board._get_player(current_player)
        opponent = board._get_player(-current_player)

        # Channels 0–20: current player's per-piece planes
        for piece_id in range(1, 22):
            representation[piece_id - 1] = (current.piece_placement_board == piece_id)

        # Channels 21–41: opponent's per-piece planes
        for piece_id in range(1, 22):
            representation[21 + piece_id - 1] = (opponent.piece_placement_board == piece_id)

        # Channels 42–43: aggregate planes
        representation[42] = (current.piece_placement_board > 0)
        representation[43] = (opponent.piece_placement_board > 0)

        return representation

    def decode(self, encoded_board):
        """Not needed for BlokusDuo — board object is the source of truth."""
        raise NotImplementedError(
            "BlokusDuo maintains internal state; decode is not needed."
        )

    @property
    def num_channels(self) -> int:
        return 44

    @property
    def encoded_shape(self) -> tuple[int, int, int]:
        return (44, self.n, self.n)
```

#### Step 14 — Wire encoder into `BlokusDuoGame` and update game methods

```python
from games.blokusduo.encoding import BlokusDuoEncoder

class BlokusDuoGame(IGame):
    def __init__(self):
        super().__init__()
        self.encoder = BlokusDuoEncoder()
        ...

    def get_canonical_form(self, board, player):
        return self.encoder.encode(self.board, player)

    def get_next_state(self, canonical_board, player, action):
        self.board.insert_piece(action=action, player_side=player)
        return self.board, -player  # Return board object, not numpy
```

Returns the board object (not a numpy representation). This fixes a pre-existing
inconsistency: the current code returns `self.board.net_representation` (numpy),
but then `get_canonical_form` tries to call `.net_representation` on it again,
which would crash. The board is managed internally by `BlokusDuoGame` anyway —
`get_canonical_form`, `get_game_ended`, and `valid_move_masking` all operate on
`self.board`.

#### Step 15 — Update BlokusDuo net (`games/blokusduo/neuralnets/net.py`)

```python
# __init__:
self.board_rows, self.board_cols = 14, 14
self.num_input_channels = 44

# First conv layer:
nn.Conv2d(in_channels=44, out_channels=config.num_channels, ...)

# forward:
x = x.view(-1, self.num_input_channels, self.board_rows, self.board_cols)
```

Everything else in the net (residual blocks, policy head, value head) is unchanged.
They work on `num_channels × 14 × 14` feature maps from the first conv onward.

Note: `conv_out` changes from `14 × 18 = 252` to `14 × 14 = 196`, which affects
the Linear layers in both heads.

#### Step 16 — Cleanup

- Remove `Player.pieces_played_encoded_region` (inventory is now implicit in
  piece_placement_board — an all-zero plane means "not yet played")
- Remove `BlokusDuoBoard.net_representation` property (replaced by encoder)
- `string_representation`: hash `self.board.as_array.tobytes()` — the raw ±1
  board plus piece placements fully determine the state.

### Parameter count impact

| Component | Current | New |
|-----------|---------|-----|
| First conv layer | `1 × C × 3 × 3` | `44 × C × 3 × 3` |
| Value head FC | `1 × 252 → C` | `1 × 196 → C` |
| Policy head FC | `2 × 252 → action_size` | `2 × 196 → action_size` |

With C=128 channels: first layer goes from 1,152 to 50,688 parameters. This is
a ~44× increase in the first layer only. All subsequent layers are identical.
For a net with ~10 residual blocks the first layer is a tiny fraction of total
parameters.

---

## Part D: Testing

#### Step 17 — Encoder/decoder tests (`tests/`)

Assumes the structural refactor's S11 (pytest setup) is already done.

**`tests/test_tictactoe/test_encoding.py`:**
- Encode/decode roundtrip: encode a known board, decode it, verify identical
- Canonical form symmetry: encoding for player 1 vs player -1 swaps channels
- Empty board: all channels zero
- Full board: every cell covered across both channels

**`tests/test_blokusduo/test_encoding.py`:**
- Encode after placing a known piece: correct channel is non-zero, others zero
- Per-piece planes are mutually exclusive (no overlap)
- Aggregate planes (42–43) equal the union of per-piece planes
- Canonical form: white-to-play and black-to-play produce swapped channel groups
- Empty board: 44 channels all zero

**`tests/test_core/test_encoding.py`:**
- `BoardEncoder` ABC cannot be instantiated directly
- Both concrete encoders satisfy the interface (`num_channels`, `encoded_shape`)

**`tests/test_tictactoe/test_net.py`:**
- Forward pass shape: (batch, 2, 3, 3) → (batch, 10) policy + (batch, 1) value
- Verify net accepts 2-channel input without errors

**`tests/test_blokusduo/test_net.py`:**
- Forward pass shape: (batch, 44, 14, 14) → correct policy + value shapes
- Verify net accepts 44-channel input without errors

---

## Part E: Documentation Update

#### Step 18 — Update `02-NEURAL-NETWORKS.md` and delete `09-BOARD-ENCODING-OPTIONS.md`

This is done **last**, once all encoding changes are implemented and tested.
At that point we have full context on what actually changed.

**Update `docs/reference/02-NEURAL-NETWORKS.md`:**

- Architecture comparison table (input shapes, parameter counts)
- Input encoding section (describe multi-channel layout for both games)
- Any diagrams showing board → net data flow
- Design decisions section (add encoder/decoder rationale)
- Remove any references to the old single-channel representations
- Add a "Board Encoding" section covering: the encoding problem (Blokus pieces
  + inventory), why per-piece spatial planes (option 3), and a brief note on
  the alternatives considered (flat ±1, 3-channel, per-piece) and why they
  were rejected. Keep it concise — a few paragraphs, not the full research doc.

**Delete `docs/reference/09-BOARD-ENCODING-OPTIONS.md`:**

The encoding options doc was a working document used to explore and choose the
encoding approach. Its useful content is now captured in this implementation
plan and will be folded into `02-NEURAL-NETWORKS.md` above. No need to keep
the exploratory doc around once the decision is documented in the right place.

---

## What stays the same

- `self.as_array` (14×14 with ±1/0) — still used by all placement validation
- `ResNetBlock` — unchanged, works on any channel count
- `action_size` — still `14 × 14 × 91 + 1` (BlokusDuo) / `3 × 3 + 1` (TicTacToe)
- `PieceManager`, `pieces.json`, all piece logic — untouched
- `no_sides`, `at_least_one_corner`, `valid_placement` — use `as_array`, unaffected
- MCTS, Coach, Arena — no structural changes (boards flow opaquely)
- TicTacToe `Board` class — untouched (game state stays as ±1 array)

---

## Open Questions

1. **`string_representation` hashing**: Hash the full multi-channel tensor
   (correct but larger) or hash just the raw game state (compact, sufficient
   since game state fully determines the net representation)? Leaning toward
   raw game state for efficiency.

2. **TicTacToe decode convention**: The decoder reconstructs channel 0 as +1,
   channel 1 as -1. This assumes the canonical form was built for the "current"
   player at encoding time. If MCTS passes the canonical board across player
   switches, we need to track or infer whose perspective it was encoded from.
   The `ndim == 3` guard + always decoding to "player 1 perspective" should
   handle this cleanly.
