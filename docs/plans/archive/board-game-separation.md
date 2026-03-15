# AlphaBlokus — Board/Game Responsibility Separation

Clarify what Board and Game classes are each responsible for, eliminate duplicated constants, move game-rule logic off the board, and decouple neural nets from the game object. Applies to both TicTacToe and BlokusDuo.

Prerequisites: Bug fixes plan (completed).

---

## Checklist

| # | Item | Effort | Done |
|---|------|--------|------|
| R1 | Net constructors take plain ints, not game objects | 30 min | ✅ |
| R2 | Eliminate duplicated constants — game derives from board | 20 min | ✅ |
| R3 | Make board per-player state accessors public (BlokusDuo) | 15 min | ✅ |
| R4 | Move `game_ended`, `game_result`, `_calculate_score` from board to game (BlokusDuo) | 45 min | ✅ |
| R5 | Move `valid_moves` from board to game (BlokusDuo) | 30 min | ✅ |
| R6 | Move `get_legal_moves`, `has_legal_moves`, `is_win` from board to game (TicTacToe) | 30 min | ✅ |
| R7 | Update IGame protocol to reflect new responsibilities | 20 min | ✅ |

**Estimated total: ~3 hours**

---

## R1. Net constructors take plain ints, not game objects

**Current state:** Both `AlphaTicTacToe` and `AlphaBlokusDuo` take a `game` object in their constructor and call `game.get_board_size()` and `game.get_action_size()`. This creates a dependency from the neural net layer up to the game layer.

**Fix:** Both nets take `board_rows: int`, `board_cols: int`, `action_size: int`, `num_input_channels: int` as constructor args. The wrapper (which already has `self.game`) resolves these values and passes them through.

**Files:**
- `games/tictactoe/neuralnets/net.py` — change `AlphaTicTacToe.__init__` signature
- `games/tictactoe/neuralnets/wrapper.py` — pass values from game
- `games/blokusduo/neuralnets/net.py` — change `AlphaBlokusDuo.__init__` signature, remove `IGame` import
- `games/blokusduo/neuralnets/wrapper.py` — pass values from game
- `tests/test_blokusduo/test_net.py` — update fixture

```python
# Before (net.py):
def __init__(self, game: IGame, config: NetConfig):
    self.board_rows, self.board_cols = game.get_board_size()
    self.action_size = game.get_action_size()

# After (net.py):
def __init__(self, board_rows: int, board_cols: int, action_size: int,
             num_input_channels: int, config: NetConfig):
    self.board_rows = board_rows
    self.board_cols = board_cols
    self.action_size = action_size
    self.num_input_channels = num_input_channels

# After (wrapper.py):
def _create_network(self) -> nn.Module:
    board = self.game.initialise_board()
    rows, cols = self.game.get_board_size()
    return AlphaBlokusDuo(
        board_rows=rows,
        board_cols=cols,
        action_size=self.game.get_action_size(),
        num_input_channels=board.num_channels,
        config=self.net_config,
    )
```

---

## R2. Eliminate duplicated constants — game derives from board

**Current state:** `BlokusDuoGame` sets `self.board_size = 14` and `self.num_planes = 91` independently. These duplicate `BlokusDuoBoard.N = 14` and `PieceManager.num_entries = 91`. Similarly, `TicTacToeGame` has `N = 3` as its own class constant, duplicating `Board.N = 3`.

**Fix:** Games derive these from their board/piece classes instead of declaring them independently.

**Files:**
- `games/blokusduo/game.py` — replace `self.board_size = 14` with `BlokusDuoBoard.N`, replace `self.num_planes = 91` with `self.piece_manager.num_entries`
- `games/tictactoe/game.py` — remove `N = 3` class constant, use `Board.N`

```python
# BlokusDuoGame (before):
self.board_size = 14
self.num_planes = 91

# BlokusDuoGame (after):
self.board_size = BlokusDuoBoard.N
self.num_planes = self.piece_manager.num_entries

# TicTacToeGame (before):
N: int = 3
# ... uses self.N everywhere

# TicTacToeGame (after):
# No class constant. Uses Board.N directly.
```

---

## R3. Make board per-player state accessors public (BlokusDuo)

**Current state:** `BlokusDuoBoard` has `_remaining_ids(player)`, `_last_piece_played(player)`, `_placement_points(player)` as private methods. Game-rule logic (scoring, game-over detection) needs these, and once that logic moves to the game (R4), the game needs public access.

**Fix:** Remove the leading underscore. These are legitimate public queries on the board state.

**Files:**
- `games/blokusduo/board.py` — rename `_remaining_ids` → `remaining_piece_ids`, `_last_piece_played` → `last_piece_played`, `_placement_points` → `placement_points`
- Update internal callers in `board.py` (`with_piece`, `game_result`)

---

## R4. Move `game_ended`, `game_result`, `_calculate_score` from board to game (BlokusDuo)

**Current state:** `BlokusDuoBoard` has `game_ended()`, `game_result(player)`, and `_calculate_score()`. These are game rules (when is the game over, who won, how is score computed), not board state.

`BlokusDuoGame.get_game_ended()` already delegates to `board.game_ended()` and `board.game_result()`. After this step, the logic lives directly on the game.

**Fix:** Move all three methods from `BlokusDuoBoard` to `BlokusDuoGame`. The game accesses board state through the public accessors from R3.

**Files:**
- `games/blokusduo/board.py` — remove `game_ended`, `game_result`, `_calculate_score`
- `games/blokusduo/game.py` — add scoring and game-over logic to `get_game_ended`

```python
# BlokusDuoGame.get_game_ended (after):
def get_game_ended(self, board: BlokusDuoBoard, player: PlayerSide) -> float:
    white_has_moves = len(board.valid_moves(1)) > 0  # uses board helper
    black_has_moves = len(board.valid_moves(-1)) > 0
    if white_has_moves or black_has_moves:
        return 0

    white_score = self._calculate_score(
        board.remaining_piece_ids(1),
        board.last_piece_played(1),
    )
    black_score = self._calculate_score(
        board.remaining_piece_ids(-1),
        board.last_piece_played(-1),
    )
    # ... same scoring logic as before
```

**Note:** `game_ended` currently raises `NotImplementedError` — this step provides the real implementation. It depends on `valid_moves` existing on the board (R5 keeps a board-level helper for this).

---

## R5. Move `valid_moves` from board to game (BlokusDuo)

**Current state:** `BlokusDuoBoard.valid_moves(player)` is a stub returning `[]`. The plan is for it to aggregate moves from the placement point cache.

**Fix:** The board keeps spatial query helpers (placement points cache, `_valid_placement`, etc.) but the method that assembles the full list of legal `Action` objects moves to the game. The game has `PieceManager` and `ActionCodec` which are needed to enumerate piece-orientations and encode actions.

The board exposes a simpler helper: the placement points cache (made public in R3). The game iterates over placement points × remaining pieces × valid orientations to build the action list.

**Files:**
- `games/blokusduo/board.py` — remove `valid_moves` stub
- `games/blokusduo/game.py` — implement `valid_moves` and `valid_move_masking` using board's placement points

```python
# On game:
def valid_moves(self, board: BlokusDuoBoard, player: PlayerSide) -> list[Action]:
    """Generate all legal moves for a player."""
    # First move logic (use cached initial actions)
    # ...
    # Subsequent moves: iterate placement points × remaining piece-orientations
    # ...

def valid_move_masking(self, board: BlokusDuoBoard, player: PlayerSide) -> NDArray:
    moves = self.valid_moves(board, player)
    mask = np.zeros(self.get_action_size())
    for action in moves:
        mask[self.action_codec.encode(action)] = 1
    if not moves:
        mask[self.action_codec.pass_action_index] = 1
    return mask
```

**Note:** The actual move generation algorithm is complex and will need its own plan. This step is about *where* the method lives, not the full implementation.

---

## R6. Move `get_legal_moves`, `has_legal_moves`, `is_win` from board to game (TicTacToe)

**Current state:** `Board` has `get_legal_moves(color)`, `has_legal_moves()`, and `is_win(color)`. These are game rules — whether a position constitutes a win, what moves are legal.

`TicTacToeGame.valid_move_masking` calls `board.get_legal_moves()`. `TicTacToeGame.get_game_ended` calls `board.is_win()` and `board.has_legal_moves()`.

**Fix:** Move all three to `TicTacToeGame`. The board keeps a simple `is_empty(x, y)` or the game reads `board.as_2d` / `board[x][y]` directly (which it already can via `__getitem__`).

**Files:**
- `games/tictactoe/board.py` — remove `get_legal_moves`, `has_legal_moves`, `is_win`
- `games/tictactoe/game.py` — move the logic there, operating on `board.as_2d` or `board[x][y]`
- `tests/test_tictactoe/test_game.py` — tests already test via the game, should mostly pass

---

## R7. Update IGame protocol to reflect new responsibilities

**Current state:** `IGame` protocol in `core/interfaces.py` documents the expected methods but doesn't enforce the Board/Game split described above.

**Fix:** Review and update docstrings on `IGame` and `IBoard` to clearly state:
- `IBoard`: state snapshot, geometry, encoding, spatial queries
- `IGame`: rules, action space, legal moves, game-over detection, symmetries

No method signature changes needed — the protocol methods already match the target design. This is a documentation pass.

**Files:**
- `core/interfaces.py` — update class and method docstrings
