# AlphaBlokus — External Interfaces

Three pieces of infrastructure sit between the AlphaBlokus agent and the outside world: a **Pentobi adapter** for automated benchmarking, a **human-playable UI**, and a shared **translation layer** that converts between internal and external representations. This document scopes all three.

---

## 1. Translation Layer (Shared Foundation)

Both the Pentobi adapter and the UI need to convert between AlphaBlokus's internal representation and human/external formats. This layer should be built first.

### Internal Representation

AlphaBlokus represents state as follows (see `blokusduo/game.py`):

| Aspect | Format |
|--------|--------|
| Board | `np.ndarray` shape (14, 14), dtype int64. 1=White, -1=Black, 0=empty |
| NN input | `np.ndarray` shape (14, 18). Board + piece-tracking columns |
| Action | `Action(piece_id, orientation, x_coordinate, y_coordinate)` |
| Coordinates | Board coords: origin bottom-left, x right, y up (0-indexed) |
| Array indices | Origin top-left, row down, col right. Conversion: `row = 13 - y`, `col = x` |
| Action space | 14 × 14 × 91 + 1 = 17,837. Last index is pass |
| Pieces | 21 pieces, IDs 1-21. 91 basis orientations via `PieceManager` |
| Players | White = 1, Black = -1 |

### Pentobi GTP Representation

Pentobi's GTP interface uses a different format:

| Aspect | Format |
|--------|--------|
| Board | Text-based, 14×14. Columns a-n, rows 1-14 |
| Coordinates | Origin bottom-left. Columns are letters (a=0, b=1, ..., n=13). Rows are 1-indexed (1=bottom) |
| Moves | Comma-separated coordinate list: `e8,d9,e9,f9,e10` |
| Players | `b` = first player (Blue), `w` = second player (White) |
| Pass | The string `pass` |

A move in Pentobi does **not** specify which piece or orientation — just the cells the piece occupies. The engine deduces the piece from the coordinates. This is elegant but means the translation layer needs to work in both directions differently.

### Translation Functions Needed

```
AlphaBlokus → Pentobi (for sending our moves):
  Action(piece_id=19, orientation=Identity, x=4, y=8)
  → compute occupied cells from piece shape + position
  → convert to Pentobi coordinates: (4,8)→e9, (3,9)→d10, etc.
  → join as comma-separated string: "e9,d10,e10,f10,e11"

Pentobi → AlphaBlokus (for receiving Pentobi's moves):
  "e9,d10,e10,f10,e11"
  → parse coordinates: [(4,8), (3,9), (4,9), (5,9), (4,10)]
  → identify piece by matching cell pattern against all 91 piece-orientations
  → determine anchor position (top-left of bounding box in board coords)
  → construct Action(piece_id, orientation, x, y)
```

**Coordinate conversion between the two systems:**

```
AlphaBlokus (x, y) → Pentobi:
  column_letter = chr(ord('a') + x)    # 0→'a', 13→'n'
  row_number    = y + 1                 # 0→1, 13→14
  result        = f"{column_letter}{row_number}"

Pentobi → AlphaBlokus (x, y):
  x = ord(column_letter) - ord('a')    # 'a'→0, 'n'→13
  y = row_number - 1                    # 1→0, 14→13
```

The coordinate systems are nearly identical (both bottom-left origin, same axis directions). The only differences are letters vs numbers for columns and 1-indexing vs 0-indexing for rows.

### Board Display (Shared by UI and Debugging)

A text renderer that converts the (14, 14) numpy board to a readable format:

```
    a b c d e f g h i j k l m n
14  . . . . . . . . . . . . . .
13  . . . . . . . . . . . . . .
12  . . . . . . . . . . . . . .
11  . . . . . W . . . . . . . .
10  . . . W W W . . . . . . . .
 9  . . . . W . . . . . . . . .
 8  . . . . . . . . . . . . . .
 ...
 2  . . . . . . . . . . . . . .
 1  . . . . . . . . . . . . . .
```

This should be a standalone utility function usable by the CLI, the Pentobi adapter (for debugging), and as a fallback renderer.

### Implementation Estimate

| Component | Effort | Notes |
|-----------|--------|-------|
| Coordinate converter (AlphaBlokus ↔ Pentobi) | ~1 hour | Straightforward mapping |
| Action → cell list (for outgoing moves) | ~2 hours | Apply piece shape at position, collect occupied cells |
| Cell list → Action (for incoming moves) | ~4 hours | Pattern matching against 91 orientations. Needs care |
| Action ↔ action index encoder/decoder | ~3 hours | Currently missing from codebase (noted in preflight fixes) |
| Board text renderer | ~1 hour | Numpy array → formatted string |
| **Total** | **~11 hours** | |

---

## 2. Pentobi Adapter

### How Pentobi Works

Pentobi is the strongest open-source Blokus AI. It is GPL-3.0 licensed, source at [github.com/enz/pentobi](https://github.com/enz/pentobi). Key facts:

- Uses MCTS with RAVE (Rapid Action Value Estimation) — no neural network
- Written in C++, built with CMake
- The GTP engine (`pentobi-gtp`) can be built without the GUI: `cmake -DPENTOBI_BUILD_GTP=ON -DPENTOBI_BUILD_GUI=OFF`
- 9 difficulty levels (1=weakest, 9=strongest). Level 9 requires ~4 GB RAM
- Communicates via the Go Text Protocol (GTP) over stdin/stdout

### GTP Protocol Overview

GTP is a simple text protocol. The controller (our code) sends commands via stdin, reads responses from stdout. Each response starts with `=` (success) or `?` (error), followed by the result, terminated by a double newline (`\n\n`).

```
Controller sends:  genmove w\n
Engine responds:   = e9,d10,e10,f10,e11\n\n
```

Key commands for our use case:

| Command | Example | Purpose |
|---------|---------|---------|
| `set_game` | `set_game Blokus Duo` | Set variant to Duo (14×14) |
| `clear_board` | `clear_board` | Reset for new game |
| `play` | `play b e8,d9,e9,f9,e10` | Submit a move for a player |
| `genmove` | `genmove w` | Ask Pentobi to generate and play a move |
| `reg_genmove` | `reg_genmove w` | Generate a move without playing it |
| `all_legal` | `all_legal b` | List all legal moves for a player |
| `final_score` | `final_score` | Get result (e.g., `B+12`) |
| `showboard` | `showboard` | Display board as text |
| `undo` | `undo` | Undo last move |
| `quit` | `quit` | Terminate engine |

### Adapter Architecture

```
PentobiAdapter
  ├── __init__(level: int, pentobi_path: str)
  │     Spawns pentobi-gtp as subprocess with --game duo --level N --quiet
  │     Sends set_game + clear_board
  │
  ├── send_command(cmd: str) → str
  │     Writes to stdin, reads until double newline from stdout
  │     Handles = / ? response parsing, raises on errors
  │
  ├── our_move(action: Action) → None
  │     Translates Action → Pentobi move string via translation layer
  │     Sends play command to inform Pentobi of our move
  │
  ├── their_move() → Action
  │     Sends genmove command
  │     Parses Pentobi's response (coordinate list or "pass")
  │     Translates to Action via translation layer
  │
  ├── play_game(our_agent, pentobi_color: str) → GameResult
  │     Full game loop alternating between our MCTS agent and Pentobi
  │     Returns winner, scores, move history, game length
  │
  ├── run_benchmark(levels: list[int], games_per_level: int) → BenchmarkResult
  │     Plays N games per level (half as each color)
  │     Computes all headline metrics (Pentobi Level, Score, Weighted Score)
  │
  ├── reset() → None
  │     Sends clear_board for a new game
  │
  └── close() → None
        Sends quit, terminates subprocess
```

### Key Implementation Details

**Subprocess management:**
```python
import subprocess

self.process = subprocess.Popen(
    [pentobi_path, '--game', 'duo', '--level', str(level), '--quiet'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=0  # unbuffered — critical for interactive I/O
)
```

**Critical gotchas:**

1. **I/O buffering.** The Pentobi docs explicitly warn about this. Child processes often use fully-buffered I/O (not line-buffered). Must flush stdin after every write. Must read stdout until the double-newline terminator — do not assume one line per response.

2. **stderr.** `pentobi-gtp` writes debug info to stderr. Using `--quiet` suppresses this, but if stderr isn't read it can fill the OS buffer and deadlock the process. Either use `--quiet` or drain stderr in a background thread.

3. **Pass handling.** In Blokus Duo, when a player has no legal moves, `genmove` returns `pass`. The game ends when both players pass consecutively.

4. **Thread safety.** For running many games in parallel, spawn multiple single-threaded `pentobi-gtp` instances rather than using one multi-threaded instance. Pentobi's docs note that >8 threads can degrade playing strength.

5. **Starting positions.** In Blokus Duo, each player's first piece must cover a specific starting point. The engine enforces this — no special handling needed on our side.

### Game Loop

```
For each benchmark game:
  1. Spawn (or reset) pentobi-gtp at target level
  2. Determine who plays which color
  3. Loop until game over:
     a. If it's our turn:
        - Get board state from our game object
        - Run MCTS to select action
        - Apply action to our game object
        - Translate action → Pentobi move string
        - Send "play <color> <move>" to Pentobi
     b. If it's Pentobi's turn:
        - Send "genmove <color>" to Pentobi
        - Receive move string (or "pass")
        - Translate Pentobi move → Action
        - Apply action to our game object
     c. Check for game over (both players passed)
  4. Get final score
  5. Record result
```

### Building Pentobi

Pentobi must be built from source (no pip install). On macOS/Linux:

```bash
git clone https://github.com/enz/pentobi.git
cd pentobi
mkdir build && cd build
cmake .. -DPENTOBI_BUILD_GTP=ON -DPENTOBI_BUILD_GUI=OFF
make -j$(nproc)
# Binary at: build/pentobi_gtp/pentobi-gtp
```

Dependencies: CMake 3.19+, C++20 compiler (GCC 12+ or Clang 15+), libatomic. No Qt needed if GUI is disabled.

### Implementation Estimate

| Component | Effort | Notes |
|-----------|--------|-------|
| Subprocess wrapper (spawn, send, receive, close) | ~3 hours | GTP protocol parsing, buffering |
| Game loop (single game, alternating turns) | ~3 hours | State sync between our game and Pentobi |
| Benchmark runner (multi-level, both colors) | ~3 hours | Parallel games, result aggregation |
| Metric computation (Pentobi Level, Score, Weighted) | ~2 hours | From evaluation plan |
| Heatmap / report generation | ~3 hours | Plotly heatmap from benchmark results |
| Testing against actual Pentobi | ~4 hours | Integration debugging, edge cases |
| **Total** | **~18 hours** | |

---

## 3. Human-Playable UI

### Approach Decision

Three viable approaches, in order of UX quality:

| Approach | UX | Effort | Best For |
|----------|-----|--------|----------|
| Web (React + FastAPI) | Excellent | ~40 hours | Demo, sharing, polished experience |
| Pygame | Good enough | ~20 hours | Development, testing, iteration |
| Terminal/CLI | Poor | ~5 hours | Debugging, scripted testing |

**Recommendation: start with Pygame, build web UI later if desired.**

Blokus is a visually complex game — pieces are polyominoes that need to be rotated, flipped, and placed precisely. A terminal UI would be painful. Pygame is the sweet spot: single Python process (no serialisation overhead, direct access to PyTorch), good enough visually, and fast to build. A web UI is a separate project that can reuse the same game logic and agent interface.

### Pygame UI Architecture

```
BlokusUI (Pygame)
  ├── Board renderer
  │     14×14 grid, colored cells for placed pieces
  │     Highlight legal placement positions
  │     Show piece preview at cursor position
  │
  ├── Piece palette
  │     Show remaining pieces for the human player
  │     Click to select, keyboard to rotate (R) / flip (F)
  │     Visual preview of current orientation
  │
  ├── Game controller
  │     Human turn: wait for valid piece placement click
  │     AI turn: run MCTS in background thread, show "thinking" indicator
  │     Manage turn alternation, pass detection, game over
  │
  └── Info panel
        Current player, scores, pieces remaining
        Move history
        AI thinking time
```

### Existing References

Several open-source Pygame Blokus implementations exist for reference:

- **[irori/blokus](https://github.com/irori/blokus)** — Browser-based Blokus Duo with excellent UX (drag-and-drop, scroll to rotate, double-click to flip). Best reference for the interaction model even though it's TypeScript/WASM.
- **[beatrice-erikson/blokus](https://github.com/beatrice-erikson/blokus)** — Pygame with clean MVC architecture (objects.py, views.py, controllers.py). Good structural reference.
- **[vineet131/Blokus](https://github.com/vineet131/Blokus)** — Pygame with pluggable AI agents via AIManager. Shows how to integrate AI into the game loop.

### Agent Interface

The UI needs a clean interface to the AI agent, independent of the UI framework:

```python
class BlokusAgent(Protocol):
    def select_move(self, game_state: BlokusGameState) -> Action:
        """Given the current state, return the chosen move."""
        ...
```

This same interface is used by:
- The Pygame UI (to get AI moves during human vs AI games)
- The Pentobi adapter (wrapping our MCTS agent)
- Self-play training (already exists in the codebase)

### Implementation Estimate

| Component | Effort | Notes |
|-----------|--------|-------|
| Board renderer (grid, pieces, highlighting) | ~4 hours | Pygame drawing primitives |
| Piece palette (selection, rotation, flip preview) | ~6 hours | Interaction design is the hard part |
| Legal move computation + highlighting | ~3 hours | Reuse existing game logic |
| Game loop (human + AI turns, threading) | ~4 hours | Background MCTS thread |
| Info panel (scores, move history) | ~2 hours | Text rendering |
| Polish (animations, sounds, start screen) | ~4 hours | Optional but nice |
| **Total** | **~23 hours** | |

---

## 4. Phasing

These three components have clear dependencies:

```
Phase A: Translation Layer (~11 hours)
  ├── Coordinate conversion
  ├── Action ↔ cell list conversion
  ├── Action ↔ action index encoding (preflight fix)
  └── Board text renderer

Phase B: Pentobi Adapter (~18 hours, depends on A)
  ├── Build pentobi-gtp from source
  ├── GTP subprocess wrapper
  ├── Game loop + benchmark runner
  └── Metric computation + heatmap

Phase C: Pygame UI (~23 hours, depends on A)
  ├── Board renderer
  ├── Piece interaction
  ├── Game controller with AI threading
  └── Info panel + polish
```

Phases B and C are independent of each other — both depend only on Phase A. The translation layer is the foundation and should be built first.

**Total effort: ~52 hours across all three phases.**

---

## 5. What We Don't Need

- **Downloading Pentobi's full source to read its internals.** The GTP protocol is a clean interface — we treat Pentobi as a black box. We only need to build the `pentobi-gtp` binary.
- **A web UI (for now).** Pygame is sufficient for development and testing. A React + FastAPI web UI could be a Phase 5 stretch goal.
- **Custom Pentobi modifications.** The GTP interface provides everything we need. No patching required.
- **Complex board state sync.** Both our game and Pentobi track their own board state. We just need to keep them in sync by sending every move to both sides. If they ever disagree, something is wrong — which is actually a useful validation check.
