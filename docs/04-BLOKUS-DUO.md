# Blokus Duo — Rules & Piece Reference

## Overview

Blokus Duo (also sold as "Blokus Travel") is a two-player abstract strategy game
played on a **14×14** square grid. Each player has a set of **21 polyomino pieces**
and attempts to place as many of them on the board as possible.

## Pieces

Each player's set of 21 pieces covers every distinct free polyomino from size 1–5:

| Size | Count | Total squares | Names in codebase |
|------|-------|---------------|--------------------|
| 1 (monomino) | 1 | 1 | 1-Piece |
| 2 (domino) | 1 | 2 | 2-Piece |
| 3 (triomino) | 2 | 6 | 3-Piece, 3L-Piece |
| 4 (tetromino) | 5 | 20 | 4-Piece, 4L-Piece, 4Z-Piece, Square-Piece, 4T-Piece |
| 5 (pentomino) | 12 | 60 | F, I, L, N, P, T, U, V, W, X, Y, Z |
| **Total** | **21** | **89** | |

Each piece can be rotated and flipped, but many orientations are equivalent.
The distinct piece-orientation combinations (basis orientations) total **91**.

## Setup

- The board is a 14×14 grid.
- Two starting squares are marked near the centre of the board (not in the corners
  like standard 4-player Blokus).
- In our implementation: White starts at **(9, 9)** and Black starts at **(4, 4)**
  using (x, y) coordinates with origin at bottom-left.

## Placement Rules

1. **First move**: each player's first piece must cover their designated starting square.
2. **Corner rule**: every subsequent piece must touch at least one corner of the
   player's own previously placed pieces.
3. **No edge adjacency**: a piece may **not** share an edge with any of the same
   player's pieces (diagonal/corner contact only).
4. **Opponent pieces**: a piece may share edges with the opponent's pieces — the
   corner-only constraint applies only to your own colour.
5. **No overlap**: pieces cannot be placed on occupied squares.
6. **No moving**: once placed, pieces cannot be moved.

## Turn Structure

Players alternate turns. On each turn a player either:
- Places one piece (following the rules above), or
- Passes if no legal placement exists.

## Game End

The game ends when **both** players are unable to place any more pieces
(either all pieces placed or no legal moves remain).

## Scoring

| Condition | Points |
|-----------|--------|
| Each unplayed square remaining | −1 |
| All 21 pieces placed | +15 |
| All 21 pieces placed **and** the last piece was the monomino | +5 bonus |

The player with the higher score wins. Equal scores are a draw.

## Key Strategic Concepts

- **Corner expansion**: since new pieces must touch your own corners, building
  outward from corners is essential for maintaining placement options.
- **Blocking**: sharing edges with opponent pieces is legal and a key defensive
  tactic — placing pieces adjacent to your opponent limits their corner access.
- **Piece ordering**: larger pieces are harder to place as the board fills up,
  so playing them early is generally preferred. Saving the monomino for last
  earns the +5 bonus.
- **Territory**: controlling open space on the board matters — being cut off
  from open regions means unplayable pieces.

## Sources

- [Mattel Official Rules (PDF)](https://service.mattel.com/instruction_sheets/BJV44-Eng.pdf)
- [UltraBoardGames — Blokus Duo](https://www.ultraboardgames.com/blokus/blokus-duo.php)

---

## Piece Catalog

In each diagram below, `#` marks a filled cell and `.` marks an empty cell within the bounding box. Orientations are labelled with their transform name from the codebase.

---

### Piece 1: 1-Piece (Monomino) — 1 orientation

```
Identity:  #
```

---

### Piece 2: 2-Piece (Domino) — 2 orientations

```
Identity:  #     Rot90:  ##
           #
```

---

### Piece 3: 3L-Piece (Corner Tromino) — 4 orientations

```
Identity:  #.     Rot90:  .#     Rot180:  ##     Rot270:  ##
           ##             ##              .#              #.
```

---

### Piece 4: 3-Piece (Straight Tromino) — 2 orientations

```
Identity:  #     Rot90:  ###
           #
           #
```

---

### Piece 5: 4-Piece (Straight Tetromino) — 2 orientations

```
Identity:  #     Rot90:  ####
           #
           #
           #
```

---

### Piece 6: 4L-Piece (L-Tetromino) — 8 orientations

```
Identity:  #.     Rot90:   ..#     Rot180:  ##     Rot270:  ###
           #.              ###              .#              #..
           ##                               .#
```
```
Flip:      .#     Flip90:  ###     Flip180: ##     Flip270: #..
           .#              ..#              #.              ###
           ##                               #.
```

---

### Piece 7: 4Z-Piece (Skew Tetromino) — 4 orientations

```
Identity:  ##.     Rot90:  .#     Flip:    .##     Flip90:  #.
           .##             ##              ##.              ##
                           #.                               .#
```

---

### Piece 8: Square-Piece (Square Tetromino) — 1 orientation

```
Identity:  ##
           ##
```

---

### Piece 9: 4T-Piece (T-Tetromino) — 4 orientations

```
Identity:  .#.     Rot90:  .#     Rot180:  ###     Rot270:  #.
           ###             ##              .#.              ##
                           .#                               #.
```

---

### Piece 10: F-Piece (F-Pentomino) — 8 orientations

```
Identity:  .##     Rot90:   #..     Rot180:  .#.     Rot270:  ..#
           ##.              ###              .##              ###
           .#.              .#.              ##.              .#.
```
```
Flip:      ##.     Flip90:  .#.     Flip180: .#.     Flip270: .#.
           .##              ###              ##.              ###
           .#.              #..              .##              ..#
```

---

### Piece 11: I-Piece (I-Pentomino) — 2 orientations

```
Identity:  #     Rot90:  #####
           #
           #
           #
           #
```

---

### Piece 12: L-Piece (L-Pentomino) — 8 orientations

```
Identity:  #.     Rot90:   ...#     Rot180:  ##     Rot270:  ####
           #.              ####              .#              #...
           #.                                .#
           ##                                .#
```
```
Flip:      .#     Flip90:  ####     Flip180: ##     Flip270: #...
           .#              ...#              #.              ####
           .#                                #.
           ##                                #.
```

---

### Piece 13: N-Piece (N-Pentomino) — 8 orientations

```
Identity:  .#     Rot90:   ##..     Rot180:  .#     Rot270:  .###
           ##              .###              .#              ##..
           #.                                ##
           #.                                #.
```
```
Flip:      #.     Flip90:  .###     Flip180: #.     Flip270: ..##
           ##              ##..              #.              ###.
           .#                                ##
           .#                                .#
```

---

### Piece 14: P-Piece (P-Pentomino) — 8 orientations

```
Identity:  ##     Rot90:   ##.     Rot180:  .#     Rot270:  ###
           ##              ###              ##              .##
           #.                               ##
```
```
Flip:      ##     Flip90:  ###     Flip180: #.     Flip270: ##.
           ##              ##.              ##              ###
           .#                               ##
```

---

### Piece 15: T-Piece (T-Pentomino) — 4 orientations

```
Identity:  ###     Rot90:  #..     Rot180:  .#.     Rot270:  ..#
           .#.             ###              .#.              ###
           .#.             #..              ###              ..#
```

---

### Piece 16: U-Piece (U-Pentomino) — 4 orientations

```
Identity:  #.#     Rot90:  ##     Rot180:  ###     Rot270:  ##
           ###             .#              #.#              #.
                           ##                               ##
```

---

### Piece 17: V-Piece (V-Pentomino) — 4 orientations

```
Identity:  ..#     Rot90:  ###     Rot180:  ###     Rot270:  #..
           ..#             ..#              #..              #..
           ###             ..#              #..              ###
```

---

### Piece 18: W-Piece (W-Pentomino) — 4 orientations

```
Identity:  ..#     Rot90:  ##.     Rot180:  .##     Rot270:  ..#
           .##             .##              ##.              ##.
           ##.             ..#              #..              #..
```

---

### Piece 19: X-Piece (X-Pentomino) — 1 orientation

```
Identity:  .#.
           ###
           .#.
```

---

### Piece 20: Y-Piece (Y-Pentomino) — 8 orientations

```
Identity:  .#     Rot90:   ####     Rot180:  #.     Rot270:  ..#.
           ##              .#..              ##              ####
           .#                                .#
           .#                                .#
```
```
Flip:      #.     Flip90:  .#..     Flip180: .#     Flip270: ####
           ##              ####              ##              ..#.
           #.                                #.
           #.                                #.
```

---

### Piece 21: Z-Piece (Z-Pentomino) — 4 orientations

```
Identity:  ##.     Rot90:   ..#     Flip:    .##     Flip90:  #..
           .#.              ###              .#.              ###
           .##              #..              ##.              ..#
```

---

## Symmetry Reduction

Each piece can be physically rotated (0°, 90°, 180°, 270°) and flipped (mirror reflection), giving up to 8 possible orientations. If a piece has some symmetry, some of these 8 are duplicates:

```
Number of unique orientations = 8 ÷ |symmetry group|

No symmetry        → 8 ÷ 1 = 8 orientations
One reflection axis → 8 ÷ 2 = 4 orientations
180° rotation only  → 8 ÷ 2 = 4 orientations
D2 (bar symmetry)   → 8 ÷ 4 = 2 orientations
D4 (square symmetry)→ 8 ÷ 8 = 1 orientation
```

The 91 orientations are the **basis orientations** — the minimal set of distinct transforms needed to represent every possible way a piece can be placed on the board.

---

## Orientation ID Mapping

In the codebase, each of the 91 piece-orientations is assigned a contiguous integer ID for action encoding. The action space is:

```
action = (row × 14 × 91) + (col × 91) + piece_orientation_id

Where piece_orientation_id ∈ [0, 90] maps to a specific (piece, orientation) pair.
```

The mapping is built by `PieceManager.populate_lookup()` in `blokusduo/pieces.py`, iterating through pieces in ID order and orientations in definition order.

| Piece | ID | Orientations | Orientation IDs |
|-------|----|-------------|-----------------|
| 1-Piece | 1 | 1 | 0 |
| 2-Piece | 2 | 2 | 1–2 |
| 3L-Piece | 3 | 4 | 3–6 |
| 3-Piece | 4 | 2 | 7–8 |
| 4-Piece | 5 | 2 | 9–10 |
| 4L-Piece | 6 | 8 | 11–18 |
| 4Z-Piece | 7 | 4 | 19–22 |
| Square | 8 | 1 | 23 |
| 4T-Piece | 9 | 4 | 24–27 |
| F | 10 | 8 | 28–35 |
| I | 11 | 2 | 36–37 |
| L | 12 | 8 | 38–45 |
| N | 13 | 8 | 46–53 |
| P | 14 | 8 | 54–61 |
| T | 15 | 4 | 62–65 |
| U | 16 | 4 | 66–69 |
| V | 17 | 4 | 70–73 |
| W | 18 | 4 | 74–77 |
| X | 19 | 1 | 78 |
| Y | 20 | 8 | 79–86 |
| Z | 21 | 4 | 87–90 |

**Note:** The IDs above assume contiguous 0-indexed assignment. The current codebase uses 1-indexed IDs with a known off-by-one gap between pieces (see [Bug Fixes §4](plans/bug-fixes.md)). After the fix, the mapping will be exactly as shown above.

---

## Piece Size Distribution

| Size | Pieces | Count | Squares | Orientations |
|------|--------|-------|---------|-------------|
| 1 (monomino) | 1-Piece | 1 | 1 | 1 |
| 2 (domino) | 2-Piece | 1 | 2 | 2 |
| 3 (tromino) | 3L, 3-Piece | 2 | 6 | 6 |
| 4 (tetromino) | 4, 4L, 4Z, Square, 4T | 5 | 20 | 19 |
| 5 (pentomino) | F, I, L, N, P, T, U, V, W, X, Y, Z | 12 | 60 | 63 |
| **Total** | | **21** | **89** | **91** |

If a player places all 21 pieces, they cover **89 squares** of the 196-square board (45.4%).
