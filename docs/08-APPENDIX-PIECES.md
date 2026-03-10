# AlphaBlokus — Appendix: Blokus Duo Pieces & Orientations

All 21 Blokus Duo pieces and their 91 unique orientations after symmetry reduction. Each player gets one copy of every piece.

---

## Summary

| # | Piece | Squares | Symmetry | Orientations | Reason |
|---|-------|---------|----------|-------------|--------|
| 1 | 1-Piece | 1 | D4 (full square) | 1 | All rotations and reflections identical |
| 2 | 2-Piece | 2 | D2 | 2 | 180° rotation + both reflections are equivalent |
| 3 | 3L-Piece | 3 | Diagonal reflection | 4 | Flip = rotation, so only 4 rotations needed |
| 4 | 3-Piece | 3 | D2 | 2 | Same symmetry as the 2-Piece |
| 5 | 4-Piece | 4 | D2 | 2 | Straight bar, horizontal or vertical |
| 6 | 4L-Piece | 4 | None | 8 | No symmetry — all 8 transforms are distinct |
| 7 | 4Z-Piece | 4 | C2 (180° rotation) | 4 | 180° rotation maps to self, but reflection gives new shape |
| 8 | Square | 4 | D4 (full square) | 1 | 2×2 square, all orientations identical |
| 9 | 4T-Piece | 4 | Vertical reflection | 4 | Flip = identity, so only 4 rotations needed |
| 10 | F | 5 | None | 8 | No symmetry |
| 11 | I | 5 | D2 | 2 | Straight bar |
| 12 | L | 5 | None | 8 | No symmetry |
| 13 | N | 5 | None | 8 | No symmetry |
| 14 | P | 5 | None | 8 | No symmetry |
| 15 | T | 5 | Vertical reflection | 4 | Flip = identity |
| 16 | U | 5 | Vertical reflection | 4 | Flip = identity |
| 17 | V | 5 | Anti-diagonal reflection | 4 | Diagonal flip = identity |
| 18 | W | 5 | Anti-diagonal reflection | 4 | Diagonal flip = identity |
| 19 | X | 5 | D4 (full square) | 1 | Plus sign, all orientations identical |
| 20 | Y | 5 | None | 8 | No symmetry |
| 21 | Z | 5 | C2 (180° rotation) | 4 | 180° rotation maps to self |

**Totals:** 21 pieces, 89 squares, 91 unique orientations

---

## How Symmetry Reduction Works

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

**Note:** The IDs above assume contiguous 0-indexed assignment. The current codebase uses 1-indexed IDs with a known off-by-one gap between pieces (see [Architecture Review §7](01-ARCHITECTURE-REVIEW.md)). After the fix, the mapping will be exactly as shown above.

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
