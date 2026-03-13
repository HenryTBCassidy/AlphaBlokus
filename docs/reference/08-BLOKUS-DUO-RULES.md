# Blokus Duo — Game Rules Reference

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
