# Board Encoding Options for the Neural Network

**Status: Option 3 (44-channel per-piece planes) has been implemented.** This document is retained as historical design context. See `02-NEURAL-NETWORKS.md` for the current architecture.

## Context

The neural net needs two categories of information:

1. **Spatial state** — which squares are occupied and by whom (14×14 grid).
2. **Piece inventory** — which of the 21 pieces each player has already played
   (or equivalently, still has available).

Knowing only the board position is insufficient: two identical board layouts can
have very different strategic values depending on what pieces remain.

---

## How AlphaZero Encodes Chess and Go

Before evaluating our options, it's useful to understand how AlphaZero solved
similar encoding problems.

### AlphaGo Zero — 17 × 19 × 19

Go has only one piece type (stones), so the encoding is simple:

- **8 planes**: current player's stones at time steps t, t−1, … t−7 (binary)
- **8 planes**: opponent's stones, same 8 time steps
- **1 plane**: colour indicator — all 1s if Black to move, all 0s if White

History is needed in Go because of the ko rule (no repeating board positions).

### AlphaZero Chess — 119 × 8 × 8

Chess has 6 piece types per player plus special rules. The 119 planes:

**Per time step (14 planes × 8 time steps = 112 planes):**
- 6 planes for current player's pieces (one binary plane per piece type: pawn,
  knight, bishop, rook, queen, king)
- 6 planes for opponent's pieces (same)
- 2 planes for position repetition counter (threefold repetition draw rule)

**Constant-valued planes (7 planes):**
- 4 castling rights planes (one per option, all cells same value)
- 1 total move count plane
- 1 no-progress counter plane (50-move rule)
- 1 side-to-move plane (all 1s or all 0s)

### Key observation: sparsity is fine

Chess piece planes are very sparse. The "white knights" plane has at most 2
ones out of 64 cells (3.1% density). The "white king" plane has exactly 1 out
of 64 (1.6%). Most of the 119 planes are mostly zeros, and it works well.

Furthermore, a 2024 DeepMind paper ("Representation Matters for Mastering
Chess") found that going from 39 input planes to 52 planes — adding derived
features like material counts, piece masks, and checking piece indicators —
improved AlphaZero by ~97 Elo. Adding more planes with richer information
helped, even though those planes were also sparse.

### What Blokus doesn't need

Neither chess nor Go history planes are relevant for Blokus. In chess, history
encodes castling eligibility and en passant. In Go, it encodes ko. In Blokus,
there are no such path-dependent rules — the board position plus piece
inventory fully determines the game state. No history needed.

### What we can learn

- **One binary plane per piece type, per player** — the standard approach.
  Conv filters can specialise for each piece type.
- **Metadata as constant planes** — castling rights, side-to-move etc are
  broadcast as filled planes so every conv filter position can see them.
- **Sparsity doesn't hurt** — the net learns to handle mostly-zero planes.
- **Richer representations win** — more informative inputs beat sparser ones,
  even if many planes are mostly empty.

---

## Option 1: Column-Padded Single Tensor (previous approach, now replaced)

```
[Black pieces (14×2)] [Board (14×14)] [White pieces (14×2)]
```

- Shape: **1 × 14 × 18** (single channel).
- The 2-column regions on each side encode 28 values (= 2 × 14). Each player has
  21 pieces, so 21 of the 28 slots are meaningful; 7 are unused padding.
- Piece `i` played → slot `i−1` set to the player's sign (±1).
- Pros: single contiguous tensor, one Conv2d input channel, simple to construct.
- Cons:
  - Conv filters sweep across the boundary between board and piece columns,
    mixing spatial information with inventory data that has no spatial meaning.
  - 7 wasted slots per side.
  - Not symmetric: Black's inventory is on the left, White's on the right.
    A 180° board rotation (valid game symmetry) puts inventory on the wrong
    sides, making data augmentation via symmetries awkward.
  - Inventory info is only visible to filters near the board edges — filters
    in the centre of the board can't see it until many layers of convolutions
    have propagated that information inward.

---

## Option 2: Late Fusion — Conv Board + FC Inventory

Separate the two information types completely. The conv layers process only
spatial data; the inventory enters at the fully-connected stage.

```
                                    ┌─────────────────┐
  Board channels (2 × 14 × 14) ──→ │  Conv backbone   │──→ flatten ──┐
  Ch 0: Current player's stones     │  (ResNet blocks) │              │
  Ch 1: Opponent's stones            └─────────────────┘              ├─→ FC layers ──→ policy + value
                                                                      │
  Inventory vector (42 elements) ─────────────────────────────────────┘
  [21 current player flags, 21 opponent flags]
```

- Board input shape: **2 × 14 × 14** (two binary channels).
- Inventory: flat vector of length 42, concatenated onto the flattened conv
  output before the FC layers in the value and policy heads.
- Splitting the board into "my stones" and "their stones" channels (instead of
  one grid with +1/−1 values) means the net doesn't need to first learn which
  sign means which player — it's already separated.

**Pros:**
- Clean separation: conv filters only see spatial patterns, no inventory noise.
- Board is 14×14 — matches the physical game, symmetries are straightforward.
- No wasted slots: 21 pieces maps to exactly 21 values per player.
- Inventory reaches the policy/value heads directly — no need to propagate
  through conv layers from the edges.
- Simplest architectural change of all the alternatives.

**Cons:**
- Conv layers cannot jointly reason about spatial patterns and inventory.
  A filter looking at a gap on the board has no idea whether you still have a
  piece that fits. That combination only happens at the FC stage.

**Net changes required:**
1. `in_channels`: 1 → 2
2. `board_rows, board_cols`: 14, 18 → 14, 14
3. Forward pass: accept board tensor `(batch, 2, 14, 14)` and inventory tensor
   `(batch, 42)` separately. After the residual blocks, flatten the conv output
   and `torch.cat` the inventory before each head's FC layers.
4. `game.py`: `net_representation` returns the 2-channel board and the inventory
   vector separately (or as a tuple).

---

## Option 3: Per-Piece Spatial Planes (44-channel)

This is the AlphaZero-style approach applied directly to Blokus. Instead of
separating board and inventory, encode *everything* as spatial planes — one
plane per piece per player, showing exactly WHERE each piece sits on the board.

```
Channels 0–20:   Current player's 21 pieces (each 14×14 binary)
                 Piece plane has 1s on the squares that piece occupies, 0s elsewhere.
                 If the piece hasn't been played yet: all zeros.

Channels 21–41:  Opponent's 21 pieces (same)

Channels 42–43:  Aggregate board planes (optional, for convenience)
                 Ch 42: union of channels 0–20  (all current player's stones)
                 Ch 43: union of channels 21–41 (all opponent's stones)
```

- Shape: **44 × 14 × 14**.
- The aggregate planes (42–43) are technically redundant (the net could learn to
  combine channels 0–20), but providing them saves the net from having to learn
  a trivial operation and the first conv layer can immediately detect "my stone
  vs their stone" patterns without needing to aggregate 21 planes.

**What a conv filter sees at any position:**

At position (5, 8), a 3×3 filter simultaneously reads all 44 channels. It knows:
- Is there any piece here? (channels 42–43)
- If so, WHICH specific piece? (channels 0–41)
- What pieces are on neighbouring squares and which pieces they are?
- Which pieces haven't been played yet? (their planes are all-zero)

This is much richer than Option 2 where the conv only sees "occupied or not."

**Why this matters for Blokus specifically:**

Piece shape and identity is strategically important. The I-piece (5×1 line) and
the X-piece (plus shape) create very different corner and blocking opportunities.
A conv filter that can see "the opponent's L-piece is here, creating corners in
these directions" can learn spatial-strategic features that a binary board plane
cannot express.

**Sparsity analysis:**

Each piece occupies 1–5 squares on a 196-cell grid:
- Monomino (1 square): 0.5% density
- Pentomino (5 squares): 2.6% density
- Early game: ~40 of 44 planes are completely zero (only 2–4 pieces placed)
- Late game: up to ~40 planes have content, but each is still very sparse

Compare with chess: the "white knights" plane is 3.1% dense, "white king" is
1.6% dense. Our per-piece planes are comparable. Chess uses 119 planes and
trains successfully despite similar sparsity levels.

The DeepMind "Representation Matters" paper directly supports this: adding
more planes with richer information improved chess performance by ~97 Elo,
even though those extra planes were also sparse. The takeaway is that the
information content matters more than the density.

**What about all the empty planes early in the game?**

When only 2 pieces have been played (turn 1 for each player), 40 of 44 planes
are all-zero. This seems wasteful, but it's fine:
- Conv filters multiply by learned weights per channel. An all-zero channel
  contributes zero to the output regardless of the weight — no noise, just
  no signal. The net learns to rely on whichever channels have content.
- The batch norm layers after each conv handle the varying activation scales
  across different game stages automatically.
- This is exactly how chess works: the "white queen" plane has at most 1 piece
  on 64 squares, and history planes for positions that haven't occurred yet
  are all zeros.

**Pros:**
- Richest spatial information — conv filters know which specific piece is where.
- Follows proven AlphaZero conventions (one plane per piece type per player).
- Naturally encodes inventory: an all-zero plane means "not yet played."
- Conv filters can learn piece-shape-aware spatial features.
- Symmetries are clean (rotate all 44 planes together).
- Backed by research: richer representations outperform sparser ones.

**Cons:**
- 44 channels is significantly more than 1 or 2. More parameters in the first
  conv layer (44 × num_filters × 3 × 3 vs 2 × num_filters × 3 × 3), meaning
  more memory and slightly more compute per forward pass.
- Requires tracking which piece occupies which squares on the board (currently
  `insert_piece` records ±1, not piece identity). Moderate code change.
- Most of the information in the aggregate planes (42–43) could be learned
  from the per-piece planes, so there's some redundancy.

**Net changes required:** All done. `in_channels` is 44, board dimensions are 14x14, `BlokusDuoBoard.as_multi_channel()` builds the `(44, 14, 14)` tensor, and piece IDs are tracked per square via `_piece_placement_board`.

---

## Option 4: FiLM Conditioning (advanced)

Feature-wise Linear Modulation: let the inventory *modulate* the conv features
rather than entering at the FC stage.

```
  Board (2 × 14 × 14) ──→ Conv block ──→ feature maps (C × 14 × 14)
                                                  │
  Inventory (42) ──→ small MLP ──→ (γ, β)        │  γ * features + β
                                     │            │
                                     └────────────┘
                                                  │
                                            ──→ ResNet blocks ──→ heads
```

- The inventory goes through a small MLP that outputs per-channel scale (γ) and
  shift (β) parameters. These are applied to the conv feature maps:
  `output = γ * features + β`.
- This means every conv layer's spatial processing is *influenced by* the
  inventory — a filter might activate differently when you have lots of pieces
  remaining vs few. The inventory shapes how the net interprets spatial patterns.
- More powerful than late fusion (Option 2) but doesn't give the conv layers
  the per-piece spatial detail that Option 3 provides.
- More complex to implement and tune.
- Well-established technique in conditional image generation (StyleGAN etc),
  less commonly used in game-playing nets.

---

## Comparison

| Approach | Input shape | Conv sees inventory | Conv sees piece identity | Complexity | Precedent |
|----------|-------------|--------------------|----|----|----|
| 1. Column-padded (current) | 1 × 14 × 18 | At edges only | No | Low | Custom |
| 2. Late fusion | 2 × 14 × 14 + vec(42) | No | No | Low | Common in hybrid nets |
| 3. Per-piece planes (44-ch) | 44 × 14 × 14 | Yes (implicitly) | Yes | Low–Medium | AlphaZero (chess, Go) |
| 4. FiLM conditioning | 2 × 14 × 14 + vec(42) | Yes (via modulation) | No | Medium | StyleGAN, conditional nets |

---

## Recommendation

The choice is primarily between **Option 2** and **Option 3**.

**Option 2 (late fusion)** is the safest, simplest change. It fixes all the
problems with the current approach (mixed spatial/inventory data, broken
symmetries, edge-only visibility). The conv layers see a clean board and the
inventory enters at the FC stage. If this is "good enough" for the game's
complexity, it's the pragmatic choice.

**Option 3 (44-channel per-piece planes)** is the richer representation and
the closest to how AlphaZero actually works. It gives conv filters the ability
to reason about specific piece shapes and positions simultaneously — which is
arguably more important in Blokus than in chess, because Blokus strategy
revolves around piece shapes, corner access, and blocking with specific pieces.
The sparsity concern is addressed by the chess precedent (comparable sparsity,
119 planes, works well) and the "Representation Matters" paper (more planes
with richer info = better performance).

**If training time is the priority**, start with Option 2 — fewer parameters,
faster per-step, and we can always upgrade later. **If representation quality
is the priority**, Option 3 is the more principled choice.

Option 4 (FiLM) is an interesting middle ground but adds architectural
complexity without the per-piece spatial detail of Option 3. Worth revisiting
if neither Option 2 nor 3 performs well enough.

### References

- [AlphaZero — Chessprogramming wiki](https://www.chessprogramming.org/AlphaZero)
- [Representation Matters for Mastering Chess (2024)](https://arxiv.org/html/2304.14918v2)
- [Leela Chess Zero — Neural Network Topology](https://lczero.org/dev/backend/nn/)
- [ChessCoach — High Level Explanation](https://chrisbutner.github.io/ChessCoach/high-level-explanation.html)
