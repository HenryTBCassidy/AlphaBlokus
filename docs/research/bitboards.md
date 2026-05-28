# Bitboards — the concept

A **bitboard** is a representation of a board-game state where each board cell maps to a single bit in an integer. Operations on the board state become bitwise operations on the integer, which CPUs execute in a single cycle. The technique is universal in strong chess / checkers / Othello engines. This doc explains what bitboards are and why they're fast, with worked examples.

**Important caveat for our project**: despite our initial assumption, **Pentobi (the strongest open-source Blokus AI) does *not* use bitboards.** It uses byte-per-cell arrays plus a precomputed move-list table keyed by `(piece, point, adj_status)`. The "bitboard" framing we started with for F2 turned out to be a poor mental model of how the leading Blokus engine actually works. See [`pentobi/move-generation.md`](pentobi/move-generation.md) for what they do instead. This doc is still useful as background for the *technique* — useful to know when reading chess / Othello sources, and to decide whether to adopt it for Blokus despite Pentobi not having done so.

---

## The core insight

A board state is fundamentally a **set of (cell, contents) pairs**. For each kind of contents (white piece, black piece, empty, etc.) you can represent "the set of cells with these contents" as a subset of all cells. A subset of N elements is equivalent to an N-bit number — bit `i` is 1 iff element `i` is in the subset.

So instead of storing a 2D array of cell states, you store **one integer per piece type**, where the integer's bits encode "which cells contain this piece type." Then:

| Set operation | Bitwise op |
|---|---|
| "Are sets A and B disjoint?" | `(A & B) == 0` |
| "Does A contain any element of B?" | `(A & B) != 0` |
| "Combine sets A and B" | `A | B` |
| "Remove B from A" | `A & ~B` |
| "How many elements in A?" | `popcount(A)` |
| "Find any element of A" | `find_first_set(A)` |

A CPU executes these in 1-4 cycles each. Compare to a Python `for cell in range(N): ...` loop that costs ~50-100 ns per iteration. **Bitboards turn ~hundreds of Python instructions into ~one machine instruction.**

---

## Worked example 1: TicTacToe in 9 bits

Let's work through a complete example slowly. We're going to take a real mid-game TicTacToe board and represent it using just two integers, then show why this lets us check "did white win?" in essentially one CPU instruction.

### Step 1: assign each cell a bit number

The board has 9 cells. We pick a numbering — any mapping from cell to a unique bit position 0-8 works, as long as we're consistent. Common choice: row-major, top-left = bit 0.

```
 Cell labels       Bit positions
 ┌───┬───┬───┐    ┌───┬───┬───┐
 │ A │ B │ C │    │ 0 │ 1 │ 2 │
 ├───┼───┼───┤    ├───┼───┼───┤
 │ D │ E │ F │ →  │ 3 │ 4 │ 5 │
 ├───┼───┼───┤    ├───┼───┼───┤
 │ G │ H │ I │    │ 6 │ 7 │ 8 │
 └───┴───┴───┘    └───┴───┴───┘
```

So cell A is bit 0, cell B is bit 1, ..., cell I is bit 8. That's the *mapping*. Lock it in.

### Step 2: encode a real board as two integers

Consider this position:

```
 ┌───┬───┬───┐
 │ X │   │ O │
 ├───┼───┼───┤
 │   │ X │   │
 ├───┼───┼───┤
 │   │   │ O │
 └───┴───┴───┘
```

White (X) has played on A and E. Black (O) has played on C and I.

We represent the board as **two separate integers**, one for each player:

- The `white` integer's bit `i` is 1 if white has played on cell `i`, else 0.
- The `black` integer's bit `i` is 1 if black has played on cell `i`, else 0.

Going cell by cell:

| Cell | Bit | White (X)? | Black (O)? |
|---|---|---|---|
| A | 0 | yes → 1 | no → 0 |
| B | 1 | no → 0 | no → 0 |
| C | 2 | no → 0 | yes → 1 |
| D | 3 | no → 0 | no → 0 |
| E | 4 | yes → 1 | no → 0 |
| F | 5 | no → 0 | no → 0 |
| G | 6 | no → 0 | no → 0 |
| H | 7 | no → 0 | no → 0 |
| I | 8 | no → 0 | yes → 1 |

Reading bits *bit 8 down to bit 0* (high-to-low, which is how we write binary numbers):

```python
white = 0b0_0001_0001       # bits 4 and 0 are set
black = 0b1_0000_0100       # bits 8 and 2 are set
```

In decimal those are 17 and 260, but **decimal is meaningless** — these integers are containers for 9 boolean flags, not numbers we ever do arithmetic on. We always think of them as bit patterns.

It can help to print them as a grid for sanity:

```python
def show(bb: int) -> str:
    return "\n".join(
        " ".join("X" if (bb >> (3*r + c)) & 1 else "." for c in range(3))
        for r in range(3)
    )

show(white)
# X . .
# . X .
# . . .

show(black)
# . . O
# . . .
# . . O
```

The two integers together carry exactly the same information as the original 3×3 grid.

### Step 3: detect a win — the key trick

Now the interesting part. **A "winning line" is a set of three cells in a row, column, or diagonal.** White wins iff white's pieces *include* one of those sets — that is, the winning set is a subset of white's set.

There are 8 winning lines. Let's pick one — the top row: cells A, B, C, i.e. bits 0, 1, 2. We encode that set as an integer the same way we encoded the board:

```python
TOP_ROW = 0b0_0000_0111       # bits 0, 1, 2 are set, all others 0
```

In grid form:

```
X X X
. . .
. . .
```

**The question "has white won via the top row?" is exactly the question "does white's set of cells contain {A, B, C}?".**

That set-containment question has a one-line answer using a bitwise AND:

```python
(white & TOP_ROW) == TOP_ROW
```

#### Why this works

`white & TOP_ROW` produces a new integer where bit `i` is set iff bit `i` is set **in both** `white` and `TOP_ROW`. In other words, it's the **intersection** of the two sets.

Then `== TOP_ROW` checks: is that intersection equal to the whole top-row set? That's exactly the definition of "top-row is a subset of white".

Worked through bit-by-bit on our board:

```
        bit:  8 7 6 5 4 3 2 1 0
       white: 0 0 0 0 1 0 0 0 1     ← cells A and E
     TOP_ROW: 0 0 0 0 0 0 1 1 1     ← cells A, B, C
     -------- ─ ─ ─ ─ ─ ─ ─ ─ ─
white & TOP:  0 0 0 0 0 0 0 0 1     ← only cell A is in both
```

Compare to `TOP_ROW` (which has bits 0,1,2 set): they're not equal. So white has *not* won via the top row. Correct — white only has 1 of the 3 needed cells.

If white had also played on B and C, the white bitboard would be `0b0_0001_0111`, and:

```
        bit:  8 7 6 5 4 3 2 1 0
       white: 0 0 0 0 1 0 1 1 1     ← cells A, B, C, E
     TOP_ROW: 0 0 0 0 0 0 1 1 1     ← cells A, B, C
     -------- ─ ─ ─ ─ ─ ─ ─ ─ ─
white & TOP:  0 0 0 0 0 0 1 1 1     ← all of A, B, C are in both
```

Now `white & TOP_ROW == TOP_ROW` is true. **White has won.**

This is the whole trick. Set-membership and set-containment questions about board cells become bitwise AND + integer compare. The CPU does both in single cycles.

#### Why this is fast

To check **all 8 winning lines**, you precompute the bitmask for each line once and store them:

```python
WINS = [
    0b0_0000_0111,  # top row    {A, B, C} = bits 0,1,2
    0b0_0011_1000,  # mid row    {D, E, F} = bits 3,4,5
    0b1_1100_0000,  # bot row    {G, H, I} = bits 6,7,8
    0b0_0100_1001,  # left col   {A, D, G} = bits 0,3,6
    0b0_1001_0010,  # mid col    {B, E, H} = bits 1,4,7
    0b1_0010_0100,  # right col  {C, F, I} = bits 2,5,8
    0b1_0001_0001,  # diag       {A, E, I} = bits 0,4,8
    0b0_0101_0100,  # anti-diag  {C, E, G} = bits 2,4,6
]

def has_won(player_bb: int) -> bool:
    return any((player_bb & w) == w for w in WINS)
```

Checking "did white win" is now **8 AND operations and 8 compares** — about 16 CPU cycles total.

Compare that to the alternative without bitboards: store the board as a 3×3 array, loop through each of the 8 winning lines, for each line loop through its 3 cells, check that each cell is white. That's 8 × 3 = 24 array accesses (each a pointer dereference + bounds check + comparison + branch), tens of memory reads, lots of branch-prediction misses. Easily 200+ CPU cycles in tight code, and 20× that if it's Python.

For TicTacToe the absolute savings are trivial. The point is the **pattern**: any "is this set of cells in this configuration?" check becomes one AND + one compare.

### Why this generalises

The TicTacToe example uses three set-theory operations that all map to bitwise ops:

| Set question | Bitwise expression | Why |
|---|---|---|
| Is set B contained in set A? | `(A & B) == B` | Intersection of A and B equals B iff every element of B is in A |
| Do sets A and B overlap? | `(A & B) != 0` | Intersection is nonempty iff at least one element is shared |
| Are A and B disjoint? | `(A & B) == 0` | Intersection is empty |
| Union of A and B | `A | B` | Element in either |
| Set difference A \ B | `A & ~B` | Element in A but not in B |

For Blokus, the questions we'll be asking are:

- "Is the footprint of piece P at position X empty?" — disjoint check → `(footprint & occupied) == 0`
- "Does piece P at position X touch a same-coloured edge?" — overlap check → `(edge_mask & own_pieces) != 0`
- "Does piece P at position X touch at least one same-coloured corner?" — overlap check → `(corner_mask & own_pieces) != 0`

Same pattern, different sets.

### Listing legal moves

To find empty cells (where you can play in TicTacToe), you compute the complement of (white ∪ black):

```python
def empty_cells(white: int, black: int) -> int:
    occupied = white | black
    return ~occupied & 0b1_1111_1111   # mask off bits beyond cell 8
```

The mask `0b1_1111_1111` (all 9 cells) limits the result to the actual board — without it, `~occupied` would have bits set above bit 8 (because Python ints are arbitrary precision), which would corrupt later operations.

To iterate the empty cells:

```python
m = empty_cells(white, black)
while m:
    cell = (m & -m).bit_length() - 1   # find the lowest set bit's position
    m &= m - 1                          # clear that bit, continue
    yield cell
```

`m & -m` isolates the lowest set bit (a classic trick — `-m` is two's complement, and the bit pattern works out that `m & -m` keeps only the rightmost 1). `m & (m-1)` clears that lowest set bit. Both are 2-cycle CPU operations. **Iterating all set bits of an integer is O(popcount), not O(width).**

This `m & -m` / `m & m-1` pair is *the* idiom for "iterate set bits". You'll see it everywhere in chess engines.

---

## Worked example 2: Chess move generation (the canonical case)

Chess fits beautifully because a chess board is 8×8 = 64 cells, exactly one machine word. The standard representation is twelve bitboards: `{white,black} × {pawns, knights, bishops, rooks, queens, kings}`.

### Why "where can my knight go from c3?" is one operation

A knight at c3 can reach a fixed set of 8 squares (or fewer, near the edge). That set is a precomputed bitmask `KNIGHT_ATTACKS[c3]`. So:

```python
def knight_moves_from(square: int, own_pieces: int) -> int:
    return KNIGHT_ATTACKS[square] & ~own_pieces
```

The lookup table has 64 entries (one per starting square), each a 64-bit integer. The whole table fits in 512 bytes — comfortably in L1 cache. **Generating all knight moves from one square = two memory loads and one AND.**

For comparison, the naive "loop through 8 candidate offsets, check each is on the board, check it's not occupied by own piece" approach is ~30-50 Python operations.

### Sliding pieces — the magic-bitboard trick

Bishops, rooks, and queens "slide" along rays until they hit something. The set of squares they can reach depends on **where the blockers are**. So a single precomputed table doesn't work — you'd need an entry for every (start_square, blocker_configuration) combination.

**Magic bitboards** solve this with a perfect hash. For each square, you find a "magic number" `M[sq]` such that:

```python
def rook_attacks(square: int, occupied: int) -> int:
    relevant_blockers = occupied & ROOK_MASK[square]   # cells on this rook's rays
    index = (relevant_blockers * M[square]) >> SHIFT[square]
    return ROOK_TABLE[square][index]
```

The multiplication-and-shift is a perfect hash from "the configuration of blockers on this rook's rays" to "the attack-set for that configuration". The tables are precomputed once; lookup is O(1).

This is *the* technique that makes chess engines fast. Finding good magic numbers is a one-time offline brute-force search.

(For Blokus we don't have sliding pieces, so we don't need magic bitboards. But the pattern of "precompute a giant table indexed by board configuration" recurs.)

---

## Worked example 3: 14×14 Blokus Duo — the size problem

Blokus Duo is 14×14 = **196 cells**. That doesn't fit in a 64-bit machine word. Three options:

### Option A: arbitrary-precision Python int (196+ bits)

```python
EMPTY_BB = 0
ALL_CELLS = (1 << 196) - 1  # bits 0..195 set, the "full board" mask

def is_cell_set(bb: int, row: int, col: int) -> bool:
    return (bb >> (row * 14 + col)) & 1 == 1
```

Python's `int` is arbitrary-precision under the hood (it's a `PyLong_Object` with a digit array). Bit operations on 196-bit ints internally loop over 3-4 64-bit "digits". You pay ~4-8× per-op overhead vs a native 64-bit int but it's still way faster than a Python `for cell in range(196)` loop, and the code reads identically to the chess case.

### Option B: array of `uint64` words

```python
import numpy as np
EMPTY_BB = np.zeros(4, dtype=np.uint64)  # 4 × 64 = 256 bits, 60 unused
```

Faster per-op (NumPy uses native uint64 ops via SIMD where possible), but every operation now requires loop-unrolled handling of the 3-4 words. Boundary cells that straddle word boundaries need careful masking. Code is more complex.

### Option C: `gmpy2.mpz`

`gmpy2` is a Python binding for the GMP arbitrary-precision integer library. It's much faster than Python's built-in int for large operations because GMP is hand-tuned assembly. Adds a dependency.

For Blokus Duo at 196 bits, the per-op overhead is small enough that **Option A (Python int) is the right starting point** — fastest to write, easy to debug, no dependencies. Optimise later if profiling shows bit-ops are the bottleneck (they probably won't be — Blokus's bottleneck is mask-table lookups, not raw bitops).

---

## The padding-bits trick

A persistent bug source in bitboards is **boundary cells**. When a knight moves from h1, the candidate squares "h0" and "h-1" don't exist on the board. Two ways to handle it:

1. Mask off the off-board bits after every operation. Cheap per op but adds a guarantee-the-caller-remembered-to-mask invariant. Brittle.
2. Embed the board inside a larger "padded" bitboard with extra rows/columns of permanently-zero cells around it. Then a knight move from h1 that "lands on h0" lands in the off-board padding, where the friendly-pieces bitboard is permanently 0 — and it just naturally drops out of the legal-moves AND. No mask needed.

This is the **padding-bits trick** (also called "sentinel bits" or "guard squares"). Standard chess engines use it. The cost is a few unused bits per word — for a 14×14 Blokus board, you'd pad to (say) 16×16 = 256 bits, of which 60 are padding.

The trick handles:
- Off-board moves silently turning into no-ops
- Edge-of-board legality checks ("can I place this piece on row 13?") becoming unconditional ANDs
- Mirror/rotate transformations preserving correctness without special cases at edges

Pentobi achieves the same robustness *without* bitboards by a different mechanism: every move is pre-numbered at startup, and off-board placements are simply not enumerated. There's a single sentinel value `Point::null()` (index equal to "one past the last cell") and unused cells in piece-cell arrays are padded with it. See [`pentobi/move-generation.md`](pentobi/move-generation.md) §2 for the details — it's an interesting alternative design choice.

---

## Mirroring and rotating bitboards

Blokus has an 8-fold symmetry group (4 rotations × 2 mirrors). To exploit it (training data augmentation, transposition tables), we need to transform bitboards under each symmetry.

For an 8×8 chess board the standard tricks are:

**Mirror left-right** (swap each byte's bits):
```python
def mirror_horizontal(bb: int) -> int:
    bb = ((bb & 0x55555555_55555555) << 1) | ((bb >> 1) & 0x55555555_55555555)
    bb = ((bb & 0x33333333_33333333) << 2) | ((bb >> 2) & 0x33333333_33333333)
    bb = ((bb & 0x0f0f0f0f_0f0f0f0f) << 4) | ((bb >> 4) & 0x0f0f0f0f_0f0f0f0f)
    return bb
```

**Mirror top-bottom** (reverse byte order — one assembly instruction on x86):
```python
def mirror_vertical(bb: int) -> int:
    return int.from_bytes(bb.to_bytes(8, 'little'), 'big')
```

**Rotate 90°** is a sequence of shifts + masks — about 6 lines, no branches.

For a 14×14 Blokus board these get more involved because cells don't align with byte boundaries. The trick most engines use is to **precompute a permutation table** mapping `bit_index → bit_index_after_transform`, then apply it by iterating the bits of the input. For a board this small, that's still fast.

---

## Move generation patterns

For our purposes the relevant pattern is:

### Pattern A: precomputed per-cell masks

For Blokus, every (piece-orientation, anchor-position) pair has a *footprint* (which cells the piece occupies), a *contact mask* (cells touching the piece's edges in same-colour-forbidden directions), and a *corner mask* (cells touching the piece's corners in same-colour-required directions).

Precompute these masks once at startup, store them in a big table. Then "is this placement legal?" is:

```python
def is_legal(footprint, contact_mask, corner_mask, own_pieces, all_pieces):
    return (
        (footprint & all_pieces) == 0          # cells are empty
        and (contact_mask & own_pieces) == 0   # no own-colour edge contact
        and (corner_mask & own_pieces) != 0    # at least one own-colour corner
    )
```

**Three integer compares** vs the current "loop over piece cells, check each" pattern.

The catch: the mask table is big. Blokus Duo has 21 pieces × ~91 orientations × 196 anchor positions × 3 masks per placement × 196 bits each = ~280 MB of masks if stored as plain bitboards.

In practice you don't enumerate by (piece, orientation, anchor) — you enumerate by **anchor point** (corners the current player can build from), so the indexing is `anchor_point → list of (piece, orientation) options at that anchor`. Much smaller.

### Pattern B: anchor-driven enumeration

Blokus has a hard constraint: every placement must touch a same-colour corner (except the first move, which goes on a fixed start square). So **the only positions worth trying** are corners adjacent to existing same-colour pieces. As the game progresses each player accumulates a list of "anchor cells" — usually 10-30 of them. You enumerate moves by iterating anchors, not by iterating every empty cell.

Pentobi uses exactly this pattern, but the anchor list is stored as a flat byte array (`Grid<bool>` plus an `ArrayList<Point>`) rather than as a bitboard. Adding a piece updates the anchor list incrementally inside `place()`: removing cells the piece occupies, removing cells now adjacent to the piece's edges, adding cells now adjacent to the piece's corners. The operations work fine with byte storage; Pentobi never needed to make the anchor set bitwise.

---

## Useful primitive operations

| Op | What it does | Native on x86 | Python equivalent |
|---|---|---|---|
| Popcount | Count set bits | `POPCNT` (1 cycle) | `int.bit_count()` (Python 3.10+) |
| Bit-scan-forward | Find lowest set bit | `BSF` / `TZCNT` (1-3 cycles) | `(x & -x).bit_length() - 1` |
| Bit-scan-reverse | Find highest set bit | `BSR` / `LZCNT` (1-3 cycles) | `x.bit_length() - 1` |
| Parallel deposit | Scatter bits | `PDEP` (BMI2, 3 cycles) | No clean Python equivalent |
| Parallel extract | Gather bits | `PEXT` (BMI2, 3 cycles) | No clean Python equivalent |

The first three are everywhere in Python via stdlib. The last two are common in cutting-edge chess engines (Leela Chess Zero, Stockfish 16+) for very fast attack generation — but there's **no good Python equivalent**, even via NumPy. For our scale, we won't need PDEP/PEXT. If F2 turns out to be inadequate we'd switch to Cython before chasing PDEP-tier optimisations.

---

## Why bitboards are bug-prone

1. **Bit ordering is a convention you have to commit to and document.** Is bit 0 the top-left corner (row-major) or the bottom-left (file-rank)? Different engines pick differently. Code that assumes one ordering and reads code that assumes another silently corrupts.
2. **Off-board bits propagate.** If `ALL_CELLS = (1 << 196) - 1` is your "valid bits" mask, every operation that might create bits above bit 195 needs to `& ALL_CELLS` to clean up. Forgetting one place causes hard-to-find drift bugs.
3. **Symmetry transforms are hand-rolled bit shuffles.** A single off-by-one in the rotate-90° function corrupts every transformed position thereafter. Tests of the form "transform-then-untransform should be identity" are essential.
4. **Magic numbers and tables are opaque.** The mask tables only make sense relative to the bit ordering you committed to. Print them as 2D grids when debugging, not as hex.
5. **Python's arbitrary-precision int hides errors that would crash in C.** Setting bit 1000 in C on a `uint64_t` is undefined behaviour; in Python it silently grows the int to 1001 bits. Bugs that would be loud crashes are silent corruption in Python.

**Mitigations:**

- Write a strict-equivalence test suite **before** the optimisation. Comparing bitboard output to current-implementation output on thousands of random positions catches almost every bug fast.
- Print bitboards as 2D grids in debug output, never as hex.
- Make bit ordering a documented constant in one place, referenced by every transform.

---

## How Pentobi handles all this — and why F2 may not be a "bitboard" plan

Pentobi specifically chose **not** to use bitboards, despite the obvious applicability. They use byte-per-cell `Grid<bool>` arrays plus a giant precomputed move-list table keyed by `(piece, point, 6-bit adj_status)` that returns ready-made candidate move lists at run time. The per-move legality check is then a 5-iteration unrolled `|=` over byte loads — not a bitwise AND of a packed mask.

See [`pentobi/move-generation.md`](pentobi/move-generation.md) for the full breakdown of what they did. The implication is that F2 of our master plan has two genuine design choices, not one:

1. **Port Pentobi's design.** Byte-per-cell state + precomputed `(piece, point, adj_status)` move-list table. Avoids bit-twiddling almost entirely. Matches the strongest open-source Blokus engine. The table is a few MB and amortises construction across a worker pool.

2. **Implement a true bitboard for Blokus.** Pack the 14×14 board into ~196 bits, do the legality check as bitwise ANDs against precomputed footprint/contact/corner masks. More ambitious; needs careful test infrastructure; potentially faster than Pentobi's design but with no precedent for Blokus. Pentobi presumably evaluated this and decided the table-lookup approach was better — worth understanding *why* before assuming bitboards are the win.

The F2 sub-plan will pick one of these (or a hybrid) based on a brief prototype comparison. For now, this concept doc and the Pentobi research notes form the input to that decision.

---

## Further reading

- [Chess Programming Wiki — Bitboards](https://www.chessprogramming.org/Bitboards) — the canonical reference. Most bitboard techniques are documented there with C examples.
- [Magic Bitboards (CPW)](https://www.chessprogramming.org/Magic_Bitboards) — the perfect-hash trick for sliding pieces.
- Stockfish source — `src/bitboard.h` and `src/bitboard.cpp`. The reference implementation for high-performance bitboard chess.
- For Blokus specifically: there's *no* published bitboard implementation we found. Pentobi's design (see [`pentobi/move-generation.md`](pentobi/move-generation.md)) deliberately uses byte arrays instead — and is the strongest open-source Blokus engine. If you go the bitboard route for Blokus, you're inventing it.
