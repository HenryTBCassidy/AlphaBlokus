# Pentobi's "bitboard" — what's actually there

The headline finding for F2: **Pentobi does not use a bitboard.** It uses a
byte-per-cell representation plus heavy precomputation. The only place a
`uintN_t` is treated as a packed feature container is `PlayoutFeatures`, and
even there the bits encode additive counters, not cell positions.

This doc documents what Pentobi actually does in each of the categories the
F2 plan asked about, so we can decide whether to copy the design, redesign,
or hybridise.

## 1. Bit layout of the board

There isn't one. The on-board cell state is stored as flat byte arrays
indexed by a small integer "point id":

- `libboardgame_base::Point<M, W, H, I>` is a wrapper over an unsigned
  integer `m_i` (`Point.h:32-81`). For Duo the type is `uint_fast16_t` (set
  by the Pentobi-side Point alias) and the range is the on-board cell count
  plus one for `Point::null()` (`Point.h:48`). It is **not** a packed
  `(x << bits) | y` coordinate; it is a sequence number assigned by
  `Geometry::init` (`Geometry.h:269-308`) as it scans the rectangle. Each
  point caches `m_x`, `m_y`, neighbours, etc. in `Geometry` (`Geometry.h:166-184`).
- The actual board state is `Board::StateBase::point_state`
  (`Board.h:335`), a `Grid<PointState>` where `PointState` is a byte-sized
  enum (1 byte per cell).
- The per-colour cell masks that the hot loops consult are
  `StateColor::forbidden` (`GridExt<bool>`, `Board.h:341`) and
  `StateColor::is_attach_point` (`Grid<bool>`, `Board.h:343`). `Grid` is a
  `T m_a[Point::range_onboard]` array (`Grid.h:96`). With `T = bool`, this
  is one byte per cell.

For Duo (14x14 = 196 cells) the `forbidden` array is therefore 197 bytes
per colour (197 because `GridExt` includes the null point,
`Grid.h:147-175`), not the 196-bit packed bitboard one might expect.

The Pentobi-side `Point` alias and per-game-size `Geometry::Iterator`
(`Geometry.h:54-69`) scan only on-board indices, so there is no padding
between rows the way a chess bitboard pads with sentinel files.

## 2. Padding bits and sentinels

There is one sentinel: `Point::null()` (`Point.h:51, 72`) is a reserved
integer value equal to `range_onboard`. The `GridExt<T>` variant carries
one extra slot for it (`Grid.h:174`). The hot loops exploit this:
`MoveInfo<MAX_SIZE>` stores its piece's cells in a fixed-size array with
unused slots set to `Point::null()` (`MoveInfo.h:38-40` and the
construction in `MoveInfo.h:34-41`), and `forbidden[Point::null()]` is
always `false`, so the inner check loop in
`State::check_forbidden<MAX_SIZE>` (`State.cpp:131-150`) iterates exactly
`MAX_SIZE` times without bounds checks regardless of the actual piece size.

There is **no per-row padding for branch-free edge checks**. The boundary
behaviour is handled at construction time: `BoardConst::create_moves`
generates a separate `MoveInfo` for every legal `(piece, orientation,
position)` triple, dropping any that go off-board
(`BoardConst.cpp:1019-1033`). So at run time the move id space already
excludes off-board placements.

## 3. Piece-with-orientation storage

Pentobi does **not** store one mask per `(piece_orientation, anchor)`
pair. It stores one `MoveInfo` per **legal move** (every legal
piece-orientation-position triple), and the move id is the index.

- A `MoveInfo<MAX_SIZE>` (`MoveInfo.h:28-54`) contains a 1-byte piece id, a
  1-byte cell count, and a fixed-size `Point m_points[MAX_SIZE]` array
  padded with `Point::null()`. `MAX_SIZE` is 5 for Classic/Duo/Junior, 6
  for Trigon, 7 for Nexos, 22 for GembloQ (`BoardConst.cpp:951-958`).
- `MoveInfoExt<MAX_ADJ_ATTACH>` (`MoveInfo.h:66-86`) follows: a
  fixed-size `Point points[MAX_ADJ_ATTACH]` plus 1-byte counts for the
  adjacent-cell prefix and the attach-cell suffix. `MAX_ADJ_ATTACH` is 16
  for Classic/Duo, 22 for Trigon, 12 for Nexos, 44 for GembloQ
  (`BoardConst.cpp:951-958`).
- `MoveInfoExt2` (`MoveInfo.h:93-123`) holds rarely-touched per-move data:
  `breaks_symmetry`, the symmetric counterpart move, the label position,
  and the scored-points array. Separated from the hot data "to improve CPU
  cache performance" (comment line 90-92).
- The three arrays are allocated as raw memory and constructed in-place
  (`BoardConst::create_move`, `BoardConst.cpp:863-878` with `new(place)`).
- Total count of distinct moves per variant is hard-coded in `Move.h:26-48`.
  For Blokus Duo: **13,729 on-board moves**. For Classic: 30,433. For
  GembloQ: 31,254. The piece-orientation count per piece (1..8 for
  Classic) is stored implicitly by emitting transforms in
  `PieceTransformsClassic.cpp:14-25` (identity, rot90, rot180, rot270,
  rot90refl, rot180refl, rot270refl, refl — 8 total).

Memory ballpark for Duo:
- `MoveInfo<5>` is 1 + 1 + 5 * sizeof(Point); with `Point` likely 2 bytes
  (`uint_fast16_t`) on common targets that's 12 bytes -> roughly 165 KB
  for 13,729 moves.
- `MoveInfoExt<16>` is 16 * 2 + 2 = 34 bytes -> ~467 KB.
- `MoveInfoExt2` is larger (`PieceInfo::max_scored_size = 22`,
  `PieceInfo.h:42-43`).

So per-variant static data is a few MB total, not the "every
(piece,position) pair" table the F2 plan considered.

## 4. The legality check, walked through

Move legality in Pentobi is **not** a single AND of two masks. The check
is split between (a) move-list pre-filtering by an `adj_status` bitmask
and (b) a small unrolled scan over the move's cells. The same-colour
edge-adjacency rule is enforced by `place()` setting `forbidden[]` on
every adjacent cell when a piece is played, so move generation only has
to look at `forbidden[]`. Walk-through:

### 4a. Compute `adj_status` for the anchor cell (Board.h:479-487)

```cpp
inline unsigned Board::get_adj_status(Point p, Color c) const {
    auto i = m_bc->get_adj_status_points(p).begin();
    auto result = static_cast<unsigned>(is_forbidden(*i, c));
    for (unsigned j = 1; j < PrecompMoves::adj_status_nu_adj; ++j)
        result |= (static_cast<unsigned>(is_forbidden(*(++i), c)) << j);
    return result;
}
```

`PrecompMoves::adj_status_nu_adj = 6` (`PrecompMoves.h:43`), so this packs
6 forbidden bits into a small integer in `[0, 63]`. The 6 cells chosen are
a near-locality of `p` (its 4 edge-neighbours then diagonal neighbours,
selected by `init_adj_status_points`, `BoardConst.cpp:1147-1199`).

### 4b. Fetch the precomputed list (`PrecompMoves::get_moves`)

```cpp
Range get_moves(Piece piece, Point p, unsigned adj_status = 0) const {
    auto& range = m_moves_range[p][adj_status][piece];
    auto begin = move_lists_begin() + range.begin();
    return {begin, begin + range.size()};
}
```

(`PrecompMoves.h:75-80`.) The list contains moves that **(i)** include
`p` as a cell, **(ii)** use a piece equal to `piece`, and **(iii)** the
construction at `BoardConst.cpp:899-902` guarantees `(adj_status & mask) ==
0`, where `mask` is the subset of those 6 cells that this particular move
also occupies. In effect, the list is pre-filtered so that none of the 6
neighbour cells coincide with another cell of the move when the relevant
`adj_status` bit is set.

This is the single most important data structure for Pentobi's speed:
a (point, adj_status, piece) tuple yields a ready-made candidate list
without any per-move bitwise check.

### 4c. The per-move loop (`State::check_forbidden`, `State.cpp:131-150`)

```cpp
auto p = get_move_info<MAX_SIZE>(mv).begin();
unsigned forbidden = is_forbidden[*p];
for (unsigned i = 1; i < MAX_SIZE; ++i)
    forbidden |= static_cast<unsigned>(is_forbidden[*(++p)]);
if (forbidden != 0) return false;
```

For each candidate move, the loop reads the up-to-`MAX_SIZE` cells of the
move and ORs the forbidden bits. The comment at `State.cpp:138-143`
explains the bitwise-OR trick: a boolean `||` would generate branches the
compiler cannot vectorise out of a 5-iteration loop; `|` does not. The
fixed `MAX_SIZE` (5 for Duo) lets the compiler unroll completely. Note:
`is_forbidden` is the per-colour `GridExt<bool>`, indexed by the small
point id, so this is one byte load per cell, not a bit test.

### 4d. The "at least one diagonal of same colour" requirement

This is **not** checked at the per-move level. Instead it is enforced by
construction:
- The first piece for a colour is anchored to the colour's starting point
  (`Board.cpp:114-122`); the gen-moves loop iterates
  `get_starting_points(c)`.
- After the first piece, the gen-moves loop iterates
  `get_attach_points(c)` (`Board.cpp:130-137`). Every cell in that list
  was added when an earlier same-colour piece was played because it was a
  diagonal of that piece (`Board.h:797-807` inside `place`). So any move
  anchored on a cell in `attach_points` is guaranteed to be diagonally
  adjacent to a same-colour piece.

### 4e. The "no edge adjacency to same colour" rule

Enforced by `place` setting `forbidden[adj_cell] = true` for the played
colour when a piece is placed (`Board.h:790-793`). At gen time it is
captured by the `is_forbidden` byte read in 4c.

So the run-time legality "AND" is implicit: choose the candidate list
keyed by `(piece, point, adj_status)`, then verify the byte mask.

## 4f. What is an "anchor"? Why those 6 specific cells?

Three terms get used interchangeably in this doc and they shouldn't be. Defining them explicitly:

- **Cell**: a single square on the 14×14 board. There are 196 of them.
- **Attach point** (a.k.a. **anchor**): a cell where the current player *could* legally start a new piece. Blokus's diagonal-corner rule means every same-colour piece (after the first) must touch an existing same-colour piece corner-to-corner. So the set of cells where a colour can start a piece is exactly the set of cells diagonally adjacent to that colour's existing pieces. Pentobi tracks this set incrementally (`StateColor::is_attach_point`, `Board.h:343`). In the mid-game each player has typically 10-30 attach points.
- **Placement** or **move**: a specific choice of `(piece, orientation, anchor)` — i.e. "put piece P, rotated to orientation O, such that one of its cells lies on anchor A". Pentobi numbers each legal placement with a unique move ID 0..13,728 for Duo.

"Enumerate legal moves" means: for each attach point, for each piece still in inventory, find the orientations that fit there given the current board state.

### Picking the 6 adj_status cells — concretely

For each anchor `P`, Pentobi precomputes a fixed list of **6 neighbour cells** whose `forbidden` status gets packed into the 6-bit `adj_status` value. The selection algorithm in `BoardConst::init_adj_status_points` (`BoardConst.cpp:1147-1199`) collects, in order:

1. The 4 edge-neighbours of `P` (cells touching `P` by an edge), in order **N, W, E, S** (from `RectGeometry::get_adj_coord`, `RectGeometry.h:75-83`).
2. Then the first 2 of `P`'s 4 diagonal-neighbours, in order **NW, SE** (from `RectGeometry::get_diag_coord`, `RectGeometry.h:86-95`).

The algorithm halts at 6 because `PrecompMoves::adj_status_nu_adj = 6`. So **NE and SW are deliberately not in `adj_status`.**

Visual (anchor `P` at centre; Pentobi uses `y` increasing downward, so "N" means `y-1`):

```
        col x-1          col x          col x+1
        ┌────────────┬────────────┬────────────┐
y-1     │    NW      │     N      │   NE       │
        │   bit 4    │   bit 0    │ (not in    │
        │            │            │ adj_status)│
        ├────────────┼────────────┼────────────┤
y       │     W      │   P        │    E       │
        │   bit 1    │ (anchor)   │   bit 2    │
        │            │            │            │
        ├────────────┼────────────┼────────────┤
y+1     │    SW      │     S      │   SE       │
        │ (not in    │   bit 3    │   bit 5    │
        │ adj_status)│            │            │
        └────────────┴────────────┴────────────┘
```

So `adj_status` for anchor `P` at `(x, y)` is a number 0-63 packed like:

```python
adj_status = (
    (forbidden[(x,   y-1)] << 0) |   # N
    (forbidden[(x-1, y  )] << 1) |   # W
    (forbidden[(x+1, y  )] << 2) |   # E
    (forbidden[(x,   y+1)] << 3) |   # S
    (forbidden[(x-1, y-1)] << 4) |   # NW
    (forbidden[(x+1, y+1)] << 5)     # SE
)
```

### Why only 2 diagonals, not all 4?

Three reasons, hinted at by `BoardConst.cpp:1149-1150` ("The order of points affects the size of the precomputed lists. The following algorithm does well but is not optimal for all geometries") and by sizing math:

1. **The precomputed table size scales with `2^adj_status_bits`.** 6 bits = 64 states. Going to 8 bits (all 4 diagonals) would 4× the table — `196 anchors × 256 × 21 pieces` instead of `196 × 64 × 21`. Still fits in memory but materially bigger.
2. **NW and SE alone cover enough filter cases.** Most Blokus pieces are short, so the diagonal cells aren't all equally likely to matter for legality. Pre-filtering on 2 of the 4 diagonals catches most cases.
3. **NE/SW exclusion is handled by the per-move check anyway.** If a piece happens to extend into NE or SW relative to its anchor, the candidate list still includes it; the `check_forbidden` loop (§4c) reads the `forbidden` byte of every cell the piece occupies, including NE/SW. So we don't *miss* illegal moves by ignoring NE/SW in `adj_status` — we just don't pre-filter as aggressively.

The choice is a memory/CPU heuristic that Pentobi tuned empirically. The cited comment explicitly calls it "not optimal for all geometries."

---

## 5. Move enumeration: the attach-point anchor

`Board::gen_moves(c, marker, moves)` (`Board.cpp:111-138`) is the
top-level enumerator:

1. If colour `c` has not played yet, iterate `get_starting_points(c)`
   instead of attach points (`Board.cpp:114-122`).
2. For each cell `p` in `m_attach_points[c]`:
   - If `forbidden[p]`, skip (attach points are not pruned eagerly —
     comment in `optimize_attach_point_lists`, `Board.cpp:482-494`).
   - Compute `adj_status` once for `p`.
   - For each `piece` still in inventory, call
     `gen_moves(c, p, piece, adj_status, marker, moves)`
     (`Board.cpp:140-149`), which iterates the precomputed list and
     applies the per-move marker plus `check_forbidden` filter.

The "anchor" optimisation is therefore the heart of Pentobi's
enumeration: the candidate set for a turn is bounded by the small list of
attach points (at most a few tens of points in midgame), not by all 196
on-board cells. The `attach_points` list is updated incrementally inside
`place()` and `optimize_attach_point_lists()` (`Board.cpp:482-494`).

The `MoveMarker` (`MoveMarker.h:17-48`) is the dedup mechanism: a single
move may appear in multiple attach-point candidate lists (a piece touches
two attach points of the same colour), so the marker rejects repeats. It
is a flat `std::array<bool, Move::range>`, i.e. ~32 KB.

## 6. Symmetry

Pentobi's symmetry handling is per-move-id, not per-bitboard-mask.

- The set of piece transforms is in `PieceTransformsClassic.cpp:14-25`
  (identity + 3 rotations + reflection variants, 8 total for Classic).
  `PieceTransformsTrigon` and `PieceTransformsGembloQ` have analogous
  fixed lists.
- During `BoardConst::create_moves`, equivalent transforms (those that
  collapse to the same shape because of a piece's own symmetry) are
  deduped into the unique-transform list `PieceInfo::m_transforms`
  (`PieceInfo.h:82-87`, doc comment lines 82-87 explicit about this).
  The 21 Blokus pieces in total produce 91 unique orientations.
- For **board** symmetry, `BoardConst::init_symmetry_info<MAX_SIZE>`
  (`BoardConst.cpp:1201-1229`) iterates every move id, applies the
  geometry's central symmetry to its cells via
  `m_symmetric_points[p]`, then searches the candidate list at the
  symmetric anchor for the reverse-ordered match. The result is stored as
  the move's `MoveInfoExt2::symmetric_move` (`MoveInfo.h:107-108`), plus
  a `breaks_symmetry` flag.
- The search uses `breaks_symmetry`/`symmetric_move` to draw
  rotational-symmetric Duo/Trigon_2 positions as ties (see
  `State::evaluate_twocolor`, `State.cpp:258-284`; `m_is_symmetry_broken`
  in State.h). It is **not** used for data augmentation in training —
  Pentobi does no neural-net training.
- There are no bitboard rotate/mirror operations in the run-time path.

## 7. End of game and score

End of game and score are tracked incrementally; no popcount is involved.

- `Board::is_game_over()` (`Board.h:208`) returns true when no colour has
  legal moves. `has_moves(c)` (`Board.cpp:236-253`) reuses the same
  attach-point + precomputed-list scan as gen-moves but returns early on
  the first found move.
- Per-colour score is stored as a `float` `StateColor::points`
  (`Board.h:351`) and incremented in `place()` by the cached
  `m_score_points[piece]` (`Board.h:772`). Bonuses (Classic
  "all-pieces-placed" +15, "one-piece-last" +5) are added on the move
  that empties `pieces_left` (`Board.h:760-768`).
- Final scores are exposed by `get_score_twocolor` / `get_score_multicolor`
  / `get_score_multiplayer` (`Board.h:603-650`) and compared via
  `get_place` (`Board.cpp:189-214`).

So Pentobi tracks "points" the cheapest possible way: one float add per
placed piece. There is no popcount-of-occupied-bits step.

## 8. What does this mean for our F2 plan?

This is the part the user explicitly said is theirs to decide, so this
section just flags the practical implications rather than recommending an
implementation. Pentobi's approach is essentially:

- Precompute every legal `(piece, orientation, anchor)` triple as a
  numbered move id, with the cell list, adjacent-cell list, and
  attach-cell list inlined into one struct per move id.
- Index those moves by `(point, 6-bit-adj-status, piece)` so move
  generation is "look up a short list, filter it by one byte read per cell
  in the piece".
- Run dynamic state as byte arrays, not bitmaps.

The "headline" optimisation is the
**precomputed-list with adj_status** trick (4b above), not anything that
would conventionally be called a bitboard. That is the design decision
worth understanding when porting.

## Verification notes

- File contents read in full: `Board.h`, `BoardConst.h`,
  `MoveInfo.h`, `Move.h`, `PrecompMoves.h`, `Piece.h`, `Grid.h`,
  `Geometry.h`, `Point.h`, `Marker.h`, `MoveMarker.h`, `PieceInfo.h`,
  `PlayoutFeatures.h`, `ScoreUtil.h`, `Variant.cpp:89-127`,
  `PieceTransformsClassic.cpp:1-100`.
- Read in parts: `Board.cpp:100-300`, `BoardConst.cpp:860-1230`,
  `State.cpp:100-300`, `SearchBase.h:1-100`.
- **Not verified personally**: the exact byte layout of
  `g_full_move_table`'s intermediate `ArrayList<Move, 44>` storage, the
  Trigon / GembloQ-specific symmetry paths, and whether the Qt GUI builds
  in any extra runtime-affecting flags. None of these are relevant to a
  Python port of move generation.
