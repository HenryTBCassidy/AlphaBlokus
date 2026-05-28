# Python port gotchas

Things in Pentobi's C++ implementation that will be awkward, slow, or
impossible to reproduce in Python — and the corresponding CPython-side
options. Scope: only the parts of Pentobi relevant to move generation and
self-play. We do not port the Qt GUI, GTP, or learning tool.

## TL;DR

The single most important takeaway is that Pentobi is **not a bitboard
engine**, so the main porting problem is not bit-trick translation but
**replicating its 2.6-million-entry precomputed move table and the 16-bit
move-id arrays of `MoveInfo`/`MoveInfoExt`** efficiently in Python.
See [move-generation.md](move-generation.md) for the
design. The list below assumes we want to keep the broad shape of
Pentobi's design.

## C++ features used

### 1. Compile-time piece-size templates

`Board::play<MAX_SIZE, MAX_ADJ_ATTACH>` and friends
(`Board.h:160-161, 741-816`) are templated on the variant's max piece
size so the per-move cell loop is fully unrolled. The dispatch lives at
the call site (`BoardConst.h:280-298`).

Python has no template specialisation. The closest options:
- Hard-code Duo's `MAX_SIZE=5, MAX_ADJ_ATTACH=16` everywhere; pay the
  one-time cost of branching on it elsewhere.
- Generate variant-specialised modules at build time. Overkill for one
  variant.
- For the actual hot loops, push them into Cython/C/numpy where the
  constant is a compile-time constant again.

### 2. The fixed-size `Point` arrays with null-padding

`MoveInfo<MAX_SIZE>::m_points` is exactly `MAX_SIZE` slots long, padded
with `Point::null()` (`MoveInfo.h:38-40`). The forbidden check
(`State.cpp:131-150`) iterates exactly `MAX_SIZE` slots and reads
`forbidden[Point::null()]`, which is always `false`. This lets the
compiler unroll without bounds checks.

Python can mimic the layout with a `numpy.ndarray` of shape
`(num_moves, MAX_SIZE)` of dtype `uint16`, and a `forbidden` array sized
one slot bigger with the null index used for padding. The unroll
benefit is gone in pure Python; it survives in numpy because reading a
fixed-width row in one vectorised access is constant-time.

### 3. `uint_least32_t` vs `uint_fast32_t`

The explicit comment in `PlayoutFeatures.h:43-47` ("logically we want
`uint_fast32_t` but that is 8 bytes on GCC/Intel and *slower*") tells you
cache footprint matters. In Python, `int` is arbitrary precision and
boxed; the rough equivalent is `numpy.uint32` arrays. **Don't** use
Python `int` for the 2.6M-entry precomputed-move-list — it would balloon
to gigabytes of `PyObject*`. Use `numpy.uint16` (move ids fit) or
`array.array("H", ...)`.

### 4. `new(place) MoveInfo<MAX_SIZE>(piece, points)` placement-new

`BoardConst::create_move` (`BoardConst.cpp:863-878`) constructs into raw
arena-allocated memory. The Python equivalent is just to allocate the
arrays up front and fill them, e.g. `numpy.empty((num_moves, MAX_SIZE),
dtype=np.uint16)` plus parallel arrays for the `Ext` fields. No
placement-new needed; the data is small enough to be flat.

### 5. The 24/8-bit packed `CompressedRange`

`PrecompMoves::CompressedRange` (`PrecompMoves.h:96-119`) packs a begin
index (24 bits) and a size (8 bits) into one `uint32_t` to keep the
range table dense (~196 × 64 × 21 entries for Duo). At 4 bytes per
entry that table is ~1 MB.

In Python the simplest equivalent is two parallel `uint32` arrays
(`begin`, `size`). The total is 8 bytes per entry instead of 4, so
roughly 2 MB. That's fine. If we cared, we could pack into one `uint32`
with bit-shifts — Python supports that — but it would just trade memory
for slower per-entry decode.

### 6. No `__builtin_popcount` / `__builtin_ctz` are used

Searched the source: no `__builtin_popcount`, `__builtin_ctz`,
`__builtin_clz`, BMI/BMI2 intrinsics, SIMD intrinsics, or `__uint128_t`
appear anywhere. The hot inner loop is byte loads and adds, not bit
manipulation. So **we do not need** `int.bit_count()` (3.10+), `gmpy2`,
or numpy bitwise ops to match Pentobi's run-time behaviour.

This is the headline gotcha: the F2 plan's worry about "Python ints
support arbitrary precision so 196-bit bitboards work but bit ops are
slower than 64-bit" is not actually relevant to a Pentobi-style port.
We would only hit it if we deliberately re-cast the design as a true
bitboard.

### 7. The `Marker` clear-by-counter trick

`libboardgame_base::Marker<P>` (`Marker.h:24-86`) avoids the
"clear-every-cell" cost between uses by storing a `unsigned m_a[range]`
and a `m_current` counter; a cell is "marked" iff `m_a[p] == m_current`.
`clear()` just decrements the counter, with one full reset every 2^32
clears.

This works fine in numpy: `m_a` is a `numpy.uint32` array and
`m_current` a Python int. Vectorised "is marked" is
`m_a[points] == m_current`. **Caveat**: the overflow handling
(`Marker.h:56-67`) needs to be preserved or you risk false-positive
"marked" reads after a long simulation.

### 8. Singleton precomputed data and global scratch

`BoardConst::get(variant)` returns a per-variant singleton
(`BoardConst.h:42-43`, header comment: "not thread-safe"). Inside
`BoardConst.cpp:27-33` there is also a file-scope `g_marker` and
`g_full_move_table` used during construction.

For our self-play ProcessPoolExecutor setup this matters:
- A Python module-level singleton built at import time is fine; each
  worker process gets its own copy. It costs ~a few MB RSS per worker
  to hold the static data.
- If we want zero-cost sharing across workers, we'd have to move the
  arrays into `multiprocessing.shared_memory` or a `mmap`'d file. That's
  doable but introduces lifetime/cleanup pain. Recommend deferring
  unless the per-worker RAM is actually a bottleneck.
- Pentobi's "not thread-safe construction" comment becomes "must
  construct once before any worker forks" in our setting. The simplest
  way is to construct on first import in the parent and rely on `fork`
  to share-COW; for `spawn` start methods (default on macOS) each worker
  rebuilds and pays the cost once.

### 9. Float types

`SearchParamConst::Float = float` (`SearchBase.h:60`) is chosen for
node-size reasons in a 32-bit fixed-point sense. The MCTS uses
`float`-accurate counters and saturates at ~2^24 simulations. We are
not porting Pentobi's MCTS, so this is moot for the move-gen-only port,
but worth knowing if we ever benchmark our visit-count arrays.

### 10. Memory layout / cache-line tricks

The only explicit cache-layout tactic is splitting per-move data across
three structs (`MoveInfo`, `MoveInfoExt`, `MoveInfoExt2`) ordered by hot
data first — see comment block at `MoveInfo.h:58-65`. In numpy this
maps to **three separate arrays** indexed by `move_id`, not one
structured dtype, so that the inner loop's per-cell read only touches
the cache lines it needs. Avoid the Pythonic temptation of a single
`dtype([...])` structured array if the hot loop only reads two fields.

There are no false-sharing-avoidance tricks, no padding for cache
alignment, no manual prefetches. (Searched the source for `alignas`,
`__attribute__((aligned))`, `prefetch`, `padding`: none.)

### 11. Little-endian assumptions

None found. The bit-packing in `PrecompMoves::CompressedRange` and
`PlayoutFeatures::IntType` is done with explicit shift / mask in C++,
not via reinterpret-casting. Portable to Python directly.

### 12. `std::array` of arrays for the precomputed table

`PrecompMoves::m_moves_range` is
`Grid<std::array<PieceMap<CompressedRange>, nu_adj_status>>`
(`PrecompMoves.h:122`). That's a 4D array indexed
`[point][adj_status][piece]` returning a packed range descriptor.

In Python: a single `numpy.ndarray` of shape
`(num_points, 64, num_pieces, 2)` with dtype `uint32` (the last
dimension being `(begin, size)`). For Duo: `196 * 64 * 21 * 2 * 4 bytes
= ~2 MB`. Cheap.

### 13. The move-list backing store

`PrecompMoves::m_move_lists` is a single
`std::array<Move, max_move_lists_sum_length>` with
`max_move_lists_sum_length = 2,628,840` (`PrecompMoves.h:48-49, 128`)
and `sizeof(Move) = 2`. Roughly 5 MB.

In Python: one `numpy.uint16` array of length ~2.6M = ~5 MB. Same.

### 14. `std::array<bool, Move::range>` MoveMarker

`MoveMarker` (`MoveMarker.h:17-48`) is a flat bool array of size
`Move::range` (32,132 for Trigon, 13,730 for Duo). It is cleared with
`std::array::fill` and set with single-element writes.

Python options:
- `numpy.zeros(num_moves, dtype=bool)` — clear is a fill, hot.
- A `bytearray` — slightly faster for single-element writes.
- The `Marker`-style counter trick is overkill here since the move-id
  space is small.

## Hot-loop translations

| Pentobi hot loop                                            | Best Python equivalent                                                                |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `forbidden \|= is_forbidden[*p++]` over `MAX_SIZE` cells    | `numpy.any(forbidden[move_cells[mv]])` per move, or vectorise over many moves at once via fancy indexing |
| `adj_status` bit-pack of 6 bools                            | Vectorise via `numpy.packbits` over the precomputed neighbour table                    |
| `PrecompMoves::get_moves(piece, p, adj_status)`             | One `numpy.uint16` slice using the `(begin, size)` table                              |
| `Marker::set` / `clear()` (counter trick)                   | `numpy.uint32` array + a generation counter; reset on overflow                       |
| `Board::play` (set forbidden + adj-forbidden + update attach) | Pure-Python with numpy boolean assignments; this is the function most likely to need Cython if it shows up in the profile |

## Multi-process gotchas

Pentobi assumes one process, multi-threaded only if
`SearchParamConst::multithread = true` (`SearchBase.h:69-72`). Our
ProcessPoolExecutor setup needs each worker to either:

1. **Inherit COW pages** (Unix `fork` start method): the static
   `BoardConst`/`PrecompMoves` are built once in the parent before
   workers spawn. Cheap RAM-wise.
2. **Rebuild on import** (macOS/Windows `spawn`): each worker pays the
   construction cost. Wall-clock cost is dominated by computing
   13,729 move infos for Duo; should be sub-second in pure Python and
   tens of ms with numpy.
3. **Share via `shared_memory`**: read-only `numpy` views over a single
   buffer. Worth doing only if the RAM cost actually hurts.

Pentobi's run-time `Search` state (`State`, the per-thread `Board`
copy, `PlayoutFeatures`, `MoveMarker`, `Marker`) is **per-thread**.
We translate that to **per-worker**.

## Things that map cleanly

- `Point` -> `int` (or a `numpy.uint16` cell index).
- `Move::IntType` -> `numpy.uint16`.
- `Grid<T>` -> `numpy.ndarray` of length `num_cells`.
- `GridExt<T>` -> `numpy.ndarray` of length `num_cells + 1` with the
  last slot being the null sentinel.
- `ArrayList<T, N>` -> a numpy buffer of length `N` plus an `int`
  current-size, or just a Python `list`. For small lists (attach
  points) Python list is fine.
- `PieceTransforms*` -> a per-piece-set Python module with the eight
  transform lookup tables.

## Things that will be painful

- Generating the 2.6M-entry move-list table at import time in pure
  Python. Vectorise this with numpy or accept a one-off load from a
  pickled / `.npy` cache.
- The per-move forbidden scan if we want to match Pentobi's
  per-simulation throughput. Pure Python will be 30-50x slower than C++.
  Profile first; consider Cython only for the inner loop if it shows up
  hot. (Per CLAUDE.md, our move-gen is already the bottleneck — the
  precomputed-list design is a real win regardless of language.)
- The construction-time symmetry sweep
  (`BoardConst::init_symmetry_info`, `BoardConst.cpp:1201-1229`) does
  a reverse-equality compare of move-cell lists. Not algorithmically
  hard, but it's quadratic-ish in moves-per-anchor and benefits from
  vectorisation.

## Where Python has an advantage

- `bytes` / `int.from_bytes` make state hashing (for our MCTS cache)
  trivial; Pentobi has nothing equivalent.
- `numpy.where` and fancy indexing replace several explicit loops in
  one expression.
- `array.array("H", ...)` plus `memoryview` slicing is as fast as the
  C++ `Move` array for sequential reads in tight inner loops if you
  hoist the slice to a local name. Not a substitute for Cython, but
  closes the gap meaningfully.

## Specific recommendations

The user's plan asked that I not over-recommend, so just three:

1. **Replicate the `(piece, point, adj_status)` precomputed table.**
   This is the actual core optimisation in Pentobi; everything else is
   bookkeeping around it.
2. **Cache it to disk** (e.g. `pentobi_tables_duo.npz`) so workers don't
   rebuild on every `spawn`-based pool start.
3. **Profile before bit-twiddling.** Pentobi itself didn't go below
   byte-per-cell; we shouldn't either until measurements say otherwise.
