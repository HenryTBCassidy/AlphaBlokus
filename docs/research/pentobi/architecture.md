# Pentobi architecture walkthrough

Where the code that matters for move generation and search lives, and how a
game is wired up at runtime. All references are to a depth-1 clone of
`enz/pentobi` (project version `31.0-dev`, `CMakeLists.txt:4-7`).

## Directory split

The project follows a `libboardgame_*` / `libpentobi_*` convention spelled out
in `CONTRIBUTING.md:18-58`:

- `libboardgame_base/` — generic primitives: `Point`, `Geometry`, `Grid`,
  `Marker`, `ArrayList`, `RandomGenerator`, sgf IO, transforms. ~6,300 LOC.
- `libboardgame_mcts/` — game-agnostic MCTS: `SearchBase`, `Tree`, `Node`,
  `LastGoodReply`, virtual loss. ~2,600 LOC.
- `libboardgame_gtp/`, `libboardgame_test/` — GTP framework and unit-test
  helpers; small.
- `libpentobi_base/` — Blokus rules and data tables: `Board`, `BoardConst`,
  `PieceInfo`, `PieceTransforms*`, geometries (`TrigonGeometry`,
  `CallistoGeometry`, `GembloQGeometry`, `NexosGeometry`), `Variant`,
  `Game`. ~10,100 LOC; this is the bulk of the project.
- `libpentobi_mcts/` — Pentobi's playing engine on top of the abstract MCTS:
  `Search`, `State`, `PriorKnowledge`, `PlayoutFeatures`, `SharedConst`,
  `History`, `LocalPoints`. ~4,600 LOC.
- `pentobi/` — Qt6 QML/Quick GUI (~6,200 LOC). Not interesting for porting.
- `pentobi_gtp/`, `libpentobi_gtp/` — optional GTP wrapper (off by default,
  `CMakeLists.txt:14-15`).
- `twogtp/` — tournament harness for two GTP engines (UNIX only).
- `learn_tool/` — offline tool that fits the move-prior parameters used by
  `libpentobi_mcts/PriorKnowledge`.
- `opening_books/` — SGF opening books shipped as data.

## Key classes and what they represent

| Concept            | Class                              | Source                              |
| ------------------ | ---------------------------------- | ----------------------------------- |
| On-board cell      | `libboardgame_base::Point<M,W,H,I>` | `libboardgame_base/Point.h:32-81`   |
| Board geometry     | `libboardgame_base::Geometry<P>`    | `libboardgame_base/Geometry.h:28-189` |
| Array indexed by point | `libboardgame_base::Grid<P,T>` | `libboardgame_base/Grid.h:64-97`    |
| Board state (per game) | `libpentobi_base::Board`        | `libpentobi_base/Board.h:36-470`    |
| Static precomputed game data | `libpentobi_base::BoardConst` | `libpentobi_base/BoardConst.h:23-225` |
| One placed move (id only) | `libpentobi_base::Move`     | `libpentobi_base/Move.h:17-83`      |
| Move payload (piece + cells)| `libpentobi_base::MoveInfo<MAX_SIZE>` | `libpentobi_base/MoveInfo.h:28-54` |
| Move adjacency/attach info | `MoveInfoExt<MAX_ADJ_ATTACH>` | `libpentobi_base/MoveInfo.h:66-86` |
| Rarely-touched move metadata | `MoveInfoExt2`            | `libpentobi_base/MoveInfo.h:93-123` |
| Piece (no orientation) | `libpentobi_base::Piece`       | `libpentobi_base/Piece.h` (74 LOC)  |
| Piece definition + transforms | `PieceInfo`                | `libpentobi_base/PieceInfo.h:33-116` |
| Set of legal transforms per piece set | `PieceTransformsClassic`, `…Trigon`, `…GembloQ` | `libpentobi_base/PieceTransformsClassic.cpp` etc. |
| Per-cell visited marker  | `libboardgame_base::Marker<P>` | `libboardgame_base/Marker.h:24-86`  |
| Per-move dedup marker    | `libpentobi_base::MoveMarker`  | `libpentobi_base/MoveMarker.h:17-48` |
| MCTS state per thread    | `libpentobi_mcts::State`       | `libpentobi_mcts/State.h:35-…` (530 LOC) |
| MCTS node                | `libboardgame_mcts::Node<…>`   | `libboardgame_mcts/Node.h`          |
| MCTS search engine       | `libboardgame_mcts::SearchBase` then `libpentobi_mcts::Search` | `libboardgame_mcts/SearchBase.h:164-…`, `libpentobi_mcts/Search.h:38-…` |
| Game-level wrapper (UI/IO) | `libpentobi_base::Game`      | `libpentobi_base/Game.h`            |
| Variant catalogue        | `libpentobi_base::Variant`     | `libpentobi_base/Variant.h`, `Variant.cpp:89-127` |

## How a game is initialised and played

Pentobi separates **static** precomputed data (board topology, all valid
moves and their masks, the precomputed move tables) from **dynamic** state
(occupied cells, attach points, remaining pieces).

1. **Variant → Geometry.** `Variant.cpp:89-127` switches on the enum and
   returns a singleton geometry: `RectGeometry<Point>::get(14, 14)` for Duo
   (line 95), `(20, 20)` for Classic (line 98), Trigon/Nexos/Callisto/GembloQ
   geometries for the rest. Geometries lazily compute adjacency and diagonal
   lists per point in `Geometry::init` (`Geometry.h:269-308`).
2. **Geometry → BoardConst (static).** `BoardConst::get(variant)` returns
   the cached static-data singleton for that variant
   (`BoardConst.h:42-43`, comment "not thread-safe"). Its constructor calls
   `create_moves()` (`BoardConst.cpp:941-972`), which for every
   (piece, orientation, anchor cell) generates the full
   `MoveInfo`/`MoveInfoExt`/`MoveInfoExt2` payload via
   `create_move<MAX_SIZE,MAX_ADJ_ATTACH>` (`BoardConst.cpp:863-939`). The
   index allocated to that move becomes its `Move::IntType` value.
3. **Precomputed move lists.** After generating each move,
   `create_moves()` iterates every on-board point and every value of
   `adj_status` (0..63 for the default `adj_status_nu_adj=6`,
   `PrecompMoves.h:40-44`), and copies the global `g_full_move_table` into
   the compact storage on `PrecompMoves::m_move_lists`
   (`BoardConst.cpp:959-968`). For Duo the total move count is 13,729
   (`Move.h:32`); the precomputed-list array fits inside
   `PrecompMoves::max_move_lists_sum_length = 2,628,840` `Move` entries
   (`PrecompMoves.h:48-49`).
4. **Board (dynamic).** `Board::init(Variant, const Setup*)`
   (`Board.h:141`, impl in `Board.cpp:289-…`) clears the per-color state
   (`m_state_color[c].forbidden.fill(false, geo)`, `is_attach_point.fill(false)`,
   piece inventories) and applies any setup pieces.
5. **A move is played.** `Board::play<MAX_SIZE, MAX_ADJ_ATTACH>(c, mv)`
   (`Board.h:741-816`) is templated on the variant's max piece size and
   adj/attach budget so the compiler can unroll inner loops. It:
   - decrements `nu_left_piece[piece]`,
   - increments `points` by the cached `m_score_points[piece]`,
   - for each on-board point of the move sets
     `point_state[*i] = PointState(c)` and **sets `forbidden[*i] = true`
     for every colour** (Blokus rule that a placed cell is forbidden for
     everyone, `Board.h:777-781`),
   - walks the move's adjacent points (edge-neighbours) and sets
     `forbidden[*i] = true` for the played colour only (the
     same-colour-no-edge rule, `Board.h:791-793`),
   - walks the move's attach points (diagonal neighbours) and appends any
     newly-attaching unforbidden ones to `m_attach_points[c]`
     (`Board.h:797-807`).
6. **Move generation.** `Board::gen_moves(c, marker, moves)`
   (`Board.cpp:111-138`) iterates the colour's attach points (or starting
   points for the first piece), computes the cell's `adj_status` from up to
   6 neighbours (`Board.h:479-487`), and for each remaining piece calls
   `m_bc->get_moves(piece, point, adj_status)` to get the precomputed move
   list. The inner loop (`Board.cpp:140-149`) just filters by the per-move
   `MoveMarker` (to dedup moves discovered from multiple anchors) and the
   per-cell `forbidden[]` mask.
7. **Search.** `libpentobi_mcts::Search` (`Search.h:38`) extends
   `libboardgame_mcts::SearchBase` (`SearchBase.h:164`). Each playout thread
   owns a `libpentobi_mcts::State` (`State.h:35`) wrapping a `Board` plus
   `PlayoutFeatures`, `PriorKnowledge`, `LastGoodReply`, and a per-colour
   `MoveMarker`. MCTS nodes live in `libboardgame_mcts::Tree<Node>`
   (`State.h:41`); RAVE and virtual loss are configurable at compile time
   via `SearchParamConst`.

## Snapshot / fast undo

`Board` deliberately has no undo (`Board.h:28-31`). Instead `take_snapshot`
copies the entire per-colour state into `m_snapshot`, and `restore_snapshot`
(`Board.h:823-845`) does `memcpy_from` on the grids and `copy_from` on the
attach-point lists. The search uses one snapshot per simulation start.

## Things that surprised me

- **No bitboard.** Despite Pentobi's reputation as a fast Blokus engine,
  there is no positional `uint64`/`uint128` bitboard anywhere. The two
  hot-path grids are `GridExt<bool> forbidden` and `Grid<bool> is_attach_point`,
  both byte-per-cell (`Board.h:341-343`). The performance win comes from
  precomputed move lists, not bitwise tricks. See
  [move-generation.md](move-generation.md).
- **`uint_least32_t` over `uint_fast32_t`.** Explicit comment in
  `PlayoutFeatures.h:43-47`: `uint_fast32_t` is 8 bytes on GCC/Intel and
  measurably slower; they pin to `uint_least32_t` (32 bits). Cache
  footprint matters more than register width.
- **Heavy use of compile-time piece-size templates.** `Board::play`,
  `Board::place`, `BoardConst::create_move`, and several check helpers are
  templated on `MAX_SIZE` and `MAX_ADJ_ATTACH` (`Board.h:160-161, 741-816`,
  `BoardConst.cpp:863-939`). The variant's max size is dispatched once at
  the call site (`BoardConst.h:280-298`) so the inner loops are
  fully-unrolled.
- **A 24/8-bit packed range descriptor** for the precomputed-move-list
  pointers (`PrecompMoves.h:96-119`): 24-bit begin index, 8-bit size, fitted
  in one `uint32`.
- **Global `g_marker` and `g_full_move_table`** used as build-time scratch
  (`BoardConst.cpp:27-33`). Comment explicitly says: not thread-safe;
  speed of construction was preferred over isolation. Singletons.
- **Per-colour `forbidden` grid is `GridExt<bool>`** (includes the
  null-point sentinel, `Board.h:341`). This lets the
  pieces-with-null-pad pattern in `MoveInfo` (`MoveInfo.h:38-40`) index it
  without a bounds check; reading `forbidden[Point::null()]` always returns
  the pre-initialised null entry. This is the closest thing to a sentinel
  trick in the codebase.
- **`Move` is a 16-bit id, not a structured payload.** All data lives in
  parallel arrays keyed by `move.to_int()` (`Move.h:21-25`). The total
  move space tops out at 32,131 for Trigon (`Move.h:28, 53`), 13,729 for
  Duo (`Move.h:32`).
