# Corpus search-space store — a position-keyed DAG in SQLite

Design + implementation plan for how the v2 Pentobi corpus stores the search space it explores.
Companion to [`pentobi-corpus-v2.md`](../pentobi-corpus-v2.md) (whose V3 row executes this plan's
checklist); literature context in
[`../research/corpus-generation-literature.md`](../../research/corpus-generation-literature.md).
Requirements, verbatim (Henry):

> a) makes it easy to tell what we've done before; b) allows us to make more training data
> systematically without accidentally repeating loads; c) gives us a measure of how we are
> expanding the search space; d) naturally matches the generating function of the data.

The short version: **one SQLite file holding a position-keyed DAG** (nodes = positions, edges =
Pentobi's ranked candidate moves, playouts hanging off the nodes an allocation **plan** assigns
games to), **plus parquet for bulk training rows** exactly as today. Identity is *positional and
content-derived* everywhere — node keys are canonical board bytes, playout seeds are hashes of
`(board_key, replica)` — so "already done" is a primary-key lookup and disjointness of new work is
structural, not bookkeeping. Since the first pass, expansion is **budget-proportional allocation
with emergent depth** (v2 plan V4): the store additionally records each allocation plan and its
per-node game targets, so "planned vs actual" is queryable and top-ups are recomputable.

---

## Checklist

Executed as row V3 of [`pentobi-corpus-v2.md`](../pentobi-corpus-v2.md).

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| S1 | `pentobi/store.py`: schema DDL, `SearchSpaceStore` open/create, symmetry-canonical node keys, insert/lookup, content-derived seeds; engine-free tests | 3 h | High | ✅ |
| S2 | Search recording: `record_search` (node fields + full ranked child list as edges, actions mapped to the key frame, mirror-pair merging); fixture-driven tests | 2 h | High | ✅ |
| S3 | Allocation planning: pure-function allocator (DAG, budget, T, R) → per-node planned games, search-on-demand mapping, mirror-pair weight merging, book-path force-insert; `plans`/`plan_nodes` persistence; tests | 4 h | High | ✅ |
| S4 | Playout registry: fulfilment-ordered scheduling against the active plan, done-marking, reconcile-from-shard-footers; tests | 2 h | High | ✅ |
| S5 | `export-opening`: DB → `opening/*.parquet` with `dag_hash` stamp; `link` outcome aggregation (recursive CTE); tests | 3 h | High | ✅ |
| S6 | Coverage report (`coverage` CLI): the metric set below + registration of the v1 corpus in `corpora` | 2 h | High | ✅ |

---

## Design

### D-a. It is a DAG keyed by position, not a tree keyed by move sequence

The generating function (requirement d) is: *at a position, run one L9 search, harvest
`move_values`, split the position's game budget among its children, recurse; play games out from
the positions whose budget stops being divisible.*
Every expensive artifact — the search, the soft target, the backed-up value — is a function of the
**position**, not of the move order that reached it. So the store keys nodes by position:

- **Dedup is exact and free.** "Have we searched this?" is a unique-index lookup on 196 canonical
  board bytes (requirement a). A transposed-into position reached by a second move order costs
  zero extra search and produces zero duplicate training rows (requirement b). Move-sequence keys
  would silently duplicate both.
- **The structure is a DAG** (multiple parents allowed), but a *layered* one: in the pass-free
  opening, depth = pieces placed is a pure function of the position, so cycles are impossible and
  "depth" is well-defined per node without qualification.
- **Cost of the DAG over a tree** is one extra concept — a node needs a *witness path* (one
  as-played action sequence from the root, stored per node) because games and validators must
  replay a concrete move sequence into the engine, and a DAG node does not have a unique one.
  That is a single JSON column, and it makes every node independently replayable and testable.

In the opening plies transpositions are probably rare — but they are not the main argument.
Positional identity is what makes playout identity (`(board_key, replica)`) survive any later
reshaping of the graph: a game generated when its start node was a playout start remains valid
and correctly attached when a larger-budget plan turns that node interior. Move-sequence identity
would tie games to a particular plan. That property is what requirement (b) actually needs.

### D-b. Node keys are symmetry-canonicalised: `min(bytes, transposed_bytes)` — **decided**

Approved by Henry (2026-07-28), and now measured rather than estimated: of the 414 legal first
moves, 10 are self-symmetric and there are **212 distinct positions** after main-diagonal
canonicalisation (51.2%); of Pentobi's 315 searched root children, **310 have their mirror also
searched** (5 self-symmetric), collapsing to **160 distinct canonical first positions** — so the
engine spends nearly half its root effort on mirror duplicates and canonicalisation reclaims it.
`get_canonical_form` is perspective-only; this key collapses the order-2 main-diagonal symmetry
that it does not. The saving is concentrated at the root (positions stay symmetric after ply 1
only if every placed piece is diagonal-symmetric — rare), and training-time augmentation in
`distill.py` regenerates the mirror of every stored row, so no information is lost.

One allocation consequence (v2 plan V4): **when a node's own position is symmetric — the root, in
practice — mirror-pair children canonicalise to one child, which must receive the pair's combined
visit weight** before temperature flattening. Otherwise each canonical opening would be charged
only half its true visit share.

Invariants that make this safe:

- `key_frame` records whether the key is the identity (0) or the transpose (1) of the as-played
  position. **Everything stored on the node and its edges lives in the key frame**: the board is
  the key itself; edge actions and harvested policy indices are mapped through
  `BlokusDuoGame.transpose_action` when `key_frame = 1`. One mapping site, at `record_search`.
- The **witness path stays in the engine (as-played) frame** — phase B replays it verbatim with
  `play`; harvested game plies are stored as-played (game rows are not symmetry-collapsed; the
  duplicate-position diagnostic reports both raw and mirror-collapsed counts).
- Training loses nothing: augmentation produces the mirror twin of every opening row, so both
  members of each collapsed pair reach the net — now with one consistent label instead of two
  independently-noisy searches.

The mapping is ~20 lines with a property test (transpose the fixture, assert identical stored
node) — S1/S2 keep that test as specified; the decision is made, the test is the guard.

### D-c. Storage technology: SQLite + parquet — and why not the alternatives

**SQLite** (stdlib `sqlite3`, zero new dependencies, one file next to the shards) for the graph,
the harvest, and all registries; **parquet** for bulk training rows, unchanged, because that is
what the trainer already consumes.

- **vs parquet-only:** the DAG build is an incremental, random-access, read-modify-write workload
  — point lookups ("is this position stored?"), partial updates (node searched, child expanded,
  playout done), and crash-safe resume. Parquet is append-only and unindexed; every "have we done
  this?" becomes a scan or a hand-rolled sidecar index, which is this design re-implemented badly.
- **vs an embedded graph library** (networkx + pickle, or similar): in-memory with monolithic
  serialisation — not crash-safe mid-build, not queryable while a run holds it, and every coverage
  question becomes bespoke code instead of a SQL one-liner. Our graph queries are trivial
  (children, frontier, one recursive aggregate); a graph engine buys nothing.
- **vs DuckDB:** closest contender (better analytics SQL), but it adds a dependency, and its
  strength — columnar scans — is what parquet already covers; its weakness — many small
  transactional writes — is exactly the build workload. SQLite with WAL handles a 12-worker
  generation run trivially (workers don't even write concurrently: the parent process owns the DB;
  workers return results).
- **No server database**, per the constraint — single box, single user; operational weight is
  a real cost and buys nothing here.

Scale check (at the measured 10 k-game / T = 2 plan: ~1.6 k searched internal nodes): nodes a few
thousand × (196 B key + ~120 B fields) ≈ 1 MB; edges ~1.6 k × ~350 children × ~40 B ≈ 25 MB;
playouts ~10 k small rows; plans negligible. A tens-of-MB SQLite file — comfortably inside "a
table and an index".

**Division of truth** (how the two stores stay consistent):

| Data | Source of truth | Derived copy |
|---|---|---|
| Graph, per-node harvest, per-edge candidates | SQLite | `opening/*.parquet` export (regenerable, stamped with `dag_hash`; stale exports are detectable and simply re-run) |
| Game training rows | `games/*.parquet` shards (atomic `.tmp`→rename, as v1) | `playouts` table (an index; reconcilable from shard footers, see D-e) |

No two-phase writes are needed: each direction has one truth and one regenerable/reconcilable
mirror.

### D-d. Schema

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- schema_version, game, board_kind, policy_size, level, engine_version,
-- created_at (plan parameters live in the plans table, not here)

CREATE TABLE nodes (
    node_id         INTEGER PRIMARY KEY,
    board_key       BLOB    NOT NULL UNIQUE,  -- min(compact, transposed compact); 196 B int8
    key_frame       INTEGER NOT NULL,         -- 0 identity / 1 key is the transpose of as-played
    depth           INTEGER NOT NULL,         -- plies from empty board (= pieces placed)
    player          INTEGER NOT NULL,         -- side to move: +1 White, -1 Black
    witness_actions TEXT    NOT NULL,         -- JSON [action, ...] root→node, as-played frame
    source          TEXT    NOT NULL,         -- 'root' | 'search' | 'book' | 'net'
    status          TEXT    NOT NULL,         -- 'pending' | 'searched'
    engine_seed     INTEGER,                  -- hash64(board_key) & 0x7fffffff (recorded for audit)
    root_visits     INTEGER,
    search_value    REAL,                     -- top child's value, side-to-move (never GTP get_value)
    search_seconds  REAL,
    searched_at     TEXT,
    outcome_mean    REAL,                     -- link pass: mean outcome (side-to-move) of subtree playouts
    outcome_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX nodes_depth_status ON nodes(depth, status);

CREATE TABLE edges (                          -- ALL move_values children of a searched node
    parent_id   INTEGER NOT NULL REFERENCES nodes(node_id),
    action      INTEGER NOT NULL,             -- in the parent's KEY frame
    rank        INTEGER NOT NULL,             -- 0-based visit rank in the parent's search
    visits      INTEGER NOT NULL,             -- mirror-pair-merged when the parent is symmetric
    visit_share REAL    NOT NULL,             -- visits / parent root_visits
    child_value REAL    NOT NULL,             -- the parent search's value for this child
    child_id    INTEGER REFERENCES nodes(node_id),  -- NULL ⇒ child not instantiated as a node
    source      TEXT    NOT NULL,             -- 'search' | 'book' | 'net'
    PRIMARY KEY (parent_id, action)
);
CREATE INDEX edges_child ON edges(child_id);

CREATE TABLE plans (                          -- one row per allocation run (v2 plan V4)
    plan_id     INTEGER PRIMARY KEY,
    created_at  TEXT    NOT NULL,
    budget      INTEGER NOT NULL,             -- total games
    temperature REAL    NOT NULL,             -- allocation T (w ∝ p^(1/T))
    min_replicas INTEGER NOT NULL,            -- R, the split threshold
    dag_hash    TEXT    NOT NULL,             -- DAG content hash when the plan was computed
    is_active   INTEGER NOT NULL DEFAULT 0    -- exactly one active plan; deficits read from it
);

CREATE TABLE plan_nodes (                     -- the allocation itself: reproducible, queryable
    plan_id       INTEGER NOT NULL REFERENCES plans(plan_id),
    node_id       INTEGER NOT NULL REFERENCES nodes(node_id),
    budget_share  REAL    NOT NULL,           -- fraction of the total budget reaching this node
    planned_games INTEGER NOT NULL,           -- games to START here (0 for internal nodes)
    PRIMARY KEY (plan_id, node_id)
);

CREATE TABLE playouts (
    node_id      INTEGER NOT NULL REFERENCES nodes(node_id),
    replica      INTEGER NOT NULL,            -- 0, 1, 2, ... — monotone per-node counter, across plans
    engine_seed  INTEGER NOT NULL,            -- hash64(board_key ‖ replica) & 0x7fffffff
    game_id      INTEGER UNIQUE,              -- global id for shard rows, assigned at schedule time
    status       TEXT    NOT NULL,            -- 'planned' | 'done'
    shard        TEXT,
    white_margin INTEGER,
    plies        INTEGER,
    completed_at TEXT,
    PRIMARY KEY (node_id, replica)
);

CREATE TABLE corpora (                        -- external datasets in the manifest, not the graph
    name TEXT PRIMARY KEY, path TEXT NOT NULL, dataset_kind TEXT NOT NULL,
    games INTEGER, positions INTEGER, notes TEXT
);
```

Design notes:

- **Candidates are rows, not blobs — and the full child list is stored.** The allocator's
  temperature flattening (`w ∝ p^(1/T)`) gives meaningful weight deep into the visit tail (at
  T = 2 a 10 k-game plan reaches ~279 of 315 first moves), so truncating the stored children would
  silently truncate the allocation. All `move_values` children (~300–420/node) are stored as
  `edges` rows: the allocation is computable from the store alone, the disagreement data is
  directly queryable, and the parquet policy target (top-32 by rank, normalised) is derived at
  export. Tail mass is derived too: `1 − sum(visits)/root_visits` over the exported 32.
- **The disagreement is first-class**: an edge's `visit_share`/`rank`/`child_value` is the
  parent's opinion; after expansion, the child row's own `search_value` is the independent
  one-ply-deeper opinion (side-to-move at the child, ≈ complementary to the parent's perspective —
  approximately, per the measured value semantics; document, don't assume exact). A view joins the
  two:

```sql
CREATE VIEW edge_disagreement AS
SELECT e.parent_id, e.action, e.rank, e.visit_share, e.child_value,
       1.0 - c.search_value AS independent_value,        -- parent-perspective, approximate
       (1.0 - c.search_value) - e.child_value AS value_gap
FROM edges e JOIN nodes c ON c.node_id = e.child_id
WHERE c.status = 'searched';
```

- **The allocation is stored; the DAG shape is not a policy.** `plan_nodes` records exactly what
  a given `(budget, T, R)` allocation assigned to every node — the reproducible plan the
  coordinator's ruling requires. The allocation itself is a **pure function of
  (DAG content, budget, T, R)** (D-e), so a plan can always be recomputed and checked against its
  stored rows; `plans.dag_hash` pins which DAG state it was computed from.
- **There are no permanent "leaves"** — a node is a *playout start* for a plan iff its
  `planned_games > 0` in that plan. Re-planning at a larger budget can turn a former start into
  an internal node; its existing playouts remain valid (positional identity) and are counted as
  that node's actuals.

### D-e. The extension protocol (requirement b)

Identity is content-derived at every level, so disjointness of later work is structural:

- **Node identity** = `board_key`. Inserting an already-present position is a no-op
  (`INSERT OR IGNORE`); its search is never repeated (`status = 'searched'`).
- **Search reproducibility**: each node's engine seed is `hash64(board_key)` — a re-search of the
  same position (single-threaded engine) reproduces the same tree, so a crash mid-search costs
  nothing and repair is idempotent.
- **Game identity** = `(board_key, replica)`, seed = `hash64(board_key ‖ replica)`, with `replica`
  a **monotone per-node counter across plans**. Scheduling new work inserts `(node, replica)` rows
  against the primary key — it *cannot* re-plan an existing pair, and replicas added months later
  (under any plan) draw seeds no earlier game used. This is v2's version of v1's "shards are a
  pure function of `(seed, game_id)`": the pure function moved from run-relative ids to
  position-content, which survives any re-planning. Crucially, identity does **not** depend on a
  node being a "leaf" — a game is "replica *r* of position *P*", full stop, so it stays valid and
  correctly attached when a larger-budget plan turns *P* into an internal node.

**Allocation is a pure function of (DAG content, budget, T, R)** — nothing else. Mapping is
*search-on-demand*: the allocator walks top-down; when it assigns ≥ 2R games to a node that is
not yet searched, that node is searched (content-derived seed, so reproducibly), its children
become available, and the walk continues. The DAG therefore grows exactly as far as the plan
needs it and no further, and re-running the allocator against the finished DAG reproduces the
plan bit-for-bit (the S3 test).

The extension cases, all through re-planning:

1. **Top-up (larger budget)**: compute a new plan at `budget = B₂ > B₁` (same T, R) and make it
   active. Child budgets scale with B, so every internal node stays internal and every node that
   remains a start has `planned_games` **monotone non-decreasing** — the new plan is a refinement,
   never a contradiction. Some former starts split: their descendants get planned games, and any
   previously-unsearched node crossing the 2R threshold gets mapped. Per-node **deficit =
   max(0, planned_active − actual)**; generation fulfils deficits. Games at now-internal nodes
   stay valid as surplus (extra samples through that subtree); total new games ≈ B₂ − B₁ plus the
   small mismatch the surplus absorbs — approximate, and honestly so: budgets are conserved per
   plan, not across plans.
2. **Different T or R**: also just a new plan (new row in `plans`); actuals carry over, deficits
   are computed against the new targets. Nothing is ever regenerated or invalidated.
3. **Resume**: interrupted mapping re-runs unsearched nodes the active plan needs; interrupted
   generation re-executes `status = 'planned'` playouts (same seeds ⇒ same games ⇒ exactly the
   missing shards are regenerated, preserving v1's resume semantics).

**Scheduling is fulfilment-driven, not counter-driven**: order all `(node, next replica)` jobs by
**(actual ÷ planned ascending, hash64(board_key))** against the active plan. Every node's
fulfilment fraction rises together — a truncated run is an even *proportional* slice of the plan
at every point (the property Henry's D5 stratified keys guaranteed, generalised to heterogeneous
per-node targets), and nodes added by a re-plan (fulfilment 0) are covered first, automatically.

**Self-description**: `dag_hash` = SHA-256 over the ordered `(board_key, root_visits, ranked
edges)` of all searched nodes. `export-opening` stamps it into the opening parquet footer (a stale
export is detectable and regenerable); game shard footers carry, per game,
`(game_id, start board_key, replica, engine_seed, witness_actions, scores)` plus the `dag_hash`
and the generating plan's `(plan_id, budget, temperature, min_replicas)` as provenance. Shards are therefore self-describing: the `playouts` table can be
rebuilt or verified from footers alone (`reconcile` — also the crash repair if a run dies between
a shard rename and its DB transaction). Games can never be silently mixed across incompatible
expansions because their identity is positional: growing the DAG never invalidates a game, and the
genuinely incompatible changes (level, engine build, board rules) are pinned in `meta` and
asserted on open.

### D-f. Coverage metrics (requirement c) and what "done" looks like

Reported by `coverage`, all cheap SQL:

| Metric | What it honestly measures |
|---|---|
| Nodes / searched / playout-starts **by depth** (depth is emergent — this table is where it becomes visible) | Structural size and mapping debt (needed-but-unsearched vs searched) |
| **Plan fulfilment**: Σ actual ÷ Σ planned overall, plus the per-node fulfilment histogram against the active plan | "Did we generate what the plan says?" — 1.0 = phase B done for this plan |
| **Distinct first moves / first positions** in the plan (vs 414 legal moves / 212 canonical positions; vs the book's 17 first moves) | The opening fan — the strategic headline, and the known coverage gap (52 of 212 canonical first positions are outside Pentobi's own root search entirely) |
| **Play-mass coverage** by depth: Σ over planned nodes of Π ancestor `visit_share` | What share of Pentobi's own L9 play-mass falls inside the plan. Caveat: visits are hyper-concentrated, so this saturates near 1 quickly; it bounds *mainline* coverage and says nothing about the deliberate flattened fan — always read next to the fan width |
| **Planned-games distribution** (min / median / max games per start; budget share by depth) | The allocation's shape — how (T, R) actually spent the budget |
| Distinct vs total stored positions in game shards (raw + mirror-collapsed) | Residual duplication the DAG can't collapse (mid-game convergence) |

"Done" for a given plan = mapping debt zero (every node the plan needs is searched) and plan
fulfilment 1.0. The coverage report plus `dag_hash` and the active plan row is the answer to
"what have we done before?" (requirement a) in one screen.

### D-g. Query examples (how this is actually used)

```sql
-- Have we searched this position?
SELECT node_id, status FROM nodes WHERE board_key = :key;

-- Planned vs actual for the active plan (the "what's left" screen)
SELECT pn.node_id, n.depth, pn.planned_games, COUNT(p.replica) AS actual
FROM plan_nodes pn
JOIN plans pl ON pl.plan_id = pn.plan_id AND pl.is_active = 1
JOIN nodes n ON n.node_id = pn.node_id
LEFT JOIN playouts p ON p.node_id = pn.node_id AND p.status = 'done'
WHERE pn.planned_games > 0
GROUP BY pn.node_id;

-- Generation work queue: lowest fulfilment first (even proportional slice at any truncation)
SELECT pn.node_id, pn.planned_games, COUNT(p.replica) AS actual,
       CAST(COUNT(p.replica) AS REAL) / pn.planned_games AS fulfilment
FROM plan_nodes pn
JOIN plans pl ON pl.plan_id = pn.plan_id AND pl.is_active = 1
LEFT JOIN playouts p ON p.node_id = pn.node_id AND p.status = 'done'
WHERE pn.planned_games > 0
GROUP BY pn.node_id HAVING actual < pn.planned_games
ORDER BY fulfilment ASC;

-- Where does Pentobi most misjudge its own candidates? (Phase-2 seeding)
SELECT * FROM edge_disagreement WHERE rank >= 10 ORDER BY value_gap DESC LIMIT 50;

-- Link pass: empirical outcomes aggregated up the DAG (WITH RECURSIVE over edges)
WITH RECURSIVE sub(node_id, root) AS (
    SELECT node_id, node_id FROM nodes
    UNION SELECT e.child_id, s.root FROM edges e JOIN sub s ON e.parent_id = s.node_id
                                     WHERE e.child_id IS NOT NULL)
SELECT s.root, AVG(p.white_margin > 0), COUNT(*)   -- sign-adjusted per side-to-move in the real pass
FROM sub s JOIN playouts p ON p.node_id = s.node_id WHERE p.status = 'done' GROUP BY s.root;
```

(The link pass carries the honest caveat from the v2 plan: an interior node's `outcome_mean`
averages continuations from *imposed* prefixes under the expansion policy's mixture, not Pentobi's
own play from that node.)

### D-h. Migration: the v1 corpus enters the manifest, not the graph

The 13 k-game v1 corpus has uniform-random unharvested openings and one-hot targets — it is not
part of the searched strong-opening space and force-fitting it into the DAG would pollute every
coverage metric with junk-opening paths. It is registered as one row in `corpora`
(`dataset_kind = 'pentobi_distill_v1'`, path, counts), so "what do we hold?" queries see it, and
the trainer keeps consuming its shards directly as the optional mid-game mix arm. Nothing else.

---

## S1. Store module: schema + keys + seeds

`src/alphablokus/games/blokusduo/pentobi/store.py`: `SearchSpaceStore` (open/create with WAL,
schema versioning via `meta`), `canonical_key(compact) -> (key_bytes, key_frame)` as
`min(bytes, ascontiguousarray(grid.T).tobytes())`, `node_seed`/`playout_seed` as the low 31 bits
of `blake2b` over the stated content, insert/lookup helpers. Engine-free tests: key symmetry
property (a board and its transpose canonicalise identically), seed stability, idempotent insert.

## S2. Search recording

`record_search(node_id, move_values)` writes the node's searched fields and the **full** child
list as `edges` rows, mapping actions through `transpose_action` when `key_frame = 1`, and
**merging mirror-pair children** (summed visits, canonical action) when the node's own position
is symmetric (D-b). Fixture-driven tests on captured `move_values` output (same fixtures as the
v2 plan's V1 parser), including a transposed-frame case asserting byte-identical stored rows and
a symmetric-root case asserting pair-merged weights.

## S3. Allocation planning

The allocator (`compute_plan(budget, temperature, min_replicas)`): pure top-down budget split over
`edges` weights (`w ∝ visit_share^(1/T)`, keep children whose renormalised budget ≥ R, node
becomes a playout start when its budget can no longer be split), persisted as a `plans` +
`plan_nodes` row set; a *mapping queue* of nodes the plan needs searched but which aren't
(`search-on-demand`); `expand_child(parent, action)` inserts the child node (compute the child
board via the rules engine, canonicalise, extend the witness path — in the as-played frame) and
sets `edges.child_id`. `insert_book_paths(lines)` force-inserts the 44 book lines' nodes/edges
with `source = 'book'` and a floor of R planned games at each book-line terminal. Tests: the
allocator is a pure function (recompute on a synthetic DAG, assert identical `plan_nodes`);
top-up monotonicity (plan at 2B refines plan at B: internal stays internal, surviving starts'
planned games never decrease); largest-remainder integer rounding conserves the budget exactly;
book floors survive any (T, R).

## S4. Playout registry

`schedule(batch_size)` takes the lowest-fulfilment `(node, next replica)` jobs against the active
plan inside one transaction (assigning `game_id`s); `mark_done(node, replica, shard, margin,
plies)`; `reconcile(shard_dir)` rebuilds/verifies `playouts` from shard footers. Tests:
fulfilment ordering (a truncated schedule is an even proportional slice), crash-window repair
(delete the DB rows, reconcile, assert identical), structural disjointness (re-planning and
re-scheduling can never duplicate a `(node, replica)` pair, including across plans).

## S5. Export + link

`export-opening` materialises one parquet row per searched node — schema per the v2 plan's V6
(top-32 normalised target, `child_values`, derived `tail_mass`, `search_value`, `depth`,
`reach_weight`, `outcome_mean/count`) — stamped with `dag_hash`; the `link` pass runs the
recursive-CTE aggregation into `nodes.outcome_mean/outcome_count` first. Tests: export
round-trips through the trainer's reader; re-export after adding a node changes `dag_hash`.

## S6. Coverage report + v1 registration

`coverage` prints the D-f table (and `--json` for the report pipeline); one-off registration of
the v1 corpus into `corpora`. Tests on a synthetic store with known counts.
