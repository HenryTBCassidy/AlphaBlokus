"""The v2 corpus search-space store: a position-keyed DAG in SQLite.

Design and rationale: ``docs/plans/corpus-search-space-store.md`` (this module is its
S1–S6). One SQLite file holds **nodes** (positions Pentobi has searched or that a plan
wants searched), **edges** (the full ranked child list of every search), **plans** (each
budget-proportional allocation and its per-node game targets) and **playouts** (the
registry of generated games). Bulk training rows stay in parquet.

Four properties the rest of the v2 pipeline leans on:

- **Identity is positional and content-derived.** A node is its board bytes; a game is
  ``(board_key, replica)`` with an engine seed hashed from exactly that pair. "Have we
  done this?" is a primary-key lookup, and new work is disjoint from old work
  structurally rather than by bookkeeping.
- **Node keys are symmetry-canonical**: ``min(compact, transposed compact)``, with
  ``key_frame`` recording which of the two the key is. Blokus Duo's symmetry group is
  order 2 (the main diagonal), and of the 414 legal first moves only 212 are distinct
  under it — Pentobi itself searches both members of 310 mirror pairs at the root, so
  canonicalising reclaims nearly half of that effort. **Everything stored against a node
  — every edge action — lives in the node's key frame**; the witness path is the one
  exception and stays in the engine's as-played frame, because phase B replays it into
  the engine verbatim.
- **The allocator is a pure function** of (DAG content, budget, T, R):
  :func:`allocate_budget` walks the graph top-down and never touches SQL, so a plan can
  be recomputed and checked against its stored rows.
- **Scheduling is fulfilment-driven**, so a truncated generation run is an even
  proportional slice of the plan rather than a prefix of it.

The store owns a :class:`~alphablokus.games.blokusduo.game.BlokusDuoGame` because half
its work is rules-engine work: canonicalising a position, replaying a witness path,
expanding a child, checking that a recorded search's children are legal where they were
recorded. It never talks to the engine — searches are handed to it as
:class:`SearchChild` lists, which is what makes the whole module testable without a
``pentobi-gtp`` binary.
"""

from __future__ import annotations

import heapq
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import blake2b, sha256
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from alphablokus.games.blokusduo.board import BlokusDuoBoard
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.pentobi.move_values import MoveValues
    from alphablokus.games.blokusduo.pentobi.translation import PentobiMoveTranslator

#: Bumped whenever the DDL below changes incompatibly.
SCHEMA_VERSION = 1

#: Board/policy format markers, shared with the corpus shards (``corpus.BOARD_KIND``).
BOARD_KIND = "compact_v1"

#: Children kept per node in the exported soft policy target (top-32 by visits covers
#: >= 96.7% of visit mass at every measured ply — v2 plan fact 6).
STORE_K = 32

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS nodes (
    node_id         INTEGER PRIMARY KEY,
    board_key       BLOB    NOT NULL UNIQUE,
    key_frame       INTEGER NOT NULL,
    depth           INTEGER NOT NULL,
    player          INTEGER NOT NULL,
    witness_actions TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    book_terminal   INTEGER NOT NULL DEFAULT 0,
    engine_seed     INTEGER,
    root_visits     INTEGER,
    search_value    REAL,
    search_seconds  REAL,
    searched_at     TEXT,
    outcome_mean    REAL,
    outcome_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS nodes_depth_status ON nodes(depth, status);

CREATE TABLE IF NOT EXISTS edges (
    parent_id   INTEGER NOT NULL REFERENCES nodes(node_id),
    action      INTEGER NOT NULL,
    rank        INTEGER NOT NULL,
    visits      INTEGER NOT NULL,
    visit_share REAL    NOT NULL,
    child_value REAL    NOT NULL,
    child_id    INTEGER REFERENCES nodes(node_id),
    source      TEXT    NOT NULL,
    PRIMARY KEY (parent_id, action)
);
CREATE INDEX IF NOT EXISTS edges_child ON edges(child_id);

CREATE TABLE IF NOT EXISTS plans (
    plan_id      INTEGER PRIMARY KEY,
    created_at   TEXT    NOT NULL,
    budget       INTEGER NOT NULL,
    temperature  REAL    NOT NULL,
    min_replicas INTEGER NOT NULL,
    dag_hash     TEXT    NOT NULL,
    is_active    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS plan_nodes (
    plan_id       INTEGER NOT NULL REFERENCES plans(plan_id),
    node_id       INTEGER NOT NULL REFERENCES nodes(node_id),
    budget_share  REAL    NOT NULL,
    planned_games INTEGER NOT NULL,
    PRIMARY KEY (plan_id, node_id)
);

CREATE TABLE IF NOT EXISTS playouts (
    node_id      INTEGER NOT NULL REFERENCES nodes(node_id),
    replica      INTEGER NOT NULL,
    engine_seed  INTEGER NOT NULL,
    game_id      INTEGER UNIQUE,
    status       TEXT    NOT NULL,
    shard        TEXT,
    white_margin INTEGER,
    plies        INTEGER,
    completed_at TEXT,
    PRIMARY KEY (node_id, replica)
);

CREATE TABLE IF NOT EXISTS corpora (
    name TEXT PRIMARY KEY, path TEXT NOT NULL, dataset_kind TEXT NOT NULL,
    games INTEGER, positions INTEGER, notes TEXT
);

CREATE VIEW IF NOT EXISTS edge_disagreement AS
SELECT e.parent_id, e.action, e.rank, e.visit_share, e.child_value,
       1.0 - c.search_value AS independent_value,
       (1.0 - c.search_value) - e.child_value AS value_gap
FROM edges e JOIN nodes c ON c.node_id = e.child_id
WHERE c.status = 'searched';
"""


class StoreError(RuntimeError):
    """Raised on a store invariant violation (incompatible meta, desynced search, ...)."""


# --------------------------------------------------------------------------- #
# Keys and content-derived seeds (S1)
# --------------------------------------------------------------------------- #


def canonical_key(compact: NDArray[np.int8]) -> tuple[bytes, int]:
    """Symmetry-canonical node key for a compact placement grid.

    Returns ``(key_bytes, key_frame)`` where the key is the byte-wise smaller of the grid
    and its main-diagonal transpose, and ``key_frame`` is 0 when the key *is* the grid
    (identity) or 1 when the key is its transpose. A self-symmetric position reports
    frame 0.

    Args:
        compact: The **side-to-move canonical** int8 14x14 placement grid, i.e.
            ``game.get_canonical_form(board, player).to_compact()``.
    """
    grid = np.ascontiguousarray(compact, dtype=np.int8)
    identity = grid.tobytes()
    transposed = np.ascontiguousarray(grid.T).tobytes()
    return (identity, 0) if identity <= transposed else (transposed, 1)


def is_symmetric_key(board_key: bytes, board_size: int) -> bool:
    """Whether a node key is its own mirror (the position is diagonally symmetric)."""
    grid = np.frombuffer(board_key, dtype=np.int8).reshape(board_size, board_size)
    return bool(board_key == np.ascontiguousarray(grid.T).tobytes())


def hash64(data: bytes) -> int:
    """Stable 64-bit content hash (blake2b) — used for seeds and for tie-break ordering."""
    return int.from_bytes(blake2b(data, digest_size=8).digest(), "big")


def node_seed(board_key: bytes) -> int:
    """Engine seed for searching a position: a function of the position alone.

    A re-search of the same position (single-threaded engine) therefore reproduces the
    same tree, so a crash mid-search costs nothing and repair is idempotent.
    """
    return hash64(board_key) & 0x7FFFFFFF


def playout_seed(board_key: bytes, replica: int) -> int:
    """Engine seed for one game: a function of ``(position, replica)``.

    Replicas added months later under any plan draw seeds no earlier game used, which is
    what makes "generate more games" structurally incapable of reproducing a game we
    already hold.
    """
    return hash64(board_key + replica.to_bytes(8, "big")) & 0x7FFFFFFF


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NodeRecord:
    """One DAG node: a position, its provenance, and its search results."""

    node_id: int
    board_key: bytes
    key_frame: int  # 0 = key is the as-played grid, 1 = key is its transpose
    depth: int  # plies from the empty board (= pieces placed while nobody has passed)
    player: int  # side to move: +1 White, -1 Black
    witness_actions: tuple[int, ...]  # root -> node, in the engine's as-played frame
    source: str  # 'root' | 'search' | 'book' | 'net'
    status: str  # 'pending' | 'searched'
    book_terminal: bool
    engine_seed: int | None
    root_visits: int | None
    search_value: float | None
    search_seconds: float | None
    searched_at: str | None
    outcome_mean: float | None
    outcome_count: int

    @property
    def is_searched(self) -> bool:
        return self.status == "searched"


@dataclass(frozen=True)
class EdgeRecord:
    """One candidate move of a searched node, in the parent's **key frame**."""

    parent_id: int
    action: int
    rank: int  # 0-based visit rank after mirror-pair merging
    visits: int
    visit_share: float
    child_value: float
    child_id: int | None
    source: str


@dataclass(frozen=True)
class SearchChild:
    """A ``move_values`` child as the engine reported it, in the **as-played** frame."""

    action: int
    visits: int
    value: float


def children_from_move_values(values: MoveValues, translator: PentobiMoveTranslator) -> tuple[SearchChild, ...]:
    """Turn a parsed ``move_values`` response into store-ready children.

    Cells are translated in the frame the engine played them in; :meth:`
    SearchSpaceStore.record_search` maps them into the node's key frame.
    """
    return tuple(
        SearchChild(
            action=translator.pentobi_to_action_index(entry.cells),
            visits=entry.visits,
            value=entry.value,
        )
        for entry in values.entries
    )


@dataclass(frozen=True)
class PlanParameters:
    """The three knobs of a budget-proportional allocation (v2 plan V4)."""

    budget: int  # B: total games
    temperature: float  # T: weights are visit_share ** (1 / T)
    min_replicas: int  # R: a child needs >= R games to survive; a node splits at >= 2R

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError(f"budget must be positive, got {self.budget}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.min_replicas < 1:
            raise ValueError(f"min_replicas must be >= 1, got {self.min_replicas}")


@dataclass(frozen=True)
class PlanRecord:
    """A persisted allocation run."""

    plan_id: int
    created_at: str
    parameters: PlanParameters
    dag_hash: str
    is_active: bool


@dataclass(frozen=True)
class PlanAllocation:
    """One node's share of a plan."""

    node_id: int
    budget_share: float  # fraction of the total budget reaching this node
    planned_games: int  # games to START here (0 for internal nodes)


@dataclass(frozen=True)
class BudgetSplit:
    """The pure allocator's output before integerisation."""

    budgets: dict[int, float]  # every node the walk reached -> real games reaching it
    starts: dict[int, float]  # playout starts -> real games starting there
    mapping_queue: tuple[int, ...]  # nodes the plan wants searched but which are not


@dataclass(frozen=True)
class PlanDraft:
    """A computed (not yet persisted) plan."""

    parameters: PlanParameters
    allocations: tuple[PlanAllocation, ...]
    mapping_queue: tuple[int, ...]

    @property
    def planned_games(self) -> int:
        return sum(allocation.planned_games for allocation in self.allocations)

    @property
    def starts(self) -> tuple[PlanAllocation, ...]:
        return tuple(allocation for allocation in self.allocations if allocation.planned_games > 0)


@dataclass(frozen=True)
class PlayoutJob:
    """One game to generate: which position, which replica, which seed."""

    node_id: int
    replica: int
    game_id: int
    engine_seed: int
    board_key: bytes
    witness_actions: tuple[int, ...]  # as-played prefix phase B replays with ``play``


@dataclass(frozen=True)
class ReconcileEntry:
    """A game as a shard footer describes it — the input to :meth:`SearchSpaceStore.reconcile`."""

    board_key: bytes
    replica: int
    game_id: int
    shard: str
    white_margin: int
    plies: int
    #: The DAG the shard was written against, when the footer records one. Lets
    #: :meth:`SearchSpaceStore.reconcile` notice a shard produced by a different corpus.
    dag_hash: str | None = None


@dataclass(frozen=True)
class ReconcileResult:
    """What :meth:`SearchSpaceStore.reconcile` had to change."""

    matched: int
    inserted: int
    updated: int
    unknown_nodes: tuple[bytes, ...]  # footers referencing positions the DAG does not hold


@dataclass(frozen=True)
class OpeningRow:
    """One exported opening-dataset row (a searched DAG node)."""

    node_id: int
    parent_id: int | None
    board: bytes  # key-frame compact board
    policy_indices: NDArray[np.int32]  # top-K children by visits, key frame
    policy_values: NDArray[np.float32]  # renormalised over the kept children
    child_values: NDArray[np.float32]  # Pentobi's per-child value, aligned to the indices
    tail_mass: float  # visit mass dropped by the top-K truncation
    search_value: float
    depth: int
    player: int
    reach_weight: float
    budget_share: float
    planned_games: int
    outcome_mean: float
    outcome_count: int


@dataclass(frozen=True)
class CoverageReport:
    """The "what have we done?" screen (store design D-f)."""

    dag_hash: str
    plan: PlanRecord | None
    nodes_by_depth: dict[int, int]
    searched_by_depth: dict[int, int]
    starts_by_depth: dict[int, int]
    planned_games: int
    actual_games: int
    mapping_debt: int  # planned nodes the plan wants searched but which are not
    fulfilment_histogram: dict[str, int]
    distinct_first_moves: int  # distinct ply-1 actions reachable in the plan
    distinct_first_positions: int  # ... after mirror canonicalisation (of 212)
    play_mass_by_depth: dict[int, float]
    planned_games_min: int
    planned_games_median: float
    planned_games_max: int
    budget_share_by_depth: dict[int, float]

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form (for ``coverage --json`` and the report pipeline)."""
        return {
            "dag_hash": self.dag_hash,
            "plan_id": self.plan.plan_id if self.plan else None,
            "plan_parameters": None
            if self.plan is None
            else {
                "budget": self.plan.parameters.budget,
                "temperature": self.plan.parameters.temperature,
                "min_replicas": self.plan.parameters.min_replicas,
            },
            "nodes_by_depth": {str(k): v for k, v in self.nodes_by_depth.items()},
            "searched_by_depth": {str(k): v for k, v in self.searched_by_depth.items()},
            "starts_by_depth": {str(k): v for k, v in self.starts_by_depth.items()},
            "planned_games": self.planned_games,
            "actual_games": self.actual_games,
            "fulfilment": self.actual_games / self.planned_games if self.planned_games else 0.0,
            "mapping_debt": self.mapping_debt,
            "fulfilment_histogram": self.fulfilment_histogram,
            "distinct_first_moves": self.distinct_first_moves,
            "distinct_first_positions": self.distinct_first_positions,
            "play_mass_by_depth": {str(k): v for k, v in self.play_mass_by_depth.items()},
            "planned_games_min": self.planned_games_min,
            "planned_games_median": self.planned_games_median,
            "planned_games_max": self.planned_games_max,
            "budget_share_by_depth": {str(k): v for k, v in self.budget_share_by_depth.items()},
        }


# --------------------------------------------------------------------------- #
# The pure allocator (S3)
# --------------------------------------------------------------------------- #


def allocate_budget(
    root: int,
    params: PlanParameters,
    *,
    depth_of: Callable[[int], int],
    is_searched: Callable[[int], bool],
    children_of: Callable[[int], Sequence[tuple[int, float]]],
    instantiate: Callable[[int, int], int],
) -> BudgetSplit:
    """Split a game budget recursively down the DAG — the v2 allocation rule.

    At a node holding budget ``b``:

    1. ``b < 2R`` ⇒ the node is a **playout start** and keeps its games.
    2. Otherwise weight its children ``w ∝ visit_share ** (1/T)`` — Pentobi's own
       opinion, flattened, never an independent evaluation — drop children whose
       renormalised share of ``b`` is below ``R``, renormalise over the survivors, and
       recurse. Dropping only ever *raises* the survivors' budgets, so one pass suffices.
    3. **Fewer than two survivors ⇒ the node is a playout start.** The v2 plan's rule
       only says "if none survive"; a *single* survivor would inherit the parent's whole
       budget undiminished and split again on the same terms, walking one forced line to
       the end of the game and never terminating. Requiring two survivors makes each
       child's budget at most ``b − R``, which bounds the descent, and costs nothing: a
       split into one piece moves the same games one ply deeper for no extra coverage.
    4. A node the walk wants to split but which has not been searched joins the
       **mapping queue** and is treated as a provisional start, so a plan computed
       against an incomplete DAG is still a usable (if shallow) plan.

    Depth is an *output*: it emerges from ``(B, T, R)`` and the shape of Pentobi's own
    visit distributions. The walk proceeds in depth order, which is well defined because
    every edge goes from depth ``d`` to ``d + 1`` — that also makes a DAG node reached
    from several parents accumulate its parents' budgets before it splits.

    Args:
        root: Node id the whole budget starts at.
        params: ``(budget, temperature, min_replicas)``.
        depth_of: Node id → plies from the empty board.
        is_searched: Node id → whether its children are known.
        children_of: Node id → ``(action in the node's key frame, visit_share)`` pairs.
            Called only for searched nodes, and must not instantiate anything.
        instantiate: ``(parent node id, action)`` → child node id. Called only for
            children that survive the split, which is what keeps the DAG as small as the
            plan needs.
    """
    exponent = 1.0 / params.temperature
    floor = float(params.min_replicas)
    pending: dict[int, dict[int, float]] = {depth_of(root): {root: float(params.budget)}}
    budgets: dict[int, float] = {}
    starts: dict[int, float] = {}
    mapping_queue: list[int] = []

    while pending:
        depth = min(pending)
        level = pending.pop(depth)
        for node_id in sorted(level):
            node_budget = level[node_id]
            budgets[node_id] = budgets.get(node_id, 0.0) + node_budget
            if node_budget < 2 * floor:
                starts[node_id] = starts.get(node_id, 0.0) + node_budget
                continue
            if not is_searched(node_id):
                mapping_queue.append(node_id)
                starts[node_id] = starts.get(node_id, 0.0) + node_budget
                continue
            shares = _surviving_children(children_of(node_id), node_budget, exponent, floor)
            if len(shares) < 2:
                starts[node_id] = starts.get(node_id, 0.0) + node_budget
                continue
            for action, child_budget in shares:
                child_id = instantiate(node_id, action)
                child_depth = depth_of(child_id)
                if child_depth <= depth:  # pragma: no cover — a pass child would loop
                    logger.warning("skipping child {} of node {}: depth did not increase", child_id, node_id)
                    continue
                bucket = pending.setdefault(child_depth, {})
                bucket[child_id] = bucket.get(child_id, 0.0) + child_budget

    return BudgetSplit(budgets=budgets, starts=starts, mapping_queue=tuple(mapping_queue))


def _surviving_children(
    children: Sequence[tuple[int, float]],
    node_budget: float,
    exponent: float,
    floor: float,
) -> list[tuple[int, float]]:
    """Flatten, drop the children below the split floor, renormalise over the rest."""
    weights = [(action, share**exponent) for action, share in children if share > 0.0]
    total = sum(weight for _, weight in weights)
    if total <= 0.0:
        return []
    survivors = [(action, weight) for action, weight in weights if node_budget * weight / total >= floor]
    surviving_total = sum(weight for _, weight in survivors)
    if surviving_total <= 0.0:
        return []
    return [(action, node_budget * weight / surviving_total) for action, weight in survivors]


def integerise_budgets(starts: dict[int, float], total: int) -> dict[int, int]:
    """Round real per-node budgets to whole games, conserving ``total`` exactly.

    Largest-remainder rounding: floor everything, then hand the leftover games to the
    largest fractional parts (ties broken by node id, so the result is deterministic).
    """
    if not starts:
        return {}
    floors = {node_id: int(budget) for node_id, budget in starts.items()}
    leftover = total - sum(floors.values())
    if leftover > 0:
        order = sorted(starts, key=lambda node_id: (-(starts[node_id] % 1.0), node_id))
        for node_id in order[:leftover]:
            floors[node_id] += 1
    return floors


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


class SearchSpaceStore:
    """The SQLite-backed search-space DAG, allocation plans and playout registry."""

    def __init__(
        self,
        path: Path,
        game: BlokusDuoGame,
        *,
        level: int | None = 9,
        engine_version: str = "unknown",
    ) -> None:
        """Open (or create) the store at ``path``.

        Args:
            path: The ``.sqlite`` file; created with its parent directories if missing.
            game: The rules engine — canonicalisation, witness replay, child expansion.
            level: Pentobi level the searches come from. Asserted on reopen: mixing
                levels in one DAG would silently mix two different teachers.
            engine_version: Engine build identifier, recorded for provenance (a mismatch
                warns rather than raises — an unknown version is not a corruption).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._game = game
        self._path = path
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(_SCHEMA)
        self._sync_meta(level=level, engine_version=engine_version)
        self._connection.commit()

    # -- lifecycle -------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def game(self) -> BlokusDuoGame:
        return self._game

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    def __enter__(self) -> SearchSpaceStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _sync_meta(self, *, level: int | None, engine_version: str) -> None:
        """Write the pinned metadata on creation; assert compatibility on reopen.

        ``level`` may be ``None`` for commands that never touch the engine (export, link,
        coverage, analyze, validate): they have no opinion about the level, so imposing
        the CLI default would make them refuse to open a store built at any other one.
        """
        expected = {
            "schema_version": str(SCHEMA_VERSION),
            "game": "blokusduo",
            "board_kind": BOARD_KIND,
            "policy_size": str(self._game.get_action_size()),
        }
        if level is not None:
            expected["level"] = str(level)
        stored = {str(row["key"]): str(row["value"]) for row in self._connection.execute("SELECT key, value FROM meta")}
        if not stored:
            if level is None:
                raise StoreError(f"{self._path.name}: refusing to create a store without a level")
            rows = [*expected.items(), ("engine_version", engine_version), ("created_at", _now())]
            self._connection.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", rows)
            return
        for key, value in expected.items():
            if stored.get(key) != value:
                raise StoreError(f"{self._path.name}: incompatible store — {key} is {stored.get(key)!r}, not {value!r}")
        if engine_version != "unknown" and stored.get("engine_version") not in ("unknown", engine_version):
            logger.warning(
                "{}: engine_version {} differs from the stored {}",
                self._path.name,
                engine_version,
                stored.get("engine_version"),
            )

    @property
    def meta(self) -> dict[str, str]:
        """The pinned metadata (schema version, game, board kind, policy size, level)."""
        return {str(row["key"]): str(row["value"]) for row in self._connection.execute("SELECT key, value FROM meta")}

    # -- nodes (S1) ------------------------------------------------------------

    def upsert_node(
        self,
        board: BlokusDuoBoard,
        player: int,
        witness_actions: Sequence[int],
        *,
        source: str = "search",
    ) -> int:
        """Insert a position (or return the existing node id for it).

        Idempotent by construction: the node key is the position, so re-inserting a
        position reached by a different move order is a no-op that returns the original
        node — including its original witness path, which stays the canonical one.

        Args:
            board: The position, in the engine's as-played frame.
            player: Side to move there (+1 White, -1 Black).
            witness_actions: One as-played action sequence from the empty board.
            source: 'root' | 'search' | 'book' | 'net'.
        """
        compact = np.asarray(self._game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
        board_key, key_frame = canonical_key(compact)
        existing = self.node_by_key(board_key)
        if existing is not None:
            return existing.node_id
        cursor = self._connection.execute(
            "INSERT INTO nodes (board_key, key_frame, depth, player, witness_actions, source, status, engine_seed) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (
                board_key,
                key_frame,
                len(witness_actions),
                player,
                json.dumps([int(action) for action in witness_actions]),
                source,
                node_seed(board_key),
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid or 0)

    def root_node(self) -> int:
        """The empty board with White to move — every witness path starts here."""
        return self.upsert_node(self._game.initialise_board(), 1, (), source="root")

    def node(self, node_id: int) -> NodeRecord:
        """Look a node up by id (raises if it does not exist)."""
        row = self._connection.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            raise StoreError(f"no node {node_id}")
        return _node_from_row(row)

    def node_by_key(self, board_key: bytes) -> NodeRecord | None:
        """Look a node up by position — the "have we been here?" query."""
        row = self._connection.execute("SELECT * FROM nodes WHERE board_key = ?", (board_key,)).fetchone()
        return None if row is None else _node_from_row(row)

    def nodes(self, *, status: str | None = None) -> list[NodeRecord]:
        """Every node, optionally filtered by status, in insertion order."""
        if status is None:
            rows = self._connection.execute("SELECT * FROM nodes ORDER BY node_id")
        else:
            rows = self._connection.execute("SELECT * FROM nodes WHERE status = ? ORDER BY node_id", (status,))
        return [_node_from_row(row) for row in rows]

    def board_at(self, node_id: int) -> tuple[BlokusDuoBoard, int]:
        """Rebuild a node's position by replaying its witness path (as-played frame).

        The DAG has no unique move order into a node, so the *stored* witness is the
        canonical one: it is what phase B replays into the engine and what every action
        recorded against the node is expressed relative to.
        """
        record = self.node(node_id)
        board = self._game.initialise_board()
        player = 1
        for action in record.witness_actions:
            board, player = self._game.get_next_state(board, player, action)
        if player != record.player:
            raise StoreError(f"node {node_id}: witness path ends with player {player}, stored {record.player}")
        return board, player

    def to_key_frame(self, record: NodeRecord, action: int) -> int:
        """Map an as-played action into a node's key frame (identity when frame 0)."""
        return self._game.transpose_action(action) if record.key_frame else action

    def from_key_frame(self, record: NodeRecord, action: int) -> int:
        """Map a key-frame action back into the node's as-played frame.

        The main-diagonal reflection is an involution, so this is the same mapping as
        :meth:`to_key_frame` — kept as a separate name because call sites read very
        differently and a silent frame confusion is the store's likeliest subtle bug.
        """
        return self._game.transpose_action(action) if record.key_frame else action

    def is_symmetric(self, record: NodeRecord) -> bool:
        """Whether the node's position is its own mirror (the root and few others)."""
        return is_symmetric_key(record.board_key, self._game.board_size)

    # -- searches (S2) ---------------------------------------------------------

    def record_search(
        self,
        node_id: int,
        children: Sequence[SearchChild],
        *,
        seconds: float | None = None,
        search_value: float | None = None,
        source: str = "search",
        validate: bool = True,
    ) -> int:
        """Store one Pentobi search: the node's fields and its **full** child list.

        Three things happen here and nowhere else:

        - **Frame mapping.** Children arrive in the engine's as-played frame and are
          stored in the node's key frame (``transpose_action`` when ``key_frame = 1``).
        - **Mirror-pair merging.** When the node's own position is symmetric — the root,
          in practice — a child and its mirror are the same canonical position, so they
          are merged into one edge carrying the pair's **combined** visits. Without this
          every canonical opening would be charged half its true visit share, and the
          allocator would systematically under-fund the openings the engine likes most.
        - **Ranking.** Edges are re-ranked by merged visits (ties by action) so ``rank``
          means the same thing at symmetric and asymmetric nodes.

        Re-recording a search replaces the node's edges (idempotent repair after a crash)
        while preserving any ``child_id`` links already established.

        Args:
            node_id: The searched node.
            children: All ``move_values`` children, as-played frame.
            seconds: Wall-clock the search took, for the cost model.
            search_value: The position's backed-up value; defaults to the top merged
                child's value (never GTP ``get_value``, which is a constant 0).
            source: Edge provenance ('search' | 'book' | 'net').
            validate: Check every child is legal at the node's replayed position — the
                guard against recording a search taken at the node's *mirror twin*,
                whose actions would be in the wrong frame.

        Returns:
            The number of edges written (post-merge).
        """
        record = self.node(node_id)
        if validate:
            self._validate_children(record, children)
        merged = self._merge_children(record, children)
        root_visits = sum(visits for _, visits, _ in merged)
        merged.sort(key=lambda item: (-item[1], item[0]))
        existing_children = {
            int(row["action"]): row["child_id"]
            for row in self._connection.execute("SELECT action, child_id FROM edges WHERE parent_id = ?", (node_id,))
        }
        # Book edges the engine did not report must survive the replace. The 44 curated
        # lines are force-inserted precisely because Pentobi may not favour them, so
        # deleting them here would orphan their children from the graph the moment their
        # parent is searched (and the root always is): the child keeps its games via
        # ``nodes.book_terminal`` but drops out of ``reach_weights``, out of ``link``'s
        # aggregation, and out of the export's ancestry, silently.
        merged_actions = {action for action, _, _ in merged}
        preserved = [
            (int(row["action"]), row["child_id"], row["child_value"])
            for row in self._connection.execute(
                "SELECT action, child_id, child_value FROM edges WHERE parent_id = ? AND source = 'book'",
                (node_id,),
            )
            if int(row["action"]) not in merged_actions
        ]
        self._connection.execute("DELETE FROM edges WHERE parent_id = ?", (node_id,))
        self._connection.executemany(
            "INSERT INTO edges (parent_id, action, rank, visits, visit_share, child_value, child_id, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    node_id,
                    action,
                    rank,
                    visits,
                    visits / root_visits if root_visits else 0.0,
                    value,
                    existing_children.get(action),
                    source,
                )
                for rank, (action, visits, value) in enumerate(merged)
            ],
        )
        self._connection.executemany(
            "INSERT INTO edges (parent_id, action, rank, visits, visit_share, child_value, child_id, source) "
            "VALUES (?, ?, ?, 0, 0.0, ?, ?, 'book')",
            [
                (node_id, action, len(merged) + offset, value, child_id)
                for offset, (action, child_id, value) in enumerate(preserved)
            ],
        )
        top_value = merged[0][2] if merged else None
        self._connection.execute(
            "UPDATE nodes SET status = 'searched', root_visits = ?, search_value = ?, search_seconds = ?, "
            "searched_at = ? WHERE node_id = ?",
            (root_visits, search_value if search_value is not None else top_value, seconds, _now(), node_id),
        )
        self._connection.commit()
        return len(merged)

    def _validate_children(self, record: NodeRecord, children: Sequence[SearchChild]) -> None:
        """Assert every reported child is legal at the node's own replayed position."""
        board, player = self.board_at(record.node_id)
        mask = self._game.valid_move_masking(board, player)
        illegal = [child.action for child in children if child.action < 0 or not mask[child.action]]
        if illegal:
            raise StoreError(
                f"node {record.node_id}: {len(illegal)} recorded children are illegal at its witness position "
                f"(first: {illegal[0]}) — the search was taken at a different position or in the wrong frame",
            )

    def _merge_children(self, record: NodeRecord, children: Sequence[SearchChild]) -> list[tuple[int, int, float]]:
        """Map children to the key frame, merging mirror pairs at symmetric nodes."""
        symmetric = self.is_symmetric(record)
        merged: dict[int, tuple[int, float, int]] = {}  # action -> (visits, value, best member's visits)
        for child in children:
            action = self.to_key_frame(record, child.action)
            if symmetric:
                action = min(action, self._game.transpose_action(action))
            previous = merged.get(action)
            if previous is None:
                merged[action] = (child.visits, child.value, child.visits)
                continue
            visits, value, best = previous
            # Keep the better-supported member's value: an unvisited mirror twin reports
            # the prior, not a search result.
            if child.visits > best:
                value = child.value
                best = child.visits
            merged[action] = (visits + child.visits, value, best)
        return [(action, visits, value) for action, (visits, value, _) in merged.items()]

    def edges(self, node_id: int) -> list[EdgeRecord]:
        """A searched node's candidate moves, best first (key frame)."""
        rows = self._connection.execute("SELECT * FROM edges WHERE parent_id = ? ORDER BY rank", (node_id,))
        return [_edge_from_row(row) for row in rows]

    def dag_hash(self) -> str:
        """Content hash over every searched node's key, root visits and ranked edges.

        Stamped into exports and plans so a stale export or a plan computed against an
        older DAG is detectable rather than merely suspected.
        """
        digest = sha256()
        rows = self._connection.execute(
            "SELECT node_id, board_key, root_visits FROM nodes WHERE status = 'searched' ORDER BY board_key",
        )
        for row in rows.fetchall():
            digest.update(bytes(row["board_key"]))
            digest.update(str(row["root_visits"]).encode())
            for edge in self._connection.execute(
                "SELECT action, visits FROM edges WHERE parent_id = ? ORDER BY rank",
                (int(row["node_id"]),),
            ):
                digest.update(f"{int(edge['action'])}:{int(edge['visits'])}".encode())
        value = digest.hexdigest()
        # Remember every hash we have ever emitted, so a shard written by a *different*
        # corpus can be spotted later. The DAG's hash changes as it grows, so a shard is
        # legitimately older than the current hash — equality is the wrong test; membership
        # of this set is the right one.
        self._connection.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, '1')",
            (f"dag_hash:{value}",),
        )
        self._connection.commit()
        return value

    # -- expansion + planning (S3) --------------------------------------------

    def expand_child(self, parent_id: int, action: int, *, source: str = "search") -> int:
        """Instantiate the child a parent's edge points at, and link the edge to it.

        Args:
            parent_id: The searched parent.
            action: The edge's action, in the **parent's key frame** (as stored).
            source: Provenance for a newly created child node.

        Returns:
            The child node's id (existing or freshly inserted — a transposed-into
            position costs no extra node and no extra search).
        """
        record = self.node(parent_id)
        board, player = self.board_at(parent_id)
        as_played = self.from_key_frame(record, action)
        child_board, child_player = self._game.get_next_state(board, player, as_played)
        child_id = self.upsert_node(
            child_board,
            child_player,
            (*record.witness_actions, as_played),
            source=source,
        )
        self._connection.execute(
            "UPDATE edges SET child_id = ? WHERE parent_id = ? AND action = ?",
            (child_id, parent_id, action),
        )
        self._connection.commit()
        return child_id

    def compute_plan(self, params: PlanParameters) -> PlanDraft:
        """Allocate ``params.budget`` games over the DAG (v2 plan V4).

        Runs :func:`allocate_budget` against the live DAG, instantiating child nodes for
        the branches that survive the split — mapping exactly as far as the plan needs
        and no further. Nodes the plan wants to split but which are unsearched come back
        in :attr:`PlanDraft.mapping_queue`; search them and recompute until the queue is
        empty (that loop is the ``plan`` CLI's job, because searching needs the engine).

        Book lines (:meth:`insert_book_paths`) get a floor of ``R`` games at each line's
        terminal node. The floor is **reserved out of the budget first**, so the total
        planned games still equal ``budget`` exactly — which also means ``budget_share``
        is measured against the full budget and the root's share sits just below 1 when
        book floors are reserved.
        """
        book_terminals = [record.node_id for record in self._book_terminals()]
        reserved = params.min_replicas * len(book_terminals)
        if reserved >= params.budget:
            raise StoreError(
                f"book floors reserve {reserved} games of a {params.budget}-game budget — raise the budget",
            )
        split = allocate_budget(
            self.root_node(),
            PlanParameters(params.budget - reserved, params.temperature, params.min_replicas),
            depth_of=lambda node_id: self.node(node_id).depth,
            is_searched=lambda node_id: self.node(node_id).is_searched,
            children_of=lambda node_id: [(edge.action, edge.visit_share) for edge in self.edges(node_id)],
            instantiate=self.expand_child,
        )
        planned = integerise_budgets(split.starts, params.budget - reserved)
        for node_id in book_terminals:
            planned[node_id] = planned.get(node_id, 0) + params.min_replicas
            split.budgets.setdefault(node_id, 0.0)
        allocations = tuple(
            PlanAllocation(
                node_id=node_id,
                budget_share=split.budgets[node_id] / params.budget,
                planned_games=planned.get(node_id, 0),
            )
            for node_id in sorted(split.budgets)
        )
        return PlanDraft(parameters=params, allocations=allocations, mapping_queue=split.mapping_queue)

    def save_plan(self, draft: PlanDraft, *, activate: bool = True) -> int:
        """Persist a plan and (by default) make it the active one.

        Exactly one plan is active; deficits, scheduling and coverage all read from it.
        Actuals are never touched: a re-plan at a different ``(B, T, R)`` adds targets,
        it never invalidates a game.
        """
        params = draft.parameters
        cursor = self._connection.execute(
            "INSERT INTO plans (created_at, budget, temperature, min_replicas, dag_hash, is_active) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (_now(), params.budget, params.temperature, params.min_replicas, self.dag_hash()),
        )
        plan_id = int(cursor.lastrowid or 0)
        self._connection.executemany(
            "INSERT INTO plan_nodes (plan_id, node_id, budget_share, planned_games) VALUES (?, ?, ?, ?)",
            [(plan_id, a.node_id, a.budget_share, a.planned_games) for a in draft.allocations],
        )
        if activate:
            self.activate_plan(plan_id)
        self._connection.commit()
        return plan_id

    def activate_plan(self, plan_id: int) -> None:
        """Make ``plan_id`` the single active plan."""
        self._connection.execute("UPDATE plans SET is_active = 0")
        self._connection.execute("UPDATE plans SET is_active = 1 WHERE plan_id = ?", (plan_id,))
        self._connection.commit()

    def active_plan(self) -> PlanRecord | None:
        """The plan generation is currently fulfilling, if any."""
        row = self._connection.execute("SELECT * FROM plans WHERE is_active = 1").fetchone()
        return None if row is None else _plan_from_row(row)

    def plan_allocations(self, plan_id: int) -> list[PlanAllocation]:
        """Every node's allocation under a plan, in node order."""
        rows = self._connection.execute(
            "SELECT node_id, budget_share, planned_games FROM plan_nodes WHERE plan_id = ? ORDER BY node_id",
            (plan_id,),
        )
        return [
            PlanAllocation(int(row["node_id"]), float(row["budget_share"]), int(row["planned_games"])) for row in rows
        ]

    def insert_book_paths(self, lines: Sequence[Sequence[int]], *, source: str = "book") -> list[int]:
        """Force-insert the opening book's lines as nodes and edges.

        Book lines are mapped regardless of what Pentobi's own search likes (the engine
        itself stays ``--nobook`` throughout — a book hit returns a move with no search
        tree and would break harvesting). Each line's terminal node is flagged so
        :meth:`compute_plan` can floor it at ``R`` games.

        Args:
            lines: One as-played action sequence per book line, from the empty board.
            source: Provenance recorded on the nodes and edges created here.

        Returns:
            One terminal node id per line, in input order.
        """
        terminals: list[int] = []
        for actions in lines:
            board = self._game.initialise_board()
            player = 1
            witness: list[int] = []
            parent_id = self.root_node()
            for action in actions:
                parent_record = self.node(parent_id)
                key_action = self.to_key_frame(parent_record, int(action))
                witness.append(int(action))
                board, player = self._game.get_next_state(board, player, int(action))
                child_id = self.upsert_node(board, player, tuple(witness), source=source)
                self._connection.execute(
                    "INSERT OR IGNORE INTO edges "
                    "(parent_id, action, rank, visits, visit_share, child_value, child_id, source) "
                    "VALUES (?, ?, ?, 0, 0.0, 0.0, ?, ?)",
                    (parent_id, key_action, self._next_edge_rank(parent_id), child_id, source),
                )
                self._connection.execute(
                    "UPDATE edges SET child_id = ? WHERE parent_id = ? AND action = ? AND child_id IS NULL",
                    (child_id, parent_id, key_action),
                )
                parent_id = child_id
            self._connection.execute("UPDATE nodes SET book_terminal = 1 WHERE node_id = ?", (parent_id,))
            terminals.append(parent_id)
        self._connection.commit()
        return terminals

    def _next_edge_rank(self, parent_id: int) -> int:
        """Rank for an edge appended to a node: after every edge already stored there."""
        row = self._connection.execute(
            "SELECT COALESCE(MAX(rank), -1) + 1 AS next FROM edges WHERE parent_id = ?",
            (parent_id,),
        ).fetchone()
        return int(row["next"])

    def _book_terminals(self) -> list[NodeRecord]:
        """Nodes flagged as the end of a book line (they carry a games floor)."""
        rows = self._connection.execute("SELECT * FROM nodes WHERE book_terminal = 1 ORDER BY node_id")
        return [_node_from_row(row) for row in rows]

    # -- playout registry (S4) -------------------------------------------------

    def schedule(self, batch_size: int) -> list[PlayoutJob]:
        """Claim the next ``batch_size`` games, lowest plan fulfilment first.

        Ordering is ``(existing games ÷ planned games) ascending, then hash64(board_key)``
        against the **active** plan, so every node's fulfilment rises together: truncate a
        run at any point and what you hold is an even proportional slice of the plan, and
        nodes a re-plan just added (fulfilment 0) are covered first automatically.

        Each job is a ``(node, replica)`` pair inserted against the playouts primary key
        — it cannot collide with an existing game, under this plan or any other.
        """
        plan = self.active_plan()
        if plan is None:
            raise StoreError("no active plan — compute and save one before scheduling games")
        rows = self._connection.execute(
            "SELECT pn.node_id AS node_id, pn.planned_games AS planned, n.board_key AS board_key, "
            "       COUNT(p.replica) AS existing, COALESCE(MAX(p.replica), -1) AS last_replica "
            "FROM plan_nodes pn "
            "JOIN nodes n ON n.node_id = pn.node_id "
            "LEFT JOIN playouts p ON p.node_id = pn.node_id "
            "WHERE pn.plan_id = ? AND pn.planned_games > 0 "
            "GROUP BY pn.node_id HAVING existing < planned",
            (plan.plan_id,),
        ).fetchall()
        next_game_id = self._next_game_id()
        jobs: list[PlayoutJob] = []
        # A heap re-keyed on **fulfilment fraction** after every assignment. Walking a
        # once-sorted list and taking one replica per node per pass is equal-*count*
        # water-filling, not equal-*fraction*: with 2–32 games per start it drives every
        # 2-game start to 100% while a 32-game mainline sits at 6%, so a truncated run
        # over-represents the flattened tail — the reverse of what the allocation is for.
        planned_by_node = {int(row["node_id"]): int(row["planned"]) for row in rows}
        remaining = {
            int(row["node_id"]): (int(row["planned"]) - int(row["existing"]), int(row["last_replica"]) + 1)
            for row in rows
        }
        heap = [
            (int(row["existing"]) / int(row["planned"]), hash64(bytes(row["board_key"])), int(row["node_id"]))
            for row in rows
        ]
        heapq.heapify(heap)
        while heap and len(jobs) < batch_size:
            _, tiebreak, node_id = heapq.heappop(heap)
            deficit, replica = remaining[node_id]
            if deficit <= 0:
                continue
            record = self.node(node_id)
            seed = playout_seed(record.board_key, replica)
            self._connection.execute(
                "INSERT INTO playouts (node_id, replica, engine_seed, game_id, status) VALUES (?, ?, ?, ?, ?)",
                (node_id, replica, seed, next_game_id, "planned"),
            )
            jobs.append(
                PlayoutJob(
                    node_id=node_id,
                    replica=replica,
                    game_id=next_game_id,
                    engine_seed=seed,
                    board_key=record.board_key,
                    witness_actions=record.witness_actions,
                ),
            )
            remaining[node_id] = (deficit - 1, replica + 1)
            next_game_id += 1
            if deficit - 1 > 0:
                done = planned_by_node[node_id] - (deficit - 1)
                heapq.heappush(heap, (done / planned_by_node[node_id], tiebreak, node_id))
        self._connection.commit()
        return jobs

    def pending_jobs(self, limit: int | None = None) -> list[PlayoutJob]:
        """Games already claimed but not finished — what a resumed run re-executes.

        Same seeds ⇒ same games ⇒ exactly the missing shards are regenerated.
        """
        query = (
            "SELECT p.node_id, p.replica, p.game_id, p.engine_seed, n.board_key, n.witness_actions "
            "FROM playouts p JOIN nodes n ON n.node_id = p.node_id "
            "WHERE p.status = 'planned' ORDER BY p.game_id"
        )
        rows = self._connection.execute(query if limit is None else f"{query} LIMIT {int(limit)}")
        return [
            PlayoutJob(
                node_id=int(row["node_id"]),
                replica=int(row["replica"]),
                game_id=int(row["game_id"]),
                engine_seed=int(row["engine_seed"]),
                board_key=bytes(row["board_key"]),
                witness_actions=tuple(json.loads(str(row["witness_actions"]))),
            )
            for row in rows
        ]

    def mark_done(self, node_id: int, replica: int, *, shard: str, white_margin: int, plies: int) -> None:
        """Record a finished game against its ``(node, replica)`` slot."""
        cursor = self._connection.execute(
            "UPDATE playouts SET status = 'done', shard = ?, white_margin = ?, plies = ?, completed_at = ? "
            "WHERE node_id = ? AND replica = ?",
            (shard, white_margin, plies, _now(), node_id, replica),
        )
        if cursor.rowcount == 0:
            raise StoreError(f"no scheduled playout ({node_id}, {replica}) to mark done")
        self._connection.commit()

    def knows_dag_hash(self, dag_hash: str) -> bool:
        """Whether this store ever emitted ``dag_hash`` (see :meth:`dag_hash`)."""
        row = self._connection.execute("SELECT 1 FROM meta WHERE key = ?", (f"dag_hash:{dag_hash}",)).fetchone()
        return row is not None

    def reconcile(self, entries: Iterable[ReconcileEntry]) -> ReconcileResult:
        """Rebuild or verify the playout registry from shard footers.

        Shards are self-describing (each game's footer carries its start ``board_key``,
        replica, seed and scores), so the ``playouts`` table is an index rather than a
        second source of truth — this is both the crash repair for a run that died
        between a shard rename and its DB transaction, and the audit that the DB and the
        shards agree.
        """
        matched = inserted = updated = 0
        unknown: list[bytes] = []
        for entry in entries:
            if entry.dag_hash is not None and not self.knows_dag_hash(entry.dag_hash):
                logger.warning(
                    "shard {} was written against DAG {} — a corpus this store has never "
                    "produced. Its games are being adopted; check the shard did not come "
                    "from another run.",
                    entry.shard,
                    entry.dag_hash[:12],
                )
            record = self.node_by_key(entry.board_key)
            if record is None:
                unknown.append(entry.board_key)
                continue
            row = self._connection.execute(
                "SELECT status, shard, white_margin, plies FROM playouts WHERE node_id = ? AND replica = ?",
                (record.node_id, entry.replica),
            ).fetchone()
            done_fields = (
                "done",
                entry.shard,
                entry.white_margin,
                entry.plies,
            )
            if row is None:
                self._connection.execute(
                    "INSERT INTO playouts (node_id, replica, engine_seed, game_id, status, shard, white_margin, "
                    "plies, completed_at) VALUES (?, ?, ?, ?, 'done', ?, ?, ?, ?)",
                    (
                        record.node_id,
                        entry.replica,
                        playout_seed(record.board_key, entry.replica),
                        entry.game_id,
                        entry.shard,
                        entry.white_margin,
                        entry.plies,
                        _now(),
                    ),
                )
                inserted += 1
                continue
            if (str(row["status"]), row["shard"], row["white_margin"], row["plies"]) == done_fields:
                matched += 1
                continue
            self._connection.execute(
                "UPDATE playouts SET status = 'done', shard = ?, white_margin = ?, plies = ?, completed_at = ?, "
                "game_id = ? WHERE node_id = ? AND replica = ?",
                (entry.shard, entry.white_margin, entry.plies, _now(), entry.game_id, record.node_id, entry.replica),
            )
            updated += 1
        self._connection.commit()
        return ReconcileResult(matched=matched, inserted=inserted, updated=updated, unknown_nodes=tuple(unknown))

    def _next_game_id(self) -> int:
        row = self._connection.execute("SELECT COALESCE(MAX(game_id), -1) + 1 AS next FROM playouts").fetchone()
        return int(row["next"])

    def playout_counts(self) -> dict[str, int]:
        """Games by status ('planned' / 'done')."""
        rows = self._connection.execute("SELECT status, COUNT(*) AS n FROM playouts GROUP BY status")
        return {str(row["status"]): int(row["n"]) for row in rows}

    # -- link + export (S5) ----------------------------------------------------

    def link(self) -> int:
        """Aggregate playout outcomes up the DAG into ``outcome_mean`` / ``outcome_count``.

        For every node, the finished games started anywhere in its subtree contribute
        their outcome **from that node's side to move** (a White-margin win is +1 for a
        node where White is to move, −1 where Black is). Honest caveat, recorded here and
        in the plan rather than in the data: an interior node's mean averages
        continuations from *imposed* prefixes under the allocation's mixture, not
        Pentobi's own play from that node.

        Returns:
            The number of nodes given a non-zero outcome count.
        """
        rows = self._connection.execute(
            "WITH RECURSIVE sub(node_id, root) AS ("
            "    SELECT node_id, node_id FROM nodes"
            "    UNION SELECT e.child_id, s.root FROM edges e JOIN sub s ON e.parent_id = s.node_id"
            "                                    WHERE e.child_id IS NOT NULL) "
            "SELECT s.root AS root, n.player AS player, "
            "       SUM(CASE WHEN p.white_margin > 0 THEN 1 WHEN p.white_margin < 0 THEN -1 ELSE 0 END) AS margin_sum, "
            "       COUNT(*) AS games "
            "FROM sub s JOIN playouts p ON p.node_id = s.node_id JOIN nodes n ON n.node_id = s.root "
            "WHERE p.status = 'done' GROUP BY s.root",
        ).fetchall()
        self._connection.execute("UPDATE nodes SET outcome_mean = NULL, outcome_count = 0")
        updates = [
            (int(row["player"]) * int(row["margin_sum"]) / int(row["games"]), int(row["games"]), int(row["root"]))
            for row in rows
        ]
        self._connection.executemany(
            "UPDATE nodes SET outcome_mean = ?, outcome_count = ? WHERE node_id = ?",
            updates,
        )
        self._connection.commit()
        return len(updates)

    def reach_weights(self) -> dict[int, float]:
        """Each node's share of Pentobi's own play mass: Π ancestor visit shares.

        Summed over DAG parents (a position reached two ways carries both paths' mass)
        and computed in depth order, which is a valid topological order because every
        edge increases depth by one.
        """
        weights: dict[int, float] = {self.root_node(): 1.0}
        rows = self._connection.execute(
            "SELECT e.parent_id AS parent_id, e.child_id AS child_id, e.visit_share AS visit_share "
            "FROM edges e JOIN nodes n ON n.node_id = e.parent_id "
            "WHERE e.child_id IS NOT NULL ORDER BY n.depth, e.parent_id, e.rank",
        )
        for row in rows:
            parent_weight = weights.get(int(row["parent_id"]), 0.0)
            child_id = int(row["child_id"])
            weights[child_id] = weights.get(child_id, 0.0) + parent_weight * float(row["visit_share"])
        return weights

    def iter_opening_rows(self, *, top_k: int = STORE_K) -> Iterator[OpeningRow]:
        """Yield one export row per searched node (the opening dataset's source).

        The soft policy target is the node's top-``top_k`` children by visits,
        renormalised to sum to 1; ``tail_mass`` records exactly what the truncation
        dropped. Everything is in the node's key frame, matching the stored board.

        Nodes with no children are **skipped**: ``search_node`` legitimately records a
        childless leaf when the engine returns an empty ``move_values`` (the side to move
        can only pass), but such a row would carry an empty policy, and the validator
        rightly rejects a target that does not sum to 1. There is no training signal in a
        position with no moves, so it does not belong in the dataset.
        """
        plan = self.active_plan()
        allocations = {a.node_id: a for a in (self.plan_allocations(plan.plan_id) if plan else [])}
        weights = self.reach_weights()
        for record in self.nodes(status="searched"):
            edges = self.edges(record.node_id)[:top_k]
            if not edges:
                continue
            visits = np.array([edge.visits for edge in edges], dtype=np.float64)
            total = float(visits.sum())
            if total <= 0.0:
                continue
            kept = total / record.root_visits if record.root_visits else 0.0
            allocation = allocations.get(record.node_id)
            parents = self._connection.execute(
                "SELECT MIN(parent_id) AS parent FROM edges WHERE child_id = ?",
                (record.node_id,),
            ).fetchone()
            yield OpeningRow(
                node_id=record.node_id,
                parent_id=None if parents["parent"] is None else int(parents["parent"]),
                board=record.board_key,
                policy_indices=np.array([edge.action for edge in edges], dtype=np.int32),
                policy_values=(visits / total if total else visits).astype(np.float32),
                child_values=np.array([edge.child_value for edge in edges], dtype=np.float32),
                tail_mass=float(max(0.0, 1.0 - kept)),
                search_value=float(record.search_value if record.search_value is not None else 0.0),
                depth=record.depth,
                player=record.player,
                reach_weight=weights.get(record.node_id, 0.0),
                budget_share=allocation.budget_share if allocation else 0.0,
                planned_games=allocation.planned_games if allocation else 0,
                outcome_mean=float(record.outcome_mean if record.outcome_mean is not None else 0.0),
                outcome_count=record.outcome_count,
            )

    # -- coverage (S6) ---------------------------------------------------------

    def register_corpus(
        self,
        name: str,
        path: str,
        dataset_kind: str,
        *,
        games: int | None = None,
        positions: int | None = None,
        notes: str | None = None,
    ) -> None:
        """Record an external dataset in the manifest (e.g. the v1 corpus, D-h).

        Registered, not grafted onto the graph: v1's uniform-random unharvested openings
        are not part of the searched strong-opening space, and forcing them in would
        pollute every coverage metric with junk-opening paths.
        """
        self._connection.execute(
            "INSERT OR REPLACE INTO corpora (name, path, dataset_kind, games, positions, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, path, dataset_kind, games, positions, notes),
        )
        self._connection.commit()

    def corpora(self) -> list[dict[str, object]]:
        """Every registered dataset."""
        return [dict(row) for row in self._connection.execute("SELECT * FROM corpora ORDER BY name")]

    def coverage(self) -> CoverageReport:
        """The store design's D-f metric set — "what have we done?" in one screen."""
        plan = self.active_plan()
        allocations = self.plan_allocations(plan.plan_id) if plan else []
        records = {record.node_id: record for record in self.nodes()}
        starts = [a for a in allocations if a.planned_games > 0]
        actuals = {
            int(row["node_id"]): int(row["n"])
            for row in self._connection.execute(
                "SELECT node_id, COUNT(*) AS n FROM playouts WHERE status = 'done' GROUP BY node_id",
            )
        }
        weights = self.reach_weights()
        planned_games = sum(a.planned_games for a in starts)
        actual_games = sum(actuals.get(a.node_id, 0) for a in starts)
        histogram = {"0%": 0, "<50%": 0, "<100%": 0, "100%": 0}
        for allocation in starts:
            fulfilment = actuals.get(allocation.node_id, 0) / allocation.planned_games
            key = "0%" if fulfilment == 0 else "100%" if fulfilment >= 1.0 else "<50%" if fulfilment < 0.5 else "<100%"
            histogram[key] += 1
        split_floor = 2 * plan.parameters.min_replicas if plan else 0
        budget = plan.parameters.budget if plan else 0
        debt = sum(
            1 for a in allocations if not records[a.node_id].is_searched and a.budget_share * budget >= split_floor
        )
        first_actions, first_positions = self._first_move_coverage({a.node_id for a in allocations})
        return CoverageReport(
            dag_hash=self.dag_hash(),
            plan=plan,
            nodes_by_depth=_count_by(records.values(), lambda record: record.depth),
            searched_by_depth=_count_by([r for r in records.values() if r.is_searched], lambda record: record.depth),
            starts_by_depth=_count_by([records[a.node_id] for a in starts], lambda record: record.depth),
            planned_games=planned_games,
            actual_games=actual_games,
            mapping_debt=debt,
            fulfilment_histogram=histogram,
            distinct_first_moves=first_actions,
            distinct_first_positions=first_positions,
            play_mass_by_depth=_sum_by(
                [(records[a.node_id].depth, weights.get(a.node_id, 0.0)) for a in allocations],
            ),
            planned_games_min=min((a.planned_games for a in starts), default=0),
            planned_games_max=max((a.planned_games for a in starts), default=0),
            planned_games_median=float(np.median([a.planned_games for a in starts])) if starts else 0.0,
            budget_share_by_depth=_sum_by([(records[a.node_id].depth, a.budget_share) for a in starts]),
        )

    def _first_move_coverage(self, plan_nodes: set[int]) -> tuple[int, int]:
        """Distinct ply-1 moves and canonical ply-1 positions the plan's nodes pass through."""
        firsts: set[int] = set()
        positions: set[bytes] = set()
        for node_id in plan_nodes:
            record = self.node(node_id)
            if not record.witness_actions:
                continue
            firsts.add(record.witness_actions[0])
            board, player = self._game.get_next_state(self._game.initialise_board(), 1, record.witness_actions[0])
            compact = np.asarray(self._game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
            positions.add(canonical_key(compact)[0])
        return len(firsts), len(positions)


# --------------------------------------------------------------------------- #
# Row decoding + small helpers
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _node_from_row(row: sqlite3.Row) -> NodeRecord:
    return NodeRecord(
        node_id=int(row["node_id"]),
        board_key=bytes(row["board_key"]),
        key_frame=int(row["key_frame"]),
        depth=int(row["depth"]),
        player=int(row["player"]),
        witness_actions=tuple(int(a) for a in json.loads(str(row["witness_actions"]))),
        source=str(row["source"]),
        status=str(row["status"]),
        book_terminal=bool(row["book_terminal"]),
        engine_seed=None if row["engine_seed"] is None else int(row["engine_seed"]),
        root_visits=None if row["root_visits"] is None else int(row["root_visits"]),
        search_value=None if row["search_value"] is None else float(row["search_value"]),
        search_seconds=None if row["search_seconds"] is None else float(row["search_seconds"]),
        searched_at=None if row["searched_at"] is None else str(row["searched_at"]),
        outcome_mean=None if row["outcome_mean"] is None else float(row["outcome_mean"]),
        outcome_count=int(row["outcome_count"]),
    )


def _edge_from_row(row: sqlite3.Row) -> EdgeRecord:
    return EdgeRecord(
        parent_id=int(row["parent_id"]),
        action=int(row["action"]),
        rank=int(row["rank"]),
        visits=int(row["visits"]),
        visit_share=float(row["visit_share"]),
        child_value=float(row["child_value"]),
        child_id=None if row["child_id"] is None else int(row["child_id"]),
        source=str(row["source"]),
    )


def _plan_from_row(row: sqlite3.Row) -> PlanRecord:
    return PlanRecord(
        plan_id=int(row["plan_id"]),
        created_at=str(row["created_at"]),
        parameters=PlanParameters(
            budget=int(row["budget"]),
            temperature=float(row["temperature"]),
            min_replicas=int(row["min_replicas"]),
        ),
        dag_hash=str(row["dag_hash"]),
        is_active=bool(row["is_active"]),
    )


def _count_by(records: Iterable[NodeRecord], key: Callable[[NodeRecord], int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        counts[key(record)] = counts.get(key(record), 0) + 1
    return dict(sorted(counts.items()))


def _sum_by(pairs: Iterable[tuple[int, float]]) -> dict[int, float]:
    totals: dict[int, float] = {}
    for key, value in pairs:
        totals[key] = totals.get(key, 0.0) + value
    return dict(sorted(totals.items()))
