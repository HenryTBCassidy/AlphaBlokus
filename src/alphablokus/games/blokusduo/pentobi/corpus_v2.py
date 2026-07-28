"""Schema v2 for the Pentobi distillation corpus: soft targets and an opening dataset.

Two datasets live under one corpus directory (plus the store DB, which is the map of what
the shards mean), both marked ``dataset_kind = "pentobi_distill_v2"``:

- **``opening/opening_{NNNNN}.parquet``** — one row per searched DAG node, materialised
  from :class:`~alphablokus.games.blokusduo.pentobi.store.SearchSpaceStore` by
  :func:`export_opening`. The DB is the source of truth; the export is regenerable and
  stamped with the store's ``dag_hash``, so a stale export is detectable rather than
  merely suspected. These are the *opening* positions — depths 1–3 exist nowhere else.
- **``games/corpus_{NNNNN}.parquet``** — one row per harvested game ply (v5/v6).

What changed from v1, and why it matters: the policy columns hold Pentobi's **whole
preference distribution** (top-32 children by visits, renormalised) instead of a one-hot
of the played move. v1 computed that distribution on every ply and threw it away. The
column *format* is unchanged (``policy_kind = "sparse_v1"``), so every downstream
densify path keeps working — a one-hot was always just a degenerate sparse target, and
``BaseNNetWrapper.loss_pi`` is already a KL against a full distribution.

Opening rows are stored in their node's **key frame** (the symmetry-canonical
orientation), board and policy indices together, so they stay self-consistent; the
trainer's order-2 augmentation regenerates the mirror of every row regardless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alphablokus.games.blokusduo.pentobi.corpus import BOARD_KIND, POLICY_KIND

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.pentobi.corpus import CorpusExample
    from alphablokus.games.blokusduo.pentobi.harvest import HarvestedGame
    from alphablokus.games.blokusduo.pentobi.store import (
        OpeningRow,
        PlanRecord,
        ReconcileEntry,
        SearchChild,
        SearchSpaceStore,
    )

#: Marks every v2 shard, opening and games alike.
DATASET_KIND = "pentobi_distill_v2"

#: Default rows per opening shard. The whole stage-1 opening dataset is a few thousand
#: rows, so this is one file in practice — the split exists for later top-ups.
OPENING_ROWS_PER_SHARD = 50_000

_OPENING_SCHEMA_FIELDS = [
    pa.field("board", pa.binary()),
    pa.field("policy_indices", pa.binary()),
    pa.field("policy_values", pa.binary()),
    pa.field("child_values", pa.binary()),
    pa.field("tail_mass", pa.float32()),
    pa.field("search_value", pa.float32()),
    pa.field("depth", pa.int32()),
    pa.field("reach_weight", pa.float32()),
    pa.field("budget_share", pa.float32()),
    pa.field("planned_games", pa.int32()),
    pa.field("node_id", pa.int64()),
    pa.field("parent_id", pa.int64()),
    pa.field("player", pa.int8()),
    pa.field("outcome_mean", pa.float32()),
    pa.field("outcome_count", pa.int32()),
]


class CorpusSchemaError(ValueError):
    """Raised when a v2 shard's contents contradict its schema or the rules engine."""


@dataclass(frozen=True)
class SoftTarget:
    """One position's soft policy target: the expert's preferences, top-K, normalised."""

    indices: NDArray[np.int32]  # actions, visit-descending
    values: NDArray[np.float32]  # visit shares over the kept children, summing to 1
    child_values: NDArray[np.float32]  # Pentobi's per-child value, aligned to the indices
    tail_mass: float  # visit mass the top-K truncation dropped

    @property
    def top_action(self) -> int:
        """``argmax(visits)`` — the move a full-strength continuation plays."""
        return int(self.indices[0])


def build_soft_target(children: Sequence[SearchChild], top_k: int = 32) -> SoftTarget:
    """Turn a search's children into the stored policy target.

    Children are ranked by visits (ties by action, so the target is deterministic), the
    top ``top_k`` are kept and renormalised, and what the truncation dropped is recorded
    rather than quietly lost — measured ≈ 0.036 of the mass at ply 1 and 0.017 at ply 2,
    with top-32 ≥ 96.7% everywhere the walk measured.

    A search whose children are all unvisited (no tree, only priors) yields a uniform
    target over them; that is honest about carrying no preference information, and the
    validator's ``support ⊆ legal`` check still applies.
    """
    if not children:
        raise CorpusSchemaError("cannot build a soft target from an empty move_values response")
    ranked = sorted(children, key=lambda child: (-child.visits, child.action))[:top_k]
    total = sum(child.visits for child in children)
    kept = sum(child.visits for child in ranked)
    if kept > 0:
        values = np.array([child.visits / kept for child in ranked], dtype=np.float32)
    else:
        values = np.full(len(ranked), 1.0 / len(ranked), dtype=np.float32)
    return SoftTarget(
        indices=np.array([child.action for child in ranked], dtype=np.int32),
        values=values,
        child_values=np.array([child.value for child in ranked], dtype=np.float32),
        tail_mass=float(1.0 - kept / total) if total > 0 else 0.0,
    )


@dataclass(frozen=True)
class OpeningShardMeta:
    """An opening shard's footer: what it holds and which DAG it came from."""

    level: int
    policy_size: int
    board_shape: tuple[int, ...]
    board_dtype: str
    dag_hash: str
    plan_id: int | None
    budget: int | None
    temperature: float | None
    min_replicas: int | None
    num_rows: int


def opening_shard_filename(index: int) -> str:
    """Canonical opening-shard filename."""
    return f"opening_{index:05d}.parquet"


def opening_shards(directory: Path) -> list[Path]:
    """All final (non-``.tmp``) opening shards in ``directory``, sorted by index."""
    return sorted(directory.glob("opening_*.parquet"))


def write_opening_shard(path: Path, rows: Sequence[OpeningRow], *, meta: OpeningShardMeta) -> int:
    """Write one opening shard atomically (``.tmp`` then rename); returns rows written."""
    metadata = {
        "dataset_kind": DATASET_KIND,
        "board_kind": BOARD_KIND,
        "board_shape": ",".join(str(d) for d in meta.board_shape),
        "board_dtype": meta.board_dtype,
        "policy_kind": POLICY_KIND,
        "policy_size": str(meta.policy_size),
        "level": str(meta.level),
        "dag_hash": meta.dag_hash,
        "plan": json.dumps(
            {
                "plan_id": meta.plan_id,
                "budget": meta.budget,
                "temperature": meta.temperature,
                "min_replicas": meta.min_replicas,
            },
        ),
    }
    schema = pa.schema(_OPENING_SCHEMA_FIELDS, metadata={k.encode(): v.encode() for k, v in metadata.items()})
    columns: dict[str, list[object]] = {name: [] for name in schema.names}
    for row in rows:
        columns["board"].append(row.board)
        columns["policy_indices"].append(row.policy_indices.astype(np.int32).tobytes())
        columns["policy_values"].append(row.policy_values.astype(np.float32).tobytes())
        columns["child_values"].append(row.child_values.astype(np.float32).tobytes())
        columns["tail_mass"].append(row.tail_mass)
        columns["search_value"].append(row.search_value)
        columns["depth"].append(row.depth)
        columns["reach_weight"].append(row.reach_weight)
        columns["budget_share"].append(row.budget_share)
        columns["planned_games"].append(row.planned_games)
        columns["node_id"].append(row.node_id)
        columns["parent_id"].append(row.parent_id)
        columns["player"].append(row.player)
        columns["outcome_mean"].append(row.outcome_mean)
        columns["outcome_count"].append(row.outcome_count)
    table = pa.Table.from_pydict(columns, schema=schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.rename(path)
    return table.num_rows


def read_opening_meta(path: Path) -> OpeningShardMeta:
    """Decode an opening shard's footer (reads only the parquet footer)."""
    raw = pq.read_schema(path).metadata or {}
    meta = {k.decode(): v.decode() for k, v in raw.items()}
    if meta.get("dataset_kind") != DATASET_KIND:
        raise CorpusSchemaError(f"{path.name}: dataset_kind={meta.get('dataset_kind')!r}, expected {DATASET_KIND!r}")
    plan = json.loads(meta.get("plan", "{}"))
    return OpeningShardMeta(
        level=int(meta["level"]),
        policy_size=int(meta["policy_size"]),
        board_shape=tuple(int(d) for d in meta["board_shape"].split(",")),
        board_dtype=meta["board_dtype"],
        dag_hash=meta["dag_hash"],
        plan_id=plan.get("plan_id"),
        budget=plan.get("budget"),
        temperature=plan.get("temperature"),
        min_replicas=plan.get("min_replicas"),
        num_rows=pq.ParquetFile(path).metadata.num_rows,
    )


def export_opening(
    store: SearchSpaceStore,
    directory: Path,
    *,
    rows_per_shard: int = OPENING_ROWS_PER_SHARD,
    top_k: int | None = None,
) -> list[Path]:
    """Materialise every searched DAG node as opening rows under ``directory``.

    Regenerable by construction: the store is the truth, and each shard is stamped with
    the ``dag_hash`` it was exported from. Stale shards from an earlier DAG are removed
    first so the directory can never mix two exports.

    Args:
        store: The search-space store to export.
        directory: Output directory (created if missing).
        rows_per_shard: Rows per file.
        top_k: Children kept in the soft target (defaults to the store's ``STORE_K``).
    """
    from alphablokus.games.blokusduo.pentobi.store import STORE_K

    directory.mkdir(parents=True, exist_ok=True)
    for stale in [*opening_shards(directory), *directory.glob("opening_*.parquet.tmp")]:
        stale.unlink()
    plan = store.active_plan()
    board_size = store.game.board_size
    written: list[Path] = []
    batch: list[OpeningRow] = []
    rows = list(store.iter_opening_rows(top_k=top_k if top_k is not None else STORE_K))
    for row in rows:
        batch.append(row)
        if len(batch) >= rows_per_shard:
            written.append(_flush_opening(store, directory, len(written), batch, plan, board_size))
            batch = []
    if batch or not written:
        written.append(_flush_opening(store, directory, len(written), batch, plan, board_size))
    return written


def _flush_opening(
    store: SearchSpaceStore,
    directory: Path,
    index: int,
    rows: list[OpeningRow],
    plan: PlanRecord | None,
    board_size: int,
) -> Path:
    """Write one opening shard with the store's provenance in its footer."""
    meta = OpeningShardMeta(
        level=int(store.meta["level"]),
        policy_size=int(store.meta["policy_size"]),
        board_shape=(board_size, board_size),
        board_dtype="int8",
        dag_hash=store.dag_hash(),
        plan_id=plan.plan_id if plan else None,
        budget=plan.parameters.budget if plan else None,
        temperature=plan.parameters.temperature if plan else None,
        min_replicas=plan.parameters.min_replicas if plan else None,
        num_rows=len(rows),
    )
    path = directory / opening_shard_filename(index)
    write_opening_shard(path, rows, meta=meta)
    return path


def opening_value(
    search_value: float,
    outcome_mean: float,
    outcome_count: int,
    *,
    target: str = "blend",
    blend_k: int = 5,
) -> float:
    """The value label for an opening row (v2 plan V9's ``--opening-value``).

    ``search`` rescales Pentobi's backed-up value into our ±1 convention (``2v − 1``, an
    approximation: the engine's values are win-probability-*like*, not calibrated).
    ``outcome`` uses the empirical mean of real L9 continuations from the node's subtree
    (the ``link`` pass). ``blend`` is the count-shrunk combination — the teacher's opinion
    at ``n = 0``, the outcomes as ``n`` grows — which is why it is the default.
    """
    rescaled = 2.0 * search_value - 1.0
    if target == "search" or outcome_count == 0:
        return rescaled
    if target == "outcome":
        return outcome_mean
    if target != "blend":
        raise CorpusSchemaError(f"unknown opening value target {target!r}")
    return (outcome_count * outcome_mean + blend_k * rescaled) / (outcome_count + blend_k)


def iter_opening_examples(
    paths: Sequence[Path],
    *,
    value_target: str = "blend",
    blend_k: int = 5,
    temperature: float = 1.0,
) -> Iterator[CorpusExample]:
    """Stream ``(board, (indices, values), value)`` training tuples from opening shards.

    The same tuple shape the self-play pipeline and the v1 corpus reader produce, so the
    SL trainer consumes all three through one code path. ``temperature`` applies the
    target softening ``p^(1/τ)`` at load, renormalised over the stored support — the
    corpus always stores τ = 1 visits, so retuning never requires regeneration.
    """
    for path in paths:
        meta = read_opening_meta(path)
        parquet_file = pq.ParquetFile(path)
        columns = ["board", "policy_indices", "policy_values", "search_value", "outcome_mean", "outcome_count"]
        for batch in parquet_file.iter_batches(columns=columns):
            for board_bytes, indices_bytes, values_bytes, search_value, outcome_mean, outcome_count in zip(
                *(batch.column(name).to_pylist() for name in columns),
                strict=True,
            ):
                board = np.frombuffer(board_bytes, dtype=np.dtype(meta.board_dtype)).reshape(meta.board_shape).copy()
                indices = np.frombuffer(indices_bytes, dtype=np.int32).copy()
                values = apply_target_temperature(np.frombuffer(values_bytes, dtype=np.float32), temperature)
                value = opening_value(
                    float(search_value),
                    float(outcome_mean),
                    int(outcome_count),
                    target=value_target,
                    blend_k=blend_k,
                )
                yield board, (indices, values), value


# --------------------------------------------------------------------------- #
# The games dataset
# --------------------------------------------------------------------------- #

_GAME_SCHEMA_FIELDS = [
    pa.field("board", pa.binary()),
    pa.field("policy_indices", pa.binary()),
    pa.field("policy_values", pa.binary()),
    pa.field("child_values", pa.binary()),
    pa.field("tail_mass", pa.float32()),
    pa.field("search_value", pa.float32()),
    pa.field("value", pa.float64()),
    pa.field("margin", pa.int32()),
    pa.field("player", pa.int8()),
    pa.field("game_id", pa.int64()),
    pa.field("ply", pa.int32()),
    pa.field("action", pa.int32()),
    pa.field("top_action", pa.int32()),
]


@dataclass(frozen=True)
class GameShardGameMeta:
    """One game's provenance in a shard footer — enough to rebuild the playout registry."""

    game_id: int
    node_id: int
    board_key: str  # hex of the start node's key
    replica: int
    engine_seed: int
    witness_actions: tuple[int, ...]
    white_score: int
    black_score: int
    plies: int

    @property
    def white_margin(self) -> int:
        return self.white_score - self.black_score


@dataclass(frozen=True)
class GameShardMeta:
    """A games shard's footer, decoded."""

    level: int
    policy_size: int
    board_shape: tuple[int, ...]
    board_dtype: str
    dag_hash: str
    plan_id: int | None
    budget: int | None
    temperature: float | None
    min_replicas: int | None
    game_sizes: tuple[int, ...]
    games: tuple[GameShardGameMeta, ...]


def game_shard_filename(index: int) -> str:
    """Canonical games-shard filename (unchanged from v1, so tooling keeps working)."""
    return f"corpus_{index:05d}.parquet"


def game_shards(directory: Path) -> list[Path]:
    """All final (non-``.tmp``) games shards in ``directory``, sorted by index."""
    return sorted(directory.glob("corpus_*.parquet"))


def write_game_shard(path: Path, games: Sequence[HarvestedGame], *, meta: GameShardMeta) -> int:
    """Write one games shard atomically; returns rows written.

    Rows are one per harvested ply, games laid out back to back in play order (the
    footer's ``game_sizes`` is the cursor), matching v1 so the grouping walk in the
    dataloader is unchanged.
    """
    metadata = {
        "dataset_kind": DATASET_KIND,
        "board_kind": BOARD_KIND,
        "board_shape": ",".join(str(d) for d in meta.board_shape),
        "board_dtype": meta.board_dtype,
        "policy_kind": POLICY_KIND,
        "policy_size": str(meta.policy_size),
        "level": str(meta.level),
        "dag_hash": meta.dag_hash,
        "plan": json.dumps(
            {
                "plan_id": meta.plan_id,
                "budget": meta.budget,
                "temperature": meta.temperature,
                "min_replicas": meta.min_replicas,
            },
        ),
        "game_sizes": ",".join(str(len(g.plies)) for g in games),
        "games_meta": json.dumps(
            [
                {
                    "game_id": g.game_id,
                    "node_id": g.node_id,
                    "board_key": g.board_key.hex(),
                    "replica": g.replica,
                    "engine_seed": g.engine_seed,
                    "witness_actions": list(g.witness_actions),
                    "white_score": g.white_score,
                    "black_score": g.black_score,
                    "plies": len(g.plies),
                }
                for g in games
            ],
        ),
    }
    schema = pa.schema(_GAME_SCHEMA_FIELDS, metadata={k.encode(): v.encode() for k, v in metadata.items()})
    columns: dict[str, list[object]] = {name: [] for name in schema.names}
    for harvested in games:
        winner = int(np.sign(harvested.white_margin))
        for ply in harvested.plies:
            columns["board"].append(ply.compact_board.tobytes())
            columns["policy_indices"].append(ply.target.indices.astype(np.int32).tobytes())
            columns["policy_values"].append(ply.target.values.astype(np.float32).tobytes())
            columns["child_values"].append(ply.target.child_values.astype(np.float32).tobytes())
            columns["tail_mass"].append(ply.target.tail_mass)
            columns["search_value"].append(ply.search_value)
            columns["value"].append(float(winner * ply.player))
            columns["margin"].append(harvested.white_margin * ply.player)
            columns["player"].append(ply.player)
            columns["game_id"].append(harvested.game_id)
            columns["ply"].append(ply.ply)
            columns["action"].append(ply.action)
            columns["top_action"].append(ply.top_action)
    table = pa.Table.from_pydict(columns, schema=schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.rename(path)
    return table.num_rows


def read_game_shard_meta(path: Path) -> GameShardMeta:
    """Decode a games shard's footer (reads only the parquet footer)."""
    raw = pq.read_schema(path).metadata or {}
    meta = {k.decode(): v.decode() for k, v in raw.items()}
    if meta.get("dataset_kind") != DATASET_KIND:
        raise CorpusSchemaError(f"{path.name}: dataset_kind={meta.get('dataset_kind')!r}, expected {DATASET_KIND!r}")
    plan = json.loads(meta.get("plan", "{}"))
    sizes = meta["game_sizes"]
    return GameShardMeta(
        level=int(meta["level"]),
        policy_size=int(meta["policy_size"]),
        board_shape=tuple(int(d) for d in meta["board_shape"].split(",")),
        board_dtype=meta["board_dtype"],
        dag_hash=meta["dag_hash"],
        plan_id=plan.get("plan_id"),
        budget=plan.get("budget"),
        temperature=plan.get("temperature"),
        min_replicas=plan.get("min_replicas"),
        game_sizes=tuple(int(s) for s in sizes.split(",")) if sizes else (),
        games=tuple(
            GameShardGameMeta(
                game_id=int(g["game_id"]),
                node_id=int(g["node_id"]),
                board_key=str(g["board_key"]),
                replica=int(g["replica"]),
                engine_seed=int(g["engine_seed"]),
                witness_actions=tuple(int(a) for a in g["witness_actions"]),
                white_score=int(g["white_score"]),
                black_score=int(g["black_score"]),
                plies=int(g["plies"]),
            )
            for g in json.loads(meta["games_meta"])
        ),
    )


def iter_game_examples(paths: Sequence[Path], *, temperature: float = 1.0) -> Iterator[CorpusExample]:
    """Stream ``(board, (indices, values), value)`` training tuples from games shards."""
    for path in paths:
        meta = read_game_shard_meta(path)
        parquet_file = pq.ParquetFile(path)
        columns = ["board", "policy_indices", "policy_values", "value"]
        for batch in parquet_file.iter_batches(columns=columns):
            for board_bytes, indices_bytes, values_bytes, value in zip(
                *(batch.column(name).to_pylist() for name in columns),
                strict=True,
            ):
                board = np.frombuffer(board_bytes, dtype=np.dtype(meta.board_dtype)).reshape(meta.board_shape).copy()
                indices = np.frombuffer(indices_bytes, dtype=np.int32).copy()
                values = apply_target_temperature(np.frombuffer(values_bytes, dtype=np.float32), temperature)
                yield board, (indices, values), float(value)


def iter_shard_playouts(directory: Path) -> Iterator[ReconcileEntry]:
    """Every game a shard directory holds, as store reconciliation entries (footers only).

    Shards are self-describing, so the playouts table can be rebuilt or verified from
    them alone — the crash repair for a run that died between a shard rename and its DB
    transaction.
    """
    from alphablokus.games.blokusduo.pentobi.store import ReconcileEntry as Entry

    for path in game_shards(directory):
        meta = read_game_shard_meta(path)
        for game in meta.games:
            yield Entry(
                board_key=bytes.fromhex(game.board_key),
                replica=game.replica,
                game_id=game.game_id,
                shard=path.name,
                white_margin=game.white_margin,
                plies=game.plies,
            )


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

#: Ply buckets the diagnostics report splits rows into (lower bound, label).
_PLY_BUCKETS = ((0, "0-3"), (4, "4-7"), (8, "8-15"), (16, "16+"))


@dataclass(frozen=True)
class CorpusReport:
    """What v2 is actually claiming, measured — the ``analyze`` output."""

    num_games: int
    num_game_rows: int
    num_opening_rows: int
    opening_row_fraction: float
    rows_by_ply_bucket: dict[str, int]
    mean_target_entropy_by_bucket: dict[str, float]
    mean_effective_moves_by_bucket: dict[str, float]
    mean_tail_mass: float
    duplicate_position_rate: float
    duplicate_position_rate_mirrored: float
    unique_starts: int
    mean_games_per_start: float
    white_win_rate: float
    draw_rate: float
    mean_absolute_margin: float

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form (for logs and the analysis CLI)."""
        return {
            "num_games": self.num_games,
            "num_game_rows": self.num_game_rows,
            "num_opening_rows": self.num_opening_rows,
            "opening_row_fraction": self.opening_row_fraction,
            "rows_by_ply_bucket": self.rows_by_ply_bucket,
            "mean_target_entropy_by_bucket": self.mean_target_entropy_by_bucket,
            "mean_effective_moves_by_bucket": self.mean_effective_moves_by_bucket,
            "mean_tail_mass": self.mean_tail_mass,
            "duplicate_position_rate": self.duplicate_position_rate,
            "duplicate_position_rate_mirrored": self.duplicate_position_rate_mirrored,
            "unique_starts": self.unique_starts,
            "mean_games_per_start": self.mean_games_per_start,
            "white_win_rate": self.white_win_rate,
            "draw_rate": self.draw_rate,
            "mean_absolute_margin": self.mean_absolute_margin,
        }


def analyze_corpus(games_dir: Path, opening_dir: Path | None = None) -> CorpusReport:
    """Measure the v2 claims: target richness, row mix, duplication, outcome balance.

    The metrics that matter here are the ones v1 failed on. **Target entropy** and
    **effective moves** (``exp(H)``) say whether the stored policy is more than a one-hot
    — the whole point of v2. The **opening row fraction** exposes the row-mix problem the
    trainer has to correct: a game harvests ~26 rows while a whole opening node is one
    row, so openings are a fraction of a percent by count despite being the strategic
    edge. **Duplicate-position rate** (raw and mirror-collapsed) is the cost of sharing
    strong openings across games — v1 measured 0% because every game had a unique random
    opening. The **White win rate** is expected to be *less* skewed than v1's 96%,
    precisely because flattened allocation plays unbalanced starts.
    """
    entropies: dict[str, list[float]] = {label: [] for _, label in _PLY_BUCKETS}
    rows_by_bucket: dict[str, int] = {label: 0 for _, label in _PLY_BUCKETS}
    boards: list[bytes] = []
    mirrored: set[bytes] = set()
    tail_masses: list[float] = []
    starts: dict[str, int] = {}
    white_wins = draws = 0
    margins: list[int] = []
    num_games = 0
    for path in game_shards(games_dir):
        meta = read_game_shard_meta(path)
        num_games += len(meta.games)
        for game_meta in meta.games:
            starts[game_meta.board_key] = starts.get(game_meta.board_key, 0) + 1
            white_wins += int(game_meta.white_margin > 0)
            draws += int(game_meta.white_margin == 0)
            margins.append(abs(game_meta.white_margin))
        table = pq.read_table(path, columns=["board", "policy_values", "tail_mass", "ply"])
        for board_bytes, values_bytes, tail_mass, ply in zip(
            table.column("board").to_pylist(),
            table.column("policy_values").to_pylist(),
            table.column("tail_mass").to_pylist(),
            table.column("ply").to_pylist(),
            strict=True,
        ):
            bucket = _ply_bucket(int(ply))
            rows_by_bucket[bucket] += 1
            entropies[bucket].append(_entropy(np.frombuffer(values_bytes, dtype=np.float32)))
            tail_masses.append(float(tail_mass))
            boards.append(board_bytes)
            mirrored.add(_mirror_key(board_bytes, meta.board_shape[0]))
    opening_rows = sum(read_opening_meta(path).num_rows for path in opening_shards(opening_dir or games_dir))
    total_rows = len(boards) + opening_rows
    return CorpusReport(
        num_games=num_games,
        num_game_rows=len(boards),
        num_opening_rows=opening_rows,
        opening_row_fraction=opening_rows / total_rows if total_rows else 0.0,
        rows_by_ply_bucket=rows_by_bucket,
        mean_target_entropy_by_bucket={k: float(np.mean(v)) if v else 0.0 for k, v in entropies.items()},
        mean_effective_moves_by_bucket={k: float(np.mean(np.exp(v))) if v else 0.0 for k, v in entropies.items()},
        mean_tail_mass=float(np.mean(tail_masses)) if tail_masses else 0.0,
        duplicate_position_rate=1.0 - len(set(boards)) / len(boards) if boards else 0.0,
        duplicate_position_rate_mirrored=1.0 - len(mirrored) / len(boards) if boards else 0.0,
        unique_starts=len(starts),
        mean_games_per_start=num_games / len(starts) if starts else 0.0,
        white_win_rate=white_wins / num_games if num_games else 0.0,
        draw_rate=draws / num_games if num_games else 0.0,
        mean_absolute_margin=float(np.mean(margins)) if margins else 0.0,
    )


def _ply_bucket(ply: int) -> str:
    label = _PLY_BUCKETS[0][1]
    for lower, name in _PLY_BUCKETS:
        if ply >= lower:
            label = name
    return label


def _entropy(values: NDArray[np.float32]) -> float:
    """Shannon entropy of a target in nats (``exp(H)`` is its effective move count)."""
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0]
    return float(-(positive * np.log(positive)).sum())


def _mirror_key(board_bytes: bytes, board_size: int) -> bytes:
    """Mirror-collapsed position key, for the duplicate-rate diagnostic."""
    grid = np.frombuffer(board_bytes, dtype=np.int8).reshape(board_size, board_size)
    return min(board_bytes, np.ascontiguousarray(grid.T).tobytes())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_game_shard(path: Path, game: BlokusDuoGame) -> int:
    """Replay every game in a shard through the rules engine and check every row.

    v1's replay is unchanged; what changed is the policy assertion. Instead of "the
    target is the one-hot of the played move" the checks are: the target sums to 1, its
    support is a **subset** of the position's legal moves (never equality — Pentobi
    searches 315 of 414 first moves), the played action is in the support, and
    ``top_action`` is the target's argmax. Boards, side-to-move, outcome, margin and the
    terminal scores are checked exactly as in v1.

    Returns the number of positions checked; raises on any mismatch.
    """
    meta = read_game_shard_meta(path)
    table = pq.read_table(path)
    rows = {name: table.column(name).to_pylist() for name in table.column_names}
    cursor = 0
    checked = 0
    for game_meta, size in zip(meta.games, meta.game_sizes, strict=True):
        context = f"game {game_meta.game_id}"
        board = game.initialise_board()
        player = 1
        for action in game_meta.witness_actions:
            _require(bool(game.valid_move_masking(board, player)[action]), path, context, "illegal witness")
            board, player = game.get_next_state(board, player, action)
        white_margin = game_meta.white_margin
        for index in range(cursor, cursor + size):
            action = int(rows["action"][index])
            label = f"row {index}"
            _require(int(rows["player"][index]) == player, path, context, f"{label}: wrong side-to-move")
            mask = game.valid_move_masking(board, player)
            _require(bool(mask[action]), path, context, f"{label}: illegal action")
            expected = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
            _require(rows["board"][index] == expected.tobytes(), path, context, f"{label}: board mismatch")
            indices = np.frombuffer(rows["policy_indices"][index], dtype=np.int32)
            values = np.frombuffer(rows["policy_values"][index], dtype=np.float32)
            _require(
                abs(float(values.sum()) - 1.0) <= 1e-5,
                path,
                context,
                f"{label}: policy does not sum to 1",
            )
            legal = set(np.flatnonzero(mask).tolist())
            _require(set(indices.tolist()) <= legal, path, context, f"{label}: support is not within legal")
            _require(action in set(indices.tolist()), path, context, f"{label}: action outside the support")
            _require(
                int(rows["top_action"][index]) == int(indices[int(np.argmax(values))]),
                path,
                context,
                f"{label}: top_action is not the target's argmax",
            )
            _require(
                int(rows["margin"][index]) == white_margin * player,
                path,
                context,
                f"{label}: margin mismatch",
            )
            _require(
                float(rows["value"][index]) == float(np.sign(white_margin) * player),
                path,
                context,
                f"{label}: value mismatch",
            )
            board, player = game.get_next_state(board, player, action)
            checked += 1
        cursor += size
        _require(game.get_game_ended(board, player) != 0, path, context, "replayed game is not terminal")
        _require(
            game.final_scores(board) == (game_meta.white_score, game_meta.black_score),
            path,
            context,
            "final scores do not match stored labels",
        )
    return checked


def validate_opening_shard(path: Path, game: BlokusDuoGame, store: SearchSpaceStore | None = None) -> int:
    """Check every opening row: real position, legal support, consistent depth.

    Opening rows carry no move sequence of their own (the DAG has no unique one), so the
    parquet-only checks are structural: the stored board rebuilds into a real position
    whose legal set contains the whole target support, the target sums to 1, and ``depth``
    equals the number of pieces on the board. Pass the ``store`` to also replay each
    node's **witness path** and confirm it lands on exactly the stored key-frame board —
    the full check the v2 plan asks for.
    """
    meta = read_opening_meta(path)
    table = pq.read_table(path)
    rows = {name: table.column(name).to_pylist() for name in table.column_names}
    for index in range(table.num_rows):
        node_id = int(rows["node_id"][index])
        compact = np.frombuffer(rows["board"][index], dtype=np.dtype(meta.board_dtype)).reshape(meta.board_shape)
        indices = np.frombuffer(rows["policy_indices"][index], dtype=np.int32)
        values = np.frombuffer(rows["policy_values"][index], dtype=np.float32)
        label = f"node {node_id}"
        _require(abs(float(values.sum()) - 1.0) <= 1e-5, path, label, "policy does not sum to 1")
        _require(len(indices) == len(values), path, label, "policy indices/values are misaligned")
        legal = set(np.flatnonzero(game.valid_move_masking(game.board_from_compact(compact), 1)).tolist())
        _require(set(indices.tolist()) <= legal, path, label, "support is not within legal")
        placed = int(np.unique(compact[compact > 0]).size + np.unique(compact[compact < 0]).size)
        _require(int(rows["depth"][index]) == placed, path, label, "depth is not the pieces placed")
        if store is not None:
            record = store.node(node_id)
            board, player = store.board_at(node_id)
            witness = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
            expected = np.ascontiguousarray(witness.T) if record.key_frame else witness
            _require(expected.tobytes() == compact.tobytes(), path, label, "witness replay mismatch")
    return table.num_rows


def _require(condition: bool, path: Path, context: str, message: str) -> None:
    """Raise a uniform validation error when ``condition`` fails."""
    if not condition:
        raise CorpusSchemaError(f"{path.name} {context}: {message}")


def apply_target_temperature(values: NDArray[np.float32], temperature: float) -> NDArray[np.float32]:
    """Soften a stored soft target: ``p^(1/τ)`` renormalised over its support.

    Honest note (v2 plan's imitation-error block): τ is *confidence softening*, not an
    error correction. It is order-preserving, so the target's argmax is unchanged, and at
    nodes where Pentobi misallocates mass among its own candidates it does not reduce —
    and can increase — the target's expected regret. What it buys is a prior that
    play-time search can override more cheaply.
    """
    if temperature <= 0:
        raise CorpusSchemaError(f"target temperature must be positive, got {temperature}")
    if temperature == 1.0:
        return np.asarray(values, dtype=np.float32).copy()
    softened = np.asarray(values, dtype=np.float64) ** (1.0 / temperature)
    total = softened.sum()
    return (softened / total if total > 0 else softened).astype(np.float32)
