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

    from alphablokus.games.blokusduo.pentobi.corpus import CorpusExample
    from alphablokus.games.blokusduo.pentobi.store import OpeningRow, PlanRecord, SearchSpaceStore

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
