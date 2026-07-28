"""Tests for the store's link pass, opening export and coverage report (S5/S6).

The DAG here is built with synthetic searches (no engine on CI) but real positions, so
the exported boards, policy supports and witness paths are all checkable against the
rules engine — which is exactly what the V6 validator will do at scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.corpus_v2 import (
    DATASET_KIND,
    apply_target_temperature,
    export_opening,
    iter_opening_examples,
    opening_shards,
    opening_value,
    read_opening_meta,
)
from alphablokus.games.blokusduo.pentobi.store import (
    PlanParameters,
    SearchChild,
    SearchSpaceStore,
)
from alphablokus.games.blokusduo.pieces import default_pieces_path

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=default_pieces_path())


@pytest.fixture
def store(game: BlokusDuoGame, tmp_path: Path) -> Iterator[SearchSpaceStore]:
    with SearchSpaceStore(tmp_path / "store.sqlite", game, level=9) as opened:
        yield opened


def _synthetic_search(store: SearchSpaceStore, game: BlokusDuoGame, node_id: int, breadth: int = 4) -> None:
    board, player = store.board_at(node_id)
    legal = [int(a) for a in np.flatnonzero(game.valid_move_masking(board, player))][:breadth]
    store.record_search(
        node_id,
        [SearchChild(action=action, visits=1000 >> i, value=0.6 - 0.01 * i) for i, action in enumerate(legal)],
    )


def _planned_dag(store: SearchSpaceStore, game: BlokusDuoGame, budget: int = 200) -> int:
    """Build a small mapped DAG the way the ``plan`` command does, and activate the plan."""
    params = PlanParameters(budget=budget, temperature=2.0, min_replicas=2)
    for _ in range(20):
        draft = store.compute_plan(params)
        if not draft.mapping_queue:
            return store.save_plan(draft)
        for node_id in draft.mapping_queue:
            _synthetic_search(store, game, node_id)
    raise AssertionError("mapping did not converge")


# --------------------------------------------------------------------------- #
# S5: the link pass
# --------------------------------------------------------------------------- #


def test_link_aggregates_outcomes_up_the_dag(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """A start's games count for it and for every ancestor, signed per side to move.

    A White win is +1 at a node where White is to move and −1 where Black is, so an
    ancestor's ``outcome_mean`` reads as "how did the side to move here actually do".
    """
    _planned_dag(store, game)
    jobs = store.schedule(6)
    for index, job in enumerate(jobs):
        store.mark_done(job.node_id, job.replica, shard="s", white_margin=5 if index % 2 == 0 else -5, plies=30)
    assert store.link() > 0

    root = store.node(store.root_node())
    assert root.outcome_count == len(jobs)  # every game is in the root's subtree
    white_wins = sum(1 for index in range(len(jobs)) if index % 2 == 0)
    assert root.outcome_mean == pytest.approx((white_wins - (len(jobs) - white_wins)) / len(jobs))

    for job in jobs:
        node = store.node(job.node_id)
        assert node.outcome_count >= 1
        assert node.outcome_mean is not None
        assert -1.0 <= node.outcome_mean <= 1.0
        if node.player == -1:  # sign is flipped for a node where Black is to move
            assert node.outcome_mean == pytest.approx(-_white_outcome(store, job.node_id))


def _white_outcome(store: SearchSpaceStore, node_id: int) -> float:
    rows = store._connection.execute(  # noqa: SLF001 — cross-checking the SQL the pass uses
        "SELECT AVG(CASE WHEN white_margin > 0 THEN 1.0 WHEN white_margin < 0 THEN -1.0 ELSE 0.0 END) AS mean "
        "FROM playouts WHERE node_id = ? AND status = 'done'",
        (node_id,),
    ).fetchone()
    return float(rows["mean"])


def test_link_is_idempotent_and_resets_stale_counts(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """Re-running the pass recomputes from scratch rather than accumulating."""
    _planned_dag(store, game)
    job = store.schedule(1)[0]
    store.mark_done(job.node_id, job.replica, shard="s", white_margin=3, plies=28)
    store.link()
    first = store.node(store.root_node())
    store.link()
    second = store.node(store.root_node())
    assert (first.outcome_mean, first.outcome_count) == (second.outcome_mean, second.outcome_count)
    assert second.outcome_count == 1


def test_reach_weight_is_the_product_of_ancestor_visit_shares(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
) -> None:
    """Play-mass coverage: the root holds 1.0 and children hold their share of it."""
    root = store.root_node()
    _synthetic_search(store, game, root, breadth=3)
    edges = store.edges(root)
    children = [store.expand_child(root, edge.action) for edge in edges]
    weights = store.reach_weights()
    assert weights[root] == pytest.approx(1.0)
    for edge, child_id in zip(edges, children, strict=True):
        assert weights[child_id] == pytest.approx(edge.visit_share)
    assert sum(weights[child] for child in children) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# S5: the opening export
# --------------------------------------------------------------------------- #


def test_export_round_trips_through_the_trainer_reader(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    tmp_path: Path,
) -> None:
    """Exported rows read back as ``(board, sparse policy, value)`` training tuples.

    Every stored board must rebuild into a real position whose legal set contains the
    whole target support — the assertion that catches a corpus/rules-engine desync, and
    the one place the key-frame convention is checked end to end through parquet.
    """
    _planned_dag(store, game)
    out = tmp_path / "opening"
    (path,) = export_opening(store, out)
    assert opening_shards(out) == [path]

    meta = read_opening_meta(path)
    assert meta.level == 9
    assert meta.dag_hash == store.dag_hash()
    assert meta.policy_size == game.get_action_size()
    assert meta.plan_id == store.active_plan().plan_id  # type: ignore[union-attr]
    assert meta.num_rows == len(store.nodes(status="searched"))

    examples = list(iter_opening_examples([path]))
    assert len(examples) == meta.num_rows
    for board_compact, (indices, values), value in examples:
        assert board_compact.dtype == np.int8
        assert values.sum() == pytest.approx(1.0, abs=1e-5)
        assert len(indices) == len(values)
        board = game.board_from_compact(board_compact)
        legal = np.flatnonzero(game.valid_move_masking(board, 1))
        assert set(indices.tolist()) <= set(legal.tolist())  # support ⊆ legal, never equality
        assert -1.0 <= value <= 1.0
        planes = game.encode_compact(board_compact)
        assert planes.shape == (44, game.board_size, game.board_size)


def test_export_carries_the_soft_target_and_its_tail(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """The stored policy is the visit distribution, truncated at top-K with the loss recorded."""
    root = store.root_node()
    _synthetic_search(store, game, root, breadth=6)
    rows = {row.node_id: row for row in store.iter_opening_rows(top_k=3)}
    row = rows[root]
    edges = store.edges(root)[:3]
    assert row.policy_indices.tolist() == [edge.action for edge in edges]
    assert row.policy_values.sum() == pytest.approx(1.0, abs=1e-6)
    assert row.policy_values[0] > row.policy_values[-1]  # a distribution, not a one-hot
    assert row.child_values.tolist() == pytest.approx([edge.child_value for edge in edges])
    assert row.tail_mass == pytest.approx(1.0 - sum(edge.visit_share for edge in edges), abs=1e-6)
    assert row.search_value == pytest.approx(store.node(root).search_value)


def test_re_export_after_a_new_search_changes_the_stamp(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    tmp_path: Path,
) -> None:
    """A stale export is detectable: the ``dag_hash`` moves when the DAG does."""
    root = store.root_node()
    _synthetic_search(store, game, root, breadth=3)
    out = tmp_path / "opening"
    (first,) = export_opening(store, out)
    before = read_opening_meta(first).dag_hash

    child = store.expand_child(root, store.edges(root)[0].action)
    _synthetic_search(store, game, child, breadth=3)
    (second,) = export_opening(store, out)
    after = read_opening_meta(second).dag_hash
    assert after != before
    assert read_opening_meta(second).num_rows == 2
    assert opening_shards(out) == [second]  # the stale export was replaced, not mixed in


def test_opening_value_blends_the_teacher_into_the_outcomes() -> None:
    """``blend`` equals the rescaled search value at n = 0 and the outcomes as n grows."""
    assert opening_value(0.75, 0.0, 0, target="blend") == pytest.approx(0.5)
    assert opening_value(0.75, -1.0, 0, target="outcome") == pytest.approx(0.5)  # no data ⇒ teacher
    assert opening_value(0.75, -1.0, 100, target="outcome") == pytest.approx(-1.0)
    assert opening_value(0.75, -1.0, 5, target="blend", blend_k=5) == pytest.approx(-0.25)
    assert opening_value(0.75, -1.0, 100, target="search") == pytest.approx(0.5)


def test_target_temperature_softens_without_reordering() -> None:
    """τ is confidence softening: order-preserving, renormalised over the stored support."""
    values = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    softened = apply_target_temperature(values, 2.0)
    assert softened.sum() == pytest.approx(1.0, abs=1e-6)
    assert list(np.argsort(-softened)) == list(np.argsort(-values))  # argmax cannot move
    assert softened[0] < values[0]
    assert softened[-1] > values[-1]
    assert apply_target_temperature(values, 1.0).tolist() == pytest.approx(values.tolist())


def test_opening_shards_are_marked_v2(store: SearchSpaceStore, game: BlokusDuoGame, tmp_path: Path) -> None:
    _synthetic_search(store, game, store.root_node(), breadth=2)
    (path,) = export_opening(store, tmp_path / "opening")
    import pyarrow.parquet as pq

    raw = pq.read_schema(path).metadata or {}
    assert raw[b"dataset_kind"].decode() == DATASET_KIND


# --------------------------------------------------------------------------- #
# S6: coverage
# --------------------------------------------------------------------------- #


def test_coverage_reports_the_plan_and_its_fulfilment(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """ "Done" for a plan = mapping debt zero and fulfilment 1.0 — both must be visible."""
    plan_id = _planned_dag(store, game)
    report = store.coverage()
    assert report.plan is not None
    assert report.plan.plan_id == plan_id
    assert report.dag_hash == store.dag_hash()
    assert report.planned_games == 200
    assert report.actual_games == 0
    assert report.mapping_debt == 0  # the plan loop searched everything it needed
    assert report.fulfilment_histogram["0%"] == len(
        [a for a in store.plan_allocations(plan_id) if a.planned_games > 0],
    )
    assert report.planned_games_min >= 1
    assert report.planned_games_max <= 200
    assert sum(report.budget_share_by_depth.values()) == pytest.approx(1.0)
    assert min(report.nodes_by_depth) == 0  # the root is depth 0
    assert report.distinct_first_moves >= 1
    assert report.distinct_first_positions <= report.distinct_first_moves

    jobs = store.schedule(10)
    for job in jobs:
        store.mark_done(job.node_id, job.replica, shard="s", white_margin=1, plies=30)
    after = store.coverage()
    assert after.actual_games == 10
    assert after.fulfilment_histogram["0%"] < report.fulfilment_histogram["0%"]
    assert after.to_dict()["fulfilment"] == pytest.approx(10 / 200)


def test_coverage_counts_mapping_debt_when_the_dag_is_incomplete(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
) -> None:
    """A plan saved against an unmapped DAG shows its debt rather than pretending."""
    _synthetic_search(store, game, store.root_node(), breadth=4)
    draft = store.compute_plan(PlanParameters(budget=200, temperature=2.0, min_replicas=2))
    assert draft.mapping_queue  # children exist but are unsearched
    store.save_plan(draft)
    assert store.coverage().mapping_debt == len(draft.mapping_queue)


def test_coverage_without_a_plan_is_still_a_report(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """The DAG can be inspected before anything is planned."""
    _synthetic_search(store, game, store.root_node(), breadth=3)
    report = store.coverage()
    assert report.plan is None
    assert report.planned_games == 0
    assert report.nodes_by_depth == {0: 1}
    assert report.to_dict()["plan_id"] is None


def test_v1_corpus_is_registered_in_the_manifest_not_the_graph(store: SearchSpaceStore) -> None:
    """v1's junk openings stay out of the DAG; the manifest still knows we hold them."""
    store.root_node()
    store.register_corpus(
        "pentobi_l9_v1",
        "temp/corpus_l9",
        "pentobi_distill_v1",
        games=13_000,
        positions=340_000,
        notes="mid-game supplement; uniform-random unharvested openings",
    )
    (row,) = store.corpora()
    assert row["dataset_kind"] == "pentobi_distill_v1"
    assert row["games"] == 13_000
    assert len(store.nodes()) == 1  # only the root — nothing was grafted onto the graph
