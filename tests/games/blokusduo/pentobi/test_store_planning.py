"""Tests for the store's allocation planner and playout registry (store design S3/S4).

Two levels. :func:`allocate_budget` is a **pure function** and is tested against a
hand-written synthetic DAG where every expected number can be worked out on paper. The
store-level plan is then built the way the real ``plan`` command builds it — compute,
search whatever lands in the mapping queue, recompute — but with synthetic visit
distributions instead of an engine, so the whole thing runs on CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.store import (
    PlanParameters,
    ReconcileEntry,
    SearchChild,
    SearchSpaceStore,
    StoreError,
    allocate_budget,
    integerise_budgets,
    playout_seed,
)
from alphablokus.games.blokusduo.pieces import default_pieces_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=default_pieces_path())


@pytest.fixture
def store(game: BlokusDuoGame, tmp_path: Path) -> Iterator[SearchSpaceStore]:
    with SearchSpaceStore(tmp_path / "store.sqlite", game, level=9) as opened:
        yield opened


# --------------------------------------------------------------------------- #
# The pure allocator (S3)
# --------------------------------------------------------------------------- #


class FakeDag:
    """A synthetic searched DAG: ``{node: [(action, visit_share), ...]}`` plus depths."""

    def __init__(self, children: dict[int, list[tuple[int, float]]], unsearched: set[int] | None = None) -> None:
        self._children = children
        self._unsearched = unsearched or set()
        self._depths = {0: 0}
        self.instantiated: list[tuple[int, int]] = []

    def depth_of(self, node_id: int) -> int:
        return self._depths[node_id]

    def is_searched(self, node_id: int) -> bool:
        """Searched unless declared otherwise; a node with no children entry is a leaf."""
        return node_id not in self._unsearched

    def children_of(self, node_id: int) -> Sequence[tuple[int, float]]:
        return self._children.get(node_id, [])

    def instantiate(self, parent_id: int, action: int) -> int:
        """Actions *are* child ids in the fake DAG — one child per action, no aliasing."""
        self.instantiated.append((parent_id, action))
        self._depths[action] = self._depths[parent_id] + 1
        return action

    def split(self, params: PlanParameters) -> object:
        return allocate_budget(
            0,
            params,
            depth_of=self.depth_of,
            is_searched=self.is_searched,
            children_of=self.children_of,
            instantiate=self.instantiate,
        )


def test_budget_is_conserved_down_the_tree() -> None:
    """Every game handed to the root arrives at exactly one playout start."""
    dag = FakeDag({0: [(1, 0.5), (2, 0.3), (3, 0.2)], 1: [(4, 0.9), (5, 0.1)]})
    split = dag.split(PlanParameters(budget=100, temperature=1.0, min_replicas=2))
    assert sum(split.starts.values()) == pytest.approx(100.0)
    assert split.mapping_queue == ()
    # Node 1's 50 games split 0.9/0.1 (both above the floor of 2); 2 and 3 are leaves.
    assert split.starts == pytest.approx({4: 45.0, 5: 5.0, 2: 30.0, 3: 20.0})


def test_a_node_below_twice_the_floor_is_a_playout_start() -> None:
    """``b < 2R`` stops the recursion — depth is an output of the budget, not an input."""
    dag = FakeDag({0: [(1, 0.5), (2, 0.5)], 1: [(3, 1.0)]})
    split = dag.split(PlanParameters(budget=3, temperature=1.0, min_replicas=2))
    assert split.starts == pytest.approx({0: 3.0})
    assert dag.instantiated == []  # nothing expanded: the DAG grows only as far as needed


def test_children_below_the_floor_are_dropped_and_the_rest_renormalised() -> None:
    """Dropping only ever raises the survivors' budgets, so one pass is enough."""
    dag = FakeDag({0: [(1, 0.8), (2, 0.19), (3, 0.01)]})
    split = dag.split(PlanParameters(budget=100, temperature=1.0, min_replicas=2))
    assert set(split.starts) == {1, 2}  # child 3's 1.0 games are below R = 2
    assert sum(split.starts.values()) == pytest.approx(100.0)
    assert split.starts[1] == pytest.approx(100 * 0.8 / 0.99)


def test_temperature_flattens_towards_the_tail() -> None:
    """``w ∝ share^(1/T)``: the sqrt default moves mass onto the under-searched tail."""
    children = {0: [(1, 0.9), (2, 0.09), (3, 0.01)]}
    raw = FakeDag(dict(children)).split(PlanParameters(10_000, 1.0, 2)).starts
    flat = FakeDag(dict(children)).split(PlanParameters(10_000, 2.0, 2)).starts
    assert raw[1] > flat[1]
    assert flat[3] > raw[3]
    assert sum(flat.values()) == pytest.approx(10_000.0)


def test_a_dag_node_accumulates_both_parents_budgets_before_splitting() -> None:
    """Depth-ordered traversal is what makes multi-parent nodes correct.

    Node 3 is reached from both 1 and 2 with 25 games each; it must split its combined
    50, not twice-split 25 (which at R = 20 would make it a leaf both times).
    """
    dag = FakeDag(
        {
            0: [(1, 0.5), (2, 0.5)],
            1: [(3, 0.5), (6, 0.5)],
            2: [(3, 0.5), (7, 0.5)],
            3: [(4, 0.6), (5, 0.4)],
        },
    )
    split = dag.split(PlanParameters(budget=100, temperature=1.0, min_replicas=10))
    assert split.budgets[3] == pytest.approx(50.0)
    assert split.starts == pytest.approx({6: 25.0, 7: 25.0, 4: 30.0, 5: 20.0})


def test_an_unsearched_node_becomes_mapping_debt_and_a_provisional_start() -> None:
    """A plan against an incomplete DAG is still usable — just shallower."""
    dag = FakeDag({0: [(1, 0.5), (2, 0.5)], 1: [(3, 1.0)]}, unsearched={1})
    split = dag.split(PlanParameters(budget=100, temperature=1.0, min_replicas=2))
    assert split.mapping_queue == (1,)
    assert split.starts == pytest.approx({1: 50.0, 2: 50.0})


def test_integerisation_conserves_the_budget_exactly() -> None:
    """Largest-remainder rounding: whole games, right total, deterministic ties."""
    starts = {1: 33.4, 2: 33.3, 3: 33.3}
    planned = integerise_budgets(starts, 100)
    assert sum(planned.values()) == 100
    assert planned == {1: 34, 2: 33, 3: 33}
    assert integerise_budgets({1: 0.5, 2: 0.5, 3: 1.0}, 2) == {1: 1, 2: 0, 3: 1}
    assert integerise_budgets({}, 10) == {}


# --------------------------------------------------------------------------- #
# The store-level plan (S3)
# --------------------------------------------------------------------------- #


def _synthetic_search(store: SearchSpaceStore, game: BlokusDuoGame, node_id: int, breadth: int = 4) -> None:
    """Record a synthetic, hyper-concentrated search at a node (no engine needed).

    Visits fall off geometrically, which is the shape Pentobi actually produces (v2 plan
    fact 6: top-1 visit share 0.4–0.98 at every ply).
    """
    board, player = store.board_at(node_id)
    legal = [int(a) for a in np.flatnonzero(game.valid_move_masking(board, player))][:breadth]
    store.record_search(
        node_id,
        [SearchChild(action=action, visits=1000 >> i, value=0.6 - 0.01 * i) for i, action in enumerate(legal)],
    )


def _map_plan(store: SearchSpaceStore, game: BlokusDuoGame, params: PlanParameters, *, breadth: int = 4) -> object:
    """Run the real ``plan`` loop: compute, search the mapping queue, repeat."""
    for _ in range(20):
        draft = store.compute_plan(params)
        if not draft.mapping_queue:
            return draft
        for node_id in draft.mapping_queue:
            _synthetic_search(store, game, node_id, breadth)
    raise AssertionError("mapping did not converge")


def test_plan_conserves_the_budget_and_emerges_in_depth(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """The plan spends exactly its budget, and its depth is an output of (B, T, R)."""
    params = PlanParameters(budget=200, temperature=2.0, min_replicas=2)
    draft = _map_plan(store, game, params)
    assert draft.planned_games == 200
    assert all(allocation.planned_games >= 0 for allocation in draft.allocations)
    depths = {store.node(a.node_id).depth for a in draft.starts}
    assert max(depths) > 1  # the budget reached past ply 1 on its own
    assert sum(a.budget_share for a in draft.starts) == pytest.approx(1.0)


def test_recomputing_the_plan_reproduces_it_exactly(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """Allocation is a pure function of (DAG content, B, T, R) — the reproducibility rule."""
    params = PlanParameters(budget=200, temperature=2.0, min_replicas=2)
    first = _map_plan(store, game, params)
    again = store.compute_plan(params)
    assert again.mapping_queue == ()
    assert again.allocations == first.allocations
    plan_id = store.save_plan(first)
    stored = store.plan_allocations(plan_id)
    assert stored == list(first.allocations)
    assert store.active_plan() is not None
    assert store.active_plan().plan_id == plan_id  # type: ignore[union-attr]


def test_a_bigger_budget_refines_the_plan_rather_than_contradicting_it(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
) -> None:
    """Top-up monotonicity: internal nodes stay internal, surviving starts never shrink.

    This is what makes "generate 5k more games" a re-plan instead of a migration — games
    already generated stay valid and attached even where a former start turned internal.
    """
    small = _map_plan(store, game, PlanParameters(budget=200, temperature=2.0, min_replicas=2))
    large = _map_plan(store, game, PlanParameters(budget=400, temperature=2.0, min_replicas=2))
    small_planned = {a.node_id: a.planned_games for a in small.allocations}
    large_planned = {a.node_id: a.planned_games for a in large.allocations}
    for node_id, planned in small_planned.items():
        if planned == 0:  # internal under the small plan must stay internal
            assert large_planned.get(node_id, 0) == 0
        elif large_planned.get(node_id, 0) > 0:  # still a start ⇒ never fewer games
            assert large_planned[node_id] >= planned
    assert large.planned_games == 400


def test_book_lines_are_mapped_and_floored(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """A book line gets ``R`` games at its terminal whatever the allocation thinks of it.

    The floor is reserved out of the budget, so the total is still exactly ``B``.
    """
    board = game.initialise_board()
    line = []
    player = 1
    for _ in range(4):  # a four-ply line the visit-weighted allocation would never reach
        action = int(np.flatnonzero(game.valid_move_masking(board, player))[-1])
        line.append(action)
        board, player = game.get_next_state(board, player, action)
    (terminal,) = store.insert_book_paths([line])
    assert store.node(terminal).book_terminal
    assert store.node(terminal).witness_actions == tuple(line)

    params = PlanParameters(budget=200, temperature=2.0, min_replicas=2)
    draft = _map_plan(store, game, params)
    planned = {a.node_id: a.planned_games for a in draft.allocations}
    assert planned[terminal] >= params.min_replicas
    assert draft.planned_games == 200


def test_a_budget_smaller_than_the_book_floors_is_rejected(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """Reserving more than the budget would silently break budget conservation."""
    board = game.initialise_board()
    action = int(np.flatnonzero(game.valid_move_masking(board, 1))[0])
    store.insert_book_paths([[action]])
    with pytest.raises(StoreError, match="book floors"):
        store.compute_plan(PlanParameters(budget=2, temperature=2.0, min_replicas=2))


# --------------------------------------------------------------------------- #
# The playout registry (S4)
# --------------------------------------------------------------------------- #


def _planned_store(store: SearchSpaceStore, game: BlokusDuoGame, budget: int = 200) -> int:
    draft = _map_plan(store, game, PlanParameters(budget=budget, temperature=2.0, min_replicas=2))
    return store.save_plan(draft)


def test_scheduling_is_an_even_proportional_slice_of_the_plan(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
) -> None:
    """Truncate a generation run anywhere and every node is at the same fulfilment ±1.

    That is the property Henry's D5 stratified opening keys guaranteed, generalised to
    heterogeneous per-node targets: a run stopped at the end of its box window holds a
    proportional slice of the plan, not its first few openings.
    """
    plan_id = _planned_store(store, game)
    planned = {a.node_id: a.planned_games for a in store.plan_allocations(plan_id) if a.planned_games > 0}
    jobs = store.schedule(int(sum(planned.values()) * 0.25))
    scheduled: dict[int, int] = {}
    for job in jobs:
        scheduled[job.node_id] = scheduled.get(job.node_id, 0) + 1
    assert set(scheduled) <= set(planned)
    fulfilments = [scheduled.get(node_id, 0) / count for node_id, count in planned.items()]
    assert max(fulfilments) - min(fulfilments) <= 0.5  # no node runs far ahead of the rest
    assert all(scheduled[node_id] <= planned[node_id] for node_id in scheduled)


def test_scheduling_never_repeats_a_node_replica_pair(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """Identity is ``(board_key, replica)`` with a content-derived seed, across plans."""
    _planned_store(store, game, budget=200)
    first = store.schedule(50)
    second = store.schedule(50)
    pairs = [(job.node_id, job.replica) for job in (*first, *second)]
    assert len(set(pairs)) == len(pairs)
    assert len({job.game_id for job in (*first, *second)}) == len(pairs)
    for job in (*first, *second):
        assert job.engine_seed == playout_seed(store.node(job.node_id).board_key, job.replica)

    # A re-plan at a larger budget adds targets; it can never re-issue existing games.
    store.save_plan(_map_plan(store, game, PlanParameters(budget=400, temperature=2.0, min_replicas=2)))
    third = store.schedule(50)
    assert not {(job.node_id, job.replica) for job in third} & set(pairs)


def test_scheduling_stops_at_the_plans_targets(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """A plan is a budget, not a suggestion: asking for more returns nothing extra."""
    plan_id = _planned_store(store, game, budget=100)
    planned = sum(a.planned_games for a in store.plan_allocations(plan_id))
    assert len(store.schedule(planned + 50)) == planned
    assert store.schedule(10) == []


def test_scheduling_without_a_plan_raises(store: SearchSpaceStore) -> None:
    with pytest.raises(StoreError, match="no active plan"):
        store.schedule(1)


def test_pending_jobs_are_what_a_resumed_run_re_executes(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """Same seeds ⇒ same games ⇒ exactly the missing shards are regenerated."""
    _planned_store(store, game, budget=100)
    jobs = store.schedule(10)
    store.mark_done(jobs[0].node_id, jobs[0].replica, shard="corpus_00000.parquet", white_margin=7, plies=30)
    pending = store.pending_jobs()
    assert len(pending) == 9
    assert (jobs[0].node_id, jobs[0].replica) not in {(job.node_id, job.replica) for job in pending}
    assert [job.engine_seed for job in pending] == [job.engine_seed for job in jobs[1:]]
    assert store.playout_counts() == {"planned": 9, "done": 1}


def test_marking_an_unscheduled_game_done_raises(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    _planned_store(store, game, budget=100)
    with pytest.raises(StoreError, match="no scheduled playout"):
        store.mark_done(1, 99, shard="x.parquet", white_margin=0, plies=0)


def test_reconcile_rebuilds_the_registry_from_shard_footers(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """Shards are the truth for games; ``playouts`` is an index that can be rebuilt.

    This is the crash repair for a run that died between a shard rename and its DB
    transaction — and the audit that the DB and the shards still agree.
    """
    _planned_store(store, game, budget=100)
    jobs = store.schedule(4)
    footers = [
        ReconcileEntry(
            board_key=job.board_key,
            replica=job.replica,
            game_id=job.game_id,
            shard="corpus_00000.parquet",
            white_margin=5 - index,
            plies=30 + index,
        )
        for index, job in enumerate(jobs)
    ]
    first = store.reconcile(footers)
    assert (first.inserted, first.updated, first.matched) == (0, 4, 0)
    assert store.playout_counts() == {"done": 4}

    again = store.reconcile(footers)
    assert (again.inserted, again.updated, again.matched) == (0, 0, 4)  # idempotent

    store._connection.execute("DELETE FROM playouts")  # noqa: SLF001 — simulating a lost DB
    store._connection.commit()  # noqa: SLF001
    rebuilt = store.reconcile(footers)
    assert (rebuilt.inserted, rebuilt.updated, rebuilt.matched) == (4, 0, 0)
    assert store.playout_counts() == {"done": 4}
    assert [job.engine_seed for job in jobs] == [playout_seed(footer.board_key, footer.replica) for footer in footers]


def test_reconcile_reports_footers_the_dag_does_not_know(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """A footer for a position outside the DAG is reported, never silently invented."""
    _planned_store(store, game, budget=100)
    result = store.reconcile(
        [ReconcileEntry(board_key=b"\x7f" * 196, replica=0, game_id=0, shard="s", white_margin=1, plies=2)],
    )
    assert result.unknown_nodes == (b"\x7f" * 196,)
