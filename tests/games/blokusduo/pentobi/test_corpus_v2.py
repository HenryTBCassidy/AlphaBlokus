"""End-to-end tests for v2 generation and schema v2 (plan rows V5/V6).

The full pipeline runs here without an engine: plan → map → schedule → play → write
shards → validate → reconcile, all driven by :class:`RandomSearchSource` through the real
rules engine. That is the same path the box will run; only the strength of the moves
differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.corpus import CorpusGenerationError
from alphablokus.games.blokusduo.pentobi.corpus_v2 import (
    CorpusSchemaError,
    GameShardMeta,
    analyze_corpus,
    build_soft_target,
    game_shard_filename,
    game_shards,
    iter_game_examples,
    iter_shard_playouts,
    read_game_shard_meta,
    validate_game_shard,
    validate_opening_shard,
    write_game_shard,
)
from alphablokus.games.blokusduo.pentobi.corpus_v2 import (
    export_opening as export_opening_dataset,
)
from alphablokus.games.blokusduo.pentobi.harvest import (
    HarvestedGame,
    RandomSearchSource,
    map_plan,
    play_planned_game,
)
from alphablokus.games.blokusduo.pentobi.store import (
    PlanParameters,
    SearchChild,
    SearchSpaceStore,
)
from alphablokus.games.blokusduo.pieces import default_pieces_path
from alphablokus.storage.sparse_policy import densify

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


def _generate(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    *,
    budget: int = 60,
    games: int = 3,
) -> list[HarvestedGame]:
    """Run the real phase A + phase B loop with the engine-free source."""
    source = RandomSearchSource(game, breadth=5)
    store.save_plan(map_plan(store, source, PlanParameters(budget, 2.0, 2)))
    return [play_planned_game(game, source, job, top_k=8) for job in store.schedule(games)]


def _shard_meta(store: SearchSpaceStore, game: BlokusDuoGame, games: list[HarvestedGame]) -> GameShardMeta:
    plan = store.active_plan()
    assert plan is not None
    return GameShardMeta(
        level=9,
        policy_size=game.get_action_size(),
        board_shape=(game.board_size, game.board_size),
        board_dtype="int8",
        dag_hash=store.dag_hash(),
        plan_id=plan.plan_id,
        budget=plan.parameters.budget,
        temperature=plan.parameters.temperature,
        min_replicas=plan.parameters.min_replicas,
        game_sizes=tuple(len(g.plies) for g in games),
        games=(),
    )


# --------------------------------------------------------------------------- #
# V5: the game loop
# --------------------------------------------------------------------------- #


def test_games_start_at_their_planned_node_and_harvest_every_ply(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
) -> None:
    """The witness prefix is replayed (never re-searched); every ply after it is harvested.

    v1 harvested only Pentobi's plies from a random 4-ply prefix and threw the opening
    away; v2's prefix is a *planned* opening whose plies already carry labels in the DAG,
    and the start position itself is searched in-game.
    """
    (harvested,) = _generate(store, game, games=1)
    record = store.node(harvested.node_id)
    assert harvested.witness_actions == record.witness_actions
    assert harvested.plies[0].ply == len(record.witness_actions)  # harvesting starts at the start node
    assert [ply.ply for ply in harvested.plies] == list(
        range(len(record.witness_actions), len(record.witness_actions) + len(harvested.plies)),
    )
    assert harvested.actions[: len(record.witness_actions)] == record.witness_actions
    assert len(harvested.plies) > 10  # a real game, played to the end


def test_every_harvested_ply_carries_a_soft_target(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """Not a one-hot: the whole preference distribution, with the truncation recorded."""
    (harvested,) = _generate(store, game, games=1)
    for ply in harvested.plies:
        assert ply.target.values.sum() == pytest.approx(1.0, abs=1e-5)
        assert ply.action == ply.top_action  # full-strength continuation: no temperature
        assert ply.action == int(ply.target.indices[0])
        assert 0.0 <= ply.target.tail_mass < 1.0
        assert len(ply.target.child_values) == len(ply.target.indices)
    assert any(len(ply.target.indices) > 1 for ply in harvested.plies)


def test_replay_of_every_game_matches_the_rules_engine(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """The stored boards, sides to move and scores are what a replay produces."""
    for harvested in _generate(store, game, games=2):
        board = game.initialise_board()
        player = 1
        by_ply = {ply.ply: ply for ply in harvested.plies}
        for index, action in enumerate(harvested.actions):
            assert game.valid_move_masking(board, player)[action] == 1
            if index in by_ply:
                ply = by_ply[index]
                assert ply.player == player
                expected = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
                assert np.array_equal(ply.compact_board, expected)
            board, player = game.get_next_state(board, player, action)
        assert game.get_game_ended(board, player) != 0
        assert game.final_scores(board) == (harvested.white_score, harvested.black_score)


def test_a_lying_engine_margin_is_caught(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """The v1 desync guard survives into v2 unchanged."""

    class LyingSource(RandomSearchSource):
        def final_white_margin(self) -> int:
            return 999

    source = LyingSource(game, breadth=4)
    store.save_plan(map_plan(store, source, PlanParameters(60, 2.0, 2)))
    with pytest.raises(CorpusGenerationError, match="margin"):
        play_planned_game(game, source, store.schedule(1)[0])


def test_a_witness_path_that_does_not_reach_its_node_is_caught(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
) -> None:
    """Each game's replayed prefix is validated against its start node's position."""
    import dataclasses

    source = RandomSearchSource(game, breadth=4)
    store.save_plan(map_plan(store, source, PlanParameters(60, 2.0, 2)))
    job = store.schedule(1)[0]
    tampered = dataclasses.replace(job, board_key=b"\x7f" * 196)
    with pytest.raises(CorpusGenerationError, match="start node"):
        play_planned_game(game, source, tampered)


def test_soft_target_truncation_records_what_it_dropped() -> None:
    """``tail_mass`` is the visit mass the top-K cut, not a rounding of the kept mass."""
    children = [SearchChild(action=index, visits=100 - 10 * index, value=0.5) for index in range(5)]
    target = build_soft_target(children, top_k=2)
    assert target.indices.tolist() == [0, 1]
    assert target.values.tolist() == pytest.approx([100 / 190, 90 / 190])
    assert target.tail_mass == pytest.approx(1.0 - 190 / 400)
    assert build_soft_target(children, top_k=99).tail_mass == pytest.approx(0.0)
    with pytest.raises(CorpusSchemaError):
        build_soft_target([], top_k=8)


def test_soft_target_of_an_unvisited_search_is_uniform() -> None:
    """All-prior children carry no preference — say so, rather than dividing by zero."""
    children = [SearchChild(action=index, visits=0, value=0.1) for index in range(4)]
    target = build_soft_target(children, top_k=8)
    assert target.values.tolist() == pytest.approx([0.25] * 4)
    assert target.tail_mass == 0.0


# --------------------------------------------------------------------------- #
# V6: schema v2 + validator
# --------------------------------------------------------------------------- #


def test_game_shard_round_trip_and_validation(store: SearchSpaceStore, game: BlokusDuoGame, tmp_path: Path) -> None:
    """Shards are self-describing, replay-validated, and readable as training examples."""
    games = _generate(store, game, games=3)
    directory = tmp_path / "games"
    directory.mkdir()
    path = directory / game_shard_filename(0)
    meta = _shard_meta(store, game, games)
    rows = write_game_shard(path, games, meta=meta)
    assert rows == sum(len(g.plies) for g in games)
    assert game_shards(directory) == [path]

    stored = read_game_shard_meta(path)
    assert stored.level == 9
    assert stored.dag_hash == store.dag_hash()
    assert stored.plan_id == meta.plan_id
    assert stored.game_sizes == tuple(len(g.plies) for g in games)
    assert [g.board_key for g in stored.games] == [g.board_key.hex() for g in games]
    assert [g.replica for g in stored.games] == [g.replica for g in games]
    assert [g.engine_seed for g in stored.games] == [g.engine_seed for g in games]
    assert [g.witness_actions for g in stored.games] == [g.witness_actions for g in games]

    assert validate_game_shard(path, game) == rows

    examples = list(iter_game_examples([path]))
    assert len(examples) == rows
    flat = [ply for g in games for ply in g.plies]
    for (board, (indices, values), value), ply in zip(examples, flat, strict=True):
        assert np.array_equal(board, ply.compact_board)
        dense = densify(indices, values, game.get_action_size())
        assert dense.sum() == pytest.approx(1.0, abs=1e-5)
        assert dense[ply.action] > 0.0
        assert value in (-1.0, 0.0, 1.0)


def test_validator_rejects_a_tampered_policy(store: SearchSpaceStore, game: BlokusDuoGame, tmp_path: Path) -> None:
    """The one-hot assertion is gone, but the target's own invariants are enforced."""
    import dataclasses

    games = _generate(store, game, games=1)
    harvested = games[0]
    first = harvested.plies[0]
    broken_target = dataclasses.replace(first.target, values=(first.target.values * 2.0).astype(np.float32))
    broken = dataclasses.replace(
        harvested,
        plies=(dataclasses.replace(first, target=broken_target), *harvested.plies[1:]),
    )
    path = tmp_path / game_shard_filename(0)
    write_game_shard(path, [broken], meta=_shard_meta(store, game, [broken]))
    with pytest.raises(CorpusSchemaError, match="sum to 1"):
        validate_game_shard(path, game)


def test_validator_rejects_a_top_action_that_is_not_the_argmax(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    tmp_path: Path,
) -> None:
    """``action`` and ``top_action`` are stored separately so a mismatch is visible."""
    import dataclasses

    (harvested,) = _generate(store, game, games=1)
    first = harvested.plies[0]
    assert len(first.target.indices) > 1
    broken = dataclasses.replace(
        harvested,
        plies=(dataclasses.replace(first, top_action=int(first.target.indices[1])), *harvested.plies[1:]),
    )
    path = tmp_path / game_shard_filename(0)
    write_game_shard(path, [broken], meta=_shard_meta(store, game, [broken]))
    with pytest.raises(CorpusSchemaError, match="top_action"):
        validate_game_shard(path, game)


def test_opening_shard_validates_against_the_store(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    tmp_path: Path,
) -> None:
    """Opening rows check out structurally *and* against a replay of their witness path."""
    _generate(store, game, games=1)
    (path,) = export_opening_dataset(store, tmp_path / "opening")
    assert validate_opening_shard(path, game) > 0
    assert validate_opening_shard(path, game, store) == len(store.nodes(status="searched"))


def test_analyze_measures_target_richness_and_row_mix(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    tmp_path: Path,
) -> None:
    """The diagnostics report the things v1 failed on, not just row counts.

    Target entropy above zero is the direct measure of "we kept more than a one-hot"; the
    opening-row fraction is the row-mix problem V9's sampling weights have to correct.
    """
    games = _generate(store, game, budget=60, games=3)
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    write_game_shard(games_dir / game_shard_filename(0), games, meta=_shard_meta(store, game, games))
    opening_dir = tmp_path / "opening"
    export_opening_dataset(store, opening_dir)

    report = analyze_corpus(games_dir, opening_dir)
    assert report.num_games == 3
    assert report.num_game_rows == sum(len(g.plies) for g in games)
    assert report.num_opening_rows == len(store.nodes(status="searched"))
    assert 0.0 < report.opening_row_fraction < 1.0
    assert sum(report.rows_by_ply_bucket.values()) == report.num_game_rows
    assert max(report.mean_target_entropy_by_bucket.values()) > 0.0  # not a pile of one-hots
    assert max(report.mean_effective_moves_by_bucket.values()) > 1.0
    assert 0.0 <= report.mean_tail_mass < 1.0
    assert 0.0 <= report.duplicate_position_rate <= 1.0
    assert report.duplicate_position_rate_mirrored >= report.duplicate_position_rate
    assert report.unique_starts >= 1
    assert report.mean_games_per_start == pytest.approx(3 / report.unique_starts)
    assert 0.0 <= report.white_win_rate <= 1.0
    assert report.to_dict()["num_games"] == 3


def test_shard_footers_reconcile_the_playout_registry(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    tmp_path: Path,
) -> None:
    """The DB is an index over the shards: losing it costs nothing but a reconcile."""
    games = _generate(store, game, games=3)
    directory = tmp_path / "games"
    directory.mkdir()
    write_game_shard(directory / game_shard_filename(0), games, meta=_shard_meta(store, game, games))

    entries = list(iter_shard_playouts(directory))
    assert len(entries) == len(games)
    result = store.reconcile(entries)
    assert result.unknown_nodes == ()
    assert store.playout_counts()["done"] == len(games)
    for harvested, entry in zip(games, entries, strict=True):
        assert entry.board_key == harvested.board_key
        assert entry.white_margin == harvested.white_margin
        assert entry.plies == len(harvested.plies)

    store.link()
    assert store.node(store.root_node()).outcome_count == len(games)
