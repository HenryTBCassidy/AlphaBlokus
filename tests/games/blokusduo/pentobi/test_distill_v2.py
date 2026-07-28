"""Tests for the v2 SL dataloader: soft targets, subtree holdout, source mixing (V9).

A tiny real v2 corpus is generated engine-free (real rules engine, real plan, real
parquet) and then streamed through the training path, so what is checked here is the
data the trainer would actually see.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.corpus_v2 import (
    GameShardMeta,
    export_opening,
    game_shard_filename,
    game_shards,
    opening_shards,
    write_game_shard,
)
from alphablokus.games.blokusduo.pentobi.distill import (
    build_training_examples,
    load_corpus_games_v2,
    load_opening_examples,
    mix_examples,
    opening_unit_for,
    partition_by_unit,
    soft_target_over_legal,
    split_opening_units,
)
from alphablokus.games.blokusduo.pentobi.harvest import RandomSearchSource, map_plan, play_planned_game
from alphablokus.games.blokusduo.pentobi.store import PlanParameters, SearchSpaceStore, canonical_key
from alphablokus.games.blokusduo.pieces import default_pieces_path
from alphablokus.storage.sparse_policy import densify

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.games.blokusduo.pentobi.distill import CorpusGameRows


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=default_pieces_path())


@pytest.fixture(scope="module")
def corpus(game: BlokusDuoGame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny real v2 corpus: a mapped plan, a handful of games, both datasets on disk."""
    directory = tmp_path_factory.mktemp("corpus_v2")
    (directory / "games").mkdir()
    with SearchSpaceStore(directory / "store.sqlite", game, level=9) as store:
        source = RandomSearchSource(game, breadth=5)
        store.save_plan(map_plan(store, source, PlanParameters(120, 2.0, 2)))
        played = [play_planned_game(game, source, job, top_k=8) for job in store.schedule(8)]
        plan = store.active_plan()
        assert plan is not None
        meta = GameShardMeta(
            level=9,
            policy_size=game.get_action_size(),
            board_shape=(game.board_size, game.board_size),
            board_dtype="int8",
            dag_hash=store.dag_hash(),
            plan_id=plan.plan_id,
            budget=plan.parameters.budget,
            temperature=plan.parameters.temperature,
            min_replicas=plan.parameters.min_replicas,
            game_sizes=tuple(len(g.plies) for g in played),
            games=(),
        )
        write_game_shard(directory / "games" / game_shard_filename(0), played, meta=meta)
        for harvested in played:
            store.mark_done(
                harvested.node_id,
                harvested.replica,
                shard=game_shard_filename(0),
                white_margin=harvested.white_margin,
                plies=len(harvested.plies),
            )
        store.link()
        export_opening(store, directory / "opening")
    return directory


@pytest.fixture(scope="module")
def games(corpus: Path, game: BlokusDuoGame) -> list[CorpusGameRows]:
    return load_corpus_games_v2(game_shards(corpus / "games"), game)


# --------------------------------------------------------------------------- #
# Soft targets
# --------------------------------------------------------------------------- #


def test_v2_games_load_with_their_stored_targets(games: list[CorpusGameRows]) -> None:
    """The distribution Pentobi computed reaches the trainer, not a synthesised one."""
    assert games
    for rows in games:
        assert rows.policies is not None
        assert len(rows.policies) == len(rows.boards)
        assert rows.opening_unit is not None
        for (indices, values), action in zip(rows.policies, rows.actions, strict=True):
            assert values.sum() == pytest.approx(1.0, abs=1e-5)
            assert action in indices.tolist()
    assert any(len(policy[0]) > 1 for rows in games for policy in (rows.policies or ()))


def test_build_examples_uses_the_stored_target_verbatim_at_tau_one(
    game: BlokusDuoGame,
    games: list[CorpusGameRows],
) -> None:
    """With ε = 0 and τ = 1 the training target *is* the stored distribution."""
    examples = build_training_examples(game, games[:1], epsilon=0.0, augment=False)
    assert games[0].policies is not None
    for (_, (indices, values), _), (stored_indices, stored_values) in zip(
        examples,
        games[0].policies,
        strict=True,
    ):
        assert indices.tolist() == stored_indices.tolist()
        assert values.tolist() == pytest.approx(stored_values.tolist())


def test_target_temperature_softens_at_load(game: BlokusDuoGame, games: list[CorpusGameRows]) -> None:
    """τ reshapes confidence at load time, so retuning it never needs regeneration."""
    sharp = build_training_examples(game, games[:1], epsilon=0.0, augment=False)
    soft = build_training_examples(game, games[:1], epsilon=0.0, augment=False, temperature=2.0)
    changed = 0
    for (_, (_, sharp_values), _), (_, (_, soft_values), _) in zip(sharp, soft, strict=True):
        assert soft_values.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.argmax(soft_values) == np.argmax(sharp_values)  # order-preserving
        if len(sharp_values) > 1 and soft_values.max() < sharp_values.max():
            changed += 1
    assert changed > 0


def test_epsilon_floors_the_target_over_the_legal_set(game: BlokusDuoGame, games: list[CorpusGameRows]) -> None:
    """The legal-set floor is still available; it just is not the default any more."""
    examples = build_training_examples(game, games[:1], epsilon=0.1, augment=False)
    board, (indices, values), _ = examples[0]
    legal = np.flatnonzero(game.valid_move_masking(game.board_from_compact(board), 1))
    assert indices.tolist() == legal.tolist()  # support widens to the whole legal set
    assert values.sum() == pytest.approx(1.0, abs=1e-5)
    assert float(values.min()) > 0.0


def test_a_target_outside_the_legal_set_is_a_desync(game: BlokusDuoGame) -> None:
    """Support ⊆ legal is asserted at load — a violation is corpus/rules desync."""
    legal = np.array([1, 2, 3], dtype=np.int32)
    policy = (np.array([1, 99], dtype=np.int32), np.array([0.5, 0.5], dtype=np.float32))
    with pytest.raises(ValueError, match="desync"):
        soft_target_over_legal(policy, legal)


def test_augmentation_transposes_the_whole_support(game: BlokusDuoGame, games: list[CorpusGameRows]) -> None:
    """Symmetry augmentation is unchanged: an arbitrary support transposes fine."""
    examples = build_training_examples(game, games[:1], epsilon=0.0, augment=True)
    (board, (indices, values), value), (twin_board, (twin_indices, twin_values), twin_value) = examples[:2]
    assert np.array_equal(twin_board, np.ascontiguousarray(board.T))
    assert twin_indices.tolist() == [game.transpose_action(int(a)) for a in indices]
    assert twin_values.tolist() == pytest.approx(values.tolist())
    assert twin_value == value
    dense = densify(twin_indices, twin_values, game.get_action_size())
    assert float(dense.sum()) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# The opening-subtree holdout
# --------------------------------------------------------------------------- #


def test_opening_unit_is_the_canonical_first_position(game: BlokusDuoGame) -> None:
    """Mirror-twin openings share a unit, so neither can leak into the other's side."""
    first = int(np.flatnonzero(game.valid_move_masking(game.initialise_board(), 1))[0])
    mirror = game.transpose_action(first)
    assert opening_unit_for(game, (first,)) == opening_unit_for(game, (mirror,))
    assert opening_unit_for(game, ()) is None
    board, player = game.get_next_state(game.initialise_board(), 1, first)
    compact = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
    assert opening_unit_for(game, (first,)) == canonical_key(compact)[0]


def test_holdout_splits_by_subtree_so_shared_openings_cannot_leak(
    games: list[CorpusGameRows],
) -> None:
    """Every game of a held-out opening goes to the holdout; none of it trains.

    This is the leak v1's game-level split would have introduced in v2: many games share
    an opening, so a game-level boundary would put identical early positions on both
    sides.
    """
    units = [rows.opening_unit for rows in games]
    holdout_units = split_opening_units(units, [1.0] * len(games), fraction=0.25, seed=3)
    assert holdout_units
    train, holdout = partition_by_unit(games, holdout_units)
    assert len(train) + len(holdout) == len(games)
    assert {rows.opening_unit for rows in train} & {rows.opening_unit for rows in holdout} == set()
    assert all(rows.opening_unit in holdout_units for rows in holdout)


def test_holdout_unit_choice_is_deterministic_and_stratified() -> None:
    """Same seed, same units; heavy and light subtrees are both represented."""
    units: list[bytes | None] = [bytes([index]) for index in range(20)]
    weights = [float(20 - index) for index in range(20)]  # a mass gradient, heavy first
    first = split_opening_units(units, weights, fraction=0.2, seed=11)
    assert first == split_opening_units(units, weights, fraction=0.2, seed=11)
    assert 0 < len(first) < len(units)
    ranks = sorted(units.index(unit) for unit in first)
    assert min(ranks) < 10 < max(ranks)  # not all from one end of the mass distribution
    assert split_opening_units(units, weights, fraction=0.0, seed=11) == set()
    assert split_opening_units([None, None], [1.0, 1.0], fraction=0.5, seed=1) == set()


def test_opening_rows_share_the_games_holdout_units(corpus: Path, game: BlokusDuoGame) -> None:
    """An opening row's unit is its depth-1 ancestor, so it lands on the games' side."""
    examples, units = load_opening_examples(opening_shards(corpus / "opening"), game)
    assert len(examples) == len(units)
    assert examples
    assert units.count(None) == 1  # only the root has no ply-1 ancestor
    game_units = {rows.opening_unit for rows in load_corpus_games_v2(game_shards(corpus / "games"), game)}
    assert {unit for unit in units if unit is not None} >= game_units


def test_opening_examples_carry_a_blended_value(corpus: Path, game: BlokusDuoGame) -> None:
    """Opening rows train on the count-shrunk blend of teacher and real outcomes."""
    blended, _ = load_opening_examples(opening_shards(corpus / "opening"), game, value_target="blend")
    teacher, _ = load_opening_examples(opening_shards(corpus / "opening"), game, value_target="search")
    outcomes, _ = load_opening_examples(opening_shards(corpus / "opening"), game, value_target="outcome")
    assert len(blended) == len(teacher) == len(outcomes)
    assert any(b != t for (_, _, b), (_, _, t) in zip(blended, teacher, strict=True))
    for board, (indices, values), value in blended:
        assert values.sum() == pytest.approx(1.0, abs=1e-5)
        legal = set(np.flatnonzero(game.valid_move_masking(game.board_from_compact(board), 1)).tolist())
        assert set(indices.tolist()) <= legal
        assert -1.0 <= value <= 1.0


# --------------------------------------------------------------------------- #
# Source mixing
# --------------------------------------------------------------------------- #


def test_mix_examples_hits_the_requested_proportions(corpus: Path, game: BlokusDuoGame) -> None:
    """An opening row must not be a 1-in-160,000 sampling event.

    Openings are ~0.6% of a v2 corpus by row count but are the strategic edge, so the mix
    weights — not the natural sizes — decide how often the net sees one.
    """
    opening, _ = load_opening_examples(opening_shards(corpus / "opening"), game)
    rows = load_corpus_games_v2(game_shards(corpus / "games"), game)
    game_examples = build_training_examples(game, rows, epsilon=0.0, augment=False)
    natural = len(opening) / (len(opening) + len(game_examples))
    assert natural < 0.2  # openings are naturally a small minority

    mixed = mix_examples({"opening": opening, "games": game_examples}, {"opening": 0.3, "games": 0.7}, seed=5)
    opening_boards = {example[0].tobytes() for example in opening}
    share = sum(1 for example in mixed if example[0].tobytes() in opening_boards) / len(mixed)
    assert share == pytest.approx(0.3, abs=0.05)


def test_mix_examples_is_deterministic_and_drops_zero_weight_sources() -> None:
    """Reproducible in the seed; a zero weight excludes a pool rather than shrinking it."""

    def pool(offset: int, value: float) -> list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]]:
        target = (np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32))
        return [(np.array([offset + index], dtype=np.int16), target, value) for index in range(50)]

    pool_a = pool(0, 1.0)
    pool_b = pool(1000, -1.0)
    first = mix_examples({"a": pool_a, "b": pool_b}, {"a": 0.5, "b": 0.5}, seed=1)
    again = mix_examples({"a": pool_a, "b": pool_b}, {"a": 0.5, "b": 0.5}, seed=1)
    assert [int(example[0][0]) for example in first] == [int(example[0][0]) for example in again]
    only_a = mix_examples({"a": pool_a, "b": pool_b}, {"a": 1.0, "b": 0.0}, seed=1)
    assert all(int(example[0][0]) < 1000 for example in only_a)
    assert mix_examples({"a": []}, {"a": 1.0}, seed=1) == []
