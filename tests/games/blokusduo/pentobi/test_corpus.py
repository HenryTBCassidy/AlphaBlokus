"""Tests for the Pentobi distillation corpus (game loop, schema, diversity, validation).

The whole loop/harvest/persist/validate pipeline runs against the *real* rules engine
with a real uniform-random move source (``RandomMoveSource``), so no ``pentobi-gtp``
binary is needed — only the engine-facing ``PentobiMoveSource`` transport is exercised
on the box instead (the pilot's ``validate`` subcommand replays every stored row).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.corpus import (
    BOARD_KIND,
    POLICY_KIND,
    CorpusGame,
    CorpusGenerationError,
    OpeningPrefixBuilder,
    RandomMoveSource,
    analyze_corpus,
    assert_unique_openings,
    compute_diversity,
    corpus_shards,
    iter_corpus_examples,
    parse_gtp_score,
    play_corpus_game,
    read_shard_meta,
    shard_filename,
    validate_shard,
    write_shard,
)
from alphablokus.games.blokusduo.pieces import default_pieces_path
from alphablokus.storage.selfplay_store import SelfPlayStore
from alphablokus.storage.sparse_policy import densify

if TYPE_CHECKING:
    from pathlib import Path

_PIECES = default_pieces_path()


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=_PIECES)


# Blokus Duo's known opening branching factor: 414 legal first placements per player.
_N_FIRST_MOVES = 414


def _play_games(game: BlokusDuoGame, n: int, *, opening_random_plies: int = 4, seed: int = 0) -> list[CorpusGame]:
    source = RandomMoveSource(game)
    builder = OpeningPrefixBuilder(game, base_seed=seed, num_plies=opening_random_plies)
    return [
        play_corpus_game(
            game,
            source,
            game_id=g,
            pentobi_seed=seed + g,
            opening_actions=builder.prefix_for(g),
        )
        for g in range(n)
    ]


# --------------------------------------------------------------------------- #
# Game loop + harvesting
# --------------------------------------------------------------------------- #


def test_play_corpus_game_harvests_only_expert_plies(game: BlokusDuoGame) -> None:
    """Opening plies are excluded from the harvest; expert plies start at ply k."""
    (g,) = _play_games(game, 1, opening_random_plies=4)
    assert len(g.opening_actions) == 4
    assert g.plies[0].ply == 4
    assert [p.ply for p in g.plies] == list(range(4, 4 + len(g.plies)))
    # The full sequence is opening + expert plies, in order.
    assert g.actions[:4] == g.opening_actions


def test_play_corpus_game_labels_match_rules_engine(game: BlokusDuoGame) -> None:
    """Replaying the full action sequence reproduces the stored boards and labels."""
    (g,) = _play_games(game, 1)
    board = game.initialise_board()
    player = 1
    by_ply = {p.ply: p for p in g.plies}
    for ply, action in enumerate(g.actions):
        assert game.valid_move_masking(board, player)[action] == 1, "stored move must be legal where stored"
        if ply in by_ply:
            harvested = by_ply[ply]
            assert harvested.player == player
            expected = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
            assert np.array_equal(harvested.compact_board, expected)
        board, player = game.get_next_state(board, player, action)
    assert game.get_game_ended(board, player) != 0
    assert game.final_scores(board) == (g.white_score, g.black_score)


def test_play_corpus_game_rejects_source_score_mismatch(game: BlokusDuoGame) -> None:
    """A source whose final margin disagrees with the rules engine raises (desync guard)."""

    class LyingSource(RandomMoveSource):
        def final_white_margin(self) -> int:
            return 999

    with pytest.raises(CorpusGenerationError, match="margin"):
        play_corpus_game(
            game,
            LyingSource(game),
            game_id=0,
            pentobi_seed=0,
            opening_actions=(),
        )


def test_parse_gtp_score() -> None:
    """B = our White → positive margin; W = our Black → negative; 0 = draw."""
    assert parse_gtp_score("B+5") == 5
    assert parse_gtp_score("W+12") == -12
    assert parse_gtp_score("0") == 0
    with pytest.raises(CorpusGenerationError):
        parse_gtp_score("resign")


# --------------------------------------------------------------------------- #
# Deterministic opening keys
# --------------------------------------------------------------------------- #


def test_first_ply_enumeration_exhaustive_and_interleaved(game: BlokusDuoGame) -> None:
    """Game i's first ply is enum[i mod 414] — an interleaved sweep: the enumeration
    matches the legal first placements exactly, any 414-game window hits 414 distinct
    first moves, and 2x414 games cover every first move exactly twice."""
    builder = OpeningPrefixBuilder(game, base_seed=0, num_plies=1)
    mask = game.valid_move_masking(game.initialise_board(), 1)
    legal = {int(a) for a in np.flatnonzero(mask)} - {game.action_codec.pass_action_index}
    assert len(builder.first_moves) == _N_FIRST_MOVES
    assert set(builder.first_moves) == legal

    firsts = [builder.prefix_for(g)[0] for g in range(2 * _N_FIRST_MOVES)]
    assert firsts == [builder.first_moves[g % _N_FIRST_MOVES] for g in range(2 * _N_FIRST_MOVES)]
    assert len(set(firsts[:_N_FIRST_MOVES])) == _N_FIRST_MOVES  # spread: no repeat until the sweep completes
    assert set(firsts) == legal  # exhaustive coverage of all 414 first moves
    assert set(Counter(firsts).values()) == {2}  # perfectly even after two full cycles


def test_opening_prefixes_deterministic_and_unique_past_cycle_boundary(game: BlokusDuoGame) -> None:
    """Two independent builders reproduce identical prefixes; zero duplicates even for
    games 414+ whose stratified first ply repeats games 0+ (plies 2..k must differ)."""
    n = _N_FIRST_MOVES + 20
    a = OpeningPrefixBuilder(game, base_seed=0, num_plies=2)
    b = OpeningPrefixBuilder(game, base_seed=0, num_plies=2)
    prefixes_a = [a.prefix_for(g) for g in range(n)]
    prefixes_b = [b.prefix_for(g) for g in range(n)]
    assert prefixes_a == prefixes_b
    assert all(len(p) == 2 for p in prefixes_a)
    assert len(set(prefixes_a)) == n


def test_opening_prefix_collision_redraws_deterministically(game: BlokusDuoGame) -> None:
    """A realised-set collision bumps the attempt sub-seed and redraws plies 2..k only:
    the redraw keeps the stratified first ply and equals the attempt-1 draw exactly."""
    probe = OpeningPrefixBuilder(game, base_seed=0, num_plies=3)
    attempt0 = probe._draw(probe.first_moves[0], 0, 0)
    attempt1 = probe._draw(probe.first_moves[0], 0, 1)
    assert attempt0 != attempt1

    builder = OpeningPrefixBuilder(game, base_seed=0, num_plies=3)
    builder._realised.add(attempt0)  # force game 0's natural draw to collide
    prefix = builder.prefix_for(0)
    assert prefix == attempt1
    assert prefix[0] == attempt0[0]  # first ply untouched — stratification survives the redraw


def test_opening_prefix_builder_enforces_ascending_walk(game: BlokusDuoGame) -> None:
    """The collision guard is defined over the full walk from game 0 — skipping raises."""
    builder = OpeningPrefixBuilder(game, base_seed=0, num_plies=4)
    builder.prefix_for(0)
    with pytest.raises(ValueError, match="in order"):
        builder.prefix_for(2)


def test_resume_reproduces_identical_opening_actions(game: BlokusDuoGame) -> None:
    """Same (base_seed, game_id) → byte-identical opening_actions on regeneration,
    exactly the resume contract (a rerun rebuilds the missing games' keys)."""
    games = _play_games(game, 4)
    builder = OpeningPrefixBuilder(game, base_seed=0, num_plies=4)
    prefixes = [builder.prefix_for(g) for g in range(4)]
    source = RandomMoveSource(game)
    for game_id in (1, 3):  # regenerate a couple of game_ids as a resume would
        regenerated = play_corpus_game(
            game, source, game_id=game_id, pentobi_seed=game_id, opening_actions=prefixes[game_id]
        )
        original = games[game_id]
        assert regenerated.opening_actions == original.opening_actions
        assert (
            np.asarray(regenerated.opening_actions, dtype=np.int32).tobytes()
            == np.asarray(original.opening_actions, dtype=np.int32).tobytes()
        )


# --------------------------------------------------------------------------- #
# Parquet schema + round-trip
# --------------------------------------------------------------------------- #


def test_schema_markers_match_selfplay_store() -> None:
    """The corpus reuses SelfPlayStore's format markers so row decoding is shared."""
    assert BOARD_KIND == SelfPlayStore.BOARD_KIND
    assert POLICY_KIND == SelfPlayStore.POLICY_KIND


def test_shard_roundtrip_and_one_hot_policies(game: BlokusDuoGame, tmp_path: Path) -> None:
    """Written shards read back with intact metadata, boards, and one-hot policies."""
    games = _play_games(game, 3)
    path = tmp_path / shard_filename(0)
    rows = write_shard(path, games, policy_size=game.get_action_size(), level=9, opening_random_plies=4)
    assert rows == sum(len(g.plies) for g in games)

    meta = read_shard_meta(path)
    assert meta.level == 9
    assert meta.opening_random_plies == 4
    assert meta.policy_size == game.get_action_size()
    assert meta.board_shape == (game.board_size, game.board_size)
    assert meta.game_sizes == tuple(len(g.plies) for g in games)
    assert tuple(g.opening_actions for g in meta.games) == tuple(g.opening_actions for g in games)

    examples = list(iter_corpus_examples([path]))
    assert len(examples) == rows
    flat = [p for g in games for p in g.plies]
    for (board, (indices, values), value), harvested in zip(examples, flat, strict=True):
        assert board.dtype == np.int8
        assert np.array_equal(board, harvested.compact_board)
        # One-hot of the played action, densifiable into the full action space.
        dense = densify(indices, values, game.get_action_size())
        assert dense.sum() == 1.0
        assert dense[harvested.action] == 1.0
        assert value in (-1.0, 0.0, 1.0)
        # The board must rebuild into the exact 44-channel net input.
        planes = game.encode_compact(board)
        assert planes.shape == (44, game.board_size, game.board_size)


def test_examples_are_symmetry_augmentable(game: BlokusDuoGame, tmp_path: Path) -> None:
    """Stored examples feed IGame.get_symmetries: the transposed one-hot policy is the
    one-hot of the transposed action, and its board is a real (transposed) board."""
    games = _play_games(game, 1)
    path = tmp_path / shard_filename(0)
    write_shard(path, games, policy_size=game.get_action_size(), level=9, opening_random_plies=4)
    board_compact, (indices, values), _value = next(iter(iter_corpus_examples([path])))
    board = game.board_from_compact(board_compact)
    dense = densify(indices, values, game.get_action_size())
    symmetries = game.get_symmetries(board, dense)
    assert len(symmetries) == 2
    _, transposed_pi = symmetries[1]
    assert transposed_pi.sum() == 1.0
    assert transposed_pi[game.transpose_action(int(indices[0]))] == 1.0


def test_validate_shard_passes_on_good_data_and_catches_corruption(game: BlokusDuoGame, tmp_path: Path) -> None:
    """The replay validator accepts a genuine shard and rejects a tampered label."""
    games = _play_games(game, 2)
    path = tmp_path / shard_filename(0)
    write_shard(path, games, policy_size=game.get_action_size(), level=9, opening_random_plies=4)
    assert validate_shard(path, game) == sum(len(g.plies) for g in games)

    # Corrupt one game's stored score labels → the replay check must fail.
    import dataclasses

    bad_games = [dataclasses.replace(games[0], white_score=games[0].white_score + 1), games[1]]
    bad_path = tmp_path / shard_filename(1)
    write_shard(bad_path, bad_games, policy_size=game.get_action_size(), level=9, opening_random_plies=4)
    with pytest.raises(CorpusGenerationError):
        validate_shard(bad_path, game)


# --------------------------------------------------------------------------- #
# Diversity metrics
# --------------------------------------------------------------------------- #


def test_compute_diversity_counts() -> None:
    """Known sequences → known uniqueness numbers (clones detected, diversity counted)."""
    sequences = [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 9, 9), (5, 6, 7, 8)]
    position_keys = [b"a", b"a", b"b", b"c"]
    openings = [(1, 2), (1, 2), (1, 2), (5, 6)]
    report = compute_diversity(sequences, position_keys, prefix_lengths=(1, 2, 4), opening_prefixes=openings)
    assert report.num_games == 4
    assert report.unique_games == 3  # one exact clone pair
    assert report.unique_game_fraction == 0.75
    assert report.unique_openings_by_prefix == {1: 2, 2: 2, 4: 3}
    assert report.unique_opening_prefixes == 2
    assert report.num_positions == 4
    assert report.unique_positions == 3


def test_analyze_corpus_over_shards(game: BlokusDuoGame, tmp_path: Path) -> None:
    """End-to-end: distinct-seed random games in two shards are all unique."""
    games = _play_games(game, 4)
    write_shard(
        tmp_path / shard_filename(0), games[:2], policy_size=game.get_action_size(), level=9, opening_random_plies=4
    )
    write_shard(
        tmp_path / shard_filename(1), games[2:], policy_size=game.get_action_size(), level=9, opening_random_plies=4
    )
    assert [p.name for p in corpus_shards(tmp_path)] == [shard_filename(0), shard_filename(1)]
    report = analyze_corpus(tmp_path)
    assert report.num_games == 4
    assert report.unique_games == 4  # distinct seeds + random play → no clones
    assert report.unique_opening_prefixes == 4  # deterministic keys: one distinct prefix per game
    assert report.num_positions == sum(len(g.plies) for g in games)


def test_zero_duplicate_openings_across_shard_boundary(game: BlokusDuoGame, tmp_path: Path) -> None:
    """The corpus-level uniqueness assertion holds globally across shards, and a
    repeated game (duplicated prefix) in another shard trips it."""
    games = _play_games(game, 6)
    for index, chunk in enumerate((games[:3], games[3:])):
        write_shard(
            tmp_path / shard_filename(index), chunk, policy_size=game.get_action_size(), level=9, opening_random_plies=4
        )
    assert assert_unique_openings(tmp_path) == 6
    assert analyze_corpus(tmp_path).unique_opening_prefixes == 6

    dup_dir = tmp_path / "dup"
    dup_dir.mkdir()
    write_shard(
        dup_dir / shard_filename(0), games[:3], policy_size=game.get_action_size(), level=9, opening_random_plies=4
    )
    write_shard(  # games[2] recurs in the second shard → a cross-shard duplicate prefix
        dup_dir / shard_filename(1),
        [games[2], *games[4:]],
        policy_size=game.get_action_size(),
        level=9,
        opening_random_plies=4,
    )
    with pytest.raises(CorpusGenerationError, match="duplicate opening prefixes"):
        assert_unique_openings(dup_dir)
