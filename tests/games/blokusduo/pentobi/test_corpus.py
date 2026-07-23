"""Tests for the Pentobi distillation corpus (game loop, schema, diversity, validation).

The whole loop/harvest/persist/validate pipeline runs against the *real* rules engine
with a real uniform-random move source (``RandomMoveSource``), so no ``pentobi-gtp``
binary is needed — only the engine-facing ``PentobiMoveSource`` transport is exercised
on the box instead (the pilot's ``validate`` subcommand replays every stored row).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.corpus import (
    BOARD_KIND,
    POLICY_KIND,
    CorpusGame,
    CorpusGenerationError,
    RandomMoveSource,
    analyze_corpus,
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


def _play_games(game: BlokusDuoGame, n: int, *, opening_random_plies: int = 4, seed: int = 0) -> list[CorpusGame]:
    source = RandomMoveSource(game)
    return [
        play_corpus_game(
            game,
            source,
            game_id=g,
            pentobi_seed=seed + g,
            opening_random_plies=opening_random_plies,
            opening_rng=np.random.default_rng((seed, g)),
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
            opening_random_plies=0,
            opening_rng=np.random.default_rng(0),
        )


def test_parse_gtp_score() -> None:
    """B = our White → positive margin; W = our Black → negative; 0 = draw."""
    assert parse_gtp_score("B+5") == 5
    assert parse_gtp_score("W+12") == -12
    assert parse_gtp_score("0") == 0
    with pytest.raises(CorpusGenerationError):
        parse_gtp_score("resign")


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
    report = compute_diversity(sequences, position_keys, prefix_lengths=(1, 2, 4))
    assert report.num_games == 4
    assert report.unique_games == 3  # one exact clone pair
    assert report.unique_game_fraction == 0.75
    assert report.unique_openings_by_prefix == {1: 2, 2: 2, 4: 3}
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
    assert report.num_positions == sum(len(g.plies) for g in games)
