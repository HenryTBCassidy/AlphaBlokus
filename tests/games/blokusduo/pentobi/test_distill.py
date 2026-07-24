"""Tests for the SL distillation dataloader (plan D6) + one real training step (D7).

Everything runs engine-free: a tiny real-schema corpus is generated with the same
``RandomMoveSource``-driven pipeline the corpus tests use (real rules engine, real
parquet shards, no ``pentobi-gtp`` binary), then streamed through the dataloader and —
for the training-step test — through the real ``BaseNNetWrapper.train`` path on a tiny
CPU net.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.config import MCTSConfig, NetConfig, RunConfig
from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper as BlokusDuoNNetWrapper
from alphablokus.games.blokusduo.pentobi.corpus import (
    OpeningPrefixBuilder,
    RandomMoveSource,
    corpus_shards,
    iter_corpus_examples,
    play_corpus_game,
    read_shard_meta,
    shard_filename,
    write_shard,
)
from alphablokus.games.blokusduo.pentobi.distill import (
    build_training_examples,
    load_corpus_games,
    sample_games,
    smooth_policy,
)
from alphablokus.games.blokusduo.pieces import default_pieces_path
from alphablokus.storage.sparse_policy import as_dense
from alphablokus.training.holdout import (
    evaluate_holdout,
    evaluate_imitation_diagnostics,
    split_games_holdout,
)

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.games.blokusduo.pentobi.distill import CorpusGameRows

_PIECES = default_pieces_path()
EPSILON = 0.1


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=_PIECES)


@pytest.fixture(scope="module")
def corpus_dir(game: BlokusDuoGame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny two-shard corpus of four random-mover games (real schema, real rules)."""
    source = RandomMoveSource(game)
    builder = OpeningPrefixBuilder(game, base_seed=0, num_plies=4)
    games = [
        play_corpus_game(game, source, game_id=g, pentobi_seed=g, opening_actions=builder.prefix_for(g))
        for g in range(4)
    ]
    directory = tmp_path_factory.mktemp("distill_corpus")
    for index, chunk in enumerate((games[:2], games[2:])):
        write_shard(
            directory / shard_filename(index),
            chunk,
            policy_size=game.get_action_size(),
            level=9,
            opening_random_plies=4,
        )
    return directory


@pytest.fixture(scope="module")
def corpus_games(corpus_dir: Path) -> list[CorpusGameRows]:
    return load_corpus_games(corpus_shards(corpus_dir))


# --------------------------------------------------------------------------- #
# Loading + game grouping
# --------------------------------------------------------------------------- #


def test_load_corpus_games_groups_rows_by_game(
    game: BlokusDuoGame, corpus_dir: Path, corpus_games: list[CorpusGameRows]
) -> None:
    """Rows group per game in shard order, aligned with the stored stream and footer."""
    assert [g.game_id for g in corpus_games] == [0, 1, 2, 3]
    sizes = [size for path in corpus_shards(corpus_dir) for size in read_shard_meta(path).game_sizes]
    assert [len(g) for g in corpus_games] == sizes

    # Index-aligned with the raw example stream: boards, values, and the one-hot action.
    flat_boards = [b for g in corpus_games for b in g.boards]
    flat_actions = [a for g in corpus_games for a in g.actions]
    flat_values = [v for g in corpus_games for v in g.values]
    for (board, (indices, values), value), grouped_board, action, grouped_value in zip(
        iter_corpus_examples(corpus_shards(corpus_dir)), flat_boards, flat_actions, flat_values, strict=True
    ):
        assert np.array_equal(board, grouped_board)
        assert indices.tolist() == [action] and values.tolist() == [1.0]
        assert value == grouped_value
    assert all(p in (-1, 1) for g in corpus_games for p in g.players)


# --------------------------------------------------------------------------- #
# Label smoothing
# --------------------------------------------------------------------------- #


def test_smoothed_targets_sum_to_one_over_exactly_the_legal_moves(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows]
) -> None:
    """Each smoothed target sums to 1, keeps ≥ 1−ε on Pentobi's move, and its support
    is exactly the position's legal set (zero mass on any illegal action)."""
    rows = corpus_games[0]
    examples = build_training_examples(game, [rows], epsilon=EPSILON, augment=False)
    assert len(examples) == len(rows)
    for (compact, pi, _value), action in zip(examples, rows.actions, strict=True):
        dense = as_dense(pi, game.get_action_size())
        assert dense.sum() == pytest.approx(1.0)
        assert dense[action] >= 1.0 - EPSILON
        legal = np.flatnonzero(game.valid_move_masking(game.board_from_compact(compact), 1))
        assert np.array_equal(np.flatnonzero(dense), legal)


def test_smooth_policy_validates_inputs() -> None:
    legal = np.array([3, 7, 9], dtype=np.int32)
    with pytest.raises(ValueError, match="not in the position's legal set"):
        smooth_policy(5, legal, 0.1)
    with pytest.raises(ValueError, match="epsilon"):
        smooth_policy(7, legal, 1.0)
    # ε = 0 degenerates to the one-hot (over an explicit legal support).
    indices, values = smooth_policy(7, legal, 0.0)
    assert indices.tolist() == [3, 7, 9]
    assert values.tolist() == [0.0, 1.0, 0.0]


# --------------------------------------------------------------------------- #
# Symmetry augmentation
# --------------------------------------------------------------------------- #


def test_augmentation_appends_the_transposed_twin(game: BlokusDuoGame, corpus_games: list[CorpusGameRows]) -> None:
    """augment=True doubles the count, and each twin equals the ``get_symmetries``
    ground truth: the transposed board with the policy mapped via ``transpose_action``."""
    rows = corpus_games[0]
    plain = build_training_examples(game, [rows], epsilon=EPSILON, augment=False)
    augmented = build_training_examples(game, [rows], epsilon=EPSILON, augment=True)
    assert len(augmented) == 2 * len(plain)

    action_size = game.get_action_size()
    for position, (original, twin) in enumerate(zip(augmented[::2], augmented[1::2], strict=True)):
        dense = as_dense(original[1], action_size)
        board = game.board_from_compact(original[0])
        _, (expected_board, expected_pi) = game.get_symmetries(board, dense)
        assert np.array_equal(twin[0], np.asarray(expected_board.to_compact(), dtype=np.int8))
        assert np.array_equal(as_dense(twin[1], action_size), expected_pi)
        # The twin's best move is the transpose of Pentobi's move, with the same mass.
        twin_dense = as_dense(twin[1], action_size)
        assert int(np.argmax(twin_dense)) == game.transpose_action(rows.actions[position])
        assert twin[2] == original[2]


# --------------------------------------------------------------------------- #
# Game-level split + shard-weighted subsampling
# --------------------------------------------------------------------------- #


def test_holdout_split_never_leaks_a_game(corpus_games: list[CorpusGameRows]) -> None:
    train, holdout = split_games_holdout(corpus_games, holdout_fraction=0.25, seed=7)
    train_ids = {g.game_id for g in train}
    holdout_ids = {g.game_id for g in holdout}
    assert train_ids.isdisjoint(holdout_ids)
    assert train_ids | holdout_ids == {0, 1, 2, 3}
    assert len(holdout) == 1


def test_sample_games_is_deterministic_and_a_subset(corpus_games: list[CorpusGameRows]) -> None:
    sampled = sample_games(corpus_games, max_games=2, seed=3)
    assert len(sampled) == 2
    assert {g.game_id for g in sampled} <= {g.game_id for g in corpus_games}
    assert [g.game_id for g in sample_games(corpus_games, 2, seed=3)] == [g.game_id for g in sampled]
    assert sample_games(corpus_games, max_games=99, seed=3) == list(corpus_games)


# --------------------------------------------------------------------------- #
# One real training step (D7's path end-to-end on CPU)
# --------------------------------------------------------------------------- #


def _tiny_run_config(tmp_path: Path) -> RunConfig:
    """A minimal Blokus run config with a tiny CPU net (fast enough for CI)."""
    return RunConfig(
        game="blokusduo",
        run_name="distill_test",
        num_generations=1,
        num_eps=1,
        temp_threshold=5,
        update_threshold=0.55,
        num_arena_matches=2,
        root_directory=tmp_path,
        load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=2, cpuct=1.0),
        net_config=NetConfig(
            learning_rate=5e-3,
            dropout=0.0,
            epochs=1,
            batch_size=8,
            cuda=False,
            num_filters=16,
            num_residual_blocks=1,
        ),
    )


def test_training_step_runs_and_fits_a_tiny_subset(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows], tmp_path: Path
) -> None:
    """Corpus examples flow through the real ``train()`` path: the loss is finite and
    held-out CE on the (trivially fittable) training subset itself decreases."""
    import torch

    torch.manual_seed(0)
    wrapper = BlokusDuoNNetWrapper(game, _tiny_run_config(tmp_path))
    examples = build_training_examples(game, [corpus_games[0]], epsilon=EPSILON, augment=False)[:12]

    before = evaluate_holdout(
        wrapper, examples, encode_fn=game.encode_compact, action_size=game.get_action_size(), batch_size=8
    )
    for generation in range(1, 4):
        wrapper.train(examples, generation=generation, metrics=None, eval_set=None)
    after = evaluate_holdout(
        wrapper, examples, encode_fn=game.encode_compact, action_size=game.get_action_size(), batch_size=8
    )

    assert np.isfinite(after.policy_ce) and np.isfinite(after.value_mse)
    assert after.policy_ce < before.policy_ce


def test_imitation_diagnostics_on_real_corpus_examples(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows], tmp_path: Path
) -> None:
    """The colour-conditional diagnostics run off the dataloader's alignment: one
    calibration row per side-to-move, bucket counts partitioning that colour's rows."""
    rows = corpus_games[1]
    examples = build_training_examples(game, [rows], epsilon=EPSILON, augment=False)
    wrapper = BlokusDuoNNetWrapper(game, _tiny_run_config(tmp_path))

    diagnostics = evaluate_imitation_diagnostics(
        wrapper,
        examples,
        list(rows.actions),
        list(rows.players),
        encode_fn=game.encode_compact,
        batch_size=8,
    )

    assert diagnostics.n_positions == len(rows)
    assert 0.0 <= diagnostics.top1_accuracy <= 1.0
    assert [c.player for c in diagnostics.calibration] == sorted(set(rows.players))
    for calibration in diagnostics.calibration:
        expected = sum(1 for p in rows.players if p == calibration.player)
        assert calibration.n_positions == expected
        assert sum(calibration.bucket_counts) == expected
        assert calibration.mean_outcome == pytest.approx(
            np.mean([v for v, p in zip(rows.values, rows.players, strict=True) if p == calibration.player])
        )
