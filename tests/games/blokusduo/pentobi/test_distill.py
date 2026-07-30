"""Tests for the SL distillation dataloader (plan D6) + one real training step (D7).

Everything runs engine-free: a tiny real-schema corpus is generated with the same
``RandomMoveSource``-driven pipeline the corpus tests use (real rules engine, real
parquet shards, no ``pentobi-gtp`` binary), then streamed through the dataloader and —
for the training-step test — through the real ``BaseNNetWrapper.train`` path on a tiny
CPU net.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pyarrow.parquet as pq
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


def test_loader_carries_the_stored_margin_per_position(corpus_dir: Path, corpus_games: list[CorpusGameRows]) -> None:
    """v1 shards store ``margin`` from the side to move; the loader must surface it.

    Cross-checked against the stored ``value``: the outcome label is the *sign* of the
    margin from that same side, so a loader that read the wrong column or lost the
    per-position player flip would disagree here (score-head plan S5).
    """
    stored = [
        int(m)
        for path in corpus_shards(corpus_dir)
        for m in pq.read_table(path, columns=["margin"]).column("margin").to_pylist()
    ]
    flat_margins = [m for g in corpus_games for m in g.margins]
    assert flat_margins == [float(m) for m in stored]

    for rows in corpus_games:
        assert len(rows.margins) == len(rows)
        for margin, value in zip(rows.margins, rows.values, strict=True):
            assert np.sign(margin) == pytest.approx(value)


def test_build_training_examples_returns_margins_aligned_with_its_examples(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows]
) -> None:
    """Alignment is the whole contract: a shifted margin trains the head on the wrong
    position and shows up in no metric. The symmetry twin shares its original's margin —
    transposing a board does not change the score."""
    rows = corpus_games[0]
    plain, plain_margins = build_training_examples(game, [rows], epsilon=EPSILON, augment=False)
    augmented, augmented_margins = build_training_examples(game, [rows], epsilon=EPSILON, augment=True)

    assert len(plain_margins) == len(plain) == len(rows)
    assert plain_margins == list(rows.margins)

    assert len(augmented_margins) == len(augmented) == 2 * len(rows)
    assert augmented_margins[::2] == list(rows.margins)
    assert augmented_margins[1::2] == list(rows.margins)


# --------------------------------------------------------------------------- #
# Label smoothing
# --------------------------------------------------------------------------- #


def test_smoothed_targets_sum_to_one_over_exactly_the_legal_moves(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows]
) -> None:
    """Each smoothed target sums to 1, keeps ≥ 1−ε on Pentobi's move, and its support
    is exactly the position's legal set (zero mass on any illegal action)."""
    rows = corpus_games[0]
    examples, _margins = build_training_examples(game, [rows], epsilon=EPSILON, augment=False)
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
    plain, _ = build_training_examples(game, [rows], epsilon=EPSILON, augment=False)
    augmented, _ = build_training_examples(game, [rows], epsilon=EPSILON, augment=True)
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
    examples = build_training_examples(game, [corpus_games[0]], epsilon=EPSILON, augment=False)[0][:12]

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
    examples, _margins = build_training_examples(game, [rows], epsilon=EPSILON, augment=False)
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


# --------------------------------------------------------------------------- #
# The SL trainer's score-head wiring, end to end (score-head plan S6)
# --------------------------------------------------------------------------- #


def _tiny_config_json(tmp_path: Path) -> Path:
    """The tiny Blokus run config as JSON, for the ``distill_sl`` CLI."""
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "game": "blokusduo",
                "run_name": "distill_score_test",
                "num_generations": 1,
                "num_eps": 1,
                "temp_threshold": 5,
                "update_threshold": 0.55,
                "num_arena_matches": 2,
                "root_directory": str(tmp_path),
                "load_model": False,
                "mcts_config": {"num_mcts_sims": 2, "cpuct": 1},
                "net_config": {
                    "learning_rate": 0.005,
                    "dropout": 0.0,
                    "epochs": 1,
                    "batch_size": 8,
                    "cuda": False,
                    "num_filters": 16,
                    "num_residual_blocks": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def _run_distill_sl(tmp_path: Path, corpus_dir: Path, *extra: str) -> dict:
    """Drive the real ``distill_sl`` CLI over the tiny corpus and read its run JSON."""
    from scripts.distill_sl import main

    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "distill.json"
    argv = [
        "distill_sl.py",
        "--config",
        str(_tiny_config_json(tmp_path)),
        "--corpus",
        str(corpus_dir),
        "--arms",
        "scratch",
        "--max-epochs",
        "1",
        "--holdout-frac",
        "0.25",
        "--ckpt-dir",
        str(tmp_path / "ckpt"),
        "--out",
        str(out),
        *extra,
    ]
    with mock.patch.object(sys, "argv", argv):
        main()
    return json.loads(out.read_text())


def test_distill_sl_reports_score_mse_and_value_skill_with_the_head_on(corpus_dir: Path, tmp_path: Path) -> None:
    """The S7 arms are compared on these two numbers, so both must reach the run JSON."""
    payload = _run_distill_sl(tmp_path, corpus_dir, "--score-head")

    arm = payload["arms"]["scratch"]
    assert payload["score_head"] is True
    assert arm["score_head"] is True
    for row in arm["curve"]:
        assert np.isfinite(row["value_skill"])
        assert row["score"] is not None
        assert np.isfinite(row["score"]["score_mse"])
        assert row["score"]["n_positions"] > 0
    assert arm["best_score"] is not None


def test_distill_sl_without_the_head_reports_no_score_and_a_smaller_net(corpus_dir: Path, tmp_path: Path) -> None:
    """The control arm: no head, no score numbers, and fewer parameters."""
    with_head = _run_distill_sl(tmp_path / "on", corpus_dir, "--score-head")
    without = _run_distill_sl(tmp_path / "off", corpus_dir)

    assert without["score_head"] is False
    assert all(row["score"] is None for row in without["arms"]["scratch"]["curve"])
    assert without["arms"]["scratch"]["best_score"] is None
    assert without["arms"]["scratch"]["num_params"] < with_head["arms"]["scratch"]["num_params"]
