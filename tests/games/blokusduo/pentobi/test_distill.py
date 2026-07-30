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
    final_ownership,
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
    plain = build_training_examples(game, [rows], epsilon=EPSILON, augment=False)
    augmented = build_training_examples(game, [rows], epsilon=EPSILON, augment=True)

    assert len(plain) == len(rows)
    assert [row.margin for row in plain] == list(rows.margins)

    assert len(augmented) == 2 * len(rows)
    assert [row.margin for row in augmented[::2]] == list(rows.margins)
    assert [row.margin for row in augmented[1::2]] == list(rows.margins)


# --------------------------------------------------------------------------- #
# Label smoothing
# --------------------------------------------------------------------------- #


def test_smoothed_targets_sum_to_one_over_exactly_the_legal_moves(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows]
) -> None:
    """Each smoothed target sums to 1, keeps ≥ 1−ε on Pentobi's move, and its support
    is exactly the position's legal set (zero mass on any illegal action)."""
    rows = corpus_games[0]
    examples = [row.example for row in build_training_examples(game, [rows], epsilon=EPSILON, augment=False)]
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
    plain = [row.example for row in build_training_examples(game, [rows], epsilon=EPSILON, augment=False)]
    augmented = [row.example for row in build_training_examples(game, [rows], epsilon=EPSILON, augment=True)]
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
    rows = build_training_examples(game, [corpus_games[0]], epsilon=EPSILON, augment=False)
    examples = [row.example for row in rows][:12]

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
    examples = [row.example for row in build_training_examples(game, [rows], epsilon=EPSILON, augment=False)]
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
    assert arm["heads"] == {"score": True, "ownership": False, "reply": False}
    for row in arm["curve"]:
        assert np.isfinite(row["value_skill"])
        assert row["aux"]["score"] is not None
        assert np.isfinite(row["aux"]["score"]["score_mse"])
        assert row["aux"]["score"]["n_positions"] > 0
        # Heads that were not built report nothing at all, never a fabricated zero.
        assert row["aux"]["ownership"] is None
        assert row["aux"]["reply"] is None
    assert arm["best_aux"]["score"] is not None


def test_distill_sl_without_the_head_reports_no_score_and_a_smaller_net(corpus_dir: Path, tmp_path: Path) -> None:
    """The control arm: no head, no score numbers, and fewer parameters."""
    with_head = _run_distill_sl(tmp_path / "on", corpus_dir, "--score-head")
    without = _run_distill_sl(tmp_path / "off", corpus_dir)

    assert without["score_head"] is False
    assert all(row["aux"]["score"] is None for row in without["arms"]["scratch"]["curve"])
    assert without["arms"]["scratch"]["best_aux"]["score"] is None
    assert without["arms"]["scratch"]["num_params"] < with_head["arms"]["scratch"]["num_params"]


def test_distill_sl_reports_the_ownership_and_reply_heads_when_asked(corpus_dir: Path, tmp_path: Path) -> None:
    """N4/N5 end to end through the real CLI: both heads train and both are measured.

    The corpus fixture is whole games, so every held-out position has a final board and
    every one but each game's last has a next ply — which is exactly what the two
    ``n_skipped`` counts must say.
    """
    payload = _run_distill_sl(tmp_path, corpus_dir, "--ownership-head", "--reply-head")

    arm = payload["arms"]["scratch"]
    assert payload["ownership_head"] is True
    assert payload["reply_head"] is True
    assert arm["heads"] == {"score": False, "ownership": True, "reply": True}
    for row in arm["curve"]:
        ownership = row["aux"]["ownership"]
        reply = row["aux"]["reply"]
        assert ownership is not None and reply is not None
        assert np.isfinite(ownership["cross_entropy"]) and 0.0 <= ownership["accuracy"] <= 1.0
        assert ownership["n_positions"] > 0 and ownership["n_skipped"] == 0
        assert np.isfinite(reply["policy_ce"]) and 0.0 <= reply["top1_accuracy"] <= 1.0
        # One masked position per held-out game: its last ply has no reply.
        assert reply["n_skipped"] == payload["holdout_leakage"]["holdout_rows"] - reply["n_positions"]
        assert reply["n_skipped"] > 0
        assert row["aux"]["score"] is None


# --------------------------------------------------------------------------- #
# Ownership targets (plan N4)
# --------------------------------------------------------------------------- #


def _replay_from_empty(game: BlokusDuoGame, corpus_dir: Path, game_id: int) -> np.ndarray:
    """The finished board's ownership map, computed by a wholly independent route.

    Replays the game from the **empty** board with its real colours — the shard footer's
    opening prefix followed by the stored plies — instead of from a canonical mid-game
    board with an alternating sign. If ``final_ownership``'s frame arithmetic were wrong
    in any way (a missing player multiply, the wrong starting parity) this map would
    disagree, and no other assertion in the suite would notice.
    """
    meta = next(
        game_meta
        for path in corpus_shards(corpus_dir)
        for game_meta in read_shard_meta(path).games
        if game_meta.game_id == game_id
    )
    rows = next(g for g in load_corpus_games(corpus_shards(corpus_dir)) if g.game_id == game_id)
    board = game.initialise_board()
    player = 1
    for action in (*meta.opening_actions, *rows.actions):
        board, player = game.get_next_state(board, player, action)
    return np.sign(np.asarray(board.to_compact(), dtype=np.int8)).astype(np.int8)


def test_final_ownership_matches_an_independent_replay_from_the_empty_board(
    game: BlokusDuoGame, corpus_dir: Path, corpus_games: list[CorpusGameRows]
) -> None:
    """White-positive, and derived from stored actions alone — no regeneration."""
    for rows in corpus_games:
        ownership = final_ownership(game, rows)
        assert ownership is not None
        assert ownership.shape == (game.board_size, game.board_size)
        assert set(np.unique(ownership).tolist()) <= {-1, 0, 1}
        assert np.array_equal(ownership, _replay_from_empty(game, corpus_dir, rows.game_id))
        # A finished Blokus board is mostly owned; an all-zero map would mean the replay
        # never placed anything and every later assertion would pass vacuously.
        assert int(np.count_nonzero(ownership)) > 40


def test_the_per_position_label_is_in_that_position_s_canonical_frame(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows]
) -> None:
    """The frame test, and the one most likely to be silently wrong.

    Pieces are never removed, so every cell already occupied in a stored (canonical)
    board keeps that same owner at the end of the game. In the position's own frame the
    mover's pieces read positive — so the ownership label at those cells must equal the
    *sign of the stored board itself*. A label left in the absolute White-positive frame
    would satisfy this for White-to-move rows and fail for every Black-to-move one.
    """
    rows = corpus_games[0]
    built = build_training_examples(game, [rows], epsilon=EPSILON, augment=False, with_ownership=True)

    assert any(player == -1 for player in rows.players), "fixture must contain Black-to-move rows"
    for row, player in zip(built, rows.players, strict=True):
        assert row.ownership is not None
        stored = np.asarray(row.example[0], dtype=np.int8)
        occupied = stored != 0
        assert np.array_equal(row.ownership[occupied], np.sign(stored[occupied]))
        # And it really is the game's final map, re-signed into this position's frame.
        assert np.array_equal(row.ownership, (final_ownership(game, rows) * player).astype(np.int8))


def test_the_symmetry_twin_takes_the_transposed_ownership_map(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows]
) -> None:
    """The twin is the transposed board, so its label is the transposed final board."""
    rows = corpus_games[1]
    built = build_training_examples(game, [rows], epsilon=EPSILON, augment=True, with_ownership=True)

    assert len(built) == 2 * len(rows)
    for original, twin in zip(built[::2], built[1::2], strict=True):
        assert original.ownership is not None and twin.ownership is not None
        assert np.array_equal(twin.ownership, original.ownership.T)
        # The invariant of the previous test must survive the transpose too.
        stored = np.asarray(twin.example[0], dtype=np.int8)
        occupied = stored != 0
        assert np.array_equal(twin.ownership[occupied], np.sign(stored[occupied]))


def test_ownership_is_not_derived_unless_asked(game: BlokusDuoGame, corpus_games: list[CorpusGameRows]) -> None:
    """Off by default: the extra game replay is paid only by the arms that use it."""
    built = build_training_examples(game, corpus_games[:1], epsilon=EPSILON, augment=True)
    assert built and all(row.ownership is None for row in built)


def test_a_game_whose_rows_stop_short_has_no_final_board(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows]
) -> None:
    """Truncated rows must mask the target, never label a half-played board as final."""
    from dataclasses import replace as replace_dataclass

    full = corpus_games[0]
    truncated = replace_dataclass(
        full,
        boards=full.boards[:5],
        actions=full.actions[:5],
        players=full.players[:5],
        values=full.values[:5],
        margins=full.margins[:5],
    )

    assert final_ownership(game, truncated) is None
    built = build_training_examples(game, [truncated], epsilon=EPSILON, augment=False, with_ownership=True)
    assert built and all(row.ownership is None for row in built)


# --------------------------------------------------------------------------- #
# Opponent-reply targets (plan N5)
# --------------------------------------------------------------------------- #


def test_the_reply_target_is_the_next_position_s_policy(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows]
) -> None:
    """The index shift, and the mask on the final ply.

    Off-by-one here is invisible in every other metric: the head would simply learn a
    slightly-wrong distribution, and the arm would read as "the technique did not help".
    """
    rows = corpus_games[0]
    built = build_training_examples(game, [rows], epsilon=EPSILON, augment=False)

    for index, row in enumerate(built[:-1]):
        assert row.reply is not None
        next_indices, next_values = built[index + 1].example[1]
        assert row.reply[0].tolist() == next_indices.tolist()
        assert row.reply[1].tolist() == next_values.tolist()
        # It is the *opponent's* actual move that carries the mass.
        assert int(row.reply[0][int(np.argmax(row.reply[1]))]) == rows.actions[index + 1]
    assert built[-1].reply is None, "the last ply of a game has no reply"


def test_no_game_borrows_the_next_game_s_first_reply(game: BlokusDuoGame, corpus_games: list[CorpusGameRows]) -> None:
    """Built per game, so a game boundary masks rather than leaking across it."""
    built = build_training_examples(game, corpus_games, epsilon=EPSILON, augment=False)

    cursor = 0
    for rows in corpus_games:
        cursor += len(rows)
        assert built[cursor - 1].reply is None, f"game {rows.game_id}'s last ply borrowed a reply"
    assert cursor == len(built)
    assert sum(1 for row in built if row.reply is None) == len(corpus_games)


def test_the_twin_s_reply_is_the_twin_of_the_next_position_s_policy(
    game: BlokusDuoGame, corpus_games: list[CorpusGameRows]
) -> None:
    """A transposed board must be answered with a transposed reply, not the original."""
    rows = corpus_games[1]
    built = build_training_examples(game, [rows], epsilon=EPSILON, augment=True)

    twins = built[1::2]
    for index, twin in enumerate(twins[:-1]):
        assert twin.reply is not None
        expected_indices, expected_values = twins[index + 1].example[1]
        assert twin.reply[0].tolist() == expected_indices.tolist()
        assert twin.reply[1].tolist() == expected_values.tolist()
        # Which is the transpose of what the un-augmented original was given.
        original_reply = built[2 * index].reply
        assert original_reply is not None
        assert twin.reply[0].tolist() == [game.transpose_action(int(a)) for a in original_reply[0]]
    assert twins[-1].reply is None
