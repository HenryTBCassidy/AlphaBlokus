"""Tests for the parallel Pentobi benchmark driver (``scripts/pentobi_benchmark``).

The Pentobi-specific execution (spinning up a real ``pentobi-gtp`` engine) is left
to the end-to-end validation — CI must not depend on the Pentobi binary. What is
tested here is everything *around* that: the pure chunking / seed-planning /
aggregation helpers, that the worker entry point is picklable (a hard requirement
for the ``spawn`` pool), and that the fan-out-then-aggregate driver is
worker-count-invariant when run over a real deterministic game in a real
``spawn`` process pool.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

from alphablokus.evaluation.arena import Arena, ColourTally
from alphablokus.games.tictactoe.game import TicTacToeGame
from scripts.pentobi_benchmark import (
    ChunkOutcome,
    _aggregate_level,
    _even_chunks,
    _plan_tasks,
    _play_chunk,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from alphablokus.interfaces import IBoard


# --------------------------------------------------------------------------- #
# _even_chunks
# --------------------------------------------------------------------------- #


def test_even_chunks_all_even_and_sum_to_even_games() -> None:
    for games in range(0, 41):
        for workers in range(1, 9):
            chunks = _even_chunks(games, workers)
            assert all(c % 2 == 0 for c in chunks), (games, workers, chunks)
            assert all(c > 0 for c in chunks), (games, workers, chunks)
            assert sum(chunks) == games - (games % 2), (games, workers, chunks)
            assert len(chunks) <= workers, (games, workers, chunks)


def test_even_chunks_are_balanced_within_two() -> None:
    # Remainder pairs are handed out one-per-worker, so chunks differ by at most 2.
    chunks = _even_chunks(20, 4)
    assert chunks == [6, 6, 4, 4]
    assert sum(chunks) == 20
    assert max(chunks) - min(chunks) <= 2


def test_even_chunks_more_workers_than_pairs_caps_chunk_count() -> None:
    # 6 games = 3 pairs: at most 3 non-empty even chunks however many workers.
    assert _even_chunks(6, 8) == [2, 2, 2]


def test_even_chunks_too_few_games_is_empty() -> None:
    assert _even_chunks(1, 4) == []
    assert _even_chunks(0, 4) == []


# --------------------------------------------------------------------------- #
# _plan_tasks
# --------------------------------------------------------------------------- #


def test_plan_tasks_seed_ranges_are_disjoint() -> None:
    # Each chunk reseeds Pentobi ``seed_base + game_index`` for game_index in
    # [0, n_games); those windows must never overlap across the whole sweep or two
    # workers would replay identical games.
    tasks = _plan_tasks(levels=[1, 5, 9], games=20, workers=4, seed=1)
    used: set[int] = set()
    for task in tasks:
        window = set(range(task.seed_base, task.seed_base + task.n_games))
        assert used.isdisjoint(window), f"seed overlap at {task}"
        used |= window


def test_plan_tasks_large_first_chunk_collects_records_alone() -> None:
    # games=20, workers=4 -> chunks [6,6,4,4]: the 6-game first chunk already
    # covers REPLAYS_PER_LEVEL (4), so no later chunk needs to collect.
    tasks = _plan_tasks(levels=[3, 7], games=20, workers=4, seed=1)
    for level in (3, 7):
        level_tasks = [t for t in tasks if t.level == level]
        assert level_tasks[0].collect_records is True
        assert all(t.collect_records is False for t in level_tasks[1:])


def test_plan_tasks_small_chunks_collect_until_replay_quota_met() -> None:
    # games=8, workers=4 -> chunks [2,2,2,2]: need the first two 2-game chunks to
    # reach REPLAYS_PER_LEVEL (4) records; the rest skip capture.
    tasks = _plan_tasks(levels=[1], games=8, workers=4, seed=1)
    collecting = [t for t in tasks if t.collect_records]
    assert sum(t.n_games for t in collecting) >= 4
    assert [t.collect_records for t in tasks] == [True, True, False, False]


def test_plan_tasks_per_level_games_match_serial_count() -> None:
    # Every level's chunks sum to the same even count the serial path would play.
    tasks = _plan_tasks(levels=[1, 2], games=21, workers=3, seed=0)
    for level in (1, 2):
        assert sum(t.n_games for t in tasks if t.level == level) == 20


# --------------------------------------------------------------------------- #
# _aggregate_level
# --------------------------------------------------------------------------- #


def _outcome(
    white: tuple[int, int, int],
    black: tuple[int, int, int],
    records: list[str] | None = None,
) -> ChunkOutcome:
    """A chunk outcome from ``(wins, losses, draws)`` per colour; records are opaque here."""
    return ChunkOutcome(
        as_white=ColourTally(games=sum(white), wins=white[0], losses=white[1], draws=white[2]),
        as_black=ColourTally(games=sum(black), wins=black[0], losses=black[1], draws=black[2]),
        records=records or [],  # type: ignore[arg-type]
    )


def test_aggregate_level_sums_counts_and_concatenates_records() -> None:
    chunk_results = [
        _outcome((2, 0, 0), (1, 1, 0), ["rec-a", "rec-b"]),
        _outcome((1, 1, 1), (1, 1, 1)),
        _outcome((1, 2, 0), (0, 2, 1), ["rec-c"]),
    ]
    agg = _aggregate_level(5, chunk_results)
    assert agg["level"] == 5
    assert agg["net_wins"] == 6
    assert agg["pentobi_wins"] == 7
    assert agg["draws"] == 3
    assert agg["games"] == 16
    assert agg["net_wins"] + agg["pentobi_wins"] + agg["draws"] == agg["games"]
    assert agg["win_rate"] == 6 / 16
    assert agg["records"] == ["rec-a", "rec-b", "rec-c"]


def test_aggregate_level_keeps_the_colour_split() -> None:
    """Both production paths used to report 0/0 here — missing telemetry as measured zeroes.

    ``evaluation/ladder_elo`` fits the first-mover advantage from this split, so a
    persisted 0/0 is not merely uninformative, it is unusable input.
    """
    agg = _aggregate_level(9, [_outcome((5, 4, 1), (2, 7, 1)), _outcome((3, 6, 1), (1, 8, 1))])
    assert agg["white_games"] == 20
    assert agg["white_wins"] == 8
    assert agg["white_draws"] == 2
    assert agg["white_score"] == (8 + 1.0) / 20
    # The net played 40 games in all, so the black half is the remainder.
    assert agg["games"] - agg["white_games"] == 20


def test_aggregate_level_empty_is_zero_rate() -> None:
    agg = _aggregate_level(1, [])
    assert agg["games"] == 0
    assert agg["win_rate"] == 0.0
    assert agg["records"] == []
    assert "white_score" not in agg  # no games played: absent, not a misleading 0.0


# --------------------------------------------------------------------------- #
# Picklability — the ``spawn`` pool pickles the callable + its args by value.
# --------------------------------------------------------------------------- #


def test_play_chunk_is_picklable_by_reference() -> None:
    assert pickle.loads(pickle.dumps(_play_chunk)) is _play_chunk


def test_play_chunk_args_are_plain_and_picklable() -> None:
    # A representative call's positional args — all plain types, no live net/engine.
    args = ("run.json", "best.pth.tar", 5, 4, 101, 400, 16, 1.0, 4, False, True, True, True)
    assert pickle.loads(pickle.dumps(args)) == args


def test_chunk_outcome_survives_the_process_boundary() -> None:
    """Workers return it by pickle, so the colour split has to travel intact."""
    outcome = _outcome((5, 4, 1), (2, 7, 1))
    assert pickle.loads(pickle.dumps(outcome)) == outcome


# --------------------------------------------------------------------------- #
# Fan-out + aggregate is worker-count-invariant (real spawn pool, real game).
#
# Uses a fully deterministic opponent so per-game outcomes are fixed — the
# aggregate W/L/D must then be identical regardless of how the games are split
# across workers. This exercises the exact chunk/seed/aggregate plumbing the real
# driver uses, without needing a Pentobi binary in CI.
# --------------------------------------------------------------------------- #


class _FirstLegalPlayer:
    """Deterministic player: always plays the lowest-index legal action."""

    def __init__(self, game: TicTacToeGame) -> None:
        self._game = game

    def __call__(self, board: IBoard) -> int:
        valids = self._game.valid_move_masking(board, 1)
        return int(np.flatnonzero(valids)[0])


def _dummy_play_chunk(n_games: int) -> ChunkOutcome:
    """Stand-in for ``_play_chunk``: play ``n_games`` deterministic TicTacToe games.

    Module-level (so the ``spawn`` pool can import + pickle it by reference) and
    Pentobi-free. Both players are deterministic, so the W/L/D split depends only
    on ``n_games`` — never on how the games were chunked across workers.
    """
    game = TicTacToeGame()
    player = _FirstLegalPlayer(game)
    as_white, as_black, _ = Arena(player, player, game).play_games_by_colour(n_games, record=False)
    return ChunkOutcome(as_white=as_white, as_black=as_black, records=[])


def _run_dummy_parallel(games: int, workers: int) -> tuple[int, int, int]:
    """Fan the deterministic dummy game out over a real ``spawn`` pool and aggregate."""
    chunks = _even_chunks(games, workers)
    ctx = mp.get_context("spawn")
    results: list[ChunkOutcome] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        results = list(pool.map(_dummy_play_chunk, chunks))
    agg = _aggregate_level(1, results)
    return agg["net_wins"], agg["pentobi_wins"], agg["draws"]


def test_parallel_aggregate_is_worker_count_invariant() -> None:
    games = 8
    serial = _run_dummy_parallel(games, 1)
    assert sum(serial) == games  # deterministic games never draw-out early / all counted
    for workers in (2, 4):
        assert _run_dummy_parallel(games, workers) == serial


# --------------------------------------------------------------------------- #
# The worker seeds both players.
#
# ``_play_chunk`` is driven for real (config load, net build, Arena, aggregation)
# with the two opponents faked out, so no Pentobi binary is needed. What it pins
# is that the worker hands its per-task seed to the net as well as to Pentobi:
# without it, a parallel run sampled openings from an entropy-seeded global RNG
# while its payload recorded ``--seed``, so two runs with identical recorded
# context played different games.
# --------------------------------------------------------------------------- #


class _FakeNetworkPlayer:
    """Records the kwargs it was constructed with, then plays the first legal move."""

    seen: list[dict[str, Any]] = []

    def __init__(self, game: Any, nnet: Any, mcts_config: Any, **kwargs: Any) -> None:  # noqa: ARG002
        type(self).seen.append(kwargs)
        self._game = game

    def __call__(self, board: IBoard) -> int:
        return int(np.flatnonzero(self._game.valid_move_masking(board, 1))[0])


class _FakePentobiPlayer:
    """Stands in for the engine-backed player: no subprocess, deterministic moves."""

    def __init__(self, game: Any, level: int, *, seed: int | None = None, nobook: bool) -> None:
        self._game = game
        self.level = level
        self.seed = seed
        self.nobook = nobook

    def __call__(self, board: IBoard) -> int:
        return int(np.flatnonzero(self._game.valid_move_masking(board, 1))[0])

    def close(self) -> None:
        pass


def _write_tictactoe_config(tmp_path: Path) -> Path:
    config = {
        "game": "tictactoe",
        "run_name": "chunk_seed_test",
        "num_generations": 1,
        "num_eps": 1,
        "temp_threshold": 5,
        "update_threshold": 0.55,
        "num_arena_matches": 2,
        "replay_buffer_games": 4,
        "root_directory": str(tmp_path),
        "load_model": False,
        "mcts_config": {"num_mcts_sims": 2, "cpuct": 1.0},
        "net_config": {
            "learning_rate": 0.001,
            "dropout": 0.3,
            "epochs": 1,
            "batch_size": 4,
            "cuda": False,
            "num_filters": 8,
            "num_residual_blocks": 1,
        },
    }
    path = tmp_path / "chunk_seed_test.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_play_chunk_seeds_the_net_with_its_task_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.pentobi_benchmark as bench

    _FakeNetworkPlayer.seen = []
    monkeypatch.setattr(bench, "NetworkPlayer", _FakeNetworkPlayer)
    monkeypatch.setattr(bench, "PentobiPlayer", _FakePentobiPlayer)

    outcome = _play_chunk(
        str(_write_tictactoe_config(tmp_path)),
        None,
        3,  # level
        4,  # n_games
        4242,  # seed_base
        2,  # sims
        1,  # batch
        1.0,  # opening_temp
        4,  # opening_moves
        True,  # cpu_net
        False,  # mps
        False,  # collect_records
        True,  # nobook
    )

    assert _FakeNetworkPlayer.seen == [{"temp": 0.0, "opening_temp": 1.0, "opening_moves": 4, "seed": 4242}]
    assert isinstance(outcome, ChunkOutcome)
    assert outcome.as_white.games == 2
    assert outcome.as_black.games == 2
