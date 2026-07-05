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

import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from alphablokus.evaluation.arena import Arena
from alphablokus.games.tictactoe.game import TicTacToeGame
from scripts.pentobi_benchmark import (
    ChunkResult,
    _aggregate_level,
    _even_chunks,
    _plan_tasks,
    _play_chunk,
)

if TYPE_CHECKING:
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


def test_aggregate_level_sums_counts_and_concatenates_records() -> None:
    chunk_results: list[ChunkResult] = [
        (3, 1, 0, ["rec-a", "rec-b"]),  # type: ignore[list-item]  # records are opaque here
        (2, 2, 2, []),
        (1, 4, 1, ["rec-c"]),  # type: ignore[list-item]
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


def test_aggregate_level_empty_is_zero_rate() -> None:
    agg = _aggregate_level(1, [])
    assert agg["games"] == 0
    assert agg["win_rate"] == 0.0
    assert agg["records"] == []


# --------------------------------------------------------------------------- #
# Picklability — the ``spawn`` pool pickles the callable + its args by value.
# --------------------------------------------------------------------------- #


def test_play_chunk_is_picklable_by_reference() -> None:
    assert pickle.loads(pickle.dumps(_play_chunk)) is _play_chunk


def test_play_chunk_args_are_plain_and_picklable() -> None:
    # A representative call's positional args — all plain types, no live net/engine.
    args = ("run.json", "best.pth.tar", 5, 4, 101, 400, 16, 1.0, 4, False, True, True)
    assert pickle.loads(pickle.dumps(args)) == args


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


def _dummy_play_chunk(n_games: int) -> ChunkResult:
    """Stand-in for ``_play_chunk``: play ``n_games`` deterministic TicTacToe games.

    Module-level (so the ``spawn`` pool can import + pickle it by reference) and
    Pentobi-free. Both players are deterministic, so the W/L/D split depends only
    on ``n_games`` — never on how the games were chunked across workers.
    """
    game = TicTacToeGame()
    player = _FirstLegalPlayer(game)
    p1_wins, p2_wins, draws, _ = Arena(player, player, game).play_games(n_games, record=False)
    return p1_wins, p2_wins, draws, []


def _run_dummy_parallel(games: int, workers: int) -> tuple[int, int, int]:
    """Fan the deterministic dummy game out over a real ``spawn`` pool and aggregate."""
    chunks = _even_chunks(games, workers)
    ctx = mp.get_context("spawn")
    results: list[ChunkResult] = []
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
