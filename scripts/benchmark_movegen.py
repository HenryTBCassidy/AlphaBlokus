"""Microbenchmark: F2 vs current move-gen, per-call.

Loads the dev_5000 cached positions, runs both implementations on each,
and reports the per-position wall-clock breakdown. Designed to give a
clear answer to "how much faster is F2?" without the noise of a full
training-pipeline benchmark.

Usage::

    uv run python -m scripts.benchmark_movegen

Output: per-implementation total time, mean/p50/p95 per-call time, and
the speedup factor. Validates that both implementations agree on every
position before reporting timings (skipping a position if they disagree
would be silent corruption).
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.movegen_runtime import get_default_generator
from tests.fixtures.blokus_positions import (
    load_cache,
    replay_to_board_and_player,
)

PIECES_PATH = Path(__file__).resolve().parent.parent / "games" / "blokusduo" / "pieces.json"
DEV_CACHE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "blokus_duo_positions" / "dev_5000.npz"


def _phase_bucket(n_moves: int) -> str:
    if n_moves == 0:
        return "empty"
    if n_moves <= 8:
        return "early"
    if n_moves <= 18:
        return "mid"
    if n_moves <= 26:
        return "late"
    return "end"


def main() -> int:
    if not DEV_CACHE.exists():
        print(f"FAIL: dev cache missing at {DEV_CACHE}")
        return 1

    print(f"Loading positions from {DEV_CACHE.name}...")
    actions_array, n_moves_array = load_cache(DEV_CACHE)
    n_positions = len(n_moves_array)
    print(f"  {n_positions:,} positions loaded")

    game = BlokusDuoGame(pieces_config_path=PIECES_PATH)

    # Warm up tables + caches by building the F2 generator.
    print("Building F2 tables (one-time cost)...")
    t0 = time.perf_counter()
    f2 = get_default_generator()
    table_build_s = time.perf_counter() - t0
    print(f"  built in {table_build_s * 1000:.0f}ms")
    print()

    # Replay each position once up front so the timed phase doesn't
    # include replay time. Two parallel lists: boards + players.
    print("Replaying positions to construct boards...")
    boards = []
    players = []
    for i in range(n_positions):
        n_moves = int(n_moves_array[i])
        sequence = actions_array[i, :n_moves]
        board, player = replay_to_board_and_player(game, sequence)
        boards.append(board)
        players.append(player)
    print(f"  {len(boards):,} boards ready")
    print()

    # ----- Time the current implementation -----
    print("Timing current implementation (game.valid_move_masking)...")
    per_call_current: list[float] = []
    current_masks: list[np.ndarray] = []
    t0 = time.perf_counter()
    for board, player in zip(boards, players, strict=True):
        call_start = time.perf_counter()
        mask = game.valid_move_masking(board, player)
        per_call_current.append(time.perf_counter() - call_start)
        current_masks.append((mask > 0).astype(np.uint8))
    current_total = time.perf_counter() - t0
    print(f"  total: {current_total:.2f}s ({n_positions} calls)")
    print()

    # ----- Time the F2 implementation -----
    print("Timing F2 implementation (movegen_runtime.valid_move_mask)...")
    per_call_f2: list[float] = []
    f2_masks: list[np.ndarray] = []
    t0 = time.perf_counter()
    for board, player in zip(boards, players, strict=True):
        call_start = time.perf_counter()
        mask = f2.valid_move_mask(game, board, player)
        per_call_f2.append(time.perf_counter() - call_start)
        f2_masks.append((mask > 0).astype(np.uint8))
    f2_total = time.perf_counter() - t0
    print(f"  total: {f2_total:.2f}s ({n_positions} calls)")
    print()

    # ----- Verify equivalence (paranoia: in case the test infra missed something) -----
    mismatches = 0
    for i, (mc, mf) in enumerate(zip(current_masks, f2_masks, strict=True)):
        if not np.array_equal(mc, mf):
            mismatches += 1
            if mismatches <= 3:
                only_c = np.where((mc == 1) & (mf == 0))[0]
                only_f = np.where((mf == 1) & (mc == 0))[0]
                print(f"  MISMATCH at position {i}: only-current={only_c[:5]}, only-f2={only_f[:5]}")
    if mismatches:
        print(f"FAIL: {mismatches} positions disagree. Halting.")
        return 1
    print(f"All {n_positions:,} masks match between implementations.")
    print()

    # ----- Report -----
    speedup = current_total / f2_total if f2_total > 0 else float("inf")
    print(f"{'=' * 60}")
    print(f"Headline: F2 is {speedup:.2f}× faster than the current implementation.")
    print(f"{'=' * 60}")
    print()
    print(f"{'Implementation':<30s} {'total':>10s} {'mean':>10s} {'p50':>10s} {'p95':>10s}")
    for name, calls in [("current", per_call_current), ("F2", per_call_f2)]:
        arr = np.array(calls)
        print(f"{name:<30s} {arr.sum():>9.2f}s {arr.mean()*1000:>9.2f}ms "
              f"{np.median(arr)*1000:>9.2f}ms {np.percentile(arr,95)*1000:>9.2f}ms")
    print()

    # ----- Per-phase breakdown -----
    print("Per-phase mean per-call time:")
    by_phase_current: dict[str, list[float]] = defaultdict(list)
    by_phase_f2: dict[str, list[float]] = defaultdict(list)
    for i in range(n_positions):
        bucket = _phase_bucket(int(n_moves_array[i]))
        by_phase_current[bucket].append(per_call_current[i])
        by_phase_f2[bucket].append(per_call_f2[i])
    print(f"  {'phase':<8s} {'n':>5s} {'current ms':>13s} {'F2 ms':>10s} {'speedup':>10s}")
    for phase in ["empty", "early", "mid", "late", "end"]:
        c = by_phase_current.get(phase, [])
        f = by_phase_f2.get(phase, [])
        if not c:
            continue
        c_mean = np.mean(c) * 1000
        f_mean = np.mean(f) * 1000
        sp = c_mean / f_mean if f_mean > 0 else float("inf")
        print(f"  {phase:<8s} {len(c):>5d} {c_mean:>12.2f}ms {f_mean:>9.2f}ms {sp:>9.2f}×")

    return 0


if __name__ == "__main__":
    sys.exit(main())
