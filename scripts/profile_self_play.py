"""Profiling harness for the self-play hot loop.

Plays self-play game(s) single-process (no worker pool, no multiprocessing
noise) and reports where time and memory go. Part of the profiling
investigation (``docs/plans/profiling-investigation.md``).

Modes:
  timing    Play N games, aggregate the MCTS episode stats into a coarse phase
            split (inference / move-gen / game-ended / other-search) plus the
            episode overhead, with each slice's Amdahl ceiling. Also reports
            tree growth and peak memory (tracemalloc + RSS).
  cprofile  Play 1 game under cProfile; print the top functions by cumulative
            and total time — function-level attribution within the loop.

Usage:
  uv run python scripts/profile_self_play.py --config <cfg> --mode timing  --games 3
  uv run python scripts/profile_self_play.py --config <cfg> --mode cprofile
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import resource
import sys
import time
import tracemalloc

import numpy as np

from core.config import load_args
from core.game_factory import instantiate_game_and_network
from core.mcts import MCTS
from core.self_play import play_self_play_episode


def _play_one(game, nnet, config, seed: int):
    """Play one self-play game at a fixed seed; return (episode_stats, n_examples, wall_s)."""
    np.random.seed(seed)
    mcts = MCTS(game, nnet, config.mcts_config)
    t0 = time.perf_counter()
    examples = play_self_play_episode(game, mcts, config.temp_threshold)
    wall = time.perf_counter() - t0
    return mcts.get_episode_stats(), len(examples), wall


def _pct(x: float, total: float) -> float:
    return 100.0 * x / total if total else 0.0


def _amdahl_ceiling(p_frac: float) -> float:
    """Max whole-game speedup if this slice (fraction p of runtime) -> 0 time."""
    return 1.0 / (1.0 - p_frac) if p_frac < 1.0 else float("inf")


def run_timing(game, nnet, config, games: int, seed: int) -> None:
    # Warm up once (CUDA context, lazy imports, first-call JIT) so the timed
    # games measure steady state, not one-off startup.
    _play_one(game, nnet, config, seed)

    tracemalloc.start()
    agg = dict(moves=0, sims=0, search=0.0, infer=0.0, movegen=0.0, gameended=0.0,
               wall=0.0, leaves=0, tree=0, examples=0)
    per_game_wall = []
    for i in range(games):
        s, nex, wall = _play_one(game, nnet, config, seed + 1 + i)
        agg["moves"] += s.num_moves
        agg["sims"] += getattr(s, "total_sims", 0)
        agg["search"] += s.total_search_time_s
        agg["infer"] += s.total_inference_time_s
        agg["movegen"] += s.total_valid_moves_time_s
        agg["gameended"] += s.total_game_ended_time_s
        agg["leaves"] += s.num_leaf_expansions
        agg["tree"] += s.tree_size
        agg["examples"] += nex
        agg["wall"] += wall
        per_game_wall.append(wall)

    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = raw_rss / 1e6 if sys.platform == "darwin" else raw_rss / 1e3  # macOS bytes / Linux KB -> MB

    wall = agg["wall"]
    other = agg["search"] - agg["infer"] - agg["movegen"] - agg["gameended"]
    overhead = wall - agg["search"]

    print("\n================ TIMING PROFILE (single-process self-play) ================")
    print(f"games={games}  total moves={agg['moves']}  total sims={agg['sims']}  "
          f"leaf expansions={agg['leaves']}")
    print(f"avg wall/game = {wall / games:.2f}s   (per-game: "
          f"{', '.join(f'{w:.2f}' for w in per_game_wall)})")
    print(f"avg tree size (states)/game = {agg['tree'] / games:.0f}   "
          f"avg examples/game = {agg['examples'] / games:.0f}")
    print(f"peak Python heap (tracemalloc) = {peak / 1e6:.1f} MB   process RSS = {rss:.0f} MB")
    print("\n  slice                     time(s)    % of game   Amdahl ceiling (slice->0)")
    print("  " + "-" * 72)
    rows = [
        ("NN inference", agg["infer"]),
        ("Move generation", agg["movegen"]),
        ("Game-ended checks", agg["gameended"]),
        ("Other search (select/UCB/expand/backprop/keying)", other),
        ("Episode overhead (symmetries/get_next_state/sampling)", overhead),
    ]
    for name, t in rows:
        p = t / wall if wall else 0.0
        print(f"  {name:<52}{t:7.2f}   {_pct(t, wall):6.1f}%      {_amdahl_ceiling(p):5.2f}x")
    print("  " + "-" * 72)
    print(f"  {'TOTAL (wall)':<52}{wall:7.2f}   100.0%")
    print(f"\n  (search subtotal {agg['search']:.2f}s = {_pct(agg['search'], wall):.1f}% of game; "
          f"the rest is episode overhead outside the MCTS loop)")
    print("==========================================================================\n")


def run_cprofile(game, nnet, config, seed: int) -> None:
    _play_one(game, nnet, config, seed)  # warm up
    pr = cProfile.Profile()
    np.random.seed(seed + 1)
    mcts = MCTS(game, nnet, config.mcts_config)
    pr.enable()
    play_self_play_episode(game, mcts, config.temp_threshold)
    pr.disable()

    for sort_key, label in [("tottime", "TOTAL time in the function itself (excl. callees)"),
                            ("cumulative", "CUMULATIVE time (incl. callees)")]:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort_key)
        ps.print_stats(25)
        print(f"\n================ cProfile — top 25 by {sort_key} ================")
        print(f"({label})")
        print(s.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile the self-play hot loop")
    ap.add_argument("--config", default="run_configurations/blokus_scaled_15.json")
    ap.add_argument("--mode", choices=["timing", "cprofile"], default="timing")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = load_args(args.config)
    game, nnet = instantiate_game_and_network(config)
    print(f"game={config.game}  sims={config.mcts_config.num_mcts_sims}  "
          f"K(batch)={config.mcts_config.mcts_batch_size}  cuda={config.net_config.cuda}  "
          f"net={config.net_config.num_filters}f x {config.net_config.num_residual_blocks}b "
          f"{config.net_config.policy_head}")

    if args.mode == "timing":
        run_timing(game, nnet, config, args.games, args.seed)
    else:
        run_cprofile(game, nnet, config, args.seed)


if __name__ == "__main__":
    main()
