"""Profiling harness for the training cycle's hot paths.

Plays self-play game(s) single-process (no worker pool, no multiprocessing
noise) and/or profiles the training step, reporting where time and memory go.
Part of the profiling investigation (``docs/plans/profiling-investigation.md``).

Modes:
  timing    Play N games; report wall/game, the coarse MCTS phase split, and
            RSS. NO tracemalloc (it distorts allocation-heavy timing).
  cprofile  Play 1 game under cProfile; print top functions by cumulative and
            total time — the honest function-level attribution.
  memory    Play 1 game under tracemalloc; report peak heap + tree growth.
  train     Time ``nnet.train()`` over a synthetic realistic-size buffer — the
            per-generation training cost (the phase MCTS work doesn't help).

  --sims overrides num_mcts_sims (the per-sim function split is sim-count
  independent, so a smaller value profiles faster).
"""
from __future__ import annotations

import argparse
import cProfile
import dataclasses
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
    np.random.seed(seed)
    mcts = MCTS(game, nnet, config.mcts_config)
    t0 = time.perf_counter()
    examples = play_self_play_episode(game, mcts, config.temp_threshold)
    wall = time.perf_counter() - t0
    return mcts.get_episode_stats(), len(examples), wall


def _pct(x: float, total: float) -> float:
    return 100.0 * x / total if total else 0.0


def _amdahl(p_frac: float) -> float:
    return 1.0 / (1.0 - p_frac) if p_frac < 1.0 else float("inf")


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3  # macOS bytes / Linux KB


def run_timing(game, nnet, config, games: int, seed: int) -> None:
    _play_one(game, nnet, config, seed)  # warm up (CUDA ctx, first-call JIT)
    agg = dict(moves=0, sims=0, search=0.0, infer=0.0, movegen=0.0,
               gameended=0.0, wall=0.0, leaves=0, tree=0, examples=0)
    per_game = []
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
        per_game.append(wall)

    wall = agg["wall"]
    other = agg["search"] - agg["infer"] - agg["movegen"] - agg["gameended"]
    overhead = wall - agg["search"]
    print("\n================ TIMING (single-process self-play, no tracemalloc) ================")
    print(f"games={games}  moves={agg['moves']}  sims={agg['sims']}  leaves={agg['leaves']}")
    print(f"avg wall/game = {wall / games:.2f}s  (per-game: {', '.join(f'{w:.1f}' for w in per_game)})")
    print(f"avg tree states/game = {agg['tree'] / games:.0f}  examples/game = {agg['examples'] / games:.0f}"
          f"  RSS={_rss_mb():.0f} MB")
    print("\n  NOTE: move-gen/game-ended timers are not wired into the batched search path,")
    print("        so they read ~0 here — cProfile gives the honest attribution.")
    print("\n  slice                          time(s)    %game    Amdahl ceiling")
    print("  " + "-" * 62)
    for name, t in [("NN inference", agg["infer"]),
                    ("Move generation (timer gap → see cProfile)", agg["movegen"]),
                    ("Game-ended (timer gap → see cProfile)", agg["gameended"]),
                    ("Other search (select/UCB/expand/backprop/key)", other),
                    ("Episode overhead", overhead)]:
        print(f"  {name:<46}{t:7.2f}  {_pct(t, wall):5.1f}%   {_amdahl(t / wall):5.2f}x")
    print("  " + "-" * 62)
    print(f"  {'TOTAL wall':<46}{wall:7.2f}  100.0%")
    print("==================================================================================\n")


def run_cprofile(game, nnet, config, seed: int) -> None:
    _play_one(game, nnet, config, seed)  # warm up
    pr = cProfile.Profile()
    np.random.seed(seed + 1)
    mcts = MCTS(game, nnet, config.mcts_config)
    pr.enable()
    play_self_play_episode(game, mcts, config.temp_threshold)
    pr.disable()
    for key, label in [("tottime", "own time, excl. callees — the optimisation targets"),
                       ("cumulative", "incl. callees")]:
        s = io.StringIO()
        pstats.Stats(pr, stream=s).strip_dirs().sort_stats(key).print_stats(22)
        print(f"\n================ cProfile top 22 by {key} ({label}) ================")
        print(s.getvalue())


def run_memory(game, nnet, config, seed: int) -> None:
    _play_one(game, nnet, config, seed)  # warm up
    tracemalloc.start()
    s, _nex, _wall = _play_one(game, nnet, config, seed + 1)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("\n================ MEMORY (one self-play game, tracemalloc) ================")
    print(f"tree states at game end = {s.tree_size}   leaf expansions = {s.num_leaf_expansions}")
    print(f"peak Python heap = {peak / 1e6:.1f} MB   process RSS = {_rss_mb():.0f} MB")
    print("==========================================================================\n")


def run_train(game, nnet, config, n_examples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    action_size = game.get_action_size()
    # Boards are stored compact now; the trainer re-encodes lazily.
    board_shape = game.initialise_board().to_compact().shape
    print(f"building {n_examples} synthetic examples (compact board {board_shape}, sparse pi)...")
    examples = []
    for _ in range(n_examples):
        board = rng.integers(-21, 22, board_shape).astype(np.int8)
        idx = rng.choice(action_size, size=150, replace=False).astype(np.int32)
        val = rng.random(150).astype(np.float32)
        val /= val.sum()
        examples.append((board, (idx, val), float(rng.uniform(-1, 1))))
    t0 = time.perf_counter()
    nnet.train(examples, generation=0)
    wall = time.perf_counter() - t0
    epochs = config.net_config.epochs
    print("\n================ TRAINING STEP (synthetic buffer) ================")
    print(f"examples={n_examples}  epochs={epochs}  batch_size={config.net_config.batch_size}")
    print(f"train() wall = {wall:.1f}s   ({wall / epochs:.1f}s/epoch)   RSS={_rss_mb():.0f} MB")
    print("==================================================================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile the training cycle's hot paths")
    ap.add_argument("--config", default="run_configurations/blokus_scaled_15.json")
    ap.add_argument("--mode", choices=["timing", "cprofile", "memory", "train"], default="timing")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--sims", type=int, default=0, help="override num_mcts_sims (0 = config value)")
    ap.add_argument("--examples", type=int, default=57000, help="train mode buffer size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = load_args(args.config)
    if args.sims:
        config = dataclasses.replace(
            config, mcts_config=dataclasses.replace(config.mcts_config, num_mcts_sims=args.sims))
    game, nnet = instantiate_game_and_network(config)
    # Match production: the Coach/workers call _maybe_enable_f2 to route move-gen
    # through the precomputed-table generator. instantiate_game does NOT, so do it here.
    f2_on = getattr(config, "use_optimised_movegen", False)
    if f2_on and hasattr(game, "enable_optimised_movegen"):
        game.enable_optimised_movegen()
    print(f"game={config.game} sims={config.mcts_config.num_mcts_sims} "
          f"K={config.mcts_config.mcts_batch_size} cuda={config.net_config.cuda} "
          f"f2_movegen={f2_on} "
          f"net={config.net_config.num_filters}f×{config.net_config.num_residual_blocks}b "
          f"{config.net_config.policy_head}")

    if args.mode == "timing":
        run_timing(game, nnet, config, args.games, args.seed)
    elif args.mode == "cprofile":
        run_cprofile(game, nnet, config, args.seed)
    elif args.mode == "memory":
        run_memory(game, nnet, config, args.seed)
    else:
        run_train(game, nnet, config, args.examples, args.seed)


if __name__ == "__main__":
    main()
