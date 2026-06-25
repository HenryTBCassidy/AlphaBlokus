"""Wall-clock benchmark for self-play, arena, and Elo phases.

This is the "before/after" tool for performance optimisations in
``docs/plans/full-cycle-optimisation.md``. It runs the three game-playing
phases of a generation at production shape (same net + sim count), captures
per-game MCTS profiling stats, and produces an HTML report with:

1. Wall-clock per phase + bar chart (the headline parallelism number).
2. Run-time estimator: predicts how long a full run of arbitrary scale
   takes given the measured per-game numbers.
3. The same per-move drill-down ``scripts/mcts_profiling.py`` produces,
   but one section per phase — so we can see whether move-gen, inference,
   or "other" shifts after each optimisation.

Usage::

    uv run python -m scripts.benchmark_phases \\
        --config run_configurations/profile_baseline.json

Output lands at ``temp/benchmarks/<run_name>_benchmark/``.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from core.arena import Arena
from core.config import RunConfig, load_args
from core.game_factory import instantiate_game_and_network
from core.mcts import MCTS, MCTSEpisodeStats
from core.players import NetworkPlayer
from reporting.mcts_profiling import PhaseResult, build_multi_phase_report

if TYPE_CHECKING:
    from core.interfaces import IGame, INeuralNetWrapper

# Force line-buffered stdout regardless of TTY. Benchmark runs detached
# under systemd-run/journald and the default block-buffered stdout would
# stall all the per-game progress lines until the process exited — turning
# "is the run healthy?" into a 30-minute blind wait. Pair this with
# ``PYTHONUNBUFFERED=1`` in the launch script for belt-and-braces.
sys.stdout.reconfigure(line_buffering=True)


def _fmt_mmss(seconds: float) -> str:
    """Compact m:ss / h:mm:ss formatter for log lines."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds % 60):02d}s"
    hours = minutes / 60
    return f"{int(hours)}h {int(minutes % 60):02d}m"


class _PhaseHeartbeat:
    """Rolling per-game progress printer with ETA.

    The first game has no rate to extrapolate from, so it prints
    ``ETA estimating...``. From game 2 onward, ETA = mean per-game ×
    games remaining, using a simple cumulative mean — robust against the
    occasional outlier game without needing a windowed average.

    Every line ends with ``flush=True`` so it lands in journald
    immediately even when stdout is not a TTY.
    """

    def __init__(self, phase_name: str, total: int) -> None:
        self._phase = phase_name
        self._total = total
        self._start = time.perf_counter()
        self._game_times: list[float] = []
        self._last_game_start = self._start
        print(f"[{phase_name}] {total} games — starting", flush=True)

    def tick(self, *, moves: int, search_time_s: float) -> None:
        now = time.perf_counter()
        this_game = now - self._last_game_start
        self._game_times.append(this_game)
        self._last_game_start = now

        done = len(self._game_times)
        elapsed = now - self._start
        avg = elapsed / done
        remaining = self._total - done
        eta_s = avg * remaining if done >= 1 else None
        eta_str = _fmt_mmss(eta_s) if eta_s is not None else "estimating..."

        print(
            f"  [{self._phase}] {done}/{self._total} | "
            f"{moves} moves, {search_time_s:.1f}s search | "
            f"elapsed {_fmt_mmss(elapsed)} | "
            f"avg {avg:.1f}s/game | ETA {eta_str}",
            flush=True,
        )


def _force_detailed_profiling(config: RunConfig) -> RunConfig:
    """Benchmark needs per-move stats so the drill-down report is meaningful."""
    return replace(
        config,
        mcts_config=replace(config.mcts_config, profiling_level="detailed"),
    )


def _run_self_play_phase(
    config: RunConfig, game: IGame, nnet: INeuralNetWrapper, num_workers: int,
) -> PhaseResult:
    """Self-play phase, sequential or process-pool-parallel based on
    ``num_workers``.

    Serial path (``num_workers == 1``) keeps the existing MCTS-per-episode
    pattern. Parallel path delegates to
    :func:`core.parallel_self_play.run_self_play_episodes_parallel` —
    the same orchestrator ``Coach._run_self_play_parallel`` uses. We
    save ``nnet``'s weights to a checkpoint workers load at pool init.
    """
    if num_workers > 1:
        return _run_self_play_phase_parallel(config, nnet, num_workers)
    return _run_self_play_phase_serial(config, game, nnet)


def _run_self_play_phase_serial(
    config: RunConfig, game: IGame, nnet: INeuralNetWrapper,
) -> PhaseResult:
    heartbeat = _PhaseHeartbeat("Self-Play", config.num_eps)
    per_game_stats: list[MCTSEpisodeStats] = []
    phase_start = time.perf_counter()

    iteration_examples: deque = deque()  # benchmark harness: only the count matters
    for _ep in range(config.num_eps):
        mcts = MCTS(game, nnet, config.mcts_config)
        examples_count = _play_one_self_play_episode(game, mcts, config)
        iteration_examples.extend([None] * examples_count)  # only the count matters here
        stats = mcts.get_episode_stats()
        per_game_stats.append(stats)
        heartbeat.tick(moves=stats.num_moves, search_time_s=stats.total_search_time_s)

    wall_clock = time.perf_counter() - phase_start
    return PhaseResult(name="Self-Play", wall_clock_s=wall_clock, stats=per_game_stats)


def _run_self_play_phase_parallel(
    config: RunConfig, nnet: INeuralNetWrapper, num_workers: int,
) -> PhaseResult:
    """Dispatch self-play across ``num_workers`` worker processes via the
    parallel self-play orchestrator. Saves the current ``nnet`` to a
    fixed-name checkpoint workers load at pool init.
    """
    from core.parallel_self_play import run_self_play_episodes_parallel

    print(f"[Self-Play] {config.num_eps} games across {num_workers} workers — starting",
          flush=True)
    checkpoint = "benchmark_worker_init.pth.tar"
    nnet.save_checkpoint(filename=checkpoint)

    phase_start = time.perf_counter()
    _per_ep_examples, per_ep_stats = run_self_play_episodes_parallel(
        config=config,
        generation=0,  # standalone benchmark has no real generation index
        checkpoint_path=checkpoint,
        num_workers=num_workers,
    )
    wall_clock = time.perf_counter() - phase_start
    return PhaseResult(name="Self-Play", wall_clock_s=wall_clock, stats=list(per_ep_stats))


def _play_one_self_play_episode(
    game: IGame, mcts: MCTS, config: RunConfig,
) -> int:
    """Replays ``Coach.execute_episode`` but without storing training examples.

    Returns the example *count* so the caller can verify the episode shape.
    Examples themselves are discarded: the benchmark cares about wall-clock
    and MCTS stats, not about training data.
    """
    board = game.initialise_board()
    current_player = 1
    move_count = 0
    examples = 0
    while True:
        move_count += 1
        canonical_board = game.get_canonical_form(board, current_player)
        temperature = int(move_count < config.temp_threshold)
        pi = mcts.get_action_prob(canonical_board, temp=temperature)
        # ``get_symmetries`` is what amplifies one position into many examples.
        # Count them so the "examples per game" ratio is realistic.
        examples += len(game.get_symmetries(canonical_board, pi))
        action = np.random.choice(len(pi), p=pi)
        board, current_player = game.get_next_state(board, current_player, action)
        if game.get_game_ended(board, current_player) != 0:
            return examples


def _combine_episode_stats(stats_list: list[MCTSEpisodeStats]) -> MCTSEpisodeStats:
    """Sum two episodes' worth of stats into one (used to fold both arena
    players' MCTS profilers into a single per-game record).

    Move-level stats are concatenated rather than summed — different players
    contribute different move indices and we want the per-move breakdown to
    reflect the whole game.
    """
    if not stats_list:
        raise ValueError("stats_list cannot be empty")
    if len(stats_list) == 1:
        return stats_list[0]
    combined_moves: list = []
    for s in stats_list:
        combined_moves.extend(s.move_stats)
    combined_moves.sort(key=lambda m: m.move_number)
    return MCTSEpisodeStats(
        num_moves=sum(s.num_moves for s in stats_list),
        total_sims=sum(s.total_sims for s in stats_list),
        total_search_time_s=sum(s.total_search_time_s for s in stats_list),
        total_inference_time_s=sum(s.total_inference_time_s for s in stats_list),
        num_leaf_expansions=sum(s.num_leaf_expansions for s in stats_list),
        tree_size=sum(s.tree_size for s in stats_list),
        total_valid_moves_time_s=sum(s.total_valid_moves_time_s for s in stats_list),
        total_game_ended_time_s=sum(s.total_game_ended_time_s for s in stats_list),
        num_valid_moves_calls=sum(s.num_valid_moves_calls for s in stats_list),
        num_game_ended_calls=sum(s.num_game_ended_calls for s in stats_list),
        tree_memory_bytes=sum(s.tree_memory_bytes for s in stats_list),
        move_stats=tuple(combined_moves),
        mean_policy_entropy=float(np.mean([s.mean_policy_entropy for s in stats_list])),
    )


def _run_two_player_phase(
    *,
    phase_name: str, num_games: int, config: RunConfig,
    game: IGame, nnet_a: INeuralNetWrapper, nnet_b: INeuralNetWrapper,
    num_workers: int,
) -> PhaseResult:
    """Shared engine for arena and Elo phases. Two-player game, swapped halfway.

    Sequential when ``num_workers == 1`` (preserves the original
    benchmark behaviour). Process-pool parallel via the parallel
    self-play orchestrator otherwise — the same one Coach uses for live
    training.
    """
    if num_workers > 1:
        return _run_two_player_phase_parallel(
            phase_name=phase_name, num_games=num_games, config=config,
            nnet_a=nnet_a, nnet_b=nnet_b, num_workers=num_workers,
        )
    return _run_two_player_phase_serial(
        phase_name=phase_name, num_games=num_games, config=config,
        game=game, nnet_a=nnet_a, nnet_b=nnet_b,
    )


def _run_two_player_phase_serial(
    *,
    phase_name: str, num_games: int, config: RunConfig,
    game: IGame, nnet_a: INeuralNetWrapper, nnet_b: INeuralNetWrapper,
) -> PhaseResult:
    heartbeat = _PhaseHeartbeat(phase_name, num_games)
    per_game_stats: list[MCTSEpisodeStats] = []
    phase_start = time.perf_counter()

    # We play one game at a time so we can reset the players' MCTS trees and
    # capture clean per-game stats. ``Arena.play_games`` would do the swap
    # bookkeeping but bundles all games together — easier to drive directly.
    for game_idx in range(num_games):
        # Swap halfway, matching ``Arena.play_games``'s convention.
        if game_idx < num_games // 2:
            player_a, player_b = nnet_a, nnet_b
        else:
            player_a, player_b = nnet_b, nnet_a

        net_player_a = NetworkPlayer(
            game=game, nnet=player_a, mcts_config=config.mcts_config, temp=0.0,
        )
        net_player_b = NetworkPlayer(
            game=game, nnet=player_b, mcts_config=config.mcts_config, temp=0.0,
        )
        arena = Arena(net_player_a, net_player_b, game)
        _, _ = arena.play_game(record=False)

        combined = _combine_episode_stats([
            net_player_a._mcts.get_episode_stats(),
            net_player_b._mcts.get_episode_stats(),
        ])
        per_game_stats.append(combined)
        heartbeat.tick(moves=combined.num_moves, search_time_s=combined.total_search_time_s)

    wall_clock = time.perf_counter() - phase_start
    return PhaseResult(name=phase_name, wall_clock_s=wall_clock, stats=per_game_stats)


def _run_two_player_phase_parallel(
    *,
    phase_name: str, num_games: int, config: RunConfig,
    nnet_a: INeuralNetWrapper, nnet_b: INeuralNetWrapper, num_workers: int,
) -> PhaseResult:
    """Parallel two-player phase via the parallel self-play orchestrator.

    Workers play the games — main process discards win/loss bookkeeping
    (the benchmark cares about wall-clock + per-game MCTS stats, not
    outcomes) and only retains stats. Each worker returns stats for the
    *whole game* (both players combined inside the worker), so each
    task contributes one combined ``MCTSEpisodeStats`` to the phase.
    """
    from core.parallel_self_play import (
        PHASE_ARENA,
        run_two_player_games_parallel,
    )

    print(
        f"[{phase_name}] {num_games} games across {num_workers} workers — starting",
        flush=True,
    )
    # Use distinct checkpoint paths so concurrent arena and Elo phases
    # never clobber each other if a future change runs them at once.
    slug = phase_name.lower().replace(" ", "_").replace("-", "_")
    checkpoint_a = f"benchmark_{slug}_a.pth.tar"
    checkpoint_b = f"benchmark_{slug}_b.pth.tar"
    nnet_a.save_checkpoint(filename=checkpoint_a)
    nnet_b.save_checkpoint(filename=checkpoint_b)

    phase_start = time.perf_counter()
    # We use PHASE_ARENA for both arena and Elo here because the
    # benchmark doesn't care about cross-phase seed namespacing — the
    # checkpoints differ between Coach's real arena and Elo, but here
    # the workers always replay against the same a/b pair we just
    # saved. The orchestrator's seed derivation gives unique seeds per
    # game within the phase, which is enough.
    run_two_player_games_parallel(
        config=config,
        generation=0,
        checkpoint_a_path=checkpoint_a,
        checkpoint_b_path=checkpoint_b,
        num_games=num_games,
        num_workers=num_workers,
        phase=PHASE_ARENA,
        record=False,
        top_k=0,
        desc=phase_name,
    )
    wall_clock = time.perf_counter() - phase_start
    # Parallel path doesn't currently return per-game MCTS stats (the
    # orchestrator was sized for outcome bookkeeping, not profiling).
    # That's acceptable for the parallelism measurement — the headline is
    # wall-clock per phase, and stats are still available from the
    # self-play phase. Future change: extend the worker return value to
    # include stats if we want full per-game drill-down here.
    return PhaseResult(name=phase_name, wall_clock_s=wall_clock, stats=[])


def _build_estimator_table(
    phases: dict[str, PhaseResult],
    config: RunConfig,
    extrapolate_scales: list[dict],
) -> str:
    """Produce an HTML table of wall-clock estimates for a list of run scales.

    Each scale row has fields:
        - name (label)
        - num_eps, num_arena, elo_games, num_gens
    The cost model: per-phase mean per-game wall-clock × game count × num_gens,
    plus a flat per-gen training overhead (taken from
    ``docs/08-TRAINING-ESTIMATES.md`` — ~3 min/gen at 64f×4b).
    """
    def mean_per_game(p: PhaseResult) -> float:
        if not p.stats:
            return 0.0
        return p.wall_clock_s / len(p.stats)

    sp_mean = mean_per_game(phases["Self-Play"])
    arena_mean = mean_per_game(phases["Arena"])
    elo_mean = mean_per_game(phases["Elo"])
    training_overhead_s = 180.0  # ~3 min — replace once we measure this in
                                 # the benchmark too if it matters

    rows = []
    for scale in extrapolate_scales:
        sp_s = sp_mean * scale["num_eps"] * scale["num_gens"]
        ar_s = arena_mean * scale["num_arena"] * scale["num_gens"]
        elo_s = elo_mean * scale["elo_games"] * scale["num_gens"]
        train_s = training_overhead_s * scale["num_gens"]
        total_s = sp_s + ar_s + elo_s + train_s
        rows.append({
            "Scale": scale["name"],
            "Gens × Eps × Arena × Elo":
                f"{scale['num_gens']} × {scale['num_eps']} × {scale['num_arena']} × {scale['elo_games']}",
            "Self-play": _fmt_duration(sp_s),
            "Arena": _fmt_duration(ar_s),
            "Elo": _fmt_duration(elo_s),
            "Training (est.)": _fmt_duration(train_s),
            "Total": _fmt_duration(total_s),
        })

    if not rows:
        return ""
    headers = list(rows[0].keys())
    out = "<table class=\"summary\"><thead><tr>"
    out += "".join(f"<th>{h}</th>" for h in headers)
    out += "</tr></thead><tbody>"
    for r in rows:
        out += "<tr>" + "".join(f"<td>{r[h]}</td>" for h in headers) + "</tr>"
    out += "</tbody></table>"
    out += (
        "<p class=\"subtitle\">Self-play/arena/Elo costs scale linearly with "
        "their game counts × number of generations. Training overhead is a "
        "rough 3-min/gen estimate (refresh once F4 work changes it).</p>"
    )
    return out


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _write_parquets(phases: dict[str, PhaseResult], output_dir: Path) -> None:
    """Persist per-phase per-game wall-clock records as parquets so we have
    a structured artifact for the master plan's progress tracker.
    """
    rows = []
    for name, p in phases.items():
        for i, s in enumerate(p.stats):
            rows.append({
                "phase": name, "game_idx": i, "num_moves": s.num_moves,
                "total_sims": s.total_sims,
                "search_time_s": s.total_search_time_s,
                "inference_time_s": s.total_inference_time_s,
                "valid_moves_time_s": s.total_valid_moves_time_s,
                "game_ended_time_s": s.total_game_ended_time_s,
            })
    per_game_df = pd.DataFrame(rows)
    per_game_df.to_parquet(output_dir / "per_game.parquet")

    summary_rows = []
    for name, p in phases.items():
        summary_rows.append({
            "phase": name, "games": len(p.stats),
            "wall_clock_s": p.wall_clock_s,
            "mean_per_game_s": p.wall_clock_s / max(len(p.stats), 1),
        })
    pd.DataFrame(summary_rows).to_parquet(output_dir / "phase_summary.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark self-play / arena / Elo phases")
    parser.add_argument(
        "--config", type=str, default="run_configurations/profile_baseline.json",
        help="Path to a run config JSON. The benchmark uses num_eps, "
        "num_arena_matches, elo_games_per_gen, and the MCTS + net configs.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write the report + parquets. Defaults to "
        "<config.root>/<run_name>_benchmark/.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Parallelism level (placeholder for F1 — sequential for now, "
        "labelled in the report so before/after comparisons are clear).",
    )
    parser.add_argument(
        "--use-f2", action="store_true",
        help="Enable the F2 precomputed-move-list move generator (Blokus only). "
        "Overrides ``use_optimised_movegen`` in the config JSON. Effective in "
        "both the main process and any spawned workers.",
    )
    parser.add_argument(
        "--mcts-batch-size", type=int, default=None,
        help="F3 batched-inference leaf batch size K. Overrides "
        "``mcts_config.mcts_batch_size``. K=1 is the pre-F3 behaviour; K>1 "
        "collects K virtual-loss-diversified leaves per single batched GPU "
        "call. Propagates to workers via the config.",
    )
    args = parser.parse_args()

    config = _force_detailed_profiling(load_args(args.config))
    if args.use_f2:
        from dataclasses import replace as dc_replace
        config = dc_replace(config, use_optimised_movegen=True)
    if args.mcts_batch_size is not None:
        from dataclasses import replace as dc_replace
        config = dc_replace(
            config,
            mcts_config=dc_replace(config.mcts_config, mcts_batch_size=args.mcts_batch_size),
        )
        print(f"[F3] MCTS batch size K={args.mcts_batch_size}", flush=True)
    output_dir = Path(args.output_dir) if args.output_dir else config.root_directory / "benchmarks" / f"{config.run_name}_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {output_dir}")

    if config.seed is not None:
        np.random.seed(config.seed)

    game, nnet = instantiate_game_and_network(config)
    nnet_opponent = nnet.__class__(game, config)  # second random-init for arena/Elo

    # Enable the optimised move generator in the main-process game if
    # requested. Workers handle this themselves via
    # ``core.parallel_self_play._maybe_enable_f2``.
    if getattr(config, "use_optimised_movegen", False) and (
        enable := getattr(game, "enable_optimised_movegen", None)
    ) is not None:
        print("[F2] Enabling optimised move generator on main game", flush=True)
        enable()

    overall_start = time.perf_counter()
    phases: dict[str, PhaseResult] = {}
    phases["Self-Play"] = _run_self_play_phase(config, game, nnet, args.num_workers)
    phases["Arena"] = _run_two_player_phase(
        phase_name="Arena", num_games=config.num_arena_matches, config=config,
        game=game, nnet_a=nnet, nnet_b=nnet_opponent, num_workers=args.num_workers,
    )
    phases["Elo"] = _run_two_player_phase(
        phase_name="Elo", num_games=config.elo_games_per_gen, config=config,
        game=game, nnet_a=nnet, nnet_b=nnet_opponent, num_workers=args.num_workers,
    )
    overall_wall = time.perf_counter() - overall_start

    _write_parquets(phases, output_dir)

    estimator_html = _build_estimator_table(
        phases, config,
        extrapolate_scales=[
            {"name": "Sanity (1×5×5×5)",
             "num_eps": 5, "num_arena": 5, "elo_games": 5, "num_gens": 1},
            {"name": "blokus_pc_second (5×80×50×20)",
             "num_eps": 80, "num_arena": 50, "elo_games": 20, "num_gens": 5},
            {"name": "Long (20×80×50×20)",
             "num_eps": 80, "num_arena": 50, "elo_games": 20, "num_gens": 20},
            {"name": "Serious (50×200×100×50)",
             "num_eps": 200, "num_arena": 100, "elo_games": 50, "num_gens": 50},
        ],
    )

    report_path = build_multi_phase_report(
        list(phases.values()),
        title="Phase benchmark",
        subtitle=(
            f"Game: {config.game} — sims: {config.mcts_config.num_mcts_sims} — "
            f"net: {config.net_config.num_filters}f×{config.net_config.num_residual_blocks}b — "
            f"workers: {args.num_workers} — total wall-clock: "
            f"{_fmt_duration(overall_wall)}"
        ),
        output_dir=output_dir,
        estimator_table_html=estimator_html,
    )

    print()
    print(f"Total wall-clock: {_fmt_duration(overall_wall)}")
    for name, p in phases.items():
        per = p.wall_clock_s / max(len(p.stats), 1)
        print(f"  {name}: {_fmt_duration(p.wall_clock_s)} "
              f"({len(p.stats)} games, {per:.2f}s/game)")
    print(f"Report → {report_path}")


if __name__ == "__main__":
    main()
