"""Benchmark a net against Pentobi and emit a stats + replays report (H5).

Plays a chosen net (MCTS) against ``pentobi-gtp`` at one level or a full 1–9 sweep,
half the games as each colour, and writes a self-contained HTML report:
- **stats** — per-level win/loss/draw, win rate + 95% Wilson CI, and the headline
  metrics from docs/05-EVALUATION.md (Pentobi Level / Score / Weighted Score);
- **replays** — a sample of games per level rendered with the *same* board renderer as
  the training arena replays (``reporting.display_blokusduo.build_game_replay_html``).

Runs net-vs-Pentobi through the existing :class:`alphablokus.evaluation.arena.Arena` via the H4
``PentobiPlayer``, so the game loop / record capture is all reused.

Usage::

    uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --level 5 --games 100
    uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --sweep --games 100
    uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --levels 1-3 --games 40
    uv run python -m scripts.pentobi_benchmark --config <run.json> --level 1 --games 4   # no --net = fresh net
    uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --sweep --workers 4

``--config`` supplies the net architecture + game + checkpoint directory; ``--net`` is the
checkpoint filename within that run's ``net_directory`` (omit to benchmark a fresh net).

**Parallelism (``--workers``).** The benchmark is *not* inference-bound — the GPU sits
near-idle while Pentobi's CPU search (which grows sharply with level) and the per-move
GTP round-trip dominate. ``--workers N`` splits the requested games across ``N`` worker
processes, each with its own net + its own ``pentobi-gtp`` engine, and aggregates the
results — a near-linear speedup up to the VRAM/core ceiling. ``--workers 1`` (the default
when ``num_parallel_workers`` is unset in the config) reproduces the serial path
bit-for-bit. Workers are spawned with the ``spawn`` start method (forking a CUDA/JAX
process deadlocks), so each is a fresh interpreter that builds its own game/net/engine
from the config path — nothing GPU-touching crosses the process boundary. On the 8 GB
3060 Ti expect ~4 GPU workers before CUDA OOM (each needs its own ~0.6–1.5 GB context);
lower ``--workers`` or pass ``--cpu-net`` to run the net on CPU and scale past the VRAM cap.

Ladder tracking (cloud-scale C11): every benchmark also drops a JSON summary into the
run's ``PentobiLadder/`` directory, which the training report renders as a "Pentobi
Ladder" section (regenerate with ``alphablokus --config <cfg> --report-only``). To ladder
several saved checkpoints, loop the script over them::

    for net in accepted_10.pth.tar accepted_20.pth.tar best.pth.tar; do
        uv run python -m scripts.pentobi_benchmark --config <run.json> --net $net --levels 1-3 --games 40
    done
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from alphablokus.config import RunConfig, load_args
from alphablokus.evaluation.arena import Arena
from alphablokus.evaluation.players import NetworkPlayer
from alphablokus.games.blokusduo.pentobi.gtp import find_pentobi_gtp
from alphablokus.games.blokusduo.pentobi.player import PentobiPlayer
from alphablokus.registry import instantiate_game, instantiate_game_and_network
from alphablokus.reporting.pentobi_ladder import parse_levels, write_ladder_result

if TYPE_CHECKING:
    from alphablokus.evaluation.arena import GameRecord

EVAL_SIMS_DEFAULT = 400
REPLAYS_PER_LEVEL = 4  # games embedded per level in the report (keeps it readable)
DEFAULT_WORKERS_WHEN_PARALLEL = 4  # VRAM-safe default on the 8 GB 3060 Ti; lower on CUDA OOM

# Type alias for a single worker's return: (net_wins, pentobi_wins, draws, records).
ChunkResult = tuple[int, int, int, "list[GameRecord]"]


def _eval_mcts_config(base, sims: int, batch: int = 1):
    """Evaluation search (IDEAS I2): strong + deterministic — flat sim schedule,
    no Dirichlet noise, temp=0 at the player.

    ``batch`` is the MCTS leaf batch size (K). K=1 is exact (no virtual-loss
    approximation); K>1 batches leaf evaluations, which is dramatically faster on
    a GPU/MPS backend (see the CPU-vs-MPS gap) at the cost of the same virtual-loss
    approximation the net trained under."""
    return replace(
        base,
        num_mcts_sims=sims,
        mcts_batch_size=batch,
        dirichlet_epsilon=0.0,
        sim_schedule="flat",
    )


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate (better than normal approx at the tails)."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _record_to_actions(game, record) -> list[dict]:
    """Convert a GameRecord into the action-dict list ``build_game_replay_html`` expects."""
    codec = game.action_codec
    actions: list[dict] = []
    for i, move in enumerate(record.moves):
        if codec.is_pass(move.action):
            actions.append({"turn": i + 1, "player": move.player, "pass": True})
            continue
        a = codec.decode(move.action)
        actions.append(
            {
                "turn": i + 1,
                "player": move.player,
                "pass": False,
                "piece_id": a.piece_id,
                "orientation": a.orientation.value,
                "x": a.x_coordinate,
                "y": a.y_coordinate,
            }
        )
    return actions


def benchmark_level(game, net_player, level: int, games: int, seed: int | None) -> dict:
    """Play ``games`` net-vs-Pentobi games at one level (half each colour) via the Arena."""
    pentobi = PentobiPlayer(game, level, seed=seed)
    try:
        # net is player1; play_games splits half/half by colour and swaps internally.
        net_wins, pentobi_wins, draws, records = Arena(
            net_player,
            pentobi,
            game,
        ).play_games(games, record=True)
    finally:
        pentobi.close()
    played = net_wins + pentobi_wins + draws
    win_rate = net_wins / played if played else 0.0
    return {
        "level": level,
        "games": played,
        "net_wins": net_wins,
        "pentobi_wins": pentobi_wins,
        "draws": draws,
        "win_rate": win_rate,
        "ci": _wilson_ci(net_wins, played),
        "records": records,
    }


# --------------------------------------------------------------------------- #
# Parallel benchmark: fan the per-level games out over a pool of worker
# processes, each with its own net + its own Pentobi engine, then aggregate.
# The pure helpers below (chunking, seed planning, aggregation) are unit-tested
# in isolation; the Pentobi-specific execution is covered by end-to-end runs.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ChunkTask:
    """One unit of parallel work: play ``n_games`` at ``level`` from ``seed_base``.

    ``collect_records`` is True only for the *first* chunk of each level — the
    report embeds just ``REPLAYS_PER_LEVEL`` games/level, so returning full
    records from every worker would waste memory and IPC.
    """

    level: int
    n_games: int
    seed_base: int
    collect_records: bool


def _even_chunks(games: int, workers: int) -> list[int]:
    """Split ``games`` into at most ``workers`` **even** chunks summing to ``games``.

    Each chunk must be even so :meth:`Arena.play_games` keeps its half-white /
    half-black colour swap per chunk (it rounds an odd count down internally).
    ``games`` is first rounded down to even to match the serial path's effective
    game count, then the even total is spread as evenly as possible; any remaining
    pairs are handed out one-per-worker (so chunks differ by at most 2).
    """
    total = games - (games % 2)  # play_games floors to even; mirror that here
    if total <= 0:
        return []
    pairs = total // 2
    workers = max(1, min(workers, pairs))
    base_pairs, extra = divmod(pairs, workers)
    chunks = [2 * (base_pairs + (1 if i < extra else 0)) for i in range(workers)]
    return [c for c in chunks if c > 0]


def _plan_tasks(levels: list[int], games: int, workers: int, seed: int) -> list[_ChunkTask]:
    """Build the full ``(level, chunk)`` task list with disjoint per-task seeds.

    Every task gets a globally-unique seed base ``seed + task_index * games`` so
    that — since each chunk plays at most ``games`` games and ``PentobiPlayer``
    reseeds ``seed_base + game_index`` per game — no two tasks ever replay the
    same Pentobi games.

    Records are collected only from the *leading* chunk(s) of each level, just
    until they cover ``REPLAYS_PER_LEVEL`` games (that is all the report embeds) —
    the rest skip record capture to save memory/IPC. At realistic sizes the first
    chunk alone suffices; the running total keeps small-chunk runs from under-filling.
    """
    tasks: list[_ChunkTask] = []
    task_index = 0
    for level in levels:
        records_collected = 0
        for n_games in _even_chunks(games, workers):
            collect = records_collected < REPLAYS_PER_LEVEL
            if collect:
                records_collected += n_games
            tasks.append(
                _ChunkTask(
                    level=level,
                    n_games=n_games,
                    seed_base=seed + task_index * games,
                    collect_records=collect,
                )
            )
            task_index += 1
    return tasks


def _aggregate_level(level: int, chunk_results: list[ChunkResult]) -> dict:
    """Sum a level's chunk results into the same dict shape as :func:`benchmark_level`."""
    net_wins = sum(r[0] for r in chunk_results)
    pentobi_wins = sum(r[1] for r in chunk_results)
    draws = sum(r[2] for r in chunk_results)
    records = [rec for r in chunk_results for rec in r[3]]
    played = net_wins + pentobi_wins + draws
    win_rate = net_wins / played if played else 0.0
    return {
        "level": level,
        "games": played,
        "net_wins": net_wins,
        "pentobi_wins": pentobi_wins,
        "draws": draws,
        "win_rate": win_rate,
        "ci": _wilson_ci(net_wins, played),
        "records": records,
    }


def _play_chunk(
    config_path: str,
    net_ckpt: str | None,
    level: int,
    n_games: int,
    seed_base: int,
    sims: int,
    batch: int,
    opening_temp: float,
    opening_moves: int,
    cpu_net: bool,
    mps: bool,
    collect_records: bool,
) -> ChunkResult:
    """Worker entry point: play ``n_games`` net-vs-Pentobi at one level, in its own process.

    Takes only plain picklable args — it rebuilds the game, net and engine itself
    so nothing GPU-touching (a live net / CUDA context) ever crosses the ``spawn``
    process boundary. This is why ``--workers`` uses ``spawn``, not ``fork``:
    forking a process that has imported Torch/JAX deadlocks.

    Device policy: run the net on CUDA when the config asks for it and it's
    available, unless ``cpu_net`` forces CPU (to scale past the VRAM cap). On the
    Mac, ``mps`` opts into Metal inference (ignored under ``cpu_net``).
    """
    import os

    if mps and not cpu_net:
        os.environ["ALPHABLOKUS_MPS"] = "1"  # opt into MPS in the wrapper (eval-only)

    import torch

    torch.set_num_threads(1)  # N workers each grabbing all cores oversubscribes the CPU

    config = load_args(config_path)
    want_cuda = config.net_config.cuda and not cpu_net and torch.cuda.is_available()
    config = replace(config, net_config=replace(config.net_config, cuda=want_cuda))

    game, nnet = instantiate_game_and_network(config)
    if net_ckpt:
        nnet.load_checkpoint(filename=net_ckpt)

    net_player = NetworkPlayer(
        game,
        nnet,
        _eval_mcts_config(config.mcts_config, sims, batch),
        temp=0.0,
        opening_temp=opening_temp,
        opening_moves=opening_moves,
    )
    pentobi = PentobiPlayer(game, level, seed=seed_base)
    try:
        net_wins, pentobi_wins, draws, records = Arena(net_player, pentobi, game).play_games(
            n_games,
            record=collect_records,
        )
    finally:
        pentobi.close()
    return net_wins, pentobi_wins, draws, records


def benchmark_levels_parallel(
    config_path: str,
    net_ckpt: str | None,
    levels: list[int],
    games: int,
    workers: int,
    sims: int,
    batch: int,
    opening_temp: float,
    opening_moves: int,
    seed: int,
    cpu_net: bool,
    mps: bool,
) -> list[dict]:
    """Run the whole level sweep across a ``spawn`` pool and aggregate per level.

    One pool serves *all* levels at once: fast low-level chunks free their worker
    to pick up slow high-level (8/9) chunks, which utilises the pool far better
    than draining level-by-level. Results are grouped back by level and returned
    in ``levels`` order, matching the serial path's per-level dict shape.
    """
    tasks = _plan_tasks(levels, games, workers, seed)
    ctx = mp.get_context("spawn")  # never fork: workers init Torch/CUDA (and JAX is imported elsewhere)
    results_by_level: dict[int, list[ChunkResult]] = {level: [] for level in levels}

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                _play_chunk,
                config_path,
                net_ckpt,
                task.level,
                task.n_games,
                task.seed_base,
                sims,
                batch,
                opening_temp,
                opening_moves,
                cpu_net,
                mps,
                task.collect_records,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            net_wins, pentobi_wins, draws, records = future.result()
            results_by_level[task.level].append((net_wins, pentobi_wins, draws, records))
            print(
                f"  level {task.level} chunk ({task.n_games} games): net {net_wins}-{pentobi_wins}-{draws}",
                flush=True,
            )

    return [_aggregate_level(level, results_by_level[level]) for level in levels]


def compute_headline_metrics(per_level: list[dict]) -> dict:
    """Pentobi Level / Score / Weighted Score per docs/05-EVALUATION.md §2."""
    beaten = [r["level"] for r in per_level if r["win_rate"] > 0.5]
    total_games = sum(r["games"] for r in per_level)
    total_wins = sum(r["net_wins"] for r in per_level)
    weighted_num = sum(r["level"] * r["net_wins"] for r in per_level)
    weighted_den = sum(r["level"] * r["games"] for r in per_level)
    return {
        "pentobi_level": max(beaten) if beaten else 0,
        "score": total_wins / total_games if total_games else 0.0,
        "weighted_score": weighted_num / weighted_den if weighted_den else 0.0,
    }


def build_report(game, per_level: list[dict], metrics: dict, header: dict, out_path: Path) -> None:
    from alphablokus.reporting.display_blokusduo import BOARD_CSS, build_game_replay_html

    rows = "".join(
        f"<tr><td>{r['level']}</td><td>{r['games']}</td>"
        f"<td>{r['net_wins']}</td><td>{r['pentobi_wins']}</td><td>{r['draws']}</td>"
        f"<td>{r['win_rate']:.0%}</td>"
        f"<td>[{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]</td></tr>"
        for r in per_level
    )
    replay_sections = []
    for r in per_level:
        games_html = "".join(
            build_game_replay_html(game, _record_to_actions(game, rec), gid)
            for gid, rec in enumerate(r["records"][:REPLAYS_PER_LEVEL])
        )
        replay_sections.append(
            f"<h3>Level {r['level']} replays "
            f"(first {min(REPLAYS_PER_LEVEL, len(r['records']))} of {len(r['records'])})</h3>"
            f"{games_html}",
        )

    style = (
        "body{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.5;color:#1a1a1a}h1{border-bottom:2px solid #333}"
        "table{border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #ccc;padding:5px 10px;"
        "text-align:center}th{background:#f6f6f6}.kpi{font-size:1.4em;font-weight:600}"
        ".meta{color:#555;font-size:.9em}" + BOARD_CSS
    )
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>Pentobi Benchmark</title>
<style>{style}</style></head><body>
<h1>Pentobi Benchmark</h1>
<p class=meta>net: {header["net"]} &middot; config: {header["config"]} &middot;
eval sims: {header["sims"]} &middot; {header["games"]} games/level &middot; {header["timestamp"]}</p>
<p class=kpi>Pentobi Level: {metrics["pentobi_level"]} &nbsp;|&nbsp;
Score: {metrics["score"]:.3f} &nbsp;|&nbsp; Weighted: {metrics["weighted_score"]:.3f}</p>
<p class=meta>Pentobi Level = highest level the net beats at &gt;50% win rate.</p>
<h2>Results by level</h2>
<table><tr><th>level</th><th>games</th><th>net W</th><th>Pentobi W</th><th>draws</th>
<th>win rate</th><th>95% CI</th></tr>{rows}</table>
<h2>Game replays</h2>
{"".join(replay_sections)}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark a net against Pentobi → HTML report")
    ap.add_argument("--config", required=True, help="Run config JSON (net arch + game + net dir)")
    ap.add_argument("--net", default=None, help="Checkpoint filename in the run's net_directory (omit = fresh net)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--level", type=int, help="Single Pentobi level (1-9)")
    group.add_argument("--sweep", action="store_true", help="Sweep all levels 1-9")
    group.add_argument("--levels", type=str, help="Level subset, e.g. '1-5' or '1,3,9'")
    ap.add_argument("--games", type=int, default=20, help="Games per level (split half/half by colour)")
    ap.add_argument("--sims", type=int, default=EVAL_SIMS_DEFAULT, help="Eval MCTS simulations")
    ap.add_argument("--seed", type=int, default=1, help="Pentobi engine base seed (per-game reseed)")
    ap.add_argument(
        "--opening-temp",
        type=float,
        default=1.0,
        help="Temperature for the net's opening plies (diversifies games; 0 = deterministic)",
    )
    ap.add_argument(
        "--opening-moves",
        type=int,
        default=4,
        help="Number of the net's opening plies sampled at --opening-temp before temp=0",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=16,
        help="MCTS leaf batch size K (1 = exact; >1 batches leaf evals, far faster on GPU/MPS)",
    )
    ap.add_argument("--out", default=None, help="Report path (default temp/benchmarks/pentobi_<net>.html)")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes for parallel play (default: config.num_parallel_workers if >1 else 4). "
        "1 = serial path, bit-for-bit as before. Lower if you see CUDA OOM.",
    )
    ap.add_argument(
        "--cpu-net",
        dest="cpu_net",
        action="store_true",
        help="Force the net onto CPU in workers (scales past the VRAM cap, slower per move)",
    )
    ap.add_argument(
        "--mps",
        dest="mps",
        action="store_true",
        default=True,
        help="Use Apple MPS (Metal) for inference when available (default on)",
    )
    ap.add_argument("--no-mps", dest="mps", action="store_false", help="Force CPU instead of MPS")
    args = ap.parse_args()

    if find_pentobi_gtp() is None:
        raise SystemExit(
            "pentobi-gtp not found — build it (docs/plans/archive/pentobi-harness.md H2) or set $PENTOBI_GTP_PATH.",
        )

    config: RunConfig = load_args(args.config)

    if args.sweep:
        levels = list(range(1, 10))
    elif args.levels:
        levels = parse_levels(args.levels)
    else:
        levels = [args.level if args.level else 1]

    # Resolve worker count: explicit --workers wins; otherwise use the config's
    # ``num_parallel_workers`` when it opts into parallelism, else the VRAM-safe
    # default. Fewer than 2 games can't be split (play_games halves per chunk),
    # so force the serial path there.
    if args.workers is not None:
        workers = max(1, args.workers)
    elif config.num_parallel_workers > 1:
        workers = config.num_parallel_workers
    else:
        workers = DEFAULT_WORKERS_WHEN_PARALLEL
    if args.games < 2:
        workers = 1

    if workers == 1:
        # --- Serial path: build the net once in-process (bit-for-bit as before). ---
        if args.mps:
            import os

            os.environ["ALPHABLOKUS_MPS"] = "1"  # opt into MPS in the wrapper (eval-only)

        import torch

        if config.net_config.cuda and not torch.cuda.is_available():
            config = replace(config, net_config=replace(config.net_config, cuda=False))
            print("[benchmark] CUDA unavailable — using MPS/CPU for the net.", flush=True)

        game, nnet = instantiate_game_and_network(config)
        if args.net:
            nnet.load_checkpoint(filename=args.net)
            print(f"[benchmark] loaded net checkpoint: {args.net}", flush=True)
        else:
            print("[benchmark] no --net given: benchmarking a fresh random-init net.", flush=True)

        net_player = NetworkPlayer(
            game,
            nnet,
            _eval_mcts_config(config.mcts_config, args.sims, args.batch),
            temp=0.0,
            opening_temp=args.opening_temp,
            opening_moves=args.opening_moves,
        )
        per_level = []
        for level in levels:
            print(f"[benchmark] level {level}: {args.games} games...", flush=True)
            r = benchmark_level(game, net_player, level, args.games, args.seed)
            print(
                f"  net {r['net_wins']}-{r['pentobi_wins']}-{r['draws']} "
                f"(win rate {r['win_rate']:.0%}, 95% CI [{r['ci'][0]:.0%}, {r['ci'][1]:.0%}])",
                flush=True,
            )
            per_level.append(r)
    else:
        # --- Parallel path: each worker builds its own net + Pentobi engine (spawn). ---
        # The parent stays GPU-clean; it only needs a game object for the report's
        # replay rendering (records decode identically on any same-type game).
        game = instantiate_game(config)
        if args.net:
            print(f"[benchmark] loading net checkpoint per worker: {args.net}", flush=True)
        else:
            print("[benchmark] no --net given: benchmarking a fresh random-init net.", flush=True)
        print(
            f"[benchmark] {workers} workers over levels {levels} "
            f"({args.games} games/level{', CPU net' if args.cpu_net else ''})...",
            flush=True,
        )
        per_level = benchmark_levels_parallel(
            config_path=args.config,
            net_ckpt=args.net,
            levels=levels,
            games=args.games,
            workers=workers,
            sims=args.sims,
            batch=args.batch,
            opening_temp=args.opening_temp,
            opening_moves=args.opening_moves,
            seed=args.seed,
            cpu_net=args.cpu_net,
            mps=args.mps,
        )
        for r in per_level:
            print(
                f"[benchmark] level {r['level']}: net {r['net_wins']}-{r['pentobi_wins']}-{r['draws']} "
                f"(win rate {r['win_rate']:.0%}, 95% CI [{r['ci'][0]:.0%}, {r['ci'][1]:.0%}])",
                flush=True,
            )

    metrics = compute_headline_metrics(per_level)
    print(
        f"[benchmark] Pentobi Level={metrics['pentobi_level']} "
        f"Score={metrics['score']:.3f} Weighted={metrics['weighted_score']:.3f}",
        flush=True,
    )

    # Persist the ladder summary where the training report picks it up.
    ladder_path = write_ladder_result(
        config.pentobi_ladder_directory,
        net=args.net or "freshnet",
        sims=args.sims,
        games_per_level=args.games,
        per_level=per_level,
        metrics=metrics,
    )
    print(f"[benchmark] ladder JSON → {ladder_path} (rendered by --report-only)", flush=True)

    out = Path(args.out or f"temp/benchmarks/pentobi_{args.net or 'freshnet'}.html")
    build_report(
        game,
        per_level,
        metrics,
        {
            "net": args.net or "fresh random-init",
            "config": args.config,
            "sims": args.sims,
            "games": args.games,
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        },
        out,
    )
    print(f"[benchmark] report → {out}", flush=True)


if __name__ == "__main__":
    main()
