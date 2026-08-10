"""Mini Pentobi ladder for external keep-best selection (post-regression-recovery P4).

Wraps ``scripts/pentobi_benchmark.py`` to ladder one or more checkpoints at a
reduced level band (default L3–L6, 50 games/level, 400 sims — ~2–3 h/checkpoint
on the box), append every result to the run's ``MiniLadder/history.json``, and
evaluate the keep-best + drift circuit-breaker logic from
``alphablokus.evaluation.ladder_selection`` over the accumulated history.

This is the run's *selection* instrument: weight flow stays continuous
(``gate_mode: "always"``) and the run's product is whatever checkpoint this
ladder crowns — the candidate-vs-incumbent arena cannot rank near-equal nets in
a game where ~93–97% of games are decided by first-mover colour
(docs/research/regression-and-next-steps.md §1.2/§4).

On a tripped drift alarm (two consecutive evaluations ≥5 pp weighted below the
best so far) the script writes ``MiniLadder/DRIFT_ALARM`` and exits with code
**3**, so a box-side loop can stop a run / page Henry without parsing output.

Usage (box)::

    uv run python scripts/mini_ladder.py --config run_configurations/<run>.json \
        --nets accepted_5.pth.tar accepted_10.pth.tar
    # later checkpoints append to the same history:
    uv run python scripts/mini_ladder.py --config run_configurations/<run>.json \
        --nets accepted_15.pth.tar

Requires ``pentobi-gtp`` (see docs/plans/archive/pentobi-harness.md; already
built on the box).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from alphablokus.config import RunConfig, load_args
from alphablokus.evaluation.ladder_selection import (
    DEFAULT_CONSECUTIVE_DROPS,
    DEFAULT_DROP,
    LadderPoint,
    checkpoint_generation,
    detect_drift,
    select_best,
)
from alphablokus.games.blokusduo.pentobi.gtp import find_pentobi_gtp
from alphablokus.reporting.pentobi_ladder import parse_levels, write_ladder_result

try:  # `uv run python scripts/mini_ladder.py` puts scripts/ on sys.path[0]
    import pentobi_benchmark
except ImportError:  # `uv run python -m scripts.mini_ladder` from the repo root
    from scripts import pentobi_benchmark  # type: ignore[no-redef]

if TYPE_CHECKING:
    from pathlib import Path

DRIFT_ALARM_EXIT_CODE = 3
DEFAULT_LEVELS = "3-6"
DEFAULT_GAMES_PER_LEVEL = 50

# The mini ladder is *only* the longitudinal instrument: it writes into the run's
# ``PentobiLadder/`` directory and its history drives keep-best and the drift
# circuit-breaker. So Pentobi's book stays off, unconditionally and with no flag to
# turn it on — every ladder number the project has ever quoted is book-free, and a
# book-on result on that series would be a different scale wearing the same name.
# ``PentobiPlayer`` requires the choice to be stated explicitly, which is what caught
# this call site: it passed no book state at all and raised TypeError.
LONGITUDINAL_NOBOOK = True


def _history_path(config: RunConfig) -> Path:
    return config.run_directory / "MiniLadder" / "history.json"


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    points: list[dict[str, Any]] = payload["points"]
    return points


def _save_history(path: Path, points: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"points": points}, indent=2), encoding="utf-8")


def _points_from_history(rows: list[dict[str, Any]]) -> list[LadderPoint]:
    return [
        LadderPoint(
            label=row["label"],
            weighted_score=float(row["weighted_score"]),
            generation=row.get("generation"),
            pentobi_level=row.get("pentobi_level"),
            score=row.get("score"),
        )
        for row in rows
    ]


def _run_one_net(config_path: str, config: RunConfig, net: str, args: argparse.Namespace) -> tuple[LadderPoint, float]:
    """Ladder one checkpoint via pentobi_benchmark's parallel sweep.

    Returns:
        ``(point, duration_s)`` — the ladder point and its wall-clock cost.
    """
    levels = parse_levels(args.levels)
    # This script only ever writes the longitudinal series, so its settings have to be
    # the ladder's: the same check the full benchmark applies to --condition ladder.
    conflict = pentobi_benchmark.condition_conflicts(
        pentobi_benchmark.CONDITION_LADDER,
        nobook=LONGITUDINAL_NOBOOK,
        sims=args.sims,
    )
    if conflict is not None:
        raise SystemExit(conflict)
    workers = args.workers if args.workers is not None else max(config.num_parallel_workers, 1)
    # Time the ladder: it is the plan's backbone measurement and its cost was
    # previously recorded nowhere, making everything downstream unschedulable.
    ladder_start = time.perf_counter()
    per_level = pentobi_benchmark.benchmark_levels_parallel(
        config_path=config_path,
        net_ckpt=net,
        levels=levels,
        games=args.games,
        workers=max(workers, 1),
        sims=args.sims,
        batch=args.batch,
        opening_temp=args.opening_temp,
        opening_moves=args.opening_moves,
        seed=args.seed,
        cpu_net=args.cpu_net,
        mps=args.mps,
        nobook=LONGITUDINAL_NOBOOK,
    )
    duration_s = time.perf_counter() - ladder_start
    metrics = pentobi_benchmark.compute_headline_metrics(per_level)
    print(
        f"[mini-ladder] {net}: ladder took {duration_s:.1f}s "
        f"({len(levels)} levels x {args.games} games at {args.sims} sims, {max(workers, 1)} workers)",
        flush=True,
    )
    # Same JSON the full benchmark writes, so the training report's Pentobi
    # Ladder section picks mini-ladder results up too — including the comparison
    # context, without which a payload cannot say what it faced (F4).
    write_ladder_result(
        config.pentobi_ladder_directory,
        net=net,
        sims=args.sims,
        games_per_level=args.games,
        per_level=per_level,
        metrics=metrics,
        duration_s=duration_s,
        condition=pentobi_benchmark.CONDITION_LADDER,
        context=pentobi_benchmark.build_context(
            argparse.Namespace(**{**vars(args), "nobook": LONGITUDINAL_NOBOOK}),
            config,
            workers=max(workers, 1),
        ),
    )
    point = LadderPoint(
        label=net,
        weighted_score=float(metrics["weighted_score"]),
        generation=checkpoint_generation(net),
        pentobi_level=int(metrics["pentobi_level"]),
        score=float(metrics["score"]),
    )
    return point, duration_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini Pentobi ladder → keep-best + drift circuit-breaker")
    parser.add_argument("--config", required=True, help="Run config JSON (net arch + game + net directory)")
    parser.add_argument(
        "--nets",
        nargs="+",
        required=True,
        help="Checkpoint filenames in the run's net_directory (e.g. accepted_5.pth.tar); absolute paths work too",
    )
    parser.add_argument("--levels", default=DEFAULT_LEVELS, help="Level band, e.g. '3-6' (default)")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_LEVEL, help="Games per level (default 50)")
    parser.add_argument("--sims", type=int, default=pentobi_benchmark.EVAL_SIMS_DEFAULT)
    parser.add_argument("--batch", type=int, default=16, help="MCTS leaf batch size K")
    parser.add_argument("--seed", type=int, default=1, help="Pentobi engine base seed")
    parser.add_argument("--opening-temp", type=float, default=1.0)
    parser.add_argument("--opening-moves", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None, help="Worker processes (default: config's)")
    parser.add_argument("--cpu-net", action="store_true", help="Run the net on CPU in workers")
    parser.add_argument("--mps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--drop", type=float, default=DEFAULT_DROP, help="Weighted-score drop that counts (default 0.05)"
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=DEFAULT_CONSECUTIVE_DROPS,
        help="Consecutive drops that trip the alarm (default 2)",
    )
    args = parser.parse_args()

    if find_pentobi_gtp() is None:
        raise SystemExit("pentobi-gtp not found — build it or set $PENTOBI_GTP_PATH (this runs on the box).")

    config = load_args(args.config)
    history_path = _history_path(config)
    history = _load_history(history_path)

    # Evaluate in generation order so "consecutive" in the drift detector means
    # consecutive checkpoints, whatever order they were passed in.
    nets = sorted(args.nets, key=lambda n: (checkpoint_generation(n) is None, checkpoint_generation(n) or 0))
    for net in nets:
        print(f"[mini-ladder] {net}: levels {args.levels}, {args.games} games/level, {args.sims} sims", flush=True)
        point, duration_s = _run_one_net(args.config, config, net, args)
        print(
            f"[mini-ladder] {net}: weighted {point.weighted_score:.3f} "
            f"(level {point.pentobi_level}, score {point.score:.3f})",
            flush=True,
        )
        row = asdict(point)
        row["sims"] = args.sims
        row["games_per_level"] = args.games
        row["levels"] = args.levels
        row["timestamp"] = datetime.now(UTC).isoformat()
        row["duration_s"] = round(duration_s, 2)
        history.append(row)
        _save_history(history_path, history)

    points = _points_from_history(history)
    best = select_best(points)
    print(f"[mini-ladder] keep-best: {best.label} (weighted {best.weighted_score:.3f}) over {len(points)} points")

    alarm = detect_drift(points, drop=args.drop, consecutive=args.consecutive)
    if alarm is None:
        print("[mini-ladder] drift circuit-breaker: OK")
        return

    flag = history_path.parent / "DRIFT_ALARM"
    flag.write_text(
        json.dumps(
            {
                "tripped_at": asdict(alarm.tripped_at),
                "best_before": asdict(alarm.best_before),
                "consecutive_drops": alarm.consecutive_drops,
                "drop": args.drop,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[mini-ladder] DRIFT ALARM: {alarm.consecutive_drops} consecutive evaluations ≥{args.drop:.0%} below "
        f"best {alarm.best_before.label} (weighted {alarm.best_before.weighted_score:.3f}) — "
        f"tripped at {alarm.tripped_at.label}. Stop the run and resume from the best checkpoint. → {flag}",
        flush=True,
    )
    sys.exit(DRIFT_ALARM_EXIT_CODE)


if __name__ == "__main__":
    main()
