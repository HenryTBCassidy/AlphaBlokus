"""Measure seconds-per-move for the net and for Pentobi, and find time parity.

Why this exists: the ladder fixes the net at ``--sims 400`` and Pentobi at
``--level N``. Both are *search-effort* settings, and the two efforts are
unrelated — so the ladder has never controlled for thinking time. Pentobi at
level 9 may search far longer per move than the net does, in which case the
headline 0.344 understates the net's strength at equal time. AlphaZero-vs-
Stockfish equalised *time per move* (hardware was disclosed, not matched), which
is the convention this script measures against.

Runs strictly single-threaded, one game at a time: per-move timings taken while
16 workers contend for one GPU measure contention, not thinking time.

Reports, per level: mean/median seconds per move for each side, and the
simulation count that would put the net at time parity (linear in sims, which
holds well for MCTS at fixed batch size — verify with ``--verify-sims``).

    uv run python scripts/measure_move_times.py \
        --config run_configurations/blokus_cloud_v3.json \
        --net accepted_40.pth.tar --levels 7,9 --games 2
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace
from pathlib import Path

from alphablokus.config import RunConfig, load_args
from alphablokus.evaluation.arena import Arena
from alphablokus.evaluation.players import NetworkPlayer
from alphablokus.games.blokusduo.pentobi.gtp import find_pentobi_gtp
from alphablokus.games.blokusduo.pentobi.player import PentobiPlayer
from alphablokus.registry import instantiate_game_and_network
from alphablokus.reporting.pentobi_ladder import parse_levels


class _Timed:
    """Wraps a player, recording wall-clock seconds for every move it is asked for.

    ``Arena`` invokes players as plain callables (``player(canonical_board)``),
    so the timing hook goes on ``__call__``. Everything else is delegated so the
    wrapper is transparent to ``Arena`` and to ``PentobiPlayer.close()``.
    """

    def __init__(self, inner: object, label: str) -> None:
        self._inner = inner
        self.label = label
        self.times: list[float] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        start = time.perf_counter()
        try:
            return self._inner(*args, **kwargs)  # type: ignore[operator]
        finally:
            self.times.append(time.perf_counter() - start)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def _summary(times: list[float]) -> dict[str, float]:
    if not times:
        return {"moves": 0, "mean": 0.0, "median": 0.0, "total": 0.0}
    return {
        "moves": len(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "total": sum(times),
    }


def _eval_mcts_config(base: object, sims: int, batch: int) -> object:
    """Pin the same search settings the ladder uses.

    ``sim_schedule`` and ``dirichlet_epsilon`` must be pinned, not inherited: the
    benchmark forces ``flat``/``0.0`` (see ``pentobi_benchmark._eval_mcts_config``),
    and a config carrying ``"branching"`` — several exist under
    ``run_configurations/archive/`` — would silently measure a different search
    than the one being calibrated for, making the parity budget wrong at source.
    """
    return replace(
        base,  # type: ignore[type-var]
        num_mcts_sims=sims,
        mcts_batch_size=batch,
        dirichlet_epsilon=0.0,
        sim_schedule="flat",
    )


def measure_level(
    game: object,
    nnet: object,
    config: RunConfig,
    level: int,
    games: int,
    sims: int,
    batch: int,
    *,
    nobook: bool,
    discard_warmup: bool = True,
) -> dict:
    """Play ``games`` net-vs-Pentobi games at ``level``, timing both sides' moves.

    ``discard_warmup`` drops the first game's timings. The net's first moves absorb
    lazy CUDA-context creation and any ``torch.compile`` tracing, which inflated the
    original L7 measurement (mean 2.12 s/move against 1.33 s at the harder L9) and
    would propagate straight into a wrong parity budget.
    """
    net_player = _Timed(
        NetworkPlayer(game, nnet, _eval_mcts_config(config.mcts_config, sims, batch), temp=0.0),
        "net",
    )
    pentobi = PentobiPlayer(game, level, seed=1, nobook=nobook)
    timed_pentobi = _Timed(pentobi, "pentobi")
    try:
        Arena(net_player, timed_pentobi, game).play_games(games, record=False)
    finally:
        pentobi.close()
    net_times, pentobi_times = net_player.times, timed_pentobi.times
    if discard_warmup and games > 1 and net_times and pentobi_times:
        # Drop roughly one game's worth of moves from the front of each series.
        net_cut, pentobi_cut = len(net_times) // games, len(pentobi_times) // games
        net_times, pentobi_times = net_times[net_cut:], pentobi_times[pentobi_cut:]
    return {
        "level": level,
        "sims": sims,
        "nobook": nobook,
        "warmup_discarded": discard_warmup and games > 1,
        "net": _summary(net_times),
        "pentobi": _summary(pentobi_times),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure per-move thinking time for the net and Pentobi")
    ap.add_argument("--config", required=True)
    ap.add_argument("--net", default=None)
    ap.add_argument("--levels", default="9", help="e.g. '7,9' or '1-9'")
    ap.add_argument("--games", type=int, default=2, help="Games per level (keep small; this is serial)")
    ap.add_argument("--sims", type=int, default=400, help="Net simulations to measure at")
    ap.add_argument("--batch", type=int, default=16, help="MCTS leaf batch size (match the ladder's)")
    ap.add_argument(
        "--verify-sims",
        action="store_true",
        help="After computing the parity sim count, re-measure at it to check the linear assumption",
    )
    book = ap.add_mutually_exclusive_group(required=True)
    book.add_argument("--nobook", dest="nobook", action="store_true", help="Pentobi without its opening book")
    book.add_argument(
        "--book",
        dest="nobook",
        action="store_false",
        help="Pentobi as shipped. Book moves return in ~0.5s against ~26s for a searched move, so "
        "the mean is bimodal — calibrate parity on the median, or on book-free moves.",
    )
    args = ap.parse_args()

    if find_pentobi_gtp() is None:
        raise SystemExit("pentobi-gtp not found — build it or set $PENTOBI_GTP_PATH.")

    config: RunConfig = load_args(args.config)
    import torch

    if config.net_config.cuda and not torch.cuda.is_available():
        config = replace(config, net_config=replace(config.net_config, cuda=False))
        print("[timing] CUDA unavailable — net on CPU, timings are NOT box-representative.", flush=True)

    game, nnet = instantiate_game_and_network(config)
    if args.net:
        nnet.load_checkpoint(filename=args.net)
        print(f"[timing] loaded {args.net}", flush=True)

    print(f"[timing] serial, batch={args.batch}, {args.games} games/level — no worker contention", flush=True)
    rows = []
    for level in parse_levels(args.levels):
        r = measure_level(game, nnet, config, level, args.games, args.sims, args.batch, nobook=args.nobook)
        n, p = r["net"], r["pentobi"]
        ratio = p["mean"] / n["mean"] if n["mean"] else float("nan")
        parity_sims = int(round(args.sims * ratio))
        r["ratio"] = ratio
        r["parity_sims"] = parity_sims
        print(
            f"  L{level}: net {n['mean']:.2f}s/move (median {n['median']:.2f}, {n['moves']} moves) | "
            f"pentobi {p['mean']:.2f}s/move (median {p['median']:.2f}, {p['moves']} moves) | "
            f"pentobi/net = {ratio:.2f}x -> time parity at ~{parity_sims} sims",
            flush=True,
        )
        if args.verify_sims and parity_sims > args.sims:
            v = measure_level(game, nnet, config, level, args.games, parity_sims, args.batch, nobook=args.nobook)
            vn, vp = v["net"], v["pentobi"]
            print(
                f"    verify @ {parity_sims} sims: net {vn['mean']:.2f}s/move vs pentobi "
                f"{vp['mean']:.2f}s/move (ratio {vp['mean'] / vn['mean']:.2f}x; 1.0 = parity)",
                flush=True,
            )
            r["verify"] = v
        rows.append(r)

    out = Path("temp/benchmarks/move_times.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(rows, indent=2))
    print(f"[timing] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
