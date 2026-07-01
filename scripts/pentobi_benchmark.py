"""Benchmark a net against Pentobi and emit a stats + replays report (H5).

Plays a chosen net (MCTS) against ``pentobi-gtp`` at one level or a full 1–9 sweep,
half the games as each colour, and writes a self-contained HTML report:
- **stats** — per-level win/loss/draw, win rate + 95% Wilson CI, and the headline
  metrics from docs/05-EVALUATION.md (Pentobi Level / Score / Weighted Score);
- **replays** — a sample of games per level rendered with the *same* board renderer as
  the training arena replays (``reporting.display_blokusduo.build_game_replay_html``).

Runs net-vs-Pentobi through the existing :class:`core.arena.Arena` via the H4
``PentobiPlayer``, so the game loop / record capture is all reused.

Usage::

    uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --level 5 --games 100
    uv run python -m scripts.pentobi_benchmark --config <run.json> --net best.pth.tar --sweep --games 100
    uv run python -m scripts.pentobi_benchmark --config <run.json> --level 1 --games 4   # no --net = fresh net

``--config`` supplies the net architecture + game + checkpoint directory; ``--net`` is the
checkpoint filename within that run's ``net_directory`` (omit to benchmark a fresh net).
"""
from __future__ import annotations

import argparse
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from core.arena import Arena
from core.config import RunConfig, load_args
from core.game_factory import instantiate_game_and_network
from core.players import NetworkPlayer
from games.blokusduo.pentobi_gtp import find_pentobi_gtp
from games.blokusduo.pentobi_player import PentobiPlayer

EVAL_SIMS_DEFAULT = 400
REPLAYS_PER_LEVEL = 4  # games embedded per level in the report (keeps it readable)


def _eval_mcts_config(base, sims: int, batch: int = 1):
    """Evaluation search (IDEAS I2): strong + deterministic — flat sim schedule,
    no Dirichlet noise, temp=0 at the player.

    ``batch`` is the MCTS leaf batch size (K). K=1 is exact (no virtual-loss
    approximation); K>1 batches leaf evaluations, which is dramatically faster on
    a GPU/MPS backend (see the CPU-vs-MPS gap) at the cost of the same virtual-loss
    approximation the net trained under."""
    return replace(
        base, num_mcts_sims=sims, mcts_batch_size=batch,
        dirichlet_epsilon=0.0, sim_schedule="flat",
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
        actions.append({
            "turn": i + 1, "player": move.player, "pass": False,
            "piece_id": a.piece_id, "orientation": a.orientation.value,
            "x": a.x_coordinate, "y": a.y_coordinate,
        })
    return actions


def benchmark_level(game, net_player, level: int, games: int, seed: int | None) -> dict:
    """Play ``games`` net-vs-Pentobi games at one level (half each colour) via the Arena."""
    pentobi = PentobiPlayer(game, level, seed=seed)
    try:
        # net is player1; play_games splits half/half by colour and swaps internally.
        net_wins, pentobi_wins, draws, records = Arena(
            net_player, pentobi, game,
        ).play_games(games, record=True)
    finally:
        pentobi.close()
    played = net_wins + pentobi_wins + draws
    win_rate = net_wins / played if played else 0.0
    return {
        "level": level, "games": played, "net_wins": net_wins,
        "pentobi_wins": pentobi_wins, "draws": draws,
        "win_rate": win_rate, "ci": _wilson_ci(net_wins, played),
        "records": records,
    }


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
    from reporting.display_blokusduo import BOARD_CSS, build_game_replay_html

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
<p class=meta>net: {header['net']} &middot; config: {header['config']} &middot;
eval sims: {header['sims']} &middot; {header['games']} games/level &middot; {header['timestamp']}</p>
<p class=kpi>Pentobi Level: {metrics['pentobi_level']} &nbsp;|&nbsp;
Score: {metrics['score']:.3f} &nbsp;|&nbsp; Weighted: {metrics['weighted_score']:.3f}</p>
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
    ap.add_argument("--games", type=int, default=20, help="Games per level (split half/half by colour)")
    ap.add_argument("--sims", type=int, default=EVAL_SIMS_DEFAULT, help="Eval MCTS simulations")
    ap.add_argument("--seed", type=int, default=1, help="Pentobi engine base seed (per-game reseed)")
    ap.add_argument("--opening-temp", type=float, default=1.0,
                    help="Temperature for the net's opening plies (diversifies games; 0 = deterministic)")
    ap.add_argument("--opening-moves", type=int, default=4,
                    help="Number of the net's opening plies sampled at --opening-temp before temp=0")
    ap.add_argument("--batch", type=int, default=16,
                    help="MCTS leaf batch size K (1 = exact; >1 batches leaf evals, far faster on GPU/MPS)")
    ap.add_argument("--out", default=None, help="Report path (default temp/benchmarks/pentobi_<net>.html)")
    ap.add_argument("--mps", dest="mps", action="store_true", default=True,
                    help="Use Apple MPS (Metal) for inference when available (default on)")
    ap.add_argument("--no-mps", dest="mps", action="store_false", help="Force CPU instead of MPS")
    args = ap.parse_args()

    if find_pentobi_gtp() is None:
        raise SystemExit(
            "pentobi-gtp not found — build it (docs/plans/pentobi-harness.md H2) "
            "or set $PENTOBI_GTP_PATH.",
        )

    if args.mps:
        import os
        os.environ["ALPHABLOKUS_MPS"] = "1"  # opt into MPS in the wrapper (eval-only)

    config: RunConfig = load_args(args.config)
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
        game, nnet, _eval_mcts_config(config.mcts_config, args.sims, args.batch), temp=0.0,
        opening_temp=args.opening_temp, opening_moves=args.opening_moves,
    )
    levels = list(range(1, 10)) if args.sweep else [args.level if args.level else 1]

    per_level = []
    for level in levels:
        print(f"[benchmark] level {level}: {args.games} games...", flush=True)
        r = benchmark_level(game, net_player, level, args.games, args.seed)
        print(f"  net {r['net_wins']}-{r['pentobi_wins']}-{r['draws']} "
              f"(win rate {r['win_rate']:.0%}, 95% CI [{r['ci'][0]:.0%}, {r['ci'][1]:.0%}])", flush=True)
        per_level.append(r)

    metrics = compute_headline_metrics(per_level)
    print(f"[benchmark] Pentobi Level={metrics['pentobi_level']} "
          f"Score={metrics['score']:.3f} Weighted={metrics['weighted_score']:.3f}", flush=True)

    out = Path(args.out or f"temp/benchmarks/pentobi_{args.net or 'freshnet'}.html")
    build_report(game, per_level, metrics, {
        "net": args.net or "fresh random-init",
        "config": args.config,
        "sims": args.sims, "games": args.games,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }, out)
    print(f"[benchmark] report → {out}", flush=True)


if __name__ == "__main__":
    main()
