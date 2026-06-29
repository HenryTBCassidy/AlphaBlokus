"""Standardised self-play benchmark — one command, one self-contained HTML report.

Runs a fixed suite and emits a single portable HTML (charts inline as base64) that
shows **where time and memory go**. Data only — no recommendations; interpretation
happens elsewhere.

Suite:
  - Single-worker self-play timing: N games, avg s/game (+ spread), examples/game.
  - Time breakdown: cProfile of one game -> category pie + top-function bar.
  - Worker scaling: games/s across {1, 2, 4, 8, 16} workers.
  - Memory: in-episode tree peak, and a training-step RSS-vs-buffer-size ramp.

Usage::

    uv run python -m scripts.benchmark                       # default config
    uv run python -m scripts.benchmark --config <cfg.json>
    uv run python -m scripts.benchmark --out temp/benchmarks/my_report.html

Reuses the live code paths (``play_self_play_episode``,
``run_self_play_episodes_parallel``, ``nnet.train``) so the numbers reflect production.
"""
from __future__ import annotations

import argparse
import base64
import cProfile
import io
import pstats
import time
import tracemalloc
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.config import RunConfig, load_args
from core.game_factory import instantiate_game_and_network
from core.mcts import MCTS
from core.parallel_self_play import run_self_play_episodes_parallel
from core.self_play import play_self_play_episode

# -- Suite parameters (edit here, not via ad-hoc args) -----------------------------
TIMING_GAMES = 5
WORKER_COUNTS = (1, 2, 4, 8, 16)
SWEEP_GAMES_PER_WORKER = 4          # sweep num_eps = this * N (>= one full batch each)
TRAIN_RAMP_EXAMPLES = (50_000, 150_000, 250_000, 350_000)
DEFAULT_CONFIG = "run_configurations/bench_workers_gpu8.json"

# cProfile category buckets, matched by (filename substring, name predicate).
_MOVEGEN_NAMES = (
    "placement", "cells_valid", "generate_valid", "valid_placement", "corner",
    "adj_status", "valid_move", "forbidden", "fill_legal", "has_valid", "game_ended",
)


def _categorise(filename: str, func: str) -> str:
    f, name = filename.lower(), func.lower()
    if "movegen_runtime" in f:
        return "move-gen"
    if "game.py" in f and any(k in name for k in _MOVEGEN_NAMES):
        return "move-gen"
    if "mcts.py" in f and ("select_action" in name or "<genexpr>" in name):
        return "UCB select"
    if "base_wrapper" in f or "conv" in name or name.startswith("predict") or "torch" in f:
        return "inference"
    if "mcts.py" in f and any(k in name for k in ("backprop", "expand", "descend", "simulate")):
        return "tree / backprop"
    if "board.py" in f or "get_next_state" in name:
        return "board / encode"
    return "other"


# -- Measurements ------------------------------------------------------------------

def measure_timing(game, nnet, config: RunConfig, n_games: int) -> dict:
    """Single-process self-play timing over ``n_games`` games."""
    per_game, examples, tree = [], [], []
    for g in range(n_games):
        np.random.seed((config.seed or 0) + g)
        mcts = MCTS(game, nnet, config.mcts_config)
        t0 = time.perf_counter()
        ex = play_self_play_episode(game, mcts, config.temp_threshold)
        per_game.append(time.perf_counter() - t0)
        examples.append(len(ex))
        tree.append(mcts.num_states())
    return {
        "n_games": n_games,
        "per_game_s": per_game,
        "avg_s": float(np.mean(per_game)),
        "std_s": float(np.std(per_game)),
        "examples_per_game": float(np.mean(examples)),
        "tree_states": float(np.mean(tree)),
    }


def measure_cprofile(game, nnet, config: RunConfig) -> dict:
    """cProfile one game; return category own-time split + top functions."""
    np.random.seed(config.seed or 0)
    mcts = MCTS(game, nnet, config.mcts_config)
    pr = cProfile.Profile()
    pr.enable()
    play_self_play_episode(game, mcts, config.temp_threshold)
    pr.disable()
    stats = pstats.Stats(pr)

    categories: dict[str, float] = {}
    rows = []
    for (filename, _lineno, func), (_cc, _nc, tt, ct, _callers) in stats.stats.items():
        cat = _categorise(filename, func)
        categories[cat] = categories.get(cat, 0.0) + tt
        rows.append((tt, ct, f"{Path(filename).name}:{func}", cat))
    rows.sort(reverse=True)
    total_tt = sum(categories.values()) or 1.0
    return {
        "categories": categories,
        "total_own_s": total_tt,
        "top_functions": [
            {"own_s": tt, "cum_s": ct, "name": name, "cat": cat}
            for tt, ct, name, cat in rows[:15]
        ],
    }


def measure_worker_sweep(config: RunConfig, worker_counts) -> dict:
    """games/s for each worker count via the real parallel self-play path."""
    nnet = instantiate_game_and_network(config)[1]
    nnet.save_checkpoint(filename="benchmark_worker_init.pth.tar")
    out = {}
    for n in worker_counts:
        eps = max(n, SWEEP_GAMES_PER_WORKER * n)
        cfg = replace(config, num_parallel_workers=n, num_eps=eps)
        t0 = time.perf_counter()
        run_self_play_episodes_parallel(
            config=cfg, generation=1,
            checkpoint_path="benchmark_worker_init.pth.tar", num_workers=n,
        )
        dt = time.perf_counter() - t0
        out[n] = {"games_per_s": eps / dt, "eps": eps, "wall_s": dt}
    return out


def measure_memory_tree(game, nnet, config: RunConfig) -> dict:
    """tracemalloc one game: peak Python heap + tree size."""
    np.random.seed(config.seed or 0)
    mcts = MCTS(game, nnet, config.mcts_config)
    tracemalloc.start()
    play_self_play_episode(game, mcts, config.temp_threshold)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"peak_heap_mb": peak / 1e6, "tree_states": mcts.num_states()}


def measure_train_ramp(config_path: str, sizes) -> dict:
    """train() peak RSS vs synthetic buffer size. Each size runs in a fresh
    subprocess (via profile_self_play) so memory doesn't accumulate across points."""
    import re
    import subprocess

    out = {}
    for n in sizes:
        proc = subprocess.run(
            ["uv", "run", "python", "-m", "scripts.profile_self_play",
             "--config", config_path, "--mode", "train", "--examples", str(n)],
            capture_output=True, text=True, timeout=900,
        )
        text = proc.stdout + proc.stderr
        m = re.search(r"RSS=(\d+)\s*MB", text)
        out[n] = {"rss_mb": int(m.group(1)) if m else None,
                  "ok": m is not None and proc.returncode == 0}
    return out


# -- HTML rendering ----------------------------------------------------------------

def _fig_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def render_html(data: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs = {}

    # Time pie
    cats = data["cprofile"]["categories"]
    items = sorted(cats.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.pie([v for _, v in items], labels=[k for k, _ in items], autopct="%d%%",
           startangle=90, textprops={"fontsize": 8})
    ax.set_title("Single-worker self-play CPU time split (cProfile own-time)")
    imgs["pie"] = _fig_b64(fig)

    # Top functions bar
    tf = data["cprofile"]["top_functions"][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([t["name"] for t in tf], [t["own_s"] for t in tf], color="#4c78a8")
    ax.set_xlabel("own time (s, 1 game)")
    ax.set_title("Top functions by own time")
    ax.tick_params(labelsize=7)
    imgs["funcs"] = _fig_b64(fig)

    # Worker scaling
    sw = data["worker_sweep"]
    ns = sorted(sw)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ns, [sw[n]["games_per_s"] for n in ns], "o-", color="#1f77b4")
    ax.set_xlabel("workers")
    ax.set_ylabel("games/s")
    ax.set_xticks(ns)
    ax.set_title("Throughput vs worker count")
    ax.grid(alpha=0.3)
    imgs["sweep"] = _fig_b64(fig)

    # Train memory ramp
    tr = data["train_ramp"]
    xs = [n for n in sorted(tr) if tr[n]["rss_mb"]]
    if xs:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot([x / 1000 for x in xs], [tr[x]["rss_mb"] / 1000 for x in xs], "o-", color="#d62728")
        ax.axhline(32, color="black", ls="--", lw=1, label="32 GB")
        ax.set_xlabel("training buffer (thousands of examples)")
        ax.set_ylabel("peak RSS (GB)")
        ax.set_title("Training-step memory vs buffer size")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        imgs["mem"] = _fig_b64(fig)

    t = data["timing"]
    hdr = data["header"]
    sweep_rows = "".join(
        f"<tr><td>{n}</td><td>{sw[n]['games_per_s']:.3f}</td><td>{sw[n]['eps']}</td></tr>"
        for n in ns)
    style = (
        "body{font-family:-apple-system,Segoe UI,sans-serif;max-width:900px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.5;color:#1a1a1a}h1{border-bottom:2px solid #333}"
        "table{border-collapse:collapse;margin:.5rem 0}th,td{border:1px solid #ccc;padding:4px 10px}"
        "img{max-width:100%;margin:.5rem 0}.meta{color:#555;font-size:.9em}"
    )
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>Self-Play Benchmark</title>
<style>{style}</style></head><body>
<h1>Self-Play Benchmark</h1>
<p class=meta>{hdr['config']} &middot; net {hdr['net']} &middot; sims {hdr['sims']}, K={hdr['k']}
&middot; commit {hdr['commit']} &middot; {hdr['timestamp']}</p>

<h2>Single-worker timing</h2>
<table><tr><th>games</th><th>avg s/game</th><th>± std</th><th>examples/game</th><th>tree states</th></tr>
<tr><td>{t['n_games']}</td><td>{t['avg_s']:.2f}</td><td>{t['std_s']:.2f}</td>
<td>{t['examples_per_game']:.0f}</td><td>{t['tree_states']:.0f}</td></tr></table>

<h2>Where the time goes (single worker)</h2>
<img src="data:image/png;base64,{imgs['pie']}">
<img src="data:image/png;base64,{imgs['funcs']}">

<h2>Throughput vs workers</h2>
<img src="data:image/png;base64,{imgs['sweep']}">
<table><tr><th>workers</th><th>games/s</th><th>games run</th></tr>{sweep_rows}</table>

<h2>Memory</h2>
<p>In-episode MCTS tree: peak Python heap <b>{data['memory']['peak_heap_mb']:.1f} MB</b>
({data['memory']['tree_states']:.0f} tree states/game).</p>
{('<img src="data:image/png;base64,' + imgs['mem'] + '">') if 'mem' in imgs else '<p>(train ramp unavailable)</p>'}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser(description="Standardised self-play benchmark -> HTML report")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = load_args(args.config)
    game, nnet = instantiate_game_and_network(config)
    if (enable := getattr(game, "enable_optimised_movegen", None)) is not None and \
            getattr(config, "use_optimised_movegen", False):
        enable()

    print("[benchmark] timing...", flush=True)
    timing = measure_timing(game, nnet, config, TIMING_GAMES)
    print("[benchmark] cProfile...", flush=True)
    cprof = measure_cprofile(game, nnet, config)
    print("[benchmark] memory (tree)...", flush=True)
    memory = measure_memory_tree(game, nnet, config)
    print("[benchmark] worker sweep...", flush=True)
    sweep = measure_worker_sweep(config, WORKER_COUNTS)
    print("[benchmark] train ramp...", flush=True)
    train_ramp = measure_train_ramp(args.config, TRAIN_RAMP_EXAMPLES)

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "?"
    data = {
        "header": {
            "config": args.config,
            "net": f"{config.net_config.num_filters}f×{config.net_config.num_residual_blocks}b",
            "sims": config.mcts_config.num_mcts_sims, "k": config.mcts_config.mcts_batch_size,
            "commit": commit,
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        },
        "timing": timing, "cprofile": cprof, "memory": memory,
        "worker_sweep": sweep, "train_ramp": train_ramp,
    }
    out = Path(args.out or f"temp/benchmarks/benchmark_{commit}.html")
    render_html(data, out)
    print(f"[benchmark] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
