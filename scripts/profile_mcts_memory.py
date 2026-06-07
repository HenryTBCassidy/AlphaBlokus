"""MCTS tree memory profiler (sub-plan M1/M2: docs/plans/mcts-memory-reduction.md).

Plays one self-play game with a single persistent MCTS (exactly as a worker does)
and attributes the tree's memory to its actual consumers, to find the biggest
reduction targets and record a baseline to measure future cuts against.

Two complementary views:
  * Deep-size pass — walks each MCTS dict (keys + values) after every move, giving
    a per-dict breakdown, bytes-per-state, and the growth curve. No tracemalloc,
    so object sizes are undistorted.
  * tracemalloc pass — replays the same game under tracemalloc and snapshots at the
    end, giving the authoritative top allocation sites by source line (no
    object-graph double-counting).

Outputs:
  * docs/research/mcts-memory-report.md   (committed: tables + findings, no images)
  * temp/mcts-memory-report.html          (gitignored: same + growth chart)
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

from core.config import load_args
from core.game_factory import instantiate_game_and_network
from core.mcts import MCTS

# The seven dicts that make up the MCTS tree (core/mcts.py).
_TREE_DICTS = (
    "q_values", "visit_counts", "state_visits", "policy_priors",
    "game_ended_cache", "valid_moves_cache", "virtual_visits",
)


def _deep_size(obj: object, seen: set[int]) -> int:
    """Recursive byte size of ``obj``, counting each object once via ``seen``.

    numpy arrays are counted as their buffer ``nbytes`` plus the wrapper; dicts
    and sequences recurse into keys/values/elements.
    """
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(obj, np.ndarray):
        # nbytes is the data buffer; +128 covers the array header. (Do NOT add
        # getsizeof, which already includes the buffer for owned arrays.)
        return int(obj.nbytes) + 128
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_size(k, seen) + _deep_size(v, seen)
    elif isinstance(obj, (tuple, list, set, frozenset)):
        for x in obj:
            size += _deep_size(x, seen)
    return size


def _dict_breakdown(mcts: MCTS) -> dict[str, tuple[int, int]]:
    """Per-dict ``(entries, bytes)``. Each dict sized with its own ``seen`` set,
    so shared state-key objects are attributed fully to every dict that holds
    them (slight over-count vs the unique total — reported alongside)."""
    out: dict[str, tuple[int, int]] = {}
    for name in _TREE_DICTS:
        d = getattr(mcts, name)
        out[name] = (len(d), _deep_size(d, set()))
    return out


def _unique_total(mcts: MCTS) -> int:
    """True bytes across all tree dicts, counting each object once."""
    seen: set[int] = set()
    return sum(_deep_size(getattr(mcts, name), seen) for name in _TREE_DICTS)


def _play_game(game, nnet, config, seed: int, record_curve: bool):
    """Play one game with a persistent MCTS (worker semantics). Returns the
    MCTS and, if requested, a per-move ``(move, unique_bytes)`` growth curve."""
    np.random.seed(seed)
    mcts = MCTS(game, nnet, config.mcts_config)
    board = game.initialise_board()
    current_player = 1
    move_count = 0
    curve: list[tuple[int, int]] = []
    while True:
        move_count += 1
        canonical = game.get_canonical_form(board, current_player)
        temp = int(move_count < config.temp_threshold)
        pi = mcts.get_action_prob(canonical, temp=temp, add_root_noise=True)
        action = np.random.choice(len(pi), p=pi)
        board, current_player = game.get_next_state(board, current_player, action)
        if record_curve:
            curve.append((move_count, _unique_total(mcts)))
        if game.get_game_ended(board, current_player) != 0:
            break
    return mcts, curve


def _mb(n: int) -> float:
    return n / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile MCTS tree memory")
    ap.add_argument("--config", default="run_configurations/blokus_scaled_15.json")
    ap.add_argument("--sims", type=int, default=0, help="override num_mcts_sims (0 = config)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = load_args(args.config)
    # Seed BOTH torch (random net init — no checkpoint is loaded) and numpy
    # (move sampling + Dirichlet noise) so the whole game, and thus the baseline
    # tree size, is reproducible run-to-run.
    torch.manual_seed(args.seed)
    # Force CPU (this profiles tree memory, which is device-independent) so it
    # runs on the Mac without CUDA.
    config = dataclasses.replace(config, net_config=dataclasses.replace(config.net_config, cuda=False))
    if args.sims:
        config = dataclasses.replace(
            config, mcts_config=dataclasses.replace(config.mcts_config, num_mcts_sims=args.sims))
    game, nnet = instantiate_game_and_network(config)
    f2 = getattr(config, "use_optimised_movegen", False)
    if f2 and hasattr(game, "enable_optimised_movegen"):
        game.enable_optimised_movegen()  # match production move-gen (affects ValidMoves dtype)

    sims = config.mcts_config.num_mcts_sims
    action_size = game.get_action_size()
    print(f"profiling MCTS memory: game={config.game} sims={sims} f2={f2} "
          f"net={config.net_config.num_filters}f×{config.net_config.num_residual_blocks}b "
          f"action_size={action_size}")

    # Pass 1: deep-size growth curve + final breakdown (no tracemalloc).
    t0 = time.perf_counter()
    mcts, curve = _play_game(game, nnet, config, args.seed, record_curve=True)
    wall = time.perf_counter() - t0
    breakdown = _dict_breakdown(mcts)

    def _vinfo(name: str) -> str:
        for v in getattr(mcts, name).values():
            if isinstance(v, np.ndarray):
                return f"{v.dtype} array, shape {v.shape}, {v.nbytes / 1024:.1f} KB each"
            return type(v).__name__
        return "empty"

    value_info = {n: _vinfo(n) for n in ("policy_priors", "valid_moves_cache")}
    unique_total = _unique_total(mcts)
    peak_bytes = max((b for _, b in curve), default=unique_total)
    n_states = len(mcts.state_visits)
    n_state_actions = len(mcts.q_values)
    moves = len(curve)

    # Pass 2: tracemalloc authoritative top allocation sites (same seed).
    tracemalloc.start(20)
    mcts_tm, _ = _play_game(game, nnet, config, args.seed, record_curve=False)  # keep tree alive for the snapshot
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    del mcts_tm
    top = snap.statistics("lineno")[:12]

    # ---- assemble report data ----
    rows = sorted(breakdown.items(), key=lambda kv: kv[1][1], reverse=True)
    bd_lines = []
    for name, (n, b) in rows:
        per = b / n if n else 0
        bd_lines.append((name, n, _mb(b), per, 100.0 * b / sum(v[1] for v in breakdown.values())))
    top_lines = []
    for st in top:
        fr = st.traceback[0]
        fn = fr.filename.split("/")[-1]
        top_lines.append((f"{fn}:{fr.lineno}", _mb(st.size), st.count))

    bytes_per_state = unique_total / n_states if n_states else 0

    _write_md(config, sims, f2, action_size, moves, n_states, n_state_actions,
              _mb(unique_total), _mb(peak_bytes), bytes_per_state, wall, bd_lines, top_lines,
              curve, value_info)
    _write_html(config, sims, moves, n_states, _mb(unique_total), _mb(peak_bytes),
                bytes_per_state, bd_lines, top_lines, curve)

    print(f"\n  game: {moves} moves, {n_states} states, {n_state_actions} state-actions ({wall:.1f}s)")
    print(f"  TREE PEAK (unique): {_mb(peak_bytes):.1f} MB   end: {_mb(unique_total):.1f} MB"
          f"   {bytes_per_state/1024:.1f} KB/state")
    for n, info in value_info.items():
        print(f"  {n} values: {info}")
    print("  per-dict (bytes):")
    for name, n, mb, per, pct in bd_lines:
        print(f"    {name:<20} {n:>8} entries  {mb:8.1f} MB  {per/1024:7.2f} KB/entry  {pct:5.1f}%")
    print("\n  wrote docs/research/mcts-memory-report.md + temp/mcts-memory-report.html")


def _write_md(config, sims, f2, action_size, moves, n_states, n_sa, total_mb, peak_mb,
              bps, wall, bd_lines, top_lines, curve, value_info) -> None:
    L = []
    L.append("# MCTS Memory — Baseline Report\n")
    L.append("Generated by `scripts/profile_mcts_memory.py` "
             "(sub-plan M2: [mcts-memory-reduction](../plans/mcts-memory-reduction.md)). "
             "One self-play game, single persistent MCTS (worker semantics). "
             "Tree memory is device-independent, so this is measured on CPU.\n")
    L.append(f"**Settings:** game `{config.game}` · {sims} sims · "
             f"net {config.net_config.num_filters}f×{config.net_config.num_residual_blocks}b · "
             f"action space {action_size} · F2 movegen {f2} · seed 42.\n")
    L.append("## Baseline (the number to beat)\n")
    L.append(f"- **Peak tree memory (per worker): {peak_mb:.1f} MB** — this is what gates "
             f"how many workers fit in RAM.\n"
             f"- End-of-game: {total_mb:.1f} MB · **{bps/1024:.1f} KB/state** · "
             f"{n_states} states · {n_sa} state-actions · {moves} moves ({wall:.1f}s).\n")
    L.append("> Treat the peak MB/worker as immutable: every later change re-runs this profile "
             "and reports the delta against it.\n")
    L.append("## Where the memory lives — per-dict breakdown\n")
    L.append("| Dict | Entries | Bytes | Per-entry | % of tree |")
    L.append("|------|--------:|------:|----------:|----------:|")
    for name, n, mb, per, pct in bd_lines:
        L.append(f"| `{name}` | {n:,} | {mb:.1f} MB | {per/1024:.2f} KB | {pct:.1f}% |")
    L.append("")
    L.append("## Python-object overhead (tracemalloc, top 12 — excludes numpy buffers)\n")
    L.append("> numpy array data buffers live outside Python's allocator, so tracemalloc can't "
             "see them — the deep-size table above is authoritative for the arrays. This table "
             "only confirms the non-array bookkeeping (dict tables, tuples, scalars) is negligible "
             "(~MB), i.e. essentially all tree memory is the two dense arrays.\n")
    L.append("| Source line | Size | Allocations |")
    L.append("|-------------|-----:|------------:|")
    for loc, mb, cnt in top_lines:
        L.append(f"| `{loc}` | {mb:.1f} MB | {cnt:,} |")
    L.append("")
    L.append("## Growth curve (unique tree bytes vs move)\n")
    L.append("| Move | Tree MB |")
    L.append("|-----:|--------:|")
    step = max(1, len(curve) // 20)
    for i in range(0, len(curve), step):
        m, b = curve[i]
        L.append(f"| {m} | {b/1e6:.1f} |")
    if curve:
        L.append(f"| {curve[-1][0]} | {curve[-1][1]/1e6:.1f} |")
    L.append("")
    L.append("## Read-out (candidate levers → feeds M3)\n")
    top_dict = bd_lines[0]
    L.append(f"- Biggest consumer: **`{top_dict[0]}`** at {top_dict[2]:.1f} MB "
             f"({top_dict[4]:.1f}% of the tree, {top_dict[3]/1024:.1f} KB/entry).")
    for n, info in value_info.items():
        L.append(f"- `{n}` stores: {info}.")
    L.append("- See `docs/plans/mcts-memory-reduction.md` M3 for the candidate cuts; "
             "the breakdown above ranks which to attack first.\n")
    Path("docs/research/mcts-memory-report.md").write_text("\n".join(L))


def _write_html(config, sims, moves, n_states, total_mb, peak_mb, bps, bd_lines, top_lines, curve) -> None:
    import plotly.graph_objects as go
    xs = [m for m, _ in curve]
    ys = [b / 1e6 for _, b in curve]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#2563eb")))
    fig.update_layout(title="MCTS tree memory growth", xaxis_title="move",
                      yaxis_title="tree MB (unique)", template="plotly_white", height=420)
    chart = fig.to_html(full_html=False, include_plotlyjs="inline")

    def table(headers, rows):
        h = "".join(f"<th>{x}</th>" for x in headers)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"

    bd_rows = [(f"<code>{n}</code>", f"{cnt:,}", f"{mb:.1f} MB", f"{per/1024:.2f} KB", f"{pct:.1f}%")
               for n, cnt, mb, per, pct in bd_lines]
    top_rows = [(f"<code>{loc}</code>", f"{mb:.1f} MB", f"{cnt:,}") for loc, mb, cnt in top_lines]
    css = (
        "body{font-family:-apple-system,Segoe UI,sans-serif;max-width:900px;margin:2rem auto;color:#1f2937}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #e5e7eb;padding:.4rem .6rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}th{background:#f9fafb}"
        ".big{font-size:1.5rem;font-weight:700;color:#2563eb}"
        "code{background:#f3f4f6;padding:.1rem .3rem;border-radius:3px}"
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>MCTS Memory Baseline</title>
<style>{css}</style></head><body>
<h1>MCTS Memory — Baseline</h1>
<p>{config.game} · {sims} sims · {moves} moves · {n_states:,} states</p>
<p class="big">Peak {peak_mb:.1f} MB/worker · {bps/1024:.1f} KB/state</p>
<p>End-of-game {total_mb:.1f} MB.</p>
<h2>Tree growth</h2>{chart}
<h2>Per-dict breakdown</h2>{table(["Dict","Entries","Bytes","Per-entry","% tree"], bd_rows)}
<h2>Top allocation sites (tracemalloc)</h2>{table(["Source line","Size","Allocations"], top_rows)}
</body></html>"""
    Path("temp").mkdir(exist_ok=True)
    Path("temp/mcts-memory-report.html").write_text(html)


if __name__ == "__main__":
    main()
