"""Reusable MCTS-profiling report builder.

The original :mod:`scripts.mcts_profiling` script had its HTML report
inline. The benchmark phase script (:mod:`scripts.benchmark_phases`)
wants the same per-move drill-down across multiple game phases
(self-play / arena / Elo), so the rendering pieces live here and both
callers reuse them.

Two entry points:

- :func:`build_single_phase_report` — one phase of games, mirrors the
  original mcts_profiling output (one set of KPIs + per-move charts).
- :func:`build_multi_phase_report` — many phases in one document, with
  a wall-clock summary at the top and a per-phase drill-down below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    from pathlib import Path

    from core.mcts import MCTSEpisodeStats


@dataclass(frozen=True)
class PhaseResult:
    """Outcome of running a benchmarked phase end-to-end.

    Attributes:
        name: Phase label used in headings (e.g. ``"Self-Play"``).
        wall_clock_s: Total wall-clock time for the phase, including
            anything outside MCTS search (player swaps, scoring, etc.).
        stats: One :class:`MCTSEpisodeStats` per game played during the
            phase. The per-game / per-move charts are built from these.
    """

    name: str
    wall_clock_s: float
    stats: list[MCTSEpisodeStats]


def _move_stats_to_df(all_stats: list[MCTSEpisodeStats]) -> pd.DataFrame:
    """Flatten per-move stats from a list of episodes into one DataFrame."""
    records = []
    for game_id, episode in enumerate(all_stats):
        for ms in episode.move_stats:
            records.append({
                "game_id": game_id,
                "move_number": ms.move_number,
                "num_sims": ms.num_sims,
                "search_time_s": ms.search_time_s,
                "inference_time_s": ms.inference_time_s,
                "valid_moves_time_s": ms.valid_moves_time_s,
                "game_ended_time_s": ms.game_ended_time_s,
                "num_leaf_expansions": ms.num_leaf_expansions,
                "num_valid_moves": ms.num_valid_moves,
            })
    return pd.DataFrame(records)


def _episode_stats_to_df(all_stats: list[MCTSEpisodeStats]) -> pd.DataFrame:
    """Flatten episode-level stats into a DataFrame."""
    records = []
    for game_id, s in enumerate(all_stats):
        records.append({
            "game_id": game_id,
            "num_moves": s.num_moves,
            "total_sims": s.total_sims,
            "total_search_time_s": s.total_search_time_s,
            "total_inference_time_s": s.total_inference_time_s,
            "total_valid_moves_time_s": s.total_valid_moves_time_s,
            "total_game_ended_time_s": s.total_game_ended_time_s,
            "num_leaf_expansions": s.num_leaf_expansions,
            "num_valid_moves_calls": s.num_valid_moves_calls,
            "num_game_ended_calls": s.num_game_ended_calls,
            "tree_size": s.tree_size,
            "tree_memory_bytes": s.tree_memory_bytes,
        })
    return pd.DataFrame(records)


def _build_kpi_cards(episode_df: pd.DataFrame, move_df: pd.DataFrame) -> tuple[dict[str, str], dict]:
    """Return ``(kpi_dict, summary_stats)`` for one phase's stats.

    ``kpi_dict`` is label→value-as-string ready for direct HTML rendering.
    ``summary_stats`` is the raw float values, useful for the multi-phase
    summary table where we want comparable numbers across phases.
    """
    avg_game_time = float(episode_df["total_search_time_s"].mean())
    avg_moves = float(episode_df["num_moves"].mean())
    avg_time_per_move = float(move_df["search_time_s"].mean()) if not move_df.empty else 0.0
    sum_sims = float(episode_df["total_sims"].sum())
    sum_search = float(episode_df["total_search_time_s"].sum())
    avg_sims_per_sec = (sum_sims / sum_search) if sum_search > 0 else 0.0
    avg_tree_memory_mb = float(episode_df["tree_memory_bytes"].mean()) / (1024 * 1024)

    kpis = {
        "Avg Search Time / Game": f"{avg_game_time:.1f}s",
        "Avg Search Time / Move": f"{avg_time_per_move * 1000:.0f}ms",
        "Simulations / Second": f"{avg_sims_per_sec:.0f}",
        "Avg Moves / Game": f"{avg_moves:.0f}",
        "Avg Tree Memory / Game": f"{avg_tree_memory_mb:.1f} MB",
    }
    summary = {
        "avg_game_time": avg_game_time,
        "avg_time_per_move_ms": avg_time_per_move * 1000,
        "sims_per_sec": avg_sims_per_sec,
        "avg_moves": avg_moves,
        "tree_memory_mb": avg_tree_memory_mb,
    }
    return kpis, summary


def _kpi_cards_html(kpis: dict[str, str]) -> str:
    parts = []
    for label, value in kpis.items():
        parts.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f"</div>"
        )
    return f'<div class="kpi-grid">{"".join(parts)}</div>'


def _build_phase_charts(
    episode_df: pd.DataFrame,
    move_df: pd.DataFrame,
) -> dict[str, go.Figure]:
    """Build the standard per-phase drill-down chart set."""
    total_inference = float(episode_df["total_inference_time_s"].sum())
    total_valid_moves = float(episode_df["total_valid_moves_time_s"].sum())
    total_game_ended = float(episode_df["total_game_ended_time_s"].sum())
    total_search = float(episode_df["total_search_time_s"].sum())
    total_other = max(0.0, total_search - total_inference - total_valid_moves - total_game_ended)

    fig_pie = go.Figure(data=[go.Pie(
        labels=["Neural Net Inference", "Move Generation", "Game-Ended Checks", "Other (UCB, tree ops)"],
        values=[total_inference, total_valid_moves, total_game_ended, total_other],
        marker_colors=["#636efa", "#00cc96", "#ef553b", "#ffa15a"],
        textinfo="label+percent", hole=0.35,
    )])
    fig_pie.update_layout(title="Time Breakdown (All Games)", height=450, template="plotly_white")

    turn_stats = move_df.groupby("move_number").agg(
        mean_search=("search_time_s", "mean"),
        mean_inference=("inference_time_s", "mean"),
        mean_valid_moves=("valid_moves_time_s", "mean"),
        mean_game_ended=("game_ended_time_s", "mean"),
        mean_num_valid=("num_valid_moves", "mean"),
        mean_leaf=("num_leaf_expansions", "mean"),
    ).reset_index() if not move_df.empty else pd.DataFrame()

    fig_timing = go.Figure()
    fig_search = go.Figure()
    fig_valid = go.Figure()
    fig_leaf = go.Figure()
    if not turn_stats.empty:
        fig_timing.add_trace(go.Scatter(
            x=turn_stats["move_number"], y=turn_stats["mean_inference"] * 1000,
            mode="lines", name="Inference", line=dict(color="#636efa", width=2),
        ))
        fig_timing.add_trace(go.Scatter(
            x=turn_stats["move_number"], y=turn_stats["mean_valid_moves"] * 1000,
            mode="lines", name="Move Generation", line=dict(color="#00cc96", width=2),
        ))
        fig_timing.add_trace(go.Scatter(
            x=turn_stats["move_number"], y=turn_stats["mean_game_ended"] * 1000,
            mode="lines", name="Game-Ended", line=dict(color="#ef553b", width=2),
        ))
        fig_search.add_trace(go.Scatter(
            x=turn_stats["move_number"], y=turn_stats["mean_search"] * 1000,
            mode="lines", name="Total Search", line=dict(color="#ab63fa", width=2),
        ))
        fig_valid.add_trace(go.Scatter(
            x=turn_stats["move_number"], y=turn_stats["mean_num_valid"],
            mode="lines", line=dict(color="#636efa", width=2),
        ))
        fig_leaf.add_trace(go.Scatter(
            x=turn_stats["move_number"], y=turn_stats["mean_leaf"],
            mode="lines", line=dict(color="#ffa15a", width=2),
        ))
    fig_timing.update_layout(
        title="Mean Time Per Move By Component",
        xaxis_title="Move Number", yaxis_title="Time (ms)",
        height=450, template="plotly_white",
    )
    fig_search.update_layout(
        title="Mean Total Search Time Per Move",
        xaxis_title="Move Number", yaxis_title="Time (ms)",
        height=450, template="plotly_white",
    )
    fig_valid.update_layout(
        title="Mean Legal Moves Per Turn (MCTS Perspective)",
        xaxis_title="Move Number", yaxis_title="Number of Legal Moves",
        height=450, template="plotly_white",
    )
    fig_leaf.update_layout(
        title="Mean Leaf Expansions Per Move",
        xaxis_title="Move Number", yaxis_title="Leaf Expansions",
        height=450, template="plotly_white",
    )

    if not move_df.empty:
        fig_hist = px.histogram(
            move_df, x="search_time_s", nbins=50,
            title="Distribution of Search Time Per Move",
            labels={"search_time_s": "Search Time (s)"},
            height=400, template="plotly_white",
        )
        fig_hist.update_layout(xaxis_title="Search Time (s)", yaxis_title="Count")
    else:
        fig_hist = go.Figure()

    return {
        "pie": fig_pie, "timing": fig_timing, "search": fig_search,
        "valid": fig_valid, "leaf": fig_leaf, "hist": fig_hist,
    }


_BASE_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1280px; margin: 0 auto; padding: 24px 32px; color: #2a3f5f;
    background: #ffffff;
}
h1 { border-bottom: 2px solid #636efa; padding-bottom: 8px; margin-bottom: 4px; }
.subtitle { color: #6b7280; font-size: 14px; margin-bottom: 24px; }
h2 { margin-top: 40px; color: #636efa; font-size: 20px; }
h3 { margin-top: 28px; color: #2a3f5f; font-size: 16px; }
.kpi-grid { display: flex; gap: 14px; margin: 20px 0 32px 0; flex-wrap: wrap; }
.kpi-card {
    flex: 1 1 180px; padding: 14px 18px; border-radius: 8px;
    background: #f8f9fb; border: 1px solid #e5e7eb;
}
.kpi-value { font-size: 26px; font-weight: 700; color: #1a1a2e; }
.kpi-label { font-size: 12px; color: #6b7280; margin-top: 2px; text-transform: uppercase;
             letter-spacing: 0.5px; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
table.summary { border-collapse: collapse; margin: 12px 0; width: 100%; }
table.summary th, table.summary td {
    text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e7eb;
    font-size: 14px;
}
table.summary th { background: #f8f9fb; }
.phase-section { border-top: 1px solid #e5e7eb; margin-top: 32px; padding-top: 12px; }
"""


def _build_phase_section(phase: PhaseResult, *, level: int = 2, charts_include_plotlyjs: bool = False) -> str:
    """Render one phase's KPI + chart block as HTML (no <html>/<head> wrapper)."""
    if not phase.stats:
        return f"<h{level}>{phase.name}</h{level}><p>No games recorded.</p>"
    episode_df = _episode_stats_to_df(phase.stats)
    move_df = _move_stats_to_df(phase.stats)
    kpis, _ = _build_kpi_cards(episode_df, move_df)
    charts = _build_phase_charts(episode_df, move_df)
    incl = charts_include_plotlyjs

    return f"""<div class="phase-section">
<h{level}>{phase.name}</h{level}>
<p class="subtitle">{len(phase.stats)} games, wall-clock {phase.wall_clock_s:.1f}s
({phase.wall_clock_s / max(len(phase.stats), 1):.2f}s/game incl. non-search overhead)</p>
{_kpi_cards_html(kpis)}
<h3>Time Breakdown</h3>
<div class="chart-grid">
    {charts["pie"].to_html(full_html=False, include_plotlyjs=incl)}
    {charts["hist"].to_html(full_html=False, include_plotlyjs=False)}
</div>
<h3>Per-Move Analysis</h3>
<div class="chart-grid">
    {charts["timing"].to_html(full_html=False, include_plotlyjs=False)}
    {charts["search"].to_html(full_html=False, include_plotlyjs=False)}
</div>
<h3>Search Characteristics</h3>
<div class="chart-grid">
    {charts["valid"].to_html(full_html=False, include_plotlyjs=False)}
    {charts["leaf"].to_html(full_html=False, include_plotlyjs=False)}
</div>
</div>"""


def build_single_phase_report(
    all_stats: list[MCTSEpisodeStats],
    *,
    title: str,
    subtitle: str,
    output_dir: Path,
    wall_clock_s: float | None = None,
) -> Path:
    """Render the single-phase report and write it to ``output_dir/report.html``.

    Used by :mod:`scripts.mcts_profiling`. Also dumps per-move and per-episode
    CSVs alongside the HTML for ad-hoc inspection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    move_df = _move_stats_to_df(all_stats)
    episode_df = _episode_stats_to_df(all_stats)

    if wall_clock_s is None:
        wall_clock_s = float(episode_df["total_search_time_s"].sum())
    phase = PhaseResult(name="Profiling", wall_clock_s=wall_clock_s, stats=all_stats)
    body = _build_phase_section(phase, level=2, charts_include_plotlyjs=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
    <style>{_BASE_CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>
{body}
</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html)

    move_df.to_csv(output_dir / "move_stats.csv", index=False)
    episode_df.to_csv(output_dir / "episode_stats.csv", index=False)
    return report_path


def build_multi_phase_report(
    phases: list[PhaseResult],
    *,
    title: str,
    subtitle: str,
    output_dir: Path,
    estimator_table_html: str | None = None,
) -> Path:
    """Render a combined report that summarises multiple phases + drills into each.

    Layout:

    1. **Wall-clock summary** — per-phase total time + games + sims/sec, side
       by side, so the F1 "before vs after" comparison is obvious at a glance.
    2. **Run-time estimate table** — optional; passed in as HTML by the
       caller (the estimator lives in :mod:`scripts.benchmark_phases`).
    3. **Per-phase drill-down** — the original mcts_profiling-style detail
       for each phase, in the same document.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for p in phases:
        if not p.stats:
            summary_rows.append({
                "Phase": p.name, "Games": 0, "Total wall-clock": "—",
                "Mean / game": "—", "Sims/sec": "—",
            })
            continue
        episode_df = _episode_stats_to_df(p.stats)
        n = len(p.stats)
        sum_sims = float(episode_df["total_sims"].sum())
        sum_search = float(episode_df["total_search_time_s"].sum())
        sims_per_sec = (sum_sims / sum_search) if sum_search > 0 else 0.0
        summary_rows.append({
            "Phase": p.name, "Games": n,
            "Total wall-clock": f"{p.wall_clock_s:.1f}s",
            "Mean / game": f"{p.wall_clock_s / n:.2f}s",
            "Sims/sec": f"{sims_per_sec:.0f}",
        })

    summary_table = "<table class=\"summary\"><thead><tr>" + "".join(
        f"<th>{k}</th>" for k in summary_rows[0]
    ) + "</tr></thead><tbody>"
    for row in summary_rows:
        summary_table += "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
    summary_table += "</tbody></table>"

    fig_phase_bar = go.Figure(data=[go.Bar(
        x=[p.name for p in phases],
        y=[p.wall_clock_s for p in phases],
        marker_color="#636efa",
        text=[f"{p.wall_clock_s:.1f}s" for p in phases],
        textposition="outside",
    )])
    fig_phase_bar.update_layout(
        title="Wall-clock by Phase",
        yaxis_title="Seconds",
        height=400, template="plotly_white",
        showlegend=False,
    )

    drilldown_blocks = []
    for _i, p in enumerate(phases):
        drilldown_blocks.append(_build_phase_section(p, level=2, charts_include_plotlyjs=False))

    estimator_html = ""
    if estimator_table_html:
        estimator_html = f"<h2>Run-time estimates</h2>\n{estimator_table_html}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
    <style>{_BASE_CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>

<h2>Phase summary</h2>
{summary_table}
{fig_phase_bar.to_html(full_html=False, include_plotlyjs=False)}

{estimator_html}

<h2>Per-phase drill-down</h2>
<p class="subtitle">Each phase below uses the same template as
<code>scripts/mcts_profiling.py</code>: KPIs, component pie, per-move
timing curves, and search characteristics.</p>
{"".join(drilldown_blocks)}
</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html)

    # Per-phase CSV dumps for ad-hoc inspection.
    for p in phases:
        if not p.stats:
            continue
        slug = p.name.lower().replace(" ", "_").replace("/", "_")
        _move_stats_to_df(p.stats).to_csv(output_dir / f"{slug}_move_stats.csv", index=False)
        _episode_stats_to_df(p.stats).to_csv(output_dir / f"{slug}_episode_stats.csv", index=False)

    return report_path
