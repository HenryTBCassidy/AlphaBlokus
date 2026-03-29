from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from plotly import graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from core.config import RunConfig

# ---------------------------------------------------------------------------
# D1: Consistent color palette and chart defaults
# ---------------------------------------------------------------------------

_COLORS = {
    "primary": "#636efa",
    "secondary": "#EF553B",
    "tertiary": "#00cc96",
    "neutral": "#aaaaaa",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "accent": "#ab63fa",
}

_PHASE_COLORS = {
    "SelfPlay": _COLORS["primary"],
    "Training": _COLORS["secondary"],
    "Arena": _COLORS["tertiary"],
}

_TEMPLATE = "plotly_white"
_FULL_WIDTH = 1200
_HALF_WIDTH = 580
_CHART_HEIGHT = 420
_GRID_HEIGHT = 380


def _apply_defaults(fig: go.Figure, *, width: int = _FULL_WIDTH, height: int = _CHART_HEIGHT) -> go.Figure:
    """Apply shared template, sizing, and margin defaults to a figure."""
    fig.update_layout(
        template=_TEMPLATE,
        width=width,
        height=height,
        margin={"t": 48, "b": 48, "l": 56, "r": 24},
    )
    return fig


# ---------------------------------------------------------------------------
# D2: KPI summary cards
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def _make_kpi_cards(
    loss_data: pd.DataFrame,
    arena_data: pd.DataFrame,
    timings_data: pd.DataFrame,
    profiling_data: pd.DataFrame,
    throughput_data: pd.DataFrame,
) -> str:
    """D2: Build HTML for the KPI card row."""
    # Final loss + delta
    sorted_loss = loss_data.sort_values(["generation", "epoch", "batch_number"])
    by_gen = sorted_loss.groupby("generation")["average_loss"].last()
    final_loss = by_gen.iloc[-1]
    first_loss = by_gen.iloc[0]
    loss_delta_pct = ((final_loss - first_loss) / first_loss) * 100

    # Accept rate
    total_arena = arena_data["wins"] + arena_data["losses"] + arena_data["draws"]
    accepted = (arena_data["wins"] / total_arena) > 0.5
    accept_count = int(accepted.sum())
    total_gens = len(arena_data)

    # Total time
    whole = timings_data[timings_data["cycle_stage"] == "WholeCycle"]
    total_time = whole["time_elapsed"].sum()

    # Self-play speed
    sp_speed = profiling_data["sims_per_second"].median()

    # Training throughput
    train_speed = throughput_data["samples_per_second"].mean()

    cards = [
        ("Final Loss", f"{final_loss:.3f}", f"{loss_delta_pct:+.0f}% vs gen 1",
         "positive" if loss_delta_pct < 0 else "negative"),
        ("Accept Rate", f"{accept_count}/{total_gens}",
         f"{100 * accept_count / total_gens:.0f}% of generations", ""),
        ("Total Time", _format_duration(total_time), f"{total_gens} generations", ""),
        ("Self-Play", f"{sp_speed:,.0f}", "sims/s (median)", ""),
        ("Training", f"{train_speed:,.0f}", "samples/s (mean)", ""),
    ]

    html_parts = []
    for label, value, context, delta_cls in cards:
        delta_class = f' class="kpi-delta {delta_cls}"' if delta_cls else ' class="kpi-delta"'
        html_parts.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'<div{delta_class}>{context}</div>'
            f"</div>"
        )
    return f'<div class="kpi-grid">{"".join(html_parts)}</div>'


# ---------------------------------------------------------------------------
# D4: Loss per generation
# ---------------------------------------------------------------------------

def _make_loss_per_generation(df: pd.DataFrame) -> go.Figure:
    """Line chart with pi_loss, v_loss, and total loss per generation."""
    sorted_df = df.sort_values(["generation", "epoch", "batch_number"])
    agg = sorted_df.groupby("generation").agg(
        pi_loss=("average_pi_loss", "last"),
        v_loss=("average_v_loss", "last"),
        total_loss=("average_loss", "last"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["generation"], y=agg["total_loss"],
        mode="lines+markers", name="Total",
        line={"width": 3, "color": _COLORS["primary"]},
    ))
    fig.add_trace(go.Scatter(
        x=agg["generation"], y=agg["pi_loss"],
        mode="lines+markers", name="Policy",
        line={"width": 2, "color": _COLORS["secondary"]},
    ))
    fig.add_trace(go.Scatter(
        x=agg["generation"], y=agg["v_loss"],
        mode="lines+markers", name="Value",
        line={"width": 2, "color": _COLORS["tertiary"]},
    ))

    # "Worse than random" band for value loss
    max_v = float(agg["v_loss"].max())
    if max_v > 1.0:
        fig.add_hrect(
            y0=1.0, y1=max(max_v * 1.05, 1.4),
            fillcolor=_COLORS["negative"], opacity=0.06, line_width=0,
            annotation_text="Value loss > 1.0: worse than random",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=_COLORS["negative"],
        )

    fig.update_layout(xaxis_title="Generation", yaxis_title="Loss", title="Loss per Generation")
    return _apply_defaults(fig)


# ---------------------------------------------------------------------------
# D5: Smoothed per-batch loss timeline
# ---------------------------------------------------------------------------

def _make_loss_timeline(df: pd.DataFrame) -> go.Figure:
    """Smoothed loss over the full training timeline with generation boundaries."""
    sorted_df = df.sort_values(["generation", "epoch", "batch_number"]).reset_index(drop=True)
    sorted_df["step"] = range(len(sorted_df))

    span = max(5, len(sorted_df) // 15)

    series = [
        ("average_loss", "Total", _COLORS["primary"]),
        ("average_pi_loss", "Policy", _COLORS["secondary"]),
        ("average_v_loss", "Value", _COLORS["tertiary"]),
    ]

    fig = go.Figure()

    for col, name, color in series:
        # Raw data (faint)
        fig.add_trace(go.Scatter(
            x=sorted_df["step"], y=sorted_df[col],
            mode="markers", marker={"size": 2, "color": color, "opacity": 0.15},
            showlegend=False, hoverinfo="skip",
        ))
        # Smoothed line
        smoothed = sorted_df[col].ewm(span=span, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=sorted_df["step"], y=smoothed,
            mode="lines", name=name,
            line={"width": 2, "color": color},
        ))

    # Generation boundary lines
    gen_boundaries = sorted_df.groupby("generation")["step"].min()
    for gen, step in gen_boundaries.items():
        if step == 0:
            continue
        fig.add_vline(
            x=step, line_dash="dot", line_color=_COLORS["neutral"], line_width=1,
            annotation_text=f"Gen {gen}", annotation_position="top",
            annotation_font_size=9, annotation_font_color=_COLORS["neutral"],
        )

    fig.update_layout(
        xaxis_title="Training Step (sequential)", yaxis_title="Loss",
        title="Per-Batch Loss (smoothed)",
    )
    return _apply_defaults(fig)


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

def _make_arena_plot(arena_data: pd.DataFrame, update_threshold: float) -> go.Figure:
    """Stacked bar (wins/draws/losses) with acceptance annotations and threshold line."""
    df = arena_data.sort_values("generation").copy()
    total = df["wins"] + df["losses"] + df["draws"]
    df["pct_wins"] = 100 * df["wins"] / total
    df["pct_draws"] = 100 * df["draws"] / total
    df["pct_losses"] = 100 * df["losses"] / total
    df["is_accepted"] = (df["wins"] / total) > update_threshold

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["generation"], y=df["pct_wins"],
        name="Wins", marker_color=_COLORS["positive"],
        customdata=df["wins"],
        hovertemplate="Wins: %{customdata} (%{y:.0f}%)<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["generation"], y=df["pct_draws"],
        name="Draws", marker_color=_COLORS["neutral"],
        customdata=df["draws"],
        hovertemplate="Draws: %{customdata} (%{y:.0f}%)<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["generation"], y=df["pct_losses"],
        name="Losses", marker_color=_COLORS["negative"],
        customdata=df["losses"],
        hovertemplate="Losses: %{customdata} (%{y:.0f}%)<extra></extra>",
    ))

    fig.add_hline(
        y=update_threshold * 100, line_dash="dash", line_color=_COLORS["primary"],
        annotation_text=f"Threshold ({update_threshold:.0%})",
        annotation_position="top left",
    )

    for _, row in df.iterrows():
        label = "Accepted" if row["is_accepted"] else "Rejected"
        color = _COLORS["positive"] if row["is_accepted"] else _COLORS["negative"]
        fig.add_annotation(
            x=row["generation"], y=102, text=label, showarrow=False,
            font={"size": 11, "color": color},
        )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Generation", yaxis_title="Percentage", yaxis_range=[0, 115],
        title="Arena: New Net vs Predecessor",
    )
    return _apply_defaults(fig)


# ---------------------------------------------------------------------------
# D6: Performance charts (half-width for grid)
# ---------------------------------------------------------------------------

def _make_timing_plot(timings_data: pd.DataFrame) -> go.Figure:
    """Stacked bar of phase durations per generation."""
    df = timings_data[timings_data["cycle_stage"] != "WholeCycle"].copy()

    fig = go.Figure()
    for phase in ["SelfPlay", "Training", "Arena"]:
        phase_df = df[df["cycle_stage"] == phase]
        fig.add_trace(go.Bar(
            x=phase_df["generation"], y=phase_df["time_elapsed"],
            name=phase, marker_color=_PHASE_COLORS[phase],
        ))

    fig.update_layout(
        barmode="stack",
        xaxis_title="Generation", yaxis_title="Time (s)",
        title="Time per Generation",
    )
    return _apply_defaults(fig, width=_HALF_WIDTH, height=_GRID_HEIGHT)


def _make_throughput_plot(throughput_data: pd.DataFrame) -> go.Figure:
    """Bar chart of average training throughput per generation."""
    agg = throughput_data.groupby("generation").agg(
        avg_throughput=("samples_per_second", "mean"),
    ).reset_index()

    fig = go.Figure(go.Bar(
        x=agg["generation"], y=agg["avg_throughput"],
        marker_color=_COLORS["primary"],
    ))
    fig.update_layout(
        xaxis_title="Generation", yaxis_title="Samples/s",
        title="Training Throughput",
    )
    return _apply_defaults(fig, width=_HALF_WIDTH, height=_GRID_HEIGHT)


def _make_resource_usage_plot(resource_data: pd.DataFrame) -> go.Figure:
    """Grouped bar of process RSS (MB) per stage per generation."""
    df = resource_data.copy()
    df["rss_mb"] = df["process_rss_bytes"] / (1024 ** 2)

    fig = go.Figure()
    for stage in ["SelfPlay", "Training", "Arena"]:
        stage_df = df[df["cycle_stage"] == stage]
        if stage_df.empty:
            continue
        fig.add_trace(go.Bar(
            x=stage_df["generation"], y=stage_df["rss_mb"],
            name=stage, marker_color=_PHASE_COLORS.get(stage, _COLORS["neutral"]),
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Generation", yaxis_title="RSS (MB)",
        title="Process Memory",
    )

    has_gpu = (
        "gpu_memory_bytes" in df.columns
        and df["gpu_memory_bytes"].notna().any()
        and (df["gpu_memory_bytes"].fillna(0) > 0).any()
    )
    if has_gpu:
        df["gpu_mb"] = df["gpu_memory_bytes"] / (1024 ** 2)
        for stage in df["cycle_stage"].unique():
            stage_df = df[df["cycle_stage"] == stage]
            fig.add_trace(go.Bar(
                x=stage_df["generation"], y=stage_df["gpu_mb"],
                name=f"{stage} (GPU)", marker_color=_PHASE_COLORS.get(stage),
                marker_pattern_shape="/",
            ))

    return _apply_defaults(fig, width=_HALF_WIDTH, height=_GRID_HEIGHT)


def _make_profiling_plot(profiling_data: pd.DataFrame) -> go.Figure:
    """Simplified profiling: game length and MCTS throughput box plots."""
    df = profiling_data.copy()
    df["generation"] = df["generation"].astype(str)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Game Length (moves)", "MCTS Throughput (sims/s)"],
        vertical_spacing=0.18,
    )

    fig.add_trace(
        go.Box(
            x=df["generation"], y=df["num_moves"],
            name="Game Length", showlegend=False,
            marker_color=_COLORS["primary"],
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(
            x=df["generation"], y=df["sims_per_second"],
            name="Sims/s", showlegend=False,
            marker_color=_COLORS["secondary"],
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_layout(title="Self-Play Profiling")
    return _apply_defaults(fig, width=_HALF_WIDTH, height=_GRID_HEIGHT)


# ---------------------------------------------------------------------------
# D7: Config table (unchanged content, moved to bottom)
# ---------------------------------------------------------------------------

def _make_config_table(config: RunConfig) -> str:
    """Build an HTML table summarising the run configuration."""
    rows = [
        ("Game", config.game),
        ("Generations", config.num_generations),
        ("Episodes / generation", config.num_eps),
        ("MCTS simulations", config.mcts_config.num_mcts_sims),
        ("CPUCT", config.mcts_config.cpuct),
        ("Arena matches", config.num_arena_matches),
        ("Update threshold", config.update_threshold),
        ("Max lookback", config.max_generations_lookback),
        ("Learning rate", config.net_config.learning_rate),
        ("Batch size", config.net_config.batch_size),
        ("Epochs", config.net_config.epochs),
        ("Residual blocks", config.net_config.num_residual_blocks),
        ("Filters", config.net_config.num_filters),
        ("Dropout", config.net_config.dropout),
        ("LR scheduler", config.net_config.lr_scheduler or "constant"),
        ("CUDA", config.net_config.cuda),
    ]
    header = "<tr><th>Parameter</th><th>Value</th></tr>"
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{header}{body}</table>"


# ---------------------------------------------------------------------------
# D3: HTML template with CSS grid, collapsible sections, descriptions
# ---------------------------------------------------------------------------

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1280px; margin: 0 auto; padding: 24px 32px; color: #2a3f5f;
    background: #ffffff;
}
h1 { border-bottom: 2px solid #636efa; padding-bottom: 8px; margin-bottom: 4px; }
.subtitle { color: #6b7280; font-size: 14px; margin-bottom: 24px; }
h2 { margin-top: 40px; color: #636efa; font-size: 20px; }
.section-desc { color: #6b7280; font-size: 13px; margin: -4px 0 16px 0; line-height: 1.5; }
section { margin-bottom: 24px; }

/* KPI cards */
.kpi-grid { display: flex; gap: 14px; margin: 20px 0 32px 0; }
.kpi-card {
    flex: 1; padding: 14px 18px; border-radius: 8px;
    background: #f8f9fb; border: 1px solid #e5e7eb;
}
.kpi-value { font-size: 26px; font-weight: 700; color: #1a1a2e; }
.kpi-label { font-size: 12px; color: #6b7280; margin-top: 2px; text-transform: uppercase;
             letter-spacing: 0.5px; }
.kpi-delta { font-size: 12px; margin-top: 4px; color: #6b7280; }
.kpi-delta.positive { color: #2ca02c; }
.kpi-delta.negative { color: #d62728; }

/* 2-column chart grid */
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

/* Collapsible sections */
details { margin: 12px 0; }
details > summary {
    cursor: pointer; font-weight: 600; color: #636efa; font-size: 14px;
    padding: 8px 0; list-style: none;
}
details > summary::before { content: "\\25B8  "; }
details[open] > summary::before { content: "\\25BE  "; }

/* Config table */
table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
th, td { border: 1px solid #e5e7eb; padding: 6px 14px; text-align: left; }
th { background-color: #f0f2f6; font-weight: 600; }

/* Chart containers */
.plotly-graph-div { margin: 0 auto; }
"""


def create_html_report(config: RunConfig) -> None:
    """Generate an interactive HTML report for a training run.

    Reads all parquet data sources and produces a single self-contained HTML file
    with Plotly charts covering training, arena, and performance metrics.

    Args:
        config: The run configuration used for this training session.
    """
    logger.info("Writing report...")
    start = time.perf_counter()

    # Read all data sources
    loss_data = pd.read_parquet(config.training_data_directory)
    arena_data = pd.read_parquet(config.arena_data_directory)
    timings_data = pd.read_parquet(config.timings_directory)
    resource_data = pd.read_parquet(config.resource_usage_directory)
    profiling_data = pd.read_parquet(config.self_play_profiling_directory)
    throughput_data = pd.read_parquet(config.training_throughput_directory)

    # Build figures
    fig_loss_gen = _make_loss_per_generation(loss_data)
    fig_loss_timeline = _make_loss_timeline(loss_data)
    fig_arena = _make_arena_plot(arena_data, config.update_threshold)
    fig_timing = _make_timing_plot(timings_data)
    fig_throughput = _make_throughput_plot(throughput_data)
    fig_resources = _make_resource_usage_plot(resource_data)
    fig_profiling = _make_profiling_plot(profiling_data)

    # Write HTML
    filename = config.report_directory / "report.html"
    filename.parent.mkdir(exist_ok=True, parents=True)

    def _chart(fig: go.Figure) -> str:
        return fig.to_html(full_html=False, include_plotlyjs=False)

    today = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    kpi_html = _make_kpi_cards(loss_data, arena_data, timings_data, profiling_data, throughput_data)
    config_html = _make_config_table(config)

    with open(filename, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>AlphaBlokus Report — {config.run_name}</title>
    <script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
    <style>{_CSS}</style>
</head>
<body>

<h1>AlphaBlokus Report</h1>
<div class="subtitle">{config.game} &middot; {config.num_generations} generations \
&middot; {today}</div>

{kpi_html}

<section>
<h2>Training</h2>
<p class="section-desc">
    Policy and value loss should decrease across generations.
    Value loss below 1.0 means the network predicts outcomes better than random.
</p>
{_chart(fig_loss_gen)}
<details>
    <summary>Per-Batch Detail</summary>
    {_chart(fig_loss_timeline)}
</details>
</section>

<section>
<h2>Arena</h2>
<p class="section-desc">
    Each generation's new network plays the incumbent.
    It must win &gt;{config.update_threshold:.0%} to be accepted.
</p>
{_chart(fig_arena)}
</section>

<section>
<h2>Performance</h2>
<p class="section-desc">
    Time and resource usage per generation.
    Training time grows as the example buffer accumulates.
</p>
<div class="chart-grid">
{_chart(fig_timing)}
{_chart(fig_throughput)}
{_chart(fig_resources)}
{_chart(fig_profiling)}
</div>
</section>

<details>
    <summary>Configuration</summary>
    {config_html}
</details>

</body>
</html>
""")

    elapsed = time.perf_counter() - start
    logger.info("Wrote report in {:.2f}s", elapsed)
