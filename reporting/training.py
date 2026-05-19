from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from plotly import graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from core.config import RunConfig


def _load_metrics(directory: Path) -> pd.DataFrame:
    """Read a hive-partitioned metrics directory and normalise the generation column.

    Pandas/PyArrow infer hive-partition keys as ``category`` dtype, which sorts
    by the order categories were inserted (effectively alphabetical: 1, 10, 11,
    ..., 2, 20). Casting to ``int`` here means every downstream ``sort_values``
    and groupby produces numerically correct results.
    """
    df = pd.read_parquet(directory)
    if "generation" in df.columns:
        df["generation"] = df["generation"].astype(int)
    return df

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


def _accepted_mask(arena_data: pd.DataFrame, update_threshold: float) -> pd.Series:
    """Return a boolean Series marking generations whose new net was accepted.

    Matches ``Coach._should_accept_new_network``: draws are excluded from the
    denominator, the win rate uses ``wins / (wins + losses)``, and the
    comparison is ``>= update_threshold`` (not a hardcoded 0.5).
    """
    decided = arena_data["wins"] + arena_data["losses"]
    win_rate = arena_data["wins"].astype(float) / decided.where(decided > 0, 1)
    return (decided > 0) & (win_rate >= update_threshold)


def _make_kpi_cards(
    loss_data: pd.DataFrame,
    arena_data: pd.DataFrame,
    timings_data: pd.DataFrame,
    profiling_data: pd.DataFrame,
    throughput_data: pd.DataFrame,
    update_threshold: float,
) -> str:
    """D2: Build HTML for the KPI card row."""
    # Final loss + delta — mean of each generation's last epoch (the
    # epoch where the network is most trained). Avoids the noise of a
    # single trailing batch and the artefact of running-mean resets.
    sorted_loss = loss_data.sort_values(["generation", "epoch", "batch_number"])
    last_epoch_per_gen = sorted_loss.groupby("generation")["epoch"].max()
    by_gen = sorted_loss[
        sorted_loss["epoch"] == sorted_loss["generation"].map(last_epoch_per_gen)
    ].groupby("generation")["total_loss"].mean()
    final_loss = by_gen.iloc[-1]
    first_loss = by_gen.iloc[0]
    loss_delta_pct = ((final_loss - first_loss) / first_loss) * 100

    # Accept rate — matches Coach._should_accept_new_network (draws excluded,
    # configured threshold used, not a hardcoded 0.5).
    accepted = _accepted_mask(arena_data, update_threshold)
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
    """Line chart with mean pi_loss, v_loss, and total_loss per generation.

    Each generation's value is the mean of its **last epoch's** per-batch
    losses — i.e. where the network is most trained for that gen. Aggregating
    raw per-batch values avoids the running-mean reset spikes that used to
    appear at epoch boundaries.
    """
    sorted_df = df.sort_values(["generation", "epoch", "batch_number"])
    last_epoch = sorted_df.groupby("generation")["epoch"].max()
    last_epoch_df = sorted_df[
        sorted_df["epoch"] == sorted_df["generation"].map(last_epoch)
    ]
    agg = last_epoch_df.groupby("generation").agg(
        pi_loss=("pi_loss", "mean"),
        v_loss=("v_loss", "mean"),
        total_loss=("total_loss", "mean"),
    ).reset_index().sort_values("generation")

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
    """Smoothed loss over the full training timeline with generation boundaries.

    Convention follows W&B / TensorBoard: raw per-batch values plotted as
    semi-transparent dots, EWM-smoothed line on top in full saturation.
    A Linear/Log button lets the reader switch the y-axis scale — useful for
    long runs where loss drops by multiple orders of magnitude.
    """
    sorted_df = df.sort_values(["generation", "epoch", "batch_number"]).reset_index(drop=True)
    sorted_df["step"] = range(len(sorted_df))

    span = max(5, len(sorted_df) // 15)

    # Raw dots at opacity 0.4 (not 0.15) so the reader can see the noise
    # underneath the smoothing — not a hidden detail, an explicit one.
    series = [
        ("total_loss", "Total", _COLORS["primary"]),
        ("pi_loss", "Policy", _COLORS["secondary"]),
        ("v_loss", "Value", _COLORS["tertiary"]),
    ]

    fig = go.Figure()

    for col, name, color in series:
        # Raw data — visible context, secondary to the smoothed line.
        fig.add_trace(go.Scatter(
            x=sorted_df["step"], y=sorted_df[col],
            mode="markers", marker={"size": 3, "color": color, "opacity": 0.25},
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
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "x": 1.0, "y": 1.15, "xanchor": "right", "yanchor": "top",
            "buttons": [
                {"label": "Linear", "method": "relayout",
                 "args": [{"yaxis.type": "linear"}]},
                {"label": "Log", "method": "relayout",
                 "args": [{"yaxis.type": "log"}]},
            ],
        }],
    )
    return _apply_defaults(fig)


def _make_per_gen_loss_curves(df: pd.DataFrame) -> go.Figure:
    """One smoothed total-loss curve per generation, coloured by generation.

    Reads at a glance: are later generations starting at lower loss? Reaching
    convergence faster? Is the curve shape consistent across gens? These are
    questions the per-batch timeline and the per-gen summary chart don't
    directly answer — this overlay does.

    X axis: training step within a generation (sequential batch index across
    that gen's epochs). Y axis: total loss (smoothed lightly with EWM).
    Colour: viridis gradient from earliest gen (deep purple) to latest gen
    (bright yellow). Colorbar on the right labels the gradient.
    """
    import plotly.colors as pc  # local import — only used here

    sorted_df = df.sort_values(["generation", "epoch", "batch_number"]).copy()
    sorted_df["step_in_gen"] = sorted_df.groupby("generation").cumcount()

    gens = sorted(sorted_df["generation"].unique())
    n_gens = len(gens)

    # Sample the Viridis colorscale at n_gens evenly-spaced points.
    sample_positions = (
        [i / max(n_gens - 1, 1) for i in range(n_gens)] if n_gens > 1 else [0.5]
    )
    palette = pc.sample_colorscale("Viridis", sample_positions)

    fig = go.Figure()
    for gen, colour in zip(gens, palette, strict=True):
        gen_df = sorted_df[sorted_df["generation"] == gen]
        if len(gen_df) < 2:
            continue
        # Light per-gen smoothing — short span so we don't over-flatten the
        # within-gen learning curve we're trying to see.
        span = max(3, len(gen_df) // 10)
        smoothed = gen_df["total_loss"].ewm(span=span, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=gen_df["step_in_gen"], y=smoothed,
            mode="lines",
            line={"width": 1.5, "color": colour},
            name=f"Gen {gen}",
            showlegend=False,
            hovertemplate=f"Gen {gen}, step %{{x}}: loss %{{y:.3f}}<extra></extra>",
        ))

    # Invisible scatter trace just to carry the colorbar.
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker={
            "size": 0.1,
            "colorscale": "Viridis",
            "cmin": min(gens), "cmax": max(gens),
            "color": [min(gens)],
            "showscale": True,
            "colorbar": {
                "title": {"text": "Generation"},
                "thickness": 12, "len": 0.85,
            },
        },
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        xaxis_title="Batch within generation",
        yaxis_title="Total loss",
        title="Loss Curves by Generation (overlaid)",
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "x": 1.0, "y": 1.15, "xanchor": "right", "yanchor": "top",
            "buttons": [
                {"label": "Linear", "method": "relayout",
                 "args": [{"yaxis.type": "linear"}]},
                {"label": "Log", "method": "relayout",
                 "args": [{"yaxis.type": "log"}]},
            ],
        }],
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
    df["is_accepted"] = _accepted_mask(df, update_threshold).values

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

    # Only annotate ACCEPTED generations — rejected is the default expectation
    # and labelling every bar caused overlap with 30+ generations.
    for _, row in df[df["is_accepted"]].iterrows():
        fig.add_annotation(
            x=row["generation"], y=108, text="✓ Accepted", showarrow=False,
            font={"size": 11, "color": _COLORS["positive"], "family": "monospace"},
        )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Generation", yaxis_title="Percentage", yaxis_range=[0, 120],
        xaxis={"type": "category", "categoryorder": "array",
               "categoryarray": [str(g) for g in df["generation"]]},
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


def _make_network_entropy_plot(entropy_data: pd.DataFrame) -> go.Figure:
    """Network policy entropy on the held-out eval set, per generation.

    Each generation contributes one point: the mean entropy across all
    training epochs in that generation. Lower entropy means the network is
    more confident in its move choices on a fixed reference set of positions.
    This is the cleanest "is the network itself learning?" signal because it
    isolates the network from MCTS noise.
    """
    df = entropy_data.copy()
    agg = df.groupby("generation").agg(
        mean=("mean_entropy", "mean"),
        std=("mean_entropy", "std"),
    ).reset_index().sort_values("generation")
    agg["std"] = agg["std"].fillna(0.0)

    fig = go.Figure()
    _mean_band_trace(
        fig, agg["generation"], agg["mean"], agg["std"],
        color=_COLORS["accent"], name="Network Entropy",
        unit="nats",
    )
    fig.update_layout(
        xaxis_title="Generation", yaxis_title="Mean entropy (nats)",
        title="Network Policy Entropy on Held-Out Set",
        xaxis={"dtick": 1 if agg["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def _make_policy_accuracy_plot(accuracy_data: pd.DataFrame) -> go.Figure:
    """Network top-1 / top-5 policy accuracy vs MCTS targets, per generation.

    Computed on the frozen eval set after every training epoch; one point
    per generation is shown (mean across epochs). Top-1 should rise from
    1/num_actions toward 1.0 as the network internalises MCTS preferences;
    Top-5 typically saturates fast on small action spaces.
    """
    df = accuracy_data.copy()
    agg = df.groupby("generation").agg(
        top1_mean=("top1_accuracy", "mean"),
        top5_mean=("top5_accuracy", "mean"),
    ).reset_index().sort_values("generation")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["generation"], y=100 * agg["top1_mean"],
        mode="lines+markers", name="Top-1",
        line={"width": 2.5, "color": _COLORS["primary"]},
        hovertemplate="Gen %{x} — top-1: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=agg["generation"], y=100 * agg["top5_mean"],
        mode="lines+markers", name="Top-5",
        line={"width": 2.5, "color": _COLORS["tertiary"], "dash": "dot"},
        hovertemplate="Gen %{x} — top-5: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Generation", yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        title="Policy Accuracy vs MCTS (held-out set)",
        xaxis={"dtick": 1 if agg["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def _make_value_calibration_plot(calibration_data: pd.DataFrame) -> go.Figure:
    """Reliability diagram for the value head, taken from the final epoch of
    the latest generation. Buckets predicted v ∈ [-1, 1] and plots mean(actual
    outcome) per bucket. Well-calibrated → points sit on the y=x diagonal.
    """
    df = calibration_data.copy()
    last_gen = df["generation"].max()
    last_epoch = df[df["generation"] == last_gen]["epoch"].max()
    latest = df[(df["generation"] == last_gen) & (df["epoch"] == last_epoch)]
    latest = latest.dropna(subset=["bucket_mean_actual"]).sort_values("bucket_center")

    fig = go.Figure()
    # Perfect-calibration diagonal
    fig.add_trace(go.Scatter(
        x=[-1, 1], y=[-1, 1],
        mode="lines", name="Perfect calibration",
        line={"dash": "dash", "color": _COLORS["neutral"], "width": 1},
        hoverinfo="skip",
    ))
    # Bucket means — marker size proportional to bucket count
    max_count = max(int(latest["bucket_count"].max()), 1)
    sizes = 6 + 24 * latest["bucket_count"] / max_count
    fig.add_trace(go.Scatter(
        x=latest["bucket_center"], y=latest["bucket_mean_actual"],
        mode="markers+lines", name=f"Gen {int(last_gen)} epoch {int(last_epoch)}",
        marker={
            "size": sizes,
            "color": _COLORS["accent"],
            "line": {"width": 1, "color": _COLORS["accent"]},
        },
        customdata=latest["bucket_count"],
        hovertemplate=(
            "Predicted ≈ %{x:.1f}, actual mean: %{y:.2f}"
            " (%{customdata} positions)<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title="Predicted value (bucket centre)",
        yaxis_title="Mean actual outcome",
        title="Value-Head Reliability (latest epoch)",
        xaxis_range=[-1.05, 1.05], yaxis_range=[-1.05, 1.05],
    )
    return _apply_defaults(fig)


def _make_resource_usage_plot(resource_data: pd.DataFrame) -> go.Figure:
    """Line chart of memory usage over generations — one line per phase.

    Replaces the previous 90-bar grouped chart, which was unreadable for runs
    with more than a few generations.
    """
    df = resource_data.copy()
    df["rss_mb"] = df["process_rss_bytes"] / (1024 ** 2)
    df = df.sort_values("generation")

    has_gpu = (
        "gpu_memory_bytes" in df.columns
        and df["gpu_memory_bytes"].notna().any()
        and (df["gpu_memory_bytes"].fillna(0) > 0).any()
    )
    if has_gpu:
        df["gpu_mb"] = df["gpu_memory_bytes"] / (1024 ** 2)

    fig = go.Figure()
    for stage in ["SelfPlay", "Training", "Arena"]:
        stage_df = df[df["cycle_stage"] == stage]
        if stage_df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=stage_df["generation"], y=stage_df["rss_mb"],
            mode="lines+markers", name=f"{stage} (RSS)",
            line={"width": 2, "color": _PHASE_COLORS.get(stage, _COLORS["neutral"])},
        ))
        if has_gpu:
            fig.add_trace(go.Scatter(
                x=stage_df["generation"], y=stage_df["gpu_mb"],
                mode="lines+markers", name=f"{stage} (GPU)",
                line={"width": 2, "dash": "dot",
                      "color": _PHASE_COLORS.get(stage, _COLORS["neutral"])},
                showlegend=True,
            ))

    fig.update_layout(
        xaxis_title="Generation", yaxis_title="Memory (MB)",
        title="Memory Usage" if has_gpu else "Process Memory (RSS)",
    )
    return _apply_defaults(fig, width=_HALF_WIDTH, height=_GRID_HEIGHT)


def _make_profiling_plot(profiling_data: pd.DataFrame) -> go.Figure:
    """Self-play profiling — mean line + std band per generation for three diagnostics.

    For each metric the headline signal is the *trend across generations*, not
    the per-gen distribution shape. Mean line + one-sigma shaded band reads at
    a glance and avoids the blocky look violins produce on near-discrete data
    (e.g. TTT game lengths only take values 5-9).

    Three rows:
    1. Game length (moves per game).
    2. MCTS throughput (sims/s).
    3. MCTS policy entropy (nats, on the pre-temperature visit distribution).
       Should drop over training as the model becomes more confident.
    """
    df = profiling_data.copy()
    if "mean_policy_entropy" not in df.columns:
        df["mean_policy_entropy"] = 0.0  # backward compat with pre-R8 runs
    agg = df.groupby("generation").agg(
        moves_mean=("num_moves", "mean"),
        moves_std=("num_moves", "std"),
        sims_mean=("sims_per_second", "mean"),
        sims_std=("sims_per_second", "std"),
        entropy_mean=("mean_policy_entropy", "mean"),
        entropy_std=("mean_policy_entropy", "std"),
    ).reset_index().sort_values("generation")
    for col in ("moves_std", "sims_std", "entropy_std"):
        agg[col] = agg[col].fillna(0.0)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Game Length (moves)",
            "MCTS Throughput (sims/s)",
            "MCTS Policy Entropy (nats)",
        ],
        vertical_spacing=0.12,
    )

    _mean_band_trace(
        fig, agg["generation"], agg["moves_mean"], agg["moves_std"],
        color=_COLORS["primary"], name="Game Length",
        unit="moves", row=1, col=1,
    )
    _mean_band_trace(
        fig, agg["generation"], agg["sims_mean"], agg["sims_std"],
        color=_COLORS["secondary"], name="Sims/s",
        unit="sims/s", row=2, col=1,
    )
    _mean_band_trace(
        fig, agg["generation"], agg["entropy_mean"], agg["entropy_std"],
        color=_COLORS["accent"], name="Policy Entropy",
        unit="nats", row=3, col=1,
    )

    fig.update_xaxes(title_text="Generation", row=3, col=1)
    for r in (1, 2, 3):
        fig.update_xaxes(
            row=r, col=1, dtick=1 if agg["generation"].max() < 40 else 5,
        )
    fig.update_layout(title="Self-Play Profiling", showlegend=False)
    return _apply_defaults(fig, width=_HALF_WIDTH, height=int(_GRID_HEIGHT * 1.4))


def _mean_band_trace(
    fig: go.Figure,
    x: pd.Series,
    mean: pd.Series,
    std: pd.Series,
    *,
    color: str,
    name: str,
    unit: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add a mean line + one-sigma shaded band to a figure.

    Pass ``row`` and ``col`` to target a subplot cell; omit them for a
    single-plot figure (no ``make_subplots`` grid).

    Hover shows ``Gen N — mean: X (unit), ± std`` for the active point.
    """
    upper = mean + std
    lower = mean - std

    target = {"row": row, "col": col} if row is not None and col is not None else {}

    # Band (drawn first so the mean line sits on top)
    fig.add_trace(
        go.Scatter(
            x=pd.concat([x, x[::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor=color,
            opacity=0.18,
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        ),
        **target,
    )
    # Mean line
    fig.add_trace(
        go.Scatter(
            x=x, y=mean,
            mode="lines+markers", name=name,
            line={"width": 2.5, "color": color},
            marker={"size": 5},
            customdata=std,
            hovertemplate=(
                f"Gen %{{x}} — mean: %{{y:.2f}} {unit}, "
                f"± %{{customdata:.2f}}<extra></extra>"
            ),
        ),
        **target,
    )


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

    # Read all data sources — _load_metrics casts the hive-partitioned
    # `generation` column from category dtype to int so sorts are numeric.
    loss_data = _load_metrics(config.training_data_directory)
    arena_data = _load_metrics(config.arena_data_directory)
    timings_data = _load_metrics(config.timings_directory)
    resource_data = _load_metrics(config.resource_usage_directory)
    profiling_data = _load_metrics(config.self_play_profiling_directory)
    throughput_data = _load_metrics(config.training_throughput_directory)
    # Optional — only populated when an eval set was built (i.e. recent runs).
    network_entropy_data = (
        _load_metrics(config.training_entropy_directory)
        if config.training_entropy_directory.exists() else None
    )
    policy_accuracy_data = (
        _load_metrics(config.policy_accuracy_directory)
        if config.policy_accuracy_directory.exists() else None
    )
    value_calibration_data = (
        _load_metrics(config.value_calibration_directory)
        if config.value_calibration_directory.exists() else None
    )

    # Build figures
    fig_loss_gen = _make_loss_per_generation(loss_data)
    fig_loss_timeline = _make_loss_timeline(loss_data)
    fig_per_gen_curves = _make_per_gen_loss_curves(loss_data)
    fig_arena = _make_arena_plot(arena_data, config.update_threshold)
    fig_timing = _make_timing_plot(timings_data)
    fig_throughput = _make_throughput_plot(throughput_data)
    fig_resources = _make_resource_usage_plot(resource_data)
    fig_profiling = _make_profiling_plot(profiling_data)
    fig_network_entropy = (
        _make_network_entropy_plot(network_entropy_data)
        if network_entropy_data is not None and not network_entropy_data.empty
        else None
    )
    fig_policy_accuracy = (
        _make_policy_accuracy_plot(policy_accuracy_data)
        if policy_accuracy_data is not None and not policy_accuracy_data.empty
        else None
    )
    fig_value_calibration = (
        _make_value_calibration_plot(value_calibration_data)
        if value_calibration_data is not None and not value_calibration_data.empty
        else None
    )

    # Write HTML
    filename = config.report_directory / "report.html"
    filename.parent.mkdir(exist_ok=True, parents=True)

    def _chart(fig: go.Figure) -> str:
        return fig.to_html(full_html=False, include_plotlyjs=False)

    today = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    kpi_html = _make_kpi_cards(
        loss_data, arena_data, timings_data, profiling_data, throughput_data,
        update_threshold=config.update_threshold,
    )
    config_html = _make_config_table(config)

    diagnostics_html = ""
    if (
        fig_network_entropy is not None
        or fig_policy_accuracy is not None
        or fig_value_calibration is not None
    ):
        parts = [
            '<section>',
            '<h2>Network Diagnostics</h2>',
            '<p class="section-desc">'
            'Per-epoch evaluation of the network alone (no MCTS) on a frozen '
            'held-out set of positions sampled from generation 1\'s self-play. '
            'These are the AlphaZero-style training health curves — they '
            'isolate the network\'s learning from MCTS noise.'
            '</p>',
        ]
        if fig_network_entropy is not None:
            parts.append(_chart(fig_network_entropy))
        if fig_policy_accuracy is not None:
            parts.append(_chart(fig_policy_accuracy))
        if fig_value_calibration is not None:
            parts.append(_chart(fig_value_calibration))
        parts.append('</section>')
        diagnostics_html = "\n".join(parts)

    network_entropy_html = ""  # rolled into diagnostics_html above

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
    <p class="section-desc" style="margin-top:18px;">
        Overlay of each generation's smoothed training curve. Colours run from
        earliest gen (purple) to latest (yellow). If later generations start
        at lower loss or converge faster, the curves drift down and steepen
        with colour — direct visual answer to "is the network getting better
        at learning the task itself over time?"
    </p>
    {_chart(fig_per_gen_curves)}
</details>
</section>

{diagnostics_html}

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
