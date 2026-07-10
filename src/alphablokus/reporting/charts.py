"""Plotly figure builders for the training-run report.

Pure functions: dataframe in, styled figure out. Shared style constants and
helpers live here so every chart in the report reads as one system.
"""

from __future__ import annotations

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

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
# KPI summary cards
# ---------------------------------------------------------------------------


def accepted_mask(arena_data: pd.DataFrame, update_threshold: float) -> pd.Series:
    """Return a boolean Series marking generations whose new net was accepted.

    Prefers the per-row ``accepted`` column persisted by
    :meth:`MetricsCollector.log_arena` — that's the ground truth direct
    from the training decision. If the column is missing (older runs
    persisted before the column existed) we fall back to recomputing via
    :func:`alphablokus.evaluation.acceptance.is_accepted_score_rule`, which is the **same
    function** the coach uses, so reporting can never diverge.
    """
    from alphablokus.evaluation.acceptance import is_accepted_score_rule

    if "accepted" in arena_data.columns:
        return arena_data["accepted"].fillna(False).astype(bool)
    return arena_data.apply(
        lambda row: is_accepted_score_rule(
            new_wins=int(row["wins"]),
            prev_wins=int(row["losses"]),
            draws=int(row["draws"]),
            threshold=update_threshold,
        ),
        axis=1,
    )


def make_loss_per_generation(df: pd.DataFrame) -> go.Figure:
    """Line chart with mean pi_loss, v_loss, and total_loss per generation.

    Each generation's value is the mean of its **last epoch's** per-batch
    losses — i.e. where the network is most trained for that gen. Aggregating
    raw per-batch values avoids the running-mean reset spikes that used to
    appear at epoch boundaries.
    """
    sorted_df = df.sort_values(["generation", "epoch", "batch_number"])
    last_epoch = sorted_df.groupby("generation")["epoch"].max()
    last_epoch_df = sorted_df[sorted_df["epoch"] == sorted_df["generation"].map(last_epoch)]
    agg = (
        last_epoch_df.groupby("generation")
        .agg(
            pi_loss=("pi_loss", "mean"),
            v_loss=("v_loss", "mean"),
            total_loss=("total_loss", "mean"),
        )
        .reset_index()
        .sort_values("generation")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["generation"],
            y=agg["total_loss"],
            mode="lines+markers",
            name="Total",
            line={"width": 3, "color": _COLORS["primary"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["generation"],
            y=agg["pi_loss"],
            mode="lines+markers",
            name="Policy",
            line={"width": 2, "color": _COLORS["secondary"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["generation"],
            y=agg["v_loss"],
            mode="lines+markers",
            name="Value",
            line={"width": 2, "color": _COLORS["tertiary"]},
        )
    )

    # "Worse than random" band for value loss
    max_v = float(agg["v_loss"].max())
    if max_v > 1.0:
        fig.add_hrect(
            y0=1.0,
            y1=max(max_v * 1.05, 1.4),
            fillcolor=_COLORS["negative"],
            opacity=0.06,
            line_width=0,
            annotation_text="Value loss > 1.0: worse than random",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=_COLORS["negative"],
        )

    fig.update_layout(xaxis_title="Generation", yaxis_title="Loss", title="Loss per Generation")
    return _apply_defaults(fig)


# ---------------------------------------------------------------------------
# Smoothed per-batch loss timeline
# ---------------------------------------------------------------------------


def make_loss_timeline(df: pd.DataFrame) -> go.Figure:
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
        fig.add_trace(
            go.Scatter(
                x=sorted_df["step"],
                y=sorted_df[col],
                mode="markers",
                marker={"size": 3, "color": color, "opacity": 0.25},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Smoothed line
        smoothed = sorted_df[col].ewm(span=span, adjust=False).mean()
        fig.add_trace(
            go.Scatter(
                x=sorted_df["step"],
                y=smoothed,
                mode="lines",
                name=name,
                line={"width": 2, "color": color},
            )
        )

    # Generation boundary lines
    gen_boundaries = sorted_df.groupby("generation")["step"].min()
    for gen, step in gen_boundaries.items():
        if step == 0:
            continue
        fig.add_vline(
            x=step,
            line_dash="dot",
            line_color=_COLORS["neutral"],
            line_width=1,
            annotation_text=f"Gen {gen}",
            annotation_position="top",
            annotation_font_size=9,
            annotation_font_color=_COLORS["neutral"],
        )

    fig.update_layout(
        xaxis_title="Training Step (sequential)",
        yaxis_title="Loss",
        title="Per-Batch Loss (smoothed)",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 1.0,
                "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [
                    {"label": "Linear", "method": "relayout", "args": [{"yaxis.type": "linear"}]},
                    {"label": "Log", "method": "relayout", "args": [{"yaxis.type": "log"}]},
                ],
            }
        ],
    )
    return _apply_defaults(fig)


def make_per_gen_loss_curves(df: pd.DataFrame) -> go.Figure:
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
    sample_positions = [i / max(n_gens - 1, 1) for i in range(n_gens)] if n_gens > 1 else [0.5]
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
        fig.add_trace(
            go.Scatter(
                x=gen_df["step_in_gen"],
                y=smoothed,
                mode="lines",
                line={"width": 1.5, "color": colour},
                name=f"Gen {gen}",
                showlegend=False,
                hovertemplate=f"Gen {gen}, step %{{x}}: loss %{{y:.3f}}<extra></extra>",
            )
        )

    # Invisible scatter trace just to carry the colorbar.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "size": 0.1,
                "colorscale": "Viridis",
                "cmin": min(gens),
                "cmax": max(gens),
                "color": [min(gens)],
                "showscale": True,
                "colorbar": {
                    "title": {"text": "Generation"},
                    "thickness": 12,
                    "len": 0.85,
                },
            },
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis_title="Batch within generation",
        yaxis_title="Total loss",
        title="Loss Curves by Generation (overlaid)",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 1.0,
                "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [
                    {"label": "Linear", "method": "relayout", "args": [{"yaxis.type": "linear"}]},
                    {"label": "Log", "method": "relayout", "args": [{"yaxis.type": "log"}]},
                ],
            }
        ],
    )
    return _apply_defaults(fig)


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------


def make_arena_plot(arena_data: pd.DataFrame, update_threshold: float) -> go.Figure:
    """Line chart of the per-generation acceptance score vs the threshold.

    The score (wins + ½ draws) is drawn as a continuous line and each
    generation is marked accepted (filled) or rejected (hollow), so the trend
    and the accept/reject pattern read at a glance even over hundreds of
    generations — the old stacked-bar-per-generation became an unreadable
    picket fence past ~30 gens. The full Wins/Losses composition is available
    as faint lines, hidden by default (toggle them on via the legend).
    """
    df = arena_data.sort_values("generation").copy()
    total = df["wins"] + df["losses"] + df["draws"]
    df["pct_wins"] = 100 * df["wins"] / total
    df["pct_losses"] = 100 * df["losses"] / total
    df["is_accepted"] = accepted_mask(df, update_threshold).values
    # Acceptance compares this SCORE (draws count as ½) to the threshold, not
    # raw wins. acceptance_score() is the very function the training loop uses,
    # so the chart and the decision cannot diverge.
    from alphablokus.evaluation.acceptance import acceptance_score

    df["pct_score"] = 100 * df.apply(
        lambda r: acceptance_score(int(r["wins"]), int(r["losses"]), int(r["draws"])),
        axis=1,
    )
    # Numeric x-axis (not categorical): Plotly then spaces gens evenly and
    # auto-thins the tick labels, which is what makes this scale to 150+ gens.
    gens = df["generation"].astype(int)

    fig = go.Figure()
    # Composition lines — full W/L picture, off by default to keep it clean.
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=df["pct_wins"],
            name="Wins %",
            mode="lines",
            line={"color": _COLORS["positive"], "width": 1},
            opacity=0.5,
            visible="legendonly",
            hovertemplate="Gen %{x} — Wins %{y:.0f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=df["pct_losses"],
            name="Losses %",
            mode="lines",
            line={"color": _COLORS["negative"], "width": 1},
            opacity=0.5,
            visible="legendonly",
            hovertemplate="Gen %{x} — Losses %{y:.0f}%<extra></extra>",
        )
    )
    # Primary: the acceptance-score line.
    fig.add_trace(
        go.Scatter(
            x=gens,
            y=df["pct_score"],
            name="Score (wins + ½ draws)",
            mode="lines",
            line={"color": "#333333", "width": 1.5},
            customdata=df[["wins", "losses", "draws"]].to_numpy(),
            hovertemplate=(
                "Gen %{x}<br>Score %{y:.0f}%<br>"
                "W%{customdata[0]} · L%{customdata[1]} · D%{customdata[2]}<extra></extra>"
            ),
        )
    )
    # Accept / reject markers on the score line — replaces the per-bar "✓
    # Accepted" text (which smeared together once accepts got dense).
    acc, rej = df[df["is_accepted"]], df[~df["is_accepted"]]
    fig.add_trace(
        go.Scatter(
            x=acc["generation"].astype(int),
            y=acc["pct_score"],
            name="Accepted",
            mode="markers",
            marker={
                "symbol": "circle",
                "size": 8,
                "color": _COLORS["positive"],
                "line": {"width": 1, "color": "#222222"},
            },
            hovertemplate="Gen %{x} ACCEPTED — score %{y:.0f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rej["generation"].astype(int),
            y=rej["pct_score"],
            name="Rejected",
            mode="markers",
            marker={"symbol": "circle-open", "size": 7, "color": _COLORS["negative"], "line": {"width": 1.5}},
            hovertemplate="Gen %{x} rejected — score %{y:.0f}%<extra></extra>",
        )
    )

    fig.add_hline(
        y=update_threshold * 100,
        line_dash="dash",
        line_color=_COLORS["primary"],
        annotation_text=f"Accept threshold ({update_threshold:.0%})",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Score / win-rate (%)",
        yaxis_range=[0, 105],
        title="Arena: New Net vs Predecessor",
    )
    return _apply_defaults(fig)


# ---------------------------------------------------------------------------
# Performance charts (half-width for grid)
# ---------------------------------------------------------------------------


def make_timing_plot(timings_data: pd.DataFrame) -> go.Figure:
    """Stacked bar of phase durations per generation."""
    df = timings_data[timings_data["cycle_stage"] != "WholeCycle"].copy()

    fig = go.Figure()
    for phase in ["SelfPlay", "Training", "Arena"]:
        phase_df = df[df["cycle_stage"] == phase]
        fig.add_trace(
            go.Bar(
                x=phase_df["generation"],
                y=phase_df["time_elapsed"],
                name=phase,
                marker_color=_PHASE_COLORS[phase],
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Generation",
        yaxis_title="Time (s)",
        title="Time per Generation",
    )
    return _apply_defaults(fig, width=_HALF_WIDTH, height=_GRID_HEIGHT)


def make_throughput_plot(throughput_data: pd.DataFrame) -> go.Figure:
    """Bar chart of average training throughput per generation."""
    agg = (
        throughput_data.groupby("generation")
        .agg(
            avg_throughput=("samples_per_second", "mean"),
        )
        .reset_index()
    )

    fig = go.Figure(
        go.Bar(
            x=agg["generation"],
            y=agg["avg_throughput"],
            marker_color=_COLORS["primary"],
        )
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Samples/s",
        title="Training Throughput",
    )
    return _apply_defaults(fig, width=_HALF_WIDTH, height=_GRID_HEIGHT)


def make_learning_rate_plot(learning_rate_data: pd.DataFrame) -> go.Figure:
    """Optimizer learning rate per generation, on a log-y axis.

    One point per generation (the mean across that generation's epochs — a
    no-op at the usual ``epochs=1``). Log-y because schedules span orders of
    magnitude; a constant schedule shows as a flat line. This is the definitive
    trace of what the optimiser actually trained at (L2).
    """
    agg = (
        learning_rate_data.groupby("generation")
        .agg(learning_rate=("learning_rate", "mean"))
        .reset_index()
        .sort_values("generation")
    )

    fig = go.Figure(
        go.Scatter(
            x=agg["generation"],
            y=agg["learning_rate"],
            mode="lines+markers",
            marker_color=_COLORS["accent"],
            name="Learning rate",
        )
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Learning rate",
        yaxis_type="log",
        title="Optimizer Learning Rate",
        xaxis={"dtick": 1 if agg["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig, width=_HALF_WIDTH, height=_GRID_HEIGHT)


def make_network_entropy_plot(entropy_data: pd.DataFrame) -> go.Figure:
    """Network policy entropy on the held-out eval set, per generation.

    Each generation contributes one point: the mean entropy across all
    training epochs in that generation. Lower entropy means the network is
    more confident in its move choices on a fixed reference set of positions.
    This is the cleanest "is the network itself learning?" signal because it
    isolates the network from MCTS noise.
    """
    df = entropy_data.copy()
    agg = (
        df.groupby("generation")
        .agg(
            mean=("mean_entropy", "mean"),
            std=("mean_entropy", "std"),
        )
        .reset_index()
        .sort_values("generation")
    )
    agg["std"] = agg["std"].fillna(0.0)

    fig = go.Figure()
    _mean_band_trace(
        fig,
        agg["generation"],
        agg["mean"],
        agg["std"],
        color=_COLORS["accent"],
        name="Network Entropy",
        unit="nats",
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Mean entropy (nats)",
        title="Network Policy Entropy on Held-Out Set",
        xaxis={"dtick": 1 if agg["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


# The replay viewer renders a full board-by-board card per move. A long run
# (e.g. 150 generations × 50 arena games × ~30 moves ≈ 230k cards) explodes
# report-gen memory — it OOM-killed run1's report during the live run. So we
# render a representative *sample*: generations spread evenly across the run
# (always keeping the first and last) and the first few games of each. The
# full record stays in the ArenaReplays parquets either way.


def make_elo_plot(rolling_elo_data: pd.DataFrame) -> go.Figure:
    """Rolling arena-derived Elo over generations — the live, non-saturating curve.

    Each generation the candidate is rated against the current arena incumbent
    (``candidate_elo = incumbent_elo + 400·log10(s/(1−s))``, from the same games
    the accept/reject arena already played); on acceptance the incumbent rolls
    forward, so the curve keeps climbing instead of flatlining at the ±1200 clamp
    the retired frozen-gen-0 metric hit. It is a *chained* estimate — rough, and
    the high-score steps are noisy on ~100 games — so read it as the live trend
    and defer to the end-of-run pool BayesElo curve for the rigorous rating.

    Accepted generations (the newly-trained net became the incumbent) are drawn
    as filled markers on the climbing line; rejected generations are hollow
    markers showing their *provisional* candidate Elo — the benchmark did not
    advance, so the next candidate is still measured against the same incumbent.
    The ``accepted`` flag is read straight from the rolling-Elo table (it's
    self-contained — no join with ArenaData needed).
    """
    df = rolling_elo_data.sort_values("generation").copy()
    df["accepted"] = df["accepted"].astype(bool)
    anchor = float(df["incumbent_elo"].iloc[0])

    fig = go.Figure()
    # Single connecting line through all gens, regardless of accept/reject.
    fig.add_trace(
        go.Scatter(
            x=df["generation"],
            y=df["rolling_elo"],
            mode="lines",
            name="",
            showlegend=False,
            line={"width": 2.5, "color": _COLORS["accent"]},
            hoverinfo="skip",
        )
    )
    hover_columns = ["elo_delta", "score_rate", "wins", "losses", "draws"]
    # Accepted-gen markers (filled) — the candidate became the new incumbent.
    accepted_df = df[df["accepted"]]
    fig.add_trace(
        go.Scatter(
            x=accepted_df["generation"],
            y=accepted_df["rolling_elo"],
            mode="markers",
            name="Accepted (new incumbent)",
            marker={"size": 9, "color": _COLORS["accent"], "symbol": "circle"},
            customdata=accepted_df[hover_columns].values,
            hovertemplate=(
                "Gen %{x} (accepted) — Elo: %{y:.0f} "
                "(%{customdata[0]:+.0f} vs incumbent)<br>"
                "Score: %{customdata[1]:.3f} "
                "(W%{customdata[2]} L%{customdata[3]} D%{customdata[4]})"
                "<extra></extra>"
            ),
        )
    )
    # Rejected-gen markers (open) — provisional candidate Elo; benchmark held.
    rejected_df = df[~df["accepted"]]
    if not rejected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=rejected_df["generation"],
                y=rejected_df["rolling_elo"],
                mode="markers",
                name="Rejected (provisional; benchmark held)",
                marker={
                    "size": 9,
                    "color": _COLORS["accent"],
                    "symbol": "circle-open",
                    "line": {"width": 2, "color": _COLORS["accent"]},
                },
                customdata=rejected_df[hover_columns].values,
                hovertemplate=(
                    "Gen %{x} (rejected — provisional candidate Elo, "
                    "benchmark not advanced) — Elo: %{y:.0f} "
                    "(%{customdata[0]:+.0f} vs incumbent)<br>"
                    "Score: %{customdata[1]:.3f} "
                    "(W%{customdata[2]} L%{customdata[3]} D%{customdata[4]})"
                    "<extra></extra>"
                ),
            )
        )
    fig.add_hline(
        y=anchor,
        line_dash="dash",
        line_color=_COLORS["neutral"],
        line_width=1,
        annotation_text=f"Anchor (start) = {anchor:.0f}",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color=_COLORS["neutral"],
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Elo rating",
        title="Rolling Arena-Derived Elo (non-saturating; anchored at start)",
        xaxis={"dtick": 1 if df["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def make_tournament_elo_plot(tournament_data: pd.DataFrame) -> go.Figure:
    """Pool BayesElo rating over generations — the non-saturating strength curve.

    Reads ``tournament_ratings.parquet`` (written by ``scripts/tournament_elo.py``):
    a sparse round-robin among the run's saved checkpoints, fit with BayesElo so
    every checkpoint gets one consistent rating on a shared scale. Unlike the
    vs-gen-0 chart above, this keeps rising until genuine convergence — it can
    separate gen 41 from gen 43 where the frozen-baseline number has flatlined at
    the ±1200 clamp. This is the DeepMind methodology and the curve to read.
    """
    df = tournament_data.sort_values("generation").copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["generation"],
            y=df["rating"],
            mode="lines+markers",
            name="Pool Elo",
            showlegend=False,
            line={"width": 2.5, "color": _COLORS["tertiary"]},
            marker={"size": 8, "color": _COLORS["tertiary"]},
            customdata=df[["n_games", "n_pairings"]].values,
            hovertemplate=(
                "Gen %{x} — pool Elo: %{y:.0f}<br>"
                "%{customdata[0]} games across %{customdata[1]} pairings"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Pool Elo rating",
        title="Pool Elo (BayesElo tournament)",
        xaxis={"dtick": 1 if df["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def make_minimax_plot(minimax_data: pd.DataFrame) -> go.Figure:
    """Vs perfect-play minimax (TTT only): draw rate and loss rate per gen.

    Against perfect play, TTT is a forced draw — so an optimal model should
    have ``draw_rate → 1.0`` and ``loss_rate → 0`` over training. Loss rate
    falling first, then draw rate rising as remaining wins disappear, is the
    canonical learning signature for a solved game.
    """
    df = minimax_data.sort_values("generation").copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["generation"],
            y=100 * df["draw_rate"],
            mode="lines+markers",
            name="Draw rate (target: 100%)",
            line={"width": 2.5, "color": _COLORS["tertiary"]},
            hovertemplate="Gen %{x} — draws: %{y:.0f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["generation"],
            y=100 * df["loss_rate"],
            mode="lines+markers",
            name="Loss rate (target: 0%)",
            line={"width": 2.5, "color": _COLORS["negative"], "dash": "dot"},
            hovertemplate="Gen %{x} — losses: %{y:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Rate (%)",
        yaxis_range=[-2, 102],
        title="Vs Perfect-Play Minimax (TTT)",
        xaxis={"dtick": 1 if df["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def make_policy_accuracy_plot(
    accuracy_data: pd.DataFrame,
    game_name: str,
) -> go.Figure:
    """Per-generation top-K agreement between the network's raw policy and two
    reference move choices on the frozen held-out set.

    **Frozen gen-1 targets** (always present) — the raw policy vs the targets
    baked into the eval set at generation 1:

    - **TTT**: minimax-optimal targets. A "hit" means the net picks a genuinely
      optimal move; top-1 should climb toward 100% as the net learns perfect play.
    - **Blokus / other**: the gen-1 64-sim MCTS visit-count argmax. This series
      *decays* over training and is **not a strength signal** — once the net
      surpasses gen-1's weak search it rightly disagrees with it (see
      docs/research/blokus-cloud-60-analysis.md §1). Kept for continuity.

    **Current-net MCTS** (present for runs that persisted compact eval boards) —
    the raw policy vs the *current* net's own search on the same positions: the
    net-vs-own-search gap, which should hold or rise as training works. This is
    the series to read as a learning-health signal.

    Computed on the frozen eval set after every training epoch; one point
    per generation is shown (mean across epochs).
    """
    df = accuracy_data.copy()
    agg_spec = {
        "top1_mean": ("top1_accuracy", "mean"),
        "top5_mean": ("top5_accuracy", "mean"),
    }
    has_mcts = "mcts_top1_accuracy" in df.columns and df["mcts_top1_accuracy"].notna().any()
    if has_mcts:
        agg_spec["mcts_top1_mean"] = ("mcts_top1_accuracy", "mean")
        agg_spec["mcts_top5_mean"] = ("mcts_top5_accuracy", "mean")
    agg = df.groupby("generation").agg(**agg_spec).reset_index().sort_values("generation")

    frozen_label = "minimax oracle" if game_name == "tictactoe" else "gen-1 MCTS targets"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["generation"],
            y=100 * agg["top1_mean"],
            mode="lines+markers",
            name=f"Top-1 vs {frozen_label}",
            line={"width": 2.5, "color": _COLORS["primary"]},
            hovertemplate="Gen %{x} — top-1 vs " + frozen_label + ": %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["generation"],
            y=100 * agg["top5_mean"],
            mode="lines+markers",
            name=f"Top-5 vs {frozen_label}",
            line={"width": 2.5, "color": _COLORS["primary"], "dash": "dot"},
            hovertemplate="Gen %{x} — top-5 vs " + frozen_label + ": %{y:.1f}%<extra></extra>",
        )
    )
    if has_mcts:
        fig.add_trace(
            go.Scatter(
                x=agg["generation"],
                y=100 * agg["mcts_top1_mean"],
                mode="lines+markers",
                name="Top-1 vs current-net MCTS",
                line={"width": 2.5, "color": _COLORS["tertiary"]},
                hovertemplate="Gen %{x} — top-1 vs current MCTS: %{y:.1f}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=agg["generation"],
                y=100 * agg["mcts_top5_mean"],
                mode="lines+markers",
                name="Top-5 vs current-net MCTS",
                line={"width": 2.5, "color": _COLORS["tertiary"], "dash": "dot"},
                hovertemplate="Gen %{x} — top-5 vs current MCTS: %{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Agreement (%)",
        yaxis_range=[0, 105],
        title="Policy Agreement: raw net vs search (held-out set)",
        xaxis={"dtick": 1 if agg["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def make_policy_value_consistency_plot(pvc_data: pd.DataFrame) -> go.Figure:
    """Policy–value consistency (PVC) over generations.

    Two agreement series between the policy head and a one-ply value lookahead
    (``Q₁(a) = −V(child)``) on the frozen held-out set, each a mean across the
    generation's training epochs:

    - **Argmax-match** (0–1): fraction of positions where the policy's best move
      is also the ``Q₁``-best move.
    - **Spearman** (−1 to 1): mean rank correlation between ``π`` and ``Q₁``
      across each position's top-K candidate moves.

    Read it as a **trend, not a target**. The policy is trained on multi-ply
    MCTS visits while ``Q₁`` is a single ply, so a healthy net rises early then
    plateaus *below* perfect agreement — the residual is roughly how much deeper
    the policy sees than one-ply value. A late drop or a persistently low level
    is the red flag (value head lagging, or policy chasing lines the value head
    doesn't support).
    """
    df = pvc_data.copy()
    agg = (
        df.groupby("generation")
        .agg(
            argmax_match_mean=("pvc_argmax_match", "mean"),
            spearman_mean=("pvc_spearman", "mean"),
        )
        .reset_index()
        .sort_values("generation")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["generation"],
            y=agg["argmax_match_mean"],
            mode="lines+markers",
            name="Argmax-match (policy top = Q₁ top)",
            line={"width": 2.5, "color": _COLORS["primary"]},
            hovertemplate="Gen %{x} — argmax-match: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["generation"],
            y=agg["spearman_mean"],
            mode="lines+markers",
            name="Spearman (π vs Q₁ ranking)",
            line={"width": 2.5, "color": _COLORS["tertiary"]},
            hovertemplate="Gen %{x} — Spearman: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.0,
        line_width=1,
        line_dash="dash",
        line_color=_COLORS["neutral"],
        annotation_text="No rank correlation",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Agreement (argmax fraction / Spearman ρ)",
        yaxis_range=[-1.05, 1.05],
        title="Policy–Value Consistency (one-ply lookahead)",
        xaxis={"dtick": 1 if agg["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def make_value_calibration_plot(
    calibration_data: pd.DataFrame,
    game_name: str,
) -> go.Figure:
    """Reliability diagram for the value head, latest epoch.

    Predicted v ∈ [-1, 1] is binned into 10 buckets; each marker is the mean
    *target* value of positions whose predicted v fell in that bucket. The
    y-axis target depends on the game:

    - **TTT**: target is the position's true minimax value ∈ ``{-1, 0, +1}``.
      Markers should hug the y=x diagonal — a winning position should be
      predicted near +1, a losing one near -1, a drawn one near 0.
    - **Blokus / other**: target is the actual game outcome ``z`` recorded
      in the eval set's self-play games. y=x is still the well-calibrated
      reference, but expect more noise since outcomes are post-hoc and
      depend on both players' downstream play.
    """
    df = calibration_data.copy()
    last_gen = df["generation"].max()
    last_epoch = df[df["generation"] == last_gen]["epoch"].max()
    latest = df[(df["generation"] == last_gen) & (df["epoch"] == last_epoch)]
    latest = latest.dropna(subset=["bucket_mean_actual"]).sort_values("bucket_center")

    if game_name == "tictactoe":
        y_label = "Mean true minimax value (-1 / 0 / +1)"
        title = "Value-Head Reliability vs Minimax (latest epoch)"
        hover_label = "minimax mean"
    else:
        y_label = "Mean actual game outcome"
        title = "Value-Head Reliability (latest epoch)"
        hover_label = "outcome mean"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-1, 1],
            y=[-1, 1],
            mode="lines",
            name="Perfect calibration (y=x)",
            line={"dash": "dash", "color": _COLORS["neutral"], "width": 1},
            hoverinfo="skip",
        )
    )
    max_count = max(int(latest["bucket_count"].max()), 1)
    sizes = 6 + 24 * latest["bucket_count"] / max_count
    fig.add_trace(
        go.Scatter(
            x=latest["bucket_center"],
            y=latest["bucket_mean_actual"],
            mode="markers+lines",
            name=f"Gen {int(last_gen)} epoch {int(last_epoch)}",
            marker={
                "size": sizes,
                "color": _COLORS["accent"],
                "line": {"width": 1, "color": _COLORS["accent"]},
            },
            customdata=latest["bucket_count"],
            hovertemplate=(
                "Predicted ≈ %{x:.1f}, " + hover_label + ": %{y:.2f} (%{customdata} positions)<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Predicted value (bucket centre)",
        yaxis_title=y_label,
        title=title,
        xaxis_range=[-1.05, 1.05],
        yaxis_range=[-1.05, 1.05],
    )
    return _apply_defaults(fig)


def make_symmetry_diagnostic_plot(symmetry_data: pd.DataFrame) -> go.Figure:
    """Per-generation KL divergence between the network's raw policy and
    its symmetric counterpart, averaged across reference positions.

    Zero is the target — a perfectly equivariant network gives the same
    distribution (modulo coordinate transformation) on a board and its
    symmetric variants. Persistent non-zero values indicate the network
    has internalised arbitrary directional biases that the symmetry-
    augmentation training signal isn't fully averaging out (the "favourite
    corner" effect Henry first spotted in the TTT report).

    Per-position lines are overlaid on the mean to surface positions that
    are particularly noisy.
    """
    df = symmetry_data.copy()
    df = df.sort_values(["generation", "position_idx", "symmetry_idx"])
    # Mean across symmetries for each (gen, position)
    per_position = (
        df.groupby(["generation", "position_idx"]).agg(position_mean_kl=("kl_divergence", "mean")).reset_index()
    )
    overall = (
        per_position.groupby("generation")
        .agg(mean_kl=("position_mean_kl", "mean"), max_kl=("position_mean_kl", "max"))
        .reset_index()
        .sort_values("generation")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=overall["generation"],
            y=overall["mean_kl"],
            mode="lines+markers",
            name="Mean across positions",
            line={"width": 2.5, "color": _COLORS["primary"]},
            hovertemplate="Gen %{x} — mean KL: %{y:.4f}<extra></extra>",
        )
    )
    for pos_idx, group in per_position.sort_values("position_idx").groupby("position_idx"):
        fig.add_trace(
            go.Scatter(
                x=group["generation"],
                y=group["position_mean_kl"],
                mode="lines",
                name=f"Position {pos_idx}",
                line={"width": 1, "dash": "dot", "color": _COLORS["neutral"]},
                opacity=0.5,
                hoverinfo="skip",
                showlegend=True,
            )
        )
    fig.add_hline(
        y=0.0,
        line_width=1,
        line_dash="dash",
        line_color=_COLORS["neutral"],
        annotation_text="Perfect equivariance",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="KL divergence (lower = more symmetric)",
        title="Network policy symmetry diagnostic",
        xaxis={"dtick": 1 if overall["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def make_resource_usage_plot(resource_data: pd.DataFrame) -> go.Figure:
    """Line chart of memory usage over generations — one line per phase.

    Replaces the previous 90-bar grouped chart, which was unreadable for runs
    with more than a few generations.
    """
    df = resource_data.copy()
    df["rss_mb"] = df["process_rss_bytes"] / (1024**2)
    df = df.sort_values("generation")

    has_gpu = (
        "gpu_memory_bytes" in df.columns
        and df["gpu_memory_bytes"].notna().any()
        and (df["gpu_memory_bytes"].fillna(0) > 0).any()
    )
    if has_gpu:
        df["gpu_mb"] = df["gpu_memory_bytes"] / (1024**2)

    fig = go.Figure()
    for stage in ["SelfPlay", "Save", "Training", "Arena"]:
        stage_df = df[df["cycle_stage"] == stage]
        if stage_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=stage_df["generation"],
                y=stage_df["rss_mb"],
                mode="lines+markers",
                name=f"{stage} (RSS)",
                line={"width": 2, "color": _PHASE_COLORS.get(stage, _COLORS["neutral"])},
            )
        )
        if has_gpu:
            fig.add_trace(
                go.Scatter(
                    x=stage_df["generation"],
                    y=stage_df["gpu_mb"],
                    mode="lines+markers",
                    name=f"{stage} (GPU)",
                    line={"width": 2, "dash": "dot", "color": _PHASE_COLORS.get(stage, _COLORS["neutral"])},
                    showlegend=True,
                )
            )

    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Memory (MB)",
        title="Memory Usage" if has_gpu else "Process Memory (RSS)",
    )
    return _apply_defaults(fig, width=_HALF_WIDTH, height=_GRID_HEIGHT)


def make_profiling_plot(profiling_data: pd.DataFrame) -> go.Figure:
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
        df["mean_policy_entropy"] = 0.0  # backward compat with older runs
    agg = (
        df.groupby("generation")
        .agg(
            moves_mean=("num_moves", "mean"),
            moves_std=("num_moves", "std"),
            sims_mean=("sims_per_second", "mean"),
            sims_std=("sims_per_second", "std"),
            entropy_mean=("mean_policy_entropy", "mean"),
            entropy_std=("mean_policy_entropy", "std"),
        )
        .reset_index()
        .sort_values("generation")
    )
    for col in ("moves_std", "sims_std", "entropy_std"):
        agg[col] = agg[col].fillna(0.0)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Game Length (moves)",
            "MCTS Throughput (sims/s)",
            "MCTS Policy Entropy (nats)",
        ],
        vertical_spacing=0.12,
    )

    _mean_band_trace(
        fig,
        agg["generation"],
        agg["moves_mean"],
        agg["moves_std"],
        color=_COLORS["primary"],
        name="Game Length",
        unit="moves",
        row=1,
        col=1,
    )
    _mean_band_trace(
        fig,
        agg["generation"],
        agg["sims_mean"],
        agg["sims_std"],
        color=_COLORS["secondary"],
        name="Sims/s",
        unit="sims/s",
        row=2,
        col=1,
    )
    _mean_band_trace(
        fig,
        agg["generation"],
        agg["entropy_mean"],
        agg["entropy_std"],
        color=_COLORS["accent"],
        name="Policy Entropy",
        unit="nats",
        row=3,
        col=1,
    )

    fig.update_xaxes(title_text="Generation", row=3, col=1)
    for r in (1, 2, 3):
        fig.update_xaxes(
            row=r,
            col=1,
            dtick=1 if agg["generation"].max() < 40 else 5,
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
            x=x,
            y=mean,
            mode="lines+markers",
            name=name,
            line={"width": 2.5, "color": color},
            marker={"size": 5},
            customdata=std,
            hovertemplate=(f"Gen %{{x}} — mean: %{{y:.2f}} {unit}, ± %{{customdata:.2f}}<extra></extra>"),
        ),
        **target,
    )


# ---------------------------------------------------------------------------
# Config table (unchanged content, moved to bottom)
# ---------------------------------------------------------------------------
