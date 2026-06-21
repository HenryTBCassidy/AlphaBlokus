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
# Consistent color palette and chart defaults
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
# KPI summary cards
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

    Prefers the per-row ``accepted`` column persisted by
    :meth:`MetricsCollector.log_arena` — that's the ground truth direct
    from the training decision. If the column is missing (older runs
    persisted before the column existed) we fall back to recomputing via
    :func:`core.acceptance.is_accepted_score_rule`, which is the **same
    function** the coach uses, so reporting can never diverge.
    """
    from core.acceptance import is_accepted_score_rule
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


def _make_kpi_cards(
    loss_data: pd.DataFrame,
    arena_data: pd.DataFrame,
    timings_data: pd.DataFrame,
    profiling_data: pd.DataFrame,
    throughput_data: pd.DataFrame,
    update_threshold: float,
) -> str:
    """Build HTML for the KPI card row."""
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
# Loss per generation
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
# Smoothed per-batch loss timeline
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
    df["is_accepted"] = _accepted_mask(df, update_threshold).values
    # Acceptance compares this SCORE (draws count as ½) to the threshold, not
    # raw wins. acceptance_score() is the very function the training loop uses,
    # so the chart and the decision cannot diverge.
    from core.acceptance import acceptance_score
    df["pct_score"] = 100 * df.apply(
        lambda r: acceptance_score(int(r["wins"]), int(r["losses"]), int(r["draws"])),
        axis=1,
    )
    # Numeric x-axis (not categorical): Plotly then spaces gens evenly and
    # auto-thins the tick labels, which is what makes this scale to 150+ gens.
    gens = df["generation"].astype(int)

    fig = go.Figure()
    # Composition lines — full W/L picture, off by default to keep it clean.
    fig.add_trace(go.Scatter(
        x=gens, y=df["pct_wins"], name="Wins %", mode="lines",
        line={"color": _COLORS["positive"], "width": 1}, opacity=0.5,
        visible="legendonly",
        hovertemplate="Gen %{x} — Wins %{y:.0f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=df["pct_losses"], name="Losses %", mode="lines",
        line={"color": _COLORS["negative"], "width": 1}, opacity=0.5,
        visible="legendonly",
        hovertemplate="Gen %{x} — Losses %{y:.0f}%<extra></extra>",
    ))
    # Primary: the acceptance-score line.
    fig.add_trace(go.Scatter(
        x=gens, y=df["pct_score"], name="Score (wins + ½ draws)", mode="lines",
        line={"color": "#333333", "width": 1.5},
        customdata=df[["wins", "losses", "draws"]].to_numpy(),
        hovertemplate=(
            "Gen %{x}<br>Score %{y:.0f}%<br>"
            "W%{customdata[0]} · L%{customdata[1]} · D%{customdata[2]}<extra></extra>"
        ),
    ))
    # Accept / reject markers on the score line — replaces the per-bar "✓
    # Accepted" text (which smeared together once accepts got dense).
    acc, rej = df[df["is_accepted"]], df[~df["is_accepted"]]
    fig.add_trace(go.Scatter(
        x=acc["generation"].astype(int), y=acc["pct_score"], name="Accepted",
        mode="markers",
        marker={"symbol": "circle", "size": 8, "color": _COLORS["positive"],
                "line": {"width": 1, "color": "#222222"}},
        hovertemplate="Gen %{x} ACCEPTED — score %{y:.0f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rej["generation"].astype(int), y=rej["pct_score"], name="Rejected",
        mode="markers",
        marker={"symbol": "circle-open", "size": 7, "color": _COLORS["negative"],
                "line": {"width": 1.5}},
        hovertemplate="Gen %{x} rejected — score %{y:.0f}%<extra></extra>",
    ))

    fig.add_hline(
        y=update_threshold * 100, line_dash="dash", line_color=_COLORS["primary"],
        annotation_text=f"Accept threshold ({update_threshold:.0%})",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title="Generation", yaxis_title="Score / win-rate (%)",
        yaxis_range=[0, 105],
        title="Arena: New Net vs Predecessor",
    )
    return _apply_defaults(fig)


# ---------------------------------------------------------------------------
# Performance charts (half-width for grid)
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


# The replay viewer renders a full board-by-board card per move. A long run
# (e.g. 150 generations × 50 arena games × ~30 moves ≈ 230k cards) explodes
# report-gen memory — it OOM-killed run1's report during the live run. So we
# render a representative *sample*: generations spread evenly across the run
# (always keeping the first and last) and the first few games of each. The
# full record stays in the ArenaReplays parquets either way.
_REPLAY_MAX_GENERATIONS = 16
_REPLAY_MAX_GAMES_PER_GEN = 6


def _evenly_sample(values: list[int], n: int) -> list[int]:
    """Pick up to ``n`` values spread evenly across ``values`` (sorted),
    always including the first and last. Returns all of them if ``len <= n``."""
    if n <= 0 or len(values) <= n:
        return list(values)
    idxs = {round(i * (len(values) - 1) / (n - 1)) for i in range(n)}
    return sorted(values[i] for i in idxs)


def _make_arena_replays_section(
    df: pd.DataFrame,
    config: RunConfig,
) -> tuple[str, str]:
    """Build the arena-replays UI as **two separate outputs**:

    1. A small **link card** that lives in the main training report —
       points the reader at the standalone replays page rather than
       inlining hundreds of game-board fragments and bloating the report.
    2. A **standalone HTML page** at ``Reporting/arena_replays.html`` with
       the full interactive replay viewer. Each turn card defaults to
       showing only the actual board after the move; clicking the turn
       header expands the top-K candidate panel for that move (the
       "hybrid" navigation Henry asked for — scroll the actuals, expand
       on click for details).

    Returns ``(link_card_html, standalone_html)``. The caller writes the
    standalone HTML to disk and inserts ``link_card_html`` in the main
    report body.

    All replay data is embedded as JSON in the standalone HTML so the
    file stays self-contained — no JS frameworks, no external fetches.
    """
    import json

    from reporting.display import get_renderer

    renderer = get_renderer(config.game)
    game = _instantiate_game(config.game)

    df = df.copy()
    df["generation"] = df["generation"].astype(int)

    # Bound the volume rendered (see module constants above) so long runs
    # don't OOM report generation. Sample generations evenly, cap games/gen.
    all_gens = sorted(df["generation"].unique())
    sampled_gens = _evenly_sample(all_gens, _REPLAY_MAX_GENERATIONS)
    df = df[
        df["generation"].isin(sampled_gens)
        & (df["game_idx"] < _REPLAY_MAX_GAMES_PER_GEN)
    ]
    if len(sampled_gens) < len(all_gens) or _REPLAY_MAX_GAMES_PER_GEN < 50:
        logger.info(
            "Arena replays: rendering {} of {} generations (evenly sampled) "
            "× up to {} games/gen to bound report size",
            len(sampled_gens), len(all_gens), _REPLAY_MAX_GAMES_PER_GEN,
        )

    df = df.sort_values(["generation", "game_idx", "move_idx"])

    games_by_gen: dict[int, list[dict]] = {}

    for (gen, game_idx), group in df.groupby(["generation", "game_idx"]):
        moves = group.sort_values("move_idx")
        first = moves.iloc[0]
        player1_was_white = bool(first["player1_was_white"])

        # Roles: Player 1 = previous best, Player 2 = new candidate (current).
        if player1_was_white:
            role_by_player = {1: "previous net", -1: "new net (this gen)"}
        else:
            role_by_player = {1: "new net (this gen)", -1: "previous net"}

        board = game.initialise_board()  # actual (non-canonical) board state
        turns_html: list[str] = []

        # Colour-name framing: TTT shows literal X/O glyphs on the board so
        # the suffix is informative; Blokus shows piece numbers so it's just
        # noise. Trim it for Blokus.
        if config.game == "tictactoe":
            colour_for_player = {1: "White (X)", -1: "Black (O)"}
        else:
            colour_for_player = {1: "White", -1: "Black"}

        for _, m in moves.iterrows():
            action = int(m["action"])
            player = int(m["player"])
            colour_name = colour_for_player[player]
            role = role_by_player[player]
            player_label = f"{colour_name} — {role}"
            turn_idx = int(m["move_idx"]) + 1

            top_k_actions = [int(a) for a in m["top_k_actions"]]
            top_k_probs = [float(p) for p in m["top_k_probs"]]
            # Defensive: drop any zero-probability entries that older runs
            # may have persisted (pre-fix in core/arena._extract_top_k).
            visited = {
                a: p for a, p in zip(top_k_actions, top_k_probs, strict=False)
                if p > 0
            }
            # Played-action probability: prefer the explicitly-persisted
            # column added in MoveRecord.played_prob; fall back to looking
            # the played action up in the top-K for older parquets.
            if "played_prob" in m and m["played_prob"] is not None:
                played_prob = float(m["played_prob"])
            else:
                played_prob = visited.get(action, 0.0)
            # Candidates panel shows the *alternatives* MCTS considered —
            # the next-best moves the model thought about other than the one
            # it actually played. Surfacing the played action via the
            # caption below it (with its own probability) avoids the
            # "the actual move isn't in the top-3" confusion that happens
            # when many actions tie at the max visit count.
            alternatives = {a: p for a, p in visited.items() if a != action}

            played_caption = _format_played_action_caption(
                game, action, played_prob, colour_name,
            )
            policy_html = renderer.render_policy_html(
                board, alternatives,
                annotation=f"{colour_name}'s top alternatives",
                current_player=player,
            )
            # Apply move and render the resulting state.
            board, _ = game.get_next_state(board, player, action)
            after_html = renderer.render_board_html(
                board, last_action=action, annotation=played_caption,
            )

            turn_card = (
                '<div class="replay-turn">'
                f'<button class="replay-turn-toggle" type="button" '
                f'onclick="alphaBlokus_toggleTurn(this)">'
                f'<span class="replay-turn-label">'
                f'Turn {turn_idx} — {player_label}'
                f'</span>'
                f'<span class="replay-turn-hint">↓ click to show top candidates</span>'
                f'</button>'
                f'<div class="replay-turn-actual">{after_html}</div>'
                f'<div class="replay-turn-candidates" hidden>{policy_html}</div>'
                "</div>"
            )
            turns_html.append(turn_card)

        # Outcome: stored from Player 1's POV (+1 = P1 won, -1 = P2 won).
        outcome = float(first["outcome"])
        if outcome > 0.5:
            winner_colour = "White" if player1_was_white else "Black"
            outcome_label = f"{winner_colour} wins — previous net"
            outcome_class = "result-prev"
        elif outcome < -0.5:
            winner_colour = "Black" if player1_was_white else "White"
            outcome_label = f"{winner_colour} wins — new net"
            outcome_class = "result-new"
        else:
            outcome_label = "Draw"
            outcome_class = "result-draw"

        # Final result banner at the bottom of the replay.
        turns_html.append(
            f'<div class="replay-result">{outcome_label}</div>'
        )

        games_by_gen.setdefault(int(gen), []).append({
            "game_idx": int(game_idx),
            "outcome": outcome,
            "outcome_label": outcome_label,
            "outcome_class": outcome_class,
            "player1_was_white": player1_was_white,
            "turns_html": turns_html,
        })

    payload = json.dumps(games_by_gen)
    gen_options = "\n".join(
        f'<option value="{g}">Generation {g}</option>'
        for g in sorted(games_by_gen)
    )
    first_gen = min(games_by_gen)
    initial_game_options = _render_game_options(games_by_gen[first_gen])

    total_games = sum(len(games) for games in games_by_gen.values())
    standalone_html = _ARENA_REPLAYS_STANDALONE_TEMPLATE.format(
        run_name=config.run_name,
        total_games=total_games,
        num_gens=len(games_by_gen),
        board_css=_blokus_board_css(config),
        gen_options=gen_options,
        initial_game_options=initial_game_options,
        payload=payload,
    )

    link_card = _ARENA_REPLAYS_LINK_CARD.format(
        total_games=total_games,
        num_gens=len(games_by_gen),
    )
    return link_card, standalone_html


def _format_played_action_caption(
    game, action_id: int, played_prob: float, colour_name: str,  # noqa: ARG001
) -> str:
    """Build the annotation for the actual-board panel — describes the move
    in human-readable terms and surfaces its raw MCTS visit probability.

    ``played_prob`` is the share of MCTS visits this action received before
    temperature sampling. With temp=0 the action played is one of the tied
    top-visit options; the displayed probability is the raw visit fraction
    so the reader can see how confident MCTS was relative to alternatives.

    A ``played_prob`` of exactly 0.0 is treated as "unknown" — that's the
    sentinel for older arena-replay parquets persisted before the explicit
    ``MoveRecord.played_prob`` field landed, where the played action may
    have fallen outside the captured top-K.
    """
    prob_suffix = (
        f" — {played_prob * 100:.1f}% of visits"
        if played_prob > 0 else " — visit % not recorded"
    )
    if game.__class__.__name__ == "BlokusDuoGame":
        if game.action_codec.is_pass(action_id):
            return f"Played: PASS{prob_suffix}"
        decoded = game.action_codec.decode(action_id)
        piece = game.piece_manager.pieces[decoded.piece_id]
        return (
            f"Played: Piece {decoded.piece_id} ({piece.name}, "
            f"{decoded.orientation.value}) at "
            f"({decoded.x_coordinate}, {decoded.y_coordinate}){prob_suffix}"
        )
    return f"Played action {action_id}{prob_suffix}"


def _blokus_board_css(config: RunConfig) -> str:
    """Inline the right per-game board CSS into the standalone page."""
    if config.game == "blokusduo":
        from reporting.display_blokusduo import BOARD_CSS
        return BOARD_CSS
    # TTT renders board styles inline via :func:`display_tictactoe`, so the
    # standalone page only needs the shared replay layout CSS.
    return ""


def _render_game_options(games: list[dict]) -> str:
    """Server-side game-dropdown options for the initial generation.

    The ``class`` on each option drives the background colour — green when the
    new (current) net won, red when the previous net won, plain for a draw.
    Browsers vary in how willing they are to style ``<option>`` directly, but
    Chromium/Firefox on desktop both honour it; falls back to no colour
    elsewhere, which is harmless.
    """
    return "\n".join(
        f'<option class="{g["outcome_class"]}" value="{g["game_idx"]}">'
        f'G{g["game_idx"] + 1} — {g["outcome_label"]}'
        f"</option>"
        for g in games
    )


def _instantiate_game(game_name: str):
    """Match main.py's mapping — needed for arena-replay board reconstruction."""
    if game_name == "tictactoe":
        from games.tictactoe.game import TicTacToeGame
        return TicTacToeGame()
    if game_name == "blokusduo":
        from pathlib import Path

        from games.blokusduo.game import BlokusDuoGame
        return BlokusDuoGame(Path("games/blokusduo/pieces.json"))
    raise ValueError(f"Unknown game: {game_name}")


_ARENA_REPLAYS_LINK_CARD = """\
<section>
<h2>Arena Game Replays</h2>
<p class="section-desc">
{total_games} recorded games across {num_gens} generations are available in
the dedicated replay viewer — board-by-board playback with expand-on-click
top-3 candidate previews per move. Pulled out of this report so the
training-metrics view stays focused.
</p>
<a href="arena_replays.html" class="open-replays-button" target="_blank">
  Open arena replay viewer →
</a>
<style>
.open-replays-button {{
    display: inline-block; margin-top: 8px;
    padding: 10px 18px; background: #636efa; color: white;
    text-decoration: none; border-radius: 6px;
    font-size: 14px; font-weight: 600;
}}
.open-replays-button:hover {{
    background: #4a55d4;
}}
</style>
</section>
"""


_ARENA_REPLAYS_STANDALONE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Arena Replays — {run_name}</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1400px; margin: 0 auto; padding: 24px 32px; color: #2a3f5f;
    background: #ffffff;
}}
h1 {{ border-bottom: 2px solid #636efa; padding-bottom: 8px; margin-bottom: 4px; }}
.subtitle {{ color: #6b7280; font-size: 14px; margin-bottom: 24px; }}

.arena-controls {{ margin: 16px 0 24px 0; font-size: 14px; }}
.arena-controls select {{
    padding: 6px 12px; font-size: 14px; margin-left: 4px;
    border: 1px solid #e5e7eb; border-radius: 6px; background: #f8f9fb;
    color: #2a3f5f; min-width: 220px;
}}
.arena-controls label {{ font-weight: 600; margin-right: 16px; }}
/* Coloured dropdown rows: green = new net won, red = previous net won. */
#alphaBlokus-game-select option.result-new {{ background: #dcfce7; color: #14532d; }}
#alphaBlokus-game-select option.result-prev {{ background: #fee2e2; color: #7f1d1d; }}
#alphaBlokus-game-select option.result-draw {{ background: transparent; color: inherit; }}

.replay-body {{ display: flex; flex-direction: column; gap: 12px; }}

.replay-turn {{
    border: 1px solid #e5e7eb; border-radius: 6px; background: #fafafa;
    overflow: hidden;
}}
.replay-turn-toggle {{
    display: flex; align-items: center; justify-content: space-between;
    width: 100%; padding: 10px 16px; border: 0;
    background: none; cursor: pointer; font-family: inherit;
    color: #2a3f5f; font-size: 13px; font-weight: 600;
    text-align: left;
}}
.replay-turn-toggle:hover {{ background: #f3f4f6; }}
.replay-turn-hint {{
    font-size: 11px; font-weight: 400; color: #6b7280; letter-spacing: 0.3px;
}}

.replay-turn-actual {{ padding: 0 16px 16px 16px; }}
.replay-turn-candidates {{
    padding: 12px 16px 16px 16px;
    border-top: 1px dashed #e5e7eb; background: #ffffff;
}}
.replay-turn-candidates[hidden] {{ display: none; }}

.replay-result {{
    padding: 16px 18px; text-align: center;
    font-size: 17px; font-weight: 700; letter-spacing: 0.4px;
    color: #1f2937; background: #f3f4f6;
    border: 1px solid #d1d5db; border-radius: 6px;
}}

/* Inlined TTT board styles — keep the policy / actual boards aligned. */
.ttt-board {{ display: inline-block; margin: 0; }}
.ttt-board table.ttt-grid {{ border-collapse: collapse; }}
.ttt-board td {{ padding: 0; }}
.ttt-board th {{
    padding: 2px 6px; font-size: 10px; color: #9ca3af;
    background: none; border: none; font-weight: normal; text-align: center;
}}
.ttt-board th.corner {{ width: 16px; }}
.ttt-board th.row-label {{ text-align: right; padding-right: 6px; }}
.ttt-board .board-annotation {{
    font-size: 11px; color: #4b5563; text-align: center;
    margin-bottom: 6px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px;
}}

{board_css}
</style>
</head>
<body>
<h1>Arena Game Replays — {run_name}</h1>
<p class="subtitle">
  {total_games} games across {num_gens} generations. Each replay shows the
  actual board after every move by default; click a turn header to expand
  the top-3 candidate previews the model was considering for that move.
  Player 1 is the previous best network and Player 2 is the new candidate
  trained this generation.
</p>

<div class="arena-controls">
    <label>Generation:
        <select id="alphaBlokus-gen-select"
                onchange="alphaBlokus_onGenChange(parseInt(this.value))">
{gen_options}
        </select>
    </label>
    <label>Game:
        <select id="alphaBlokus-game-select"
                onchange="alphaBlokus_onGameChange(parseInt(this.value))">
{initial_game_options}
        </select>
    </label>
    <label>
        <button type="button" onclick="alphaBlokus_toggleAll()">Expand / collapse all candidates</button>
    </label>
</div>
<div id="alphaBlokus-replay-body" class="replay-body"></div>

<script>
(function() {{
    const REPLAYS = {payload};

    function genGames(gen) {{ return REPLAYS[gen] || []; }}

    function onGenChange(gen) {{
        const select = document.getElementById('alphaBlokus-game-select');
        const games = genGames(gen);
        select.innerHTML = games.map(g => {{
            return '<option class="' + g.outcome_class + '" value="'
                 + g.game_idx + '">G' + (g.game_idx + 1)
                 + ' — ' + g.outcome_label + '</option>';
        }}).join('');
        if (games.length > 0) {{
            select.value = games[0].game_idx;
            renderGame(gen, games[0].game_idx);
        }} else {{
            document.getElementById('alphaBlokus-replay-body').innerHTML = '';
        }}
    }}

    function onGameChange(gameIdx) {{
        const gen = parseInt(document.getElementById('alphaBlokus-gen-select').value);
        renderGame(gen, gameIdx);
    }}

    function renderGame(gen, gameIdx) {{
        const games = genGames(gen);
        const game = games.find(g => g.game_idx === gameIdx);
        if (!game) return;
        document.getElementById('alphaBlokus-replay-body').innerHTML =
            game.turns_html.join('');
    }}

    function toggleTurn(buttonEl) {{
        const turn = buttonEl.parentElement;
        const candidates = turn.querySelector('.replay-turn-candidates');
        if (!candidates) return;
        candidates.hidden = !candidates.hidden;
        turn.classList.toggle('expanded', !candidates.hidden);
        const hint = buttonEl.querySelector('.replay-turn-hint');
        if (hint) hint.textContent = candidates.hidden
            ? '↓ click to show top candidates'
            : '↑ hide candidates';
    }}

    function toggleAll() {{
        const turns = document.querySelectorAll('.replay-turn');
        const allHidden = Array.from(turns).every(t => {{
            const c = t.querySelector('.replay-turn-candidates');
            return !c || c.hidden;
        }});
        turns.forEach(t => {{
            const c = t.querySelector('.replay-turn-candidates');
            if (!c) return;
            c.hidden = !allHidden;
            t.classList.toggle('expanded', allHidden);
            const hint = t.querySelector('.replay-turn-hint');
            if (hint) hint.textContent = allHidden
                ? '↑ hide candidates'
                : '↓ click to show top candidates';
        }});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
        const genSelect = document.getElementById('alphaBlokus-gen-select');
        if (!genSelect) return;
        const gen = parseInt(genSelect.value);
        const games = genGames(gen);
        if (games.length > 0) {{
            renderGame(gen, games[0].game_idx);
        }}
    }});

    window.alphaBlokus_onGenChange = onGenChange;
    window.alphaBlokus_onGameChange = onGameChange;
    window.alphaBlokus_toggleTurn = toggleTurn;
    window.alphaBlokus_toggleAll = toggleAll;
}})();
</script>
</body>
</html>
"""


def _make_elo_plot(
    elo_data: pd.DataFrame,
    arena_data: pd.DataFrame | None = None,
) -> go.Figure:
    """Absolute Elo rating over generations, anchored at the gen-0 baseline.

    Important interpretation detail: Elo evaluation runs *after* the
    accept/reject decision each gen. When a gen is rejected, ``self.nnet``
    has been reverted to the previous-best checkpoint — so the rated
    network is the *previous accepted* net, not the just-trained one.
    Two consecutive rejected gens produce two Elo points for the *same*
    underlying network (differing only by 20-game sampling noise).

    Rejected gens are drawn with an open marker + dashed line segment to
    make this visible — solid filled markers = the gen's newly-trained
    accepted net, hollow markers = a reverted (previous-best) net being
    re-evaluated.

    When ``arena_data`` is not provided (e.g. when called for older runs
    without the column), all gens are plotted as accepted.
    """
    df = elo_data.sort_values("generation").copy()
    baseline = int(df["baseline_rating"].iloc[0])

    if arena_data is not None and not arena_data.empty:
        # The 'accepted' column is read straight off arena_data when present;
        # the threshold passed here is only used by the fallback path for
        # older runs without the column persisted.
        mask = _accepted_mask(arena_data, update_threshold=0.5)
        accepted_lookup = pd.Series(
            mask.values, index=arena_data["generation"].astype(int).values,
        )
        df["accepted"] = (
            df["generation"].astype(int).map(accepted_lookup).fillna(True).astype(bool)
        )
    else:
        df["accepted"] = True

    fig = go.Figure()
    # Single connecting line through all gens, regardless of accept/reject.
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["elo_rating"],
        mode="lines", name="", showlegend=False,
        line={"width": 2.5, "color": _COLORS["accent"]},
        hoverinfo="skip",
    ))
    # Accepted-gen markers (filled).
    accepted_df = df[df["accepted"]]
    fig.add_trace(go.Scatter(
        x=accepted_df["generation"], y=accepted_df["elo_rating"],
        mode="markers", name="Accepted (newly-trained net)",
        marker={"size": 9, "color": _COLORS["accent"], "symbol": "circle"},
        customdata=accepted_df[["elo_diff", "score_rate", "wins", "losses", "draws"]].values,
        hovertemplate=(
            "Gen %{x} (accepted) — Elo: %{y:.0f} "
            "(%{customdata[0]:+.0f} vs baseline)<br>"
            "Score: %{customdata[1]:.3f} "
            "(W%{customdata[2]} L%{customdata[3]} D%{customdata[4]})"
            "<extra></extra>"
        ),
    ))
    # Rejected-gen markers (open) — these re-evaluate the previous accepted net.
    rejected_df = df[~df["accepted"]]
    if not rejected_df.empty:
        fig.add_trace(go.Scatter(
            x=rejected_df["generation"], y=rejected_df["elo_rating"],
            mode="markers", name="Rejected (re-evaluation of prev best)",
            marker={"size": 9, "color": _COLORS["accent"], "symbol": "circle-open",
                    "line": {"width": 2, "color": _COLORS["accent"]}},
            customdata=rejected_df[["elo_diff", "score_rate", "wins", "losses", "draws"]].values,
            hovertemplate=(
                "Gen %{x} (rejected — rating shown is the reverted "
                "previous-best net) — Elo: %{y:.0f} "
                "(%{customdata[0]:+.0f} vs baseline)<br>"
                "Score: %{customdata[1]:.3f} "
                "(W%{customdata[2]} L%{customdata[3]} D%{customdata[4]})"
                "<extra></extra>"
            ),
        ))
    fig.add_hline(
        y=baseline, line_dash="dash", line_color=_COLORS["neutral"], line_width=1,
        annotation_text=f"Baseline (gen 0) = {baseline}",
        annotation_position="bottom right",
        annotation_font_size=10, annotation_font_color=_COLORS["neutral"],
    )
    fig.update_layout(
        xaxis_title="Generation", yaxis_title="Elo rating",
        title="Elo Rating vs Frozen Gen-0 Baseline",
        xaxis={"dtick": 1 if df["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def _make_minimax_plot(minimax_data: pd.DataFrame) -> go.Figure:
    """Vs perfect-play minimax (TTT only): draw rate and loss rate per gen.

    Against perfect play, TTT is a forced draw — so an optimal model should
    have ``draw_rate → 1.0`` and ``loss_rate → 0`` over training. Loss rate
    falling first, then draw rate rising as remaining wins disappear, is the
    canonical learning signature for a solved game.
    """
    df = minimax_data.sort_values("generation").copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["generation"], y=100 * df["draw_rate"],
        mode="lines+markers", name="Draw rate (target: 100%)",
        line={"width": 2.5, "color": _COLORS["tertiary"]},
        hovertemplate="Gen %{x} — draws: %{y:.0f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=100 * df["loss_rate"],
        mode="lines+markers", name="Loss rate (target: 0%)",
        line={"width": 2.5, "color": _COLORS["negative"], "dash": "dot"},
        hovertemplate="Gen %{x} — losses: %{y:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Generation", yaxis_title="Rate (%)",
        yaxis_range=[-2, 102],
        title="Vs Perfect-Play Minimax (TTT)",
        xaxis={"dtick": 1 if df["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def _make_policy_accuracy_plot(
    accuracy_data: pd.DataFrame, game_name: str,
) -> go.Figure:
    """Per-generation top-K agreement between the network's raw policy and
    the eval-set target. ``game_name`` switches the framing only — the
    numbers come from the same diagnostic either way.

    - **TTT**: target is *minimax-optimal*. A "hit" means the net picks an
      action that is genuinely optimal under perfect play. Top-1 should
      climb toward 100% as the net internalises perfect play.
    - **Blokus / other**: target is the MCTS visit-count argmax recorded in
      gen-1 self-play. Top-1 measures how often the raw net agrees with
      what search arrived at — AlphaGo Zero's Figure 3b in spirit.

    Computed on the frozen eval set after every training epoch; one point
    per generation is shown (mean across epochs).
    """
    df = accuracy_data.copy()
    agg = df.groupby("generation").agg(
        top1_mean=("top1_accuracy", "mean"),
        top5_mean=("top5_accuracy", "mean"),
    ).reset_index().sort_values("generation")

    if game_name == "tictactoe":
        title = "Policy Agreement vs Minimax Oracle (held-out set)"
    else:
        title = "Policy Agreement vs MCTS (held-out set)"

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
        xaxis_title="Generation", yaxis_title="Agreement (%)",
        yaxis_range=[0, 105], title=title,
        xaxis={"dtick": 1 if agg["generation"].max() < 40 else 5},
    )
    return _apply_defaults(fig)


def _make_value_calibration_plot(
    calibration_data: pd.DataFrame, game_name: str,
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
    fig.add_trace(go.Scatter(
        x=[-1, 1], y=[-1, 1],
        mode="lines", name="Perfect calibration (y=x)",
        line={"dash": "dash", "color": _COLORS["neutral"], "width": 1},
        hoverinfo="skip",
    ))
    max_count = max(int(latest["bucket_count"].max()), 1)
    sizes = 6 + 24 * latest["bucket_count"] / max_count
    fig.add_trace(go.Scatter(
        x=latest["bucket_center"], y=latest["bucket_mean_actual"],
        mode="markers+lines", name=f"Gen {int(last_gen)} epoch {int(last_epoch)}",
        marker={
            "size": sizes, "color": _COLORS["accent"],
            "line": {"width": 1, "color": _COLORS["accent"]},
        },
        customdata=latest["bucket_count"],
        hovertemplate=(
            "Predicted ≈ %{x:.1f}, " + hover_label + ": %{y:.2f}"
            " (%{customdata} positions)<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title="Predicted value (bucket centre)",
        yaxis_title=y_label, title=title,
        xaxis_range=[-1.05, 1.05], yaxis_range=[-1.05, 1.05],
    )
    return _apply_defaults(fig)


def _make_symmetry_diagnostic_plot(symmetry_data: pd.DataFrame) -> go.Figure:
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
        df.groupby(["generation", "position_idx"])
        .agg(position_mean_kl=("kl_divergence", "mean"))
        .reset_index()
    )
    overall = (
        per_position.groupby("generation")
        .agg(mean_kl=("position_mean_kl", "mean"),
             max_kl=("position_mean_kl", "max"))
        .reset_index()
        .sort_values("generation")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=overall["generation"], y=overall["mean_kl"],
        mode="lines+markers", name="Mean across positions",
        line={"width": 2.5, "color": _COLORS["primary"]},
        hovertemplate="Gen %{x} — mean KL: %{y:.4f}<extra></extra>",
    ))
    for pos_idx, group in per_position.sort_values("position_idx").groupby("position_idx"):
        fig.add_trace(go.Scatter(
            x=group["generation"], y=group["position_mean_kl"],
            mode="lines", name=f"Position {pos_idx}",
            line={"width": 1, "dash": "dot", "color": _COLORS["neutral"]},
            opacity=0.5, hoverinfo="skip", showlegend=True,
        ))
    fig.add_hline(
        y=0.0, line_width=1, line_dash="dash", line_color=_COLORS["neutral"],
        annotation_text="Perfect equivariance",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="Generation", yaxis_title="KL divergence (lower = more symmetric)",
        title="Network policy symmetry diagnostic",
        xaxis={"dtick": 1 if overall["generation"].max() < 40 else 5},
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
        df["mean_policy_entropy"] = 0.0  # backward compat with older runs
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
# Config table (unchanged content, moved to bottom)
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
# HTML template with CSS grid, collapsible sections, descriptions
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
    arena_replays_data = (
        _load_metrics(config.arena_replays_directory)
        if config.arena_replays_directory.exists() else None
    )
    policy_accuracy_data = (
        _load_metrics(config.policy_accuracy_directory)
        if config.policy_accuracy_directory.exists() else None
    )
    value_calibration_data = (
        _load_metrics(config.value_calibration_directory)
        if config.value_calibration_directory.exists() else None
    )
    elo_data = (
        _load_metrics(config.elo_ratings_directory)
        if config.elo_ratings_directory.exists() else None
    )
    minimax_data = (
        _load_metrics(config.minimax_results_directory)
        if config.minimax_results_directory.exists() else None
    )
    symmetry_data = (
        _load_metrics(config.symmetry_diagnostic_directory)
        if config.symmetry_diagnostic_directory.exists() else None
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
        _make_policy_accuracy_plot(policy_accuracy_data, config.game)
        if policy_accuracy_data is not None and not policy_accuracy_data.empty
        else None
    )
    fig_value_calibration = (
        _make_value_calibration_plot(value_calibration_data, config.game)
        if value_calibration_data is not None and not value_calibration_data.empty
        else None
    )
    fig_elo = (
        _make_elo_plot(elo_data, arena_data)
        if elo_data is not None and not elo_data.empty
        else None
    )
    fig_minimax = (
        _make_minimax_plot(minimax_data)
        if minimax_data is not None and not minimax_data.empty
        else None
    )
    fig_symmetry = (
        _make_symmetry_diagnostic_plot(symmetry_data)
        if symmetry_data is not None and not symmetry_data.empty
        else None
    )
    if arena_replays_data is not None and not arena_replays_data.empty:
        arena_replays_html, arena_replays_standalone = _make_arena_replays_section(
            arena_replays_data, config,
        )
    else:
        arena_replays_html, arena_replays_standalone = "", ""

    # Write HTML
    filename = config.report_directory / "report.html"
    filename.parent.mkdir(exist_ok=True, parents=True)

    # Standalone arena replays page lives alongside the main report. Linked
    # from the main report via the `_ARENA_REPLAYS_LINK_CARD` button.
    if arena_replays_standalone:
        replays_path = config.report_directory / "arena_replays.html"
        replays_path.write_text(arena_replays_standalone, encoding="utf-8")
        logger.info("Wrote arena replay viewer to {}", replays_path)

    def _chart(fig: go.Figure) -> str:
        return fig.to_html(full_html=False, include_plotlyjs=False)

    today = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    kpi_html = _make_kpi_cards(
        loss_data, arena_data, timings_data, profiling_data, throughput_data,
        update_threshold=config.update_threshold,
    )
    config_html = _make_config_table(config)

    strength_html = ""
    if fig_elo is not None or fig_minimax is not None:
        parts = [
            '<section>',
            '<h2>Strength vs Fixed Baselines</h2>',
            '<p class="section-desc">'
            'External strength measurements: the active network (after the '
            'accept/reject decision) plays a fixed gen-0 opponent. Filled '
            'markers = accepted gen (rating is for the just-trained net); '
            'open markers = rejected gen (rating is for the reverted '
            'previous-best net — so two adjacent open markers measure the '
            '<em>same</em> network and differ only by 20-game sampling '
            'noise). Minimax (TTT only) is the absolute '
            '"is this optimal?" signal.<br>'
            '<strong>Caveat:</strong> at high MCTS sim counts the search '
            'dominates the network signal, so the trained-net-vs-random '
            'gap is squeezed and per-gen swings are dominated by the '
            'small (20-game) sample. Treat the absolute level as noisy; '
            'trust the trend over many gens. The reliable training-progress '
            'signals are <em>policy agreement</em> and <em>value loss</em>.'
            '</p>',
        ]
        if fig_elo is not None:
            parts.append(_chart(fig_elo))
        if fig_minimax is not None:
            parts.append(_chart(fig_minimax))
        parts.append('</section>')
        strength_html = "\n".join(parts)

    diagnostics_html = ""
    if (
        fig_network_entropy is not None
        or fig_policy_accuracy is not None
        or fig_value_calibration is not None
        or fig_symmetry is not None
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
        if fig_symmetry is not None:
            parts.append(_chart(fig_symmetry))
        parts.append('</section>')
        diagnostics_html = "\n".join(parts)

    network_entropy_html = ""  # rolled into diagnostics_html above

    with open(filename, "w", encoding="utf-8") as f:
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
    Each generation's new network plays the incumbent. It is accepted when its
    score &mdash; wins plus half of draws &mdash; reaches {config.update_threshold:.0%}
    (the black tick on each bar), not on raw wins alone.
</p>
{_chart(fig_arena)}
</section>

{strength_html}

{arena_replays_html}

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
