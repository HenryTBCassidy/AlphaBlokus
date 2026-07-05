from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    from plotly import graph_objects as go

    from alphablokus.config import RunConfig


from alphablokus.reporting.arena_replays import build_arena_replays_section, load_sampled_replays
from alphablokus.reporting.charts import (
    accepted_mask,
    make_arena_plot,
    make_elo_plot,
    make_loss_per_generation,
    make_loss_timeline,
    make_minimax_plot,
    make_network_entropy_plot,
    make_per_gen_loss_curves,
    make_policy_accuracy_plot,
    make_profiling_plot,
    make_resource_usage_plot,
    make_symmetry_diagnostic_plot,
    make_throughput_plot,
    make_timing_plot,
    make_tournament_elo_plot,
    make_value_calibration_plot,
)
from alphablokus.reporting.pentobi_ladder import build_pentobi_ladder_section


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
    update_threshold: float,
) -> str:
    """Build HTML for the KPI card row."""
    # Final loss + delta — mean of each generation's last epoch (the
    # epoch where the network is most trained). Avoids the noise of a
    # single trailing batch and the artefact of running-mean resets.
    sorted_loss = loss_data.sort_values(["generation", "epoch", "batch_number"])
    last_epoch_per_gen = sorted_loss.groupby("generation")["epoch"].max()
    by_gen = (
        sorted_loss[sorted_loss["epoch"] == sorted_loss["generation"].map(last_epoch_per_gen)]
        .groupby("generation")["total_loss"]
        .mean()
    )
    final_loss = by_gen.iloc[-1]
    first_loss = by_gen.iloc[0]
    loss_delta_pct = ((final_loss - first_loss) / first_loss) * 100

    # Accept rate — matches Coach._should_accept_new_network (draws excluded,
    # configured threshold used, not a hardcoded 0.5).
    accepted = accepted_mask(arena_data, update_threshold)
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
        (
            "Final Loss",
            f"{final_loss:.3f}",
            f"{loss_delta_pct:+.0f}% vs gen 1",
            "positive" if loss_delta_pct < 0 else "negative",
        ),
        ("Accept Rate", f"{accept_count}/{total_gens}", f"{100 * accept_count / total_gens:.0f}% of generations", ""),
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
            f"<div{delta_class}>{context}</div>"
            f"</div>"
        )
    return f'<div class="kpi-grid">{"".join(html_parts)}</div>'


# ---------------------------------------------------------------------------
# Loss per generation
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
        ("Replay buffer (games)", config.replay_buffer_games),
        ("Buffer staleness (gens)", round(config.replay_buffer_games / max(config.num_eps, 1), 1)),
        (
            "Emergent reuse (E×B/F)",
            round(config.net_config.epochs * config.replay_buffer_games / max(config.num_eps, 1), 1),
        ),
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
        _load_metrics(config.training_entropy_directory) if config.training_entropy_directory.exists() else None
    )
    # Replays are partition-filtered down to the sampled generations × capped
    # games the viewer actually renders — never the whole (unbounded) history.
    arena_replays_data = (
        load_sampled_replays(config.arena_replays_directory) if config.arena_replays_directory.exists() else None
    )
    policy_accuracy_data = (
        _load_metrics(config.policy_accuracy_directory) if config.policy_accuracy_directory.exists() else None
    )
    value_calibration_data = (
        _load_metrics(config.value_calibration_directory) if config.value_calibration_directory.exists() else None
    )
    elo_data = _load_metrics(config.elo_ratings_directory) if config.elo_ratings_directory.exists() else None
    # Pool BayesElo ratings: a single file written by scripts/tournament_elo.py,
    # absent for runs that never ran the post-hoc tournament.
    tournament_ratings_path = config.tournament_directory / "tournament_ratings.parquet"
    tournament_data = pd.read_parquet(tournament_ratings_path) if tournament_ratings_path.exists() else None
    minimax_data = (
        _load_metrics(config.minimax_results_directory) if config.minimax_results_directory.exists() else None
    )
    symmetry_data = (
        _load_metrics(config.symmetry_diagnostic_directory) if config.symmetry_diagnostic_directory.exists() else None
    )

    # Build figures
    fig_loss_gen = make_loss_per_generation(loss_data)
    fig_loss_timeline = make_loss_timeline(loss_data)
    fig_per_gen_curves = make_per_gen_loss_curves(loss_data)
    fig_arena = make_arena_plot(arena_data, config.update_threshold)
    fig_timing = make_timing_plot(timings_data)
    fig_throughput = make_throughput_plot(throughput_data)
    fig_resources = make_resource_usage_plot(resource_data)
    fig_profiling = make_profiling_plot(profiling_data)
    fig_network_entropy = (
        make_network_entropy_plot(network_entropy_data)
        if network_entropy_data is not None and not network_entropy_data.empty
        else None
    )
    fig_policy_accuracy = (
        make_policy_accuracy_plot(policy_accuracy_data, config.game)
        if policy_accuracy_data is not None and not policy_accuracy_data.empty
        else None
    )
    fig_value_calibration = (
        make_value_calibration_plot(value_calibration_data, config.game)
        if value_calibration_data is not None and not value_calibration_data.empty
        else None
    )
    fig_elo = make_elo_plot(elo_data, arena_data) if elo_data is not None and not elo_data.empty else None
    fig_tournament_elo = (
        make_tournament_elo_plot(tournament_data) if tournament_data is not None and not tournament_data.empty else None
    )
    fig_minimax = make_minimax_plot(minimax_data) if minimax_data is not None and not minimax_data.empty else None
    fig_symmetry = (
        make_symmetry_diagnostic_plot(symmetry_data) if symmetry_data is not None and not symmetry_data.empty else None
    )
    if arena_replays_data is not None and not arena_replays_data.empty:
        arena_replays_html, arena_replays_standalone = build_arena_replays_section(
            arena_replays_data,
            config,
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
        loss_data,
        arena_data,
        timings_data,
        profiling_data,
        throughput_data,
        update_threshold=config.update_threshold,
    )
    config_html = _make_config_table(config)

    pentobi_ladder_html = build_pentobi_ladder_section(config.pentobi_ladder_directory)

    strength_html = ""
    if fig_tournament_elo is not None or fig_elo is not None or fig_minimax is not None:
        parts = [
            "<section>",
            "<h2>Strength vs Fixed Baselines</h2>",
        ]
        if fig_tournament_elo is not None:
            parts.append(
                '<p class="section-desc">'
                "<strong>Pool Elo (BayesElo)</strong> is the canonical strength "
                "curve: a sparse round-robin among the run's saved checkpoints, "
                "fit so every checkpoint shares one rating scale. It keeps rising "
                "until genuine convergence, where the vs-gen-0 number below "
                "flatlines once the net beats gen-0 ~100% of the time (the ±1200 "
                "clamp). Read the pool curve; treat vs-gen-0 as an early-training "
                "signal only.</p>"
            )
            parts.append(_chart(fig_tournament_elo))
        parts.append(
            '<p class="section-desc">'
            "External strength measurements: the active network (after the "
            "accept/reject decision) plays a fixed gen-0 opponent. Filled "
            "markers = accepted gen (rating is for the just-trained net); "
            "open markers = rejected gen (rating is for the reverted "
            "previous-best net — so two adjacent open markers measure the "
            "<em>same</em> network and differ only by 20-game sampling "
            "noise). Minimax (TTT only) is the absolute "
            '"is this optimal?" signal.<br>'
            "<strong>Caveat:</strong> at high MCTS sim counts the search "
            "dominates the network signal, so the trained-net-vs-random "
            "gap is squeezed and per-gen swings are dominated by the "
            "small (20-game) sample. Treat the absolute level as noisy; "
            "trust the trend over many gens. The reliable training-progress "
            "signals are <em>policy agreement</em> and <em>value loss</em>."
            "</p>"
        )
        if fig_elo is not None:
            parts.append(_chart(fig_elo))
        if fig_minimax is not None:
            parts.append(_chart(fig_minimax))
        parts.append("</section>")
        strength_html = "\n".join(parts)

    diagnostics_html = ""
    if (
        fig_network_entropy is not None
        or fig_policy_accuracy is not None
        or fig_value_calibration is not None
        or fig_symmetry is not None
    ):
        parts = [
            "<section>",
            "<h2>Network Diagnostics</h2>",
            '<p class="section-desc">'
            "Per-epoch evaluation of the network alone (no MCTS) on a frozen "
            "held-out set of positions sampled from generation 1's self-play. "
            "These are the AlphaZero-style training health curves — they "
            "isolate the network's learning from MCTS noise."
            "</p>",
        ]
        if fig_network_entropy is not None:
            parts.append(_chart(fig_network_entropy))
        if fig_policy_accuracy is not None:
            parts.append(_chart(fig_policy_accuracy))
        if fig_value_calibration is not None:
            parts.append(_chart(fig_value_calibration))
        if fig_symmetry is not None:
            parts.append(_chart(fig_symmetry))
        parts.append("</section>")
        diagnostics_html = "\n".join(parts)

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

{pentobi_ladder_html}

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
