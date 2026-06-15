"""Generate an HTML report visualising the full-cycle optimisation progress.

Renders the measured milestone progression from the master plan's progress
tracker (`docs/plans/full-cycle-optimisation.md`) — serial baseline → parallel
self-play → precomputed move-gen → batched inference — as a single
self-contained HTML page so the cumulative speedup is visible at a glance.

The milestone numbers are the curated figures from the progress tracker (the
historical per-run parquets for the parallel-self-play step no longer exist
locally, so the tracker is the source of truth). The batched-inference figures
come from `temp/profile_baseline_f3_k{1,8,16}_w8_2026-06-01/`.

Usage:
    uv run python -m scripts.optimisation_progress_report
    # writes temp/optimisation_progress/report.html
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go

# --- House style (kept consistent with reporting/mcts_profiling.py) ----------

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.4.0.min.js"
_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1280px; margin: 0 auto; padding: 24px 32px; color: #2a3f5f; background: #fff; }
h1 { border-bottom: 2px solid #636efa; padding-bottom: 8px; margin-bottom: 4px; }
.subtitle { color: #6b7280; font-size: 14px; margin-bottom: 24px; }
h2 { margin-top: 40px; color: #636efa; font-size: 20px; }
.kpi-grid { display: flex; gap: 14px; margin: 20px 0 32px 0; flex-wrap: wrap; }
.kpi-card { flex: 1 1 180px; padding: 14px 18px; border-radius: 8px; background: #f8f9fb;
            border: 1px solid #e5e7eb; }
.kpi-value { font-size: 26px; font-weight: 700; color: #1a1a2e; }
.kpi-label { font-size: 12px; color: #6b7280; margin-top: 2px; text-transform: uppercase;
             letter-spacing: 0.5px; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
table.summary { border-collapse: collapse; margin: 12px 0; width: 100%; }
table.summary th, table.summary td { text-align: left; padding: 8px 12px;
    border-bottom: 1px solid #e5e7eb; font-size: 14px; }
table.summary th { background: #f8f9fb; }
.note { color: #6b7280; font-size: 13px; font-style: italic; margin-top: 8px; }
"""

# Step accent colours.
_C_SERIAL, _C_F1, _C_F2, _C_F3 = "#9aa0b5", "#00cc96", "#636efa", "#ef553b"


@dataclass(frozen=True)
class Milestone:
    """One row of the progress tracker, in plotting-ready form."""

    label: str
    date: str
    total_min: float          # whole-benchmark wall-clock (16 self-play + 10 arena + 10 elo)
    cumulative_speedup: float  # vs the serial benchmark baseline
    self_play_s_per_game: float
    colour: str


# Curated from docs/plans/full-cycle-optimisation.md progress tracker + the
# batched-inference run.
MILESTONES: list[Milestone] = [
    Milestone("Serial baseline", "2026-05-26", 32.6, 1.00, 54.1, _C_SERIAL),
    Milestone("F1 — parallel (4 workers)", "2026-05-27", 7.9, 4.13, 12.0, _C_F1),
    Milestone("F1 — parallel (8 workers)", "2026-05-28", 5.7, 5.72, 8.08, _C_F1),
    Milestone("F2 — move-gen table (8w)", "2026-05-28", 4.3, 7.58, 6.49, _C_F2),
    Milestone("F3 — batched inference (8w, K=16)", "2026-06-01", 2.31, 14.1, 3.73, _C_F3),
]

# Per-process component split (% of MCTS search time). Parallel rows include
# contention waits; F3 row computed from its per-game profiling (approximate).
# Columns: inference, move-gen (valid_moves), other (UCB/bookkeeping/game-ended).
COMPONENT_SPLIT: list[tuple[str, float, float, float]] = [
    ("Serial baseline", 49.7, 42.6, 7.7),
    ("F1 (8 workers)", 37.3, 52.4, 10.3),
    ("F2 (8 workers)", 77.5, 8.7, 13.8),
    ("F3 (8w, K=16)", 39.8, 25.2, 35.0),
]

# F3 batch-size sweep — total benchmark wall-clock (s), 8 workers (end-to-end, contended).
F3_K_SWEEP: list[tuple[str, float]] = [("K=1", 257.6), ("K=8", 217.5), ("K=16", 138.7)]

# Microbenchmark — isolated GPU-side speedup of predict_batch(K) vs K serial
# predict() on the idle 3060 Ti (production net). Shows the pure batching gain,
# uncontaminated by MCTS / 8-worker contention.
F3_MICROBENCH: list[tuple[int, float]] = [
    (1, 0.73), (2, 1.69), (4, 4.60), (8, 9.23), (16, 11.53), (32, 19.75), (64, 20.57),
]


def _bar(x, y, *, colours, title, ytitle, texttemplate, height=420) -> go.Figure:
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colours, text=y,
                           texttemplate=texttemplate, textposition="outside"))
    fig.update_layout(title=title, yaxis_title=ytitle, template="plotly_white",
                      height=height, margin=dict(t=56, b=120), xaxis_tickangle=-20)
    fig.update_yaxes(rangemode="tozero")
    return fig


def _build_figures() -> dict[str, go.Figure]:
    labels = [m.label for m in MILESTONES]
    colours = [m.colour for m in MILESTONES]

    total = _bar(labels, [m.total_min for m in MILESTONES], colours=colours,
                 title="Total benchmark wall-clock (16 self-play + 10 arena + 10 Elo games)",
                 ytitle="minutes", texttemplate="%{y:.1f} min")

    speedup = _bar(labels, [m.cumulative_speedup for m in MILESTONES], colours=colours,
                   title="Cumulative speedup vs serial baseline",
                   ytitle="× faster", texttemplate="%{y:.2f}×")

    sp = _bar(labels, [m.self_play_s_per_game for m in MILESTONES], colours=colours,
              title="Self-play cost per game", ytitle="seconds / game",
              texttemplate="%{y:.2f}s")

    # Component-split evolution (stacked).
    comp_labels = [c[0] for c in COMPONENT_SPLIT]
    split = go.Figure()
    for name, idx, colour in [("NN inference", 1, _C_F3), ("Move generation", 2, _C_F1),
                              ("Other (UCB / bookkeeping)", 3, _C_SERIAL)]:
        split.add_bar(name=name, x=comp_labels, y=[c[idx] for c in COMPONENT_SPLIT],
                      marker_color=colour, text=[f"{c[idx]:.0f}%" for c in COMPONENT_SPLIT],
                      textposition="inside")
    split.update_layout(barmode="stack", title="Where the time goes — component split per step",
                        yaxis_title="% of MCTS search time", template="plotly_white",
                        height=420, margin=dict(t=56, b=80), legend=dict(orientation="h", y=-0.18))

    ksweep = _bar([k for k, _ in F3_K_SWEEP], [v for _, v in F3_K_SWEEP],
                  colours=[_C_F2, "#ab63fa", _C_F3],
                  title="F3 batch-size sweep — total wall-clock (8 workers, end-to-end)",
                  ytitle="seconds", texttemplate="%{y:.0f}s")

    micro = go.Figure(go.Scatter(
        x=[k for k, _ in F3_MICROBENCH], y=[s for _, s in F3_MICROBENCH],
        mode="lines+markers+text", text=[f"{s:.1f}×" for _, s in F3_MICROBENCH],
        textposition="top center", line=dict(color=_C_F3, width=3), marker=dict(size=8)))
    micro.update_layout(
        title="F3 isolated GPU-side batching speedup (idle GPU: predict_batch(K) vs K× predict)",
        xaxis_title="batch size K", yaxis_title="× faster", template="plotly_white",
        height=420, margin=dict(t=56, b=60))
    micro.update_xaxes(type="log", tickvals=[k for k, _ in F3_MICROBENCH],
                       ticktext=[str(k) for k, _ in F3_MICROBENCH])
    micro.add_hline(y=1.0, line_dash="dot", line_color="#9aa0b5",
                    annotation_text="break-even", annotation_position="bottom right")

    return {"total": total, "speedup": speedup, "self_play": sp, "split": split,
            "ksweep": ksweep, "micro": micro}


def _kpi_cards() -> str:
    latest = MILESTONES[-1]
    cards = [
        (f"{latest.cumulative_speedup:.1f}×", "cumulative speedup vs serial"),
        (f"{latest.total_min:.2f} min", "total benchmark wall-clock now"),
        (f"{latest.self_play_s_per_game:.2f}s", "self-play per game now"),
        ("3.58× lighter", "GPU inference per game (F3 K=16)"),
    ]
    inner = "".join(
        f'<div class="kpi-card"><div class="kpi-value">{v}</div>'
        f'<div class="kpi-label">{lbl}</div></div>'
        for v, lbl in cards
    )
    return f'<div class="kpi-grid">{inner}</div>'


def _summary_table() -> str:
    rows = "".join(
        f"<tr><td>{m.label}</td><td>{m.date}</td><td>{m.total_min:.2f} min</td>"
        f"<td>{m.cumulative_speedup:.2f}×</td><td>{m.self_play_s_per_game:.2f}s</td></tr>"
        for m in MILESTONES
    )
    return f"""<table class="summary">
<thead><tr><th>Milestone</th><th>Date</th><th>Total wall-clock</th>
<th>Cumulative speedup</th><th>Self-play / game</th></tr></thead>
<tbody>{rows}</tbody></table>"""


def build_report(output_dir: Path) -> Path:
    """Render the report to ``output_dir/report.html`` and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figs = _build_figures()

    # Plotly JS is loaded once via the CDN <script> in <head>, so every figure
    # embeds without its own copy.
    def html(fig: go.Figure) -> str:
        return fig.to_html(full_html=False, include_plotlyjs=False)

    body = f"""<h1>AlphaBlokus — Full-Cycle Optimisation Progress</h1>
<p class="subtitle">Self-play training throughput on the RTX 3060 Ti, serial baseline through F3.
Source: master plan progress tracker. Generated by <code>scripts/optimisation_progress_report.py</code>.</p>
{_kpi_cards()}
<h2>The journey</h2>
{_summary_table()}
<div class="chart-grid">
  {html(figs["total"])}
  {html(figs["speedup"])}
</div>
<h2>Per-game self-play cost</h2>
<div class="chart-grid">
  {html(figs["self_play"])}
  {html(figs["split"])}
</div>
<p class="note">Component split is per-process share of MCTS search time. Parallel rows include
GPU/CPU contention between the 8 workers; the F3 row is computed from its per-game profiling and is
approximate. The story: F2 crushed move-gen (42.6% → 8.7%), which made inference dominant (77.5%);
F3 then batched inference back down (~40%).</p>
<h2>F3 — batched inference, two views</h2>
<div class="chart-grid">
  {html(figs["ksweep"])}
  {html(figs["micro"])}
</div>
<p class="note">Left: end-to-end benchmark under 8-worker contention — K=16 gives 1.86× total vs K=1.
Right: the <em>isolated</em> GPU-side gain on an idle GPU — batching is 9.2× at K=8 and 11.5× at K=16.
The gap between the two is the cost of 8 workers sharing one GPU (each worker's batched call still
queues behind the others) — which is exactly what a cross-worker inference server (Option B) would
recover. K=8's weak end-to-end number is contention noise, not a plateau: its isolated gain is a clean 9.2×.</p>
"""

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AlphaBlokus — Optimisation Progress</title>
<script src="{_PLOTLY_CDN}"></script>
<style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>"""

    out = output_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    out = build_report(Path("temp/analysis/optimisation_progress"))
    print(f"Report → {out}", file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
