# Report Redesign

Ground-up redesign of the HTML training report. The current report (from the
[report-modernization](archive/report-modernization.md) plan) has the right data and charts but
the presentation layer — layout, hierarchy, styling, context — needs a rethink.

This plan is informed by:
- **AlphaGo / AlphaGo Zero / AlphaZero / MuZero papers** — Elo-vs-time line charts with annotated
  reference lines, restrained 2-3 color palettes, white backgrounds, multi-panel layouts
- **W&B / TensorBoard / MLflow dashboards** — KPI cards at top, 2-column grids, smoothed loss
  curves with raw data behind, collapsible sections, `plotly_white` template
- **`docs/05-EVALUATION.md`** — what metrics matter and what "healthy" looks like

**Scope:** Presentation-layer redesign only. No new data collection or metrics computation.
Same 7 parquet data sources, same `create_html_report(config)` entry point.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| D1 | Switch all charts to `plotly_white` template and consistent color palette | 20 min | High | ✅ |
| D2 | Add KPI summary cards at the top of the report | 30 min | High | ✅ |
| D3 | Restructure HTML layout: CSS grid, collapsible sections, section descriptions | 40 min | High | ✅ |
| D4 | Redesign loss-per-generation chart | 15 min | Medium | ✅ |
| D5 | Replace per-batch scatter with smoothed line chart | 30 min | Medium | ✅ |
| D6 | Resize performance charts for 2-column grid | 15 min | Medium | ✅ |
| D7 | Move config table to collapsible section at bottom | 10 min | Low | ✅ |
| D8 | Smoke test, visual review, polish | 20 min | High | ✅ |

---

## D1. Switch all charts to `plotly_white` template and consistent color palette

**Current state:** Charts use the default Plotly template with grey `#E5ECF6` backgrounds. Colors
are a mix of Plotly defaults, hardcoded hex, and `px.colors.qualitative.Plotly`. Every DeepMind
paper and every professional ML dashboard uses a white background with a restrained palette.

**Fix:** Define a module-level palette and apply `plotly_white` to every figure.

```python
# Palette — used consistently across all charts
_COLORS = {
    "primary": "#636efa",     # Blue — main metric, primary series
    "secondary": "#EF553B",   # Red-orange — secondary series, losses
    "tertiary": "#00cc96",    # Green — tertiary series, wins
    "neutral": "#aaaaaa",     # Grey — draws, inactive, background
    "accent": "#ab63fa",      # Purple — highlights, annotations
}

_PHASE_COLORS = {
    "SelfPlay": _COLORS["primary"],
    "Training": _COLORS["secondary"],
    "Arena": _COLORS["tertiary"],
}
```

Apply to every figure via:
```python
fig.update_layout(template="plotly_white")
```

Also remove the hardcoded `marker_color` values scattered through chart functions and replace
with palette references. The loss-per-generation chart's dashed/dotted lines should use the
palette too (total=primary, pi=secondary, v=tertiary).

---

## D2. Add KPI summary cards at the top of the report

**Current state:** The report opens with "Run: smoke_test — 3 generations" then immediately
shows the config table. No at-a-glance summary of training health.

**Fix:** Add a row of 4-5 KPI cards between the header and the first section. Each card shows
a metric value, label, and delta/context indicator.

**Cards (computed from existing parquet data):**

| Card | Value | Context | Source |
|------|-------|---------|--------|
| Final Loss | Last `average_loss` from final generation | Delta vs first generation (e.g. "▼ 32%") | TrainingData |
| Accept Rate | Accepted generations / total generations | e.g. "2 of 3 generations" | ArenaData |
| Total Time | Sum of all WholeCycle timings | Formatted as seconds/minutes | Timings |
| Self-Play Speed | Median `sims_per_second` across all episodes | "sims/s" | SelfPlayProfiling |
| Training Speed | Mean `samples_per_second` across all epochs | "samples/s" | TrainingThroughput |

**Implementation:** A `_make_kpi_cards(...)` function that takes the dataframes and returns an
HTML string. CSS flexbox row with cards as equal-width items.

```css
.kpi-grid {
    display: flex;
    gap: 16px;
    margin: 24px 0;
}
.kpi-card {
    flex: 1;
    padding: 16px 20px;
    border-radius: 8px;
    background: #f8f9fb;
    border: 1px solid #e5e7eb;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #1a1a2e;
}
.kpi-label {
    font-size: 13px;
    color: #6b7280;
    margin-top: 4px;
}
.kpi-delta {
    font-size: 13px;
    margin-top: 4px;
}
.kpi-delta.positive { color: #2ca02c; }
.kpi-delta.negative { color: #d62728; }
```

---

## D3. Restructure HTML layout: CSS grid, collapsible sections, section descriptions

**Current state:** All charts are full-width `<section>` blocks stacked vertically. No grid,
no collapsible sections, no descriptive text. The page is long and offers no guidance on what
to look for.

**Fix — layout structure:**

```
Header + subtitle (game, generations, date)
KPI cards (D2)

── Training ───────────────────────────────
  Description text
  [Loss per Generation — full width]
  ▸ Per-Batch Detail (collapsed)
    [Smoothed loss lines — full width]
  [Throughput — full width]

── Arena ──────────────────────────────────
  Description text
  [Arena stacked bar — full width]

── Performance ────────────────────────────
  Description text
  ┌──────────────────┐  ┌──────────────────┐
  │ Time/Generation   │  │ Throughput        │
  └──────────────────┘  └──────────────────┘
  ┌──────────────────┐  ┌──────────────────┐
  │ Memory            │  │ MCTS Profiling    │
  └──────────────────┘  └──────────────────┘

▸ Configuration (collapsed by default)
  [Config table]
```

**CSS grid for 2-column performance layout:**

```css
.chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
```

**Collapsible sections** use native `<details><summary>`:

```html
<details>
  <summary>Per-Batch Detail</summary>
  <!-- chart here -->
</details>
```

Style the `<summary>` to look like a clickable section header (cursor pointer, chevron).

**Section descriptions** — 1-2 sentences under each `<h2>` explaining what to look for,
drawn from `docs/05-EVALUATION.md`:

- **Training:** "Policy and value loss should decrease across generations. Value loss below
  1.0 means the network predicts outcomes better than random."
- **Arena:** "Each generation's new network plays the incumbent. It must win above the
  threshold to be accepted."
- **Performance:** "Time and resource usage per generation. Training time grows as the
  example buffer accumulates."

---

## D4. Redesign loss-per-generation chart

**Current state:** Three lines (total, pi, v) with different dash styles on the default
template. The total line is bold, pi/v are dashed/dotted. Works but is visually basic.

**Fix:**
- Switch to `plotly_white` (D1)
- Use solid lines for all three with distinct palette colors (total=primary, pi=secondary, v=tertiary)
- Increase line width slightly (2px for pi/v, 3px for total)
- Add markers at each generation point
- Add a subtle grey annotation band if value loss is above 1.0 (the "worse than random" zone
  from the eval doc), with a text label "Worse than random"

The annotation band is a `fig.add_hrect()`:
```python
fig.add_hrect(y0=1.0, y1=max_y, fillcolor="#d62728", opacity=0.05,
              line_width=0, annotation_text="Worse than random",
              annotation_position="top left")
```

Only add this if the value loss ever exceeds 1.0 (it will in early training).

---

## D5. Replace per-batch scatter with smoothed line chart

**Current state:** 1x3 faceted scatter plot with 3px dots coloured by epoch. With hundreds of
data points across generations, this is noisy and hard to interpret. The industry standard
(W&B, TensorBoard) is a smoothed line with the raw data as a faint trace behind.

**Fix:** Replace with a single full-width chart showing smoothed loss over the full training
timeline (sequential batch index across all generations).

- X-axis: sequential batch index (0 to total batches across all generations)
- Y-axis: loss value
- Three smoothed lines: pi_loss, v_loss, total_loss (same palette as D4)
- Raw data shown as faint points (opacity ~0.15) behind the smoothed lines
- Vertical dashed lines at generation boundaries with generation number labels
- Smoothing via exponential moving average (pandas `.ewm(span=...)`)

This is much more readable than the current scatter and follows the W&B/TensorBoard pattern
exactly. The generation boundaries let you see where new self-play data entered training.

Wrap in a `<details>` element so it's collapsed by default — the per-generation summary
chart (D4) is the primary view, this is the drill-down.

---

## D6. Resize performance charts for 2-column grid

**Current state:** All performance charts are 1200px wide, stacked vertically. The timing,
throughput, memory, and profiling charts would work better in a 2x2 grid.

**Fix:**
- Add a `half_width` parameter to chart functions, defaulting chart width to 580px
  (fits 2 columns in the 1200px container with 16px gap)
- Reduce chart height to 400px for grid charts (from 500-700px)
- Wrap the 4 performance charts in a `<div class="chart-grid">`:
  - Top-left: timing stacked bar
  - Top-right: throughput bar
  - Bottom-left: memory grouped bar
  - Bottom-right: MCTS profiling (simplified — see below)

**Simplify MCTS profiling for grid:** The current 2x2 box plot subplot is too dense for a
half-width card. Replace with a simpler 2-row layout:
- Top: game length box plot (the most meaningful metric)
- Bottom: sims/sec box plot (the key performance metric)
- Drop inference fraction and tree size from the main view (they're niche — could go in a
  collapsible detail section if needed)

---

## D7. Move config table to collapsible section at bottom

**Current state:** Config table is the first section after the header, taking prime visual
real estate.

**Fix:** Move to the very end of the report, wrapped in `<details>` collapsed by default:

```html
<details>
  <summary><h2>Configuration</h2></summary>
  <!-- config table -->
</details>
```

Nobody opens a training report to check the learning rate. They want to know if training is
working. Config is reference material — useful when you need it, not when you don't.

---

## D8. Smoke test, visual review, polish

Run the smoke test config and open the report. Check:
- KPI cards render correctly with proper values and deltas
- Charts use white background consistently
- 2-column grid renders correctly (no overflow, no squished charts)
- Collapsible sections expand/collapse properly
- Section descriptions are readable
- Loss smoothing looks right (not too smooth, not too noisy)
- Colors are consistent across all charts
- No visual regressions from the previous report version

Open in both Chrome and Safari to check cross-browser rendering.
