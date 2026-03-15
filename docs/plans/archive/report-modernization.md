# Report Modernization

Modernize the HTML training report to use all available metrics data and fix structural issues.
The existing report (`reporting.py`) was written early and only uses 3 of 7 data sources.
This plan adds the missing charts, fixes the broken HTML, and improves the layout.

**Scope boundary:** This plan covers charts derived from data the framework already collects.
BayesElo computation, external benchmark reporting (Pentobi heatmaps, composite scores), and
the TicTacToe solver are **out of scope** — see the [Recommendation](#recommendation-tictactoe-solver--benchmark-architecture) section at the bottom for phasing.

**Prerequisite:** Familiarity with `docs/05-EVALUATION.md` (what we want to measure) and `docs/07-DATA-STORAGE.md` (what data exists).

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| R1 | Fix HTML document structure | 20 min | High | ✅ |
| R2 | Add run config summary table | 20 min | High | ✅ |
| R3 | Split loss curves into pi_loss and v_loss | 20 min | High | ✅ |
| R4 | Enhance arena chart with acceptance annotations and draws | 30 min | High | ✅ |
| R5 | Replace timing subplots with stacked bar chart | 20 min | Medium | ✅ |
| R6 | Add resource usage chart | 30 min | Medium | ✅ |
| R7 | Add self-play profiling charts | 40 min | Medium | ✅ |
| R8 | Add training throughput chart | 20 min | Medium | ✅ |
| R9 | Smoke test and verify report renders correctly | 15 min | High | ✅ |

---

## R1. Fix HTML document structure

**Current state:** The report writes `</body></html>` in the header, then appends chart divs after the document close. The Plotly CDN script is loaded separately in every chart div. There are no section headings, no CSS, and no semantic structure.

**Fix:** Proper HTML5 document with:
- Single `<html>`, `<head>`, `<body>` structure
- Plotly JS loaded once in `<head>`
- Section headings (`<h2>`) grouping related charts
- Minimal inline CSS for readability (max-width container, font, spacing)
- Charts wrapped in `<section>` elements

**Report sections (in order):**

1. **Header** — Title, run name, generation count
2. **Overview** — Run config summary table (R2)
3. **Training** — Loss curves (R3), training throughput (R8)
4. **Arena** — Win/loss/draw with annotations (R4)
5. **Performance** — Timing breakdown (R5), resource usage (R6), self-play profiling (R7)

**Plotly embedding:** Switch from `include_plotlyjs='cdn'` (repeated per chart) to loading the CDN script once in `<head>`, then using `include_plotlyjs=False` for each chart's `to_html()` call.

---

## R2. Add run config summary table

**Current state:** The report shows only the run name. No context about what parameters produced this run.

**Fix:** Add an HTML table at the top showing key config values:

| Parameter | Value |
|-----------|-------|
| Game | tictactoe |
| Generations | 3 |
| Episodes per generation | 15 |
| MCTS simulations | 25 |
| Arena matches | 6 |
| Update threshold | 0.55 |
| Learning rate | 0.001 |
| Batch size | 32 |
| Residual blocks | 2 |

Pull these from the `RunConfig` object that's already passed to `create_html_report()`. No new data source needed.

---

## R3. Split loss curves into pi_loss and v_loss

**Current state:** Both loss charts show only `average_loss` (= `average_pi_loss + average_v_loss`). The eval doc says to track pi_loss and v_loss separately because they diagnose different problems (policy not learning vs value not calibrating).

**Fix for Figure 1 (loss per generation):**
- Three lines: `average_pi_loss`, `average_v_loss`, `average_loss` (total)
- Keep total as the primary (solid, bold), pi and v as secondary (dashed)
- Title: "Loss per Generation"

**Fix for Figure 2 (loss by batch):**
- Replace the single scatter with a 1x3 faceted subplot: pi_loss, v_loss, total_loss
- Keep epoch as color dimension, generation in hover
- Title: "Per-Batch Loss by Component"

Data is already in the TrainingData parquet (`average_pi_loss`, `average_v_loss` columns). No collection changes needed.

---

## R4. Enhance arena chart with acceptance annotations and draws

**Current state:** Shows only win% and loss% as stacked bars. No draw%, no indication of whether the model was accepted or rejected, no threshold line.

**Fix:**
- Three-segment stacked bar: wins, draws, losses (green, grey, red)
- Horizontal line at `update_threshold` (from config) — the acceptance boundary
- Text annotation above each bar: "Accepted" or "Rejected"
- Derive acceptance from: `wins / (wins + losses + draws) > update_threshold`
- The threshold needs to be passed to the report function (add parameter or pass full config)

Data is in ArenaData parquet. The only new input is the threshold value from config.

---

## R5. Replace timing subplots with stacked bar chart

**Current state:** Four separate bar subplots (SelfPlay, Training, Arena, WholeCycle) with shared y-axis. Hard to compare phases within a generation.

**Fix:** Single stacked bar chart:
- X: generation
- Y: time (seconds)
- Segments: SelfPlay, Training, Arena (stacked)
- Drop the WholeCycle segment (it's the sum of the others, redundant in a stacked chart)
- Color: one color per phase
- Title: "Time per Generation"

Same Timings parquet data, just a different visualization.

---

## R6. Add resource usage chart

**Current state:** ResourceUsage parquet exists with `cycle_stage`, `process_rss_bytes`, `gpu_memory_bytes` columns. Not shown in report.

**Fix:** Grouped bar chart:
- X: generation
- Y: memory in MB (divide bytes by 1024^2)
- Groups: one bar per cycle_stage (SelfPlay, Training, Arena)
- Title: "Process Memory (RSS) per Generation"
- If GPU memory is non-null for any row, add a second chart for GPU memory

This is important for Blokus — the 14x14 board with 44 channels will use significantly more memory than TicTacToe. Tracking this early catches memory leaks.

---

## R7. Add self-play profiling charts

**Current state:** SelfPlayProfiling parquet has per-episode data: `num_moves`, `total_sims`, `sims_per_second`, `inference_fraction`, `tree_size`. Not shown in report.

**Fix:** 2x2 subplot grid:

1. **Game Length** (top-left): Box plot of `num_moves` per generation. Shows distribution and whether games get longer as the net improves.
2. **MCTS Throughput** (top-right): Box plot of `sims_per_second` per generation. Key performance metric — should stay stable or improve.
3. **Inference Fraction** (bottom-left): Box plot of `inference_fraction` per generation. Shows how much of MCTS time is neural net evaluation vs tree operations.
4. **Tree Size** (bottom-right): Box plot of `tree_size` per generation. Shows search depth/breadth.

Box plots because we have 15 episodes per generation — enough to show distribution. For runs with many more episodes, could switch to violin plots.

Title: "Self-Play Profiling"

---

## R8. Add training throughput chart

**Current state:** TrainingThroughput parquet has per-epoch data: `num_examples`, `epoch_time_s`, `samples_per_second`. Not shown in report.

**Fix:** Dual-axis line chart:
- Left Y: `samples_per_second` (line with markers)
- Right Y: `epoch_time_s` (dashed line)
- X: epoch index (sequential across all generations, with generation boundaries marked)
- Title: "Training Throughput"

Alternative (simpler): Bar chart of average `samples_per_second` per generation. This is probably cleaner for the report — save the per-epoch detail for notebook analysis.

Go with the simpler bar chart: average throughput per generation.

---

## R9. Smoke test and verify report renders correctly

Run the smoke test config and verify:
- Report HTML is valid (no unclosed tags)
- All charts render (no blank sections)
- Plotly interactivity works (hover, zoom)
- Section headings and config table display correctly
- Test with both file:// and a local HTTP server

---

## Recommendation: TicTacToe Solver & Benchmark Architecture

Henry asked whether it's worth building a TicTacToe solver as a Pentobi substitute to test the benchmark reporting pipeline. Here's the analysis.

### What we'd get from a TicTacToe solver

The eval doc (`05-EVALUATION.md`) describes several visualizations that require an **external benchmark opponent**:

- **Pentobi Heatmap** — win rate by level by generation (the "money plot")
- **Composite scores** — Pentobi Level, Pentobi Score, Pentobi Weighted Score over time
- **BayesElo curve** — internal Elo from arena results (doesn't need external opponent, but is closely related)
- **Win-Rate Profile** — per-level win rates

None of these can be built or tested with just self-play data. A TicTacToe solver with difficulty levels would give us a stand-in opponent to develop and test all this reporting code.

### How it would work

TicTacToe is solved — minimax produces perfect play in microseconds. We can create artificial difficulty levels:

| Level | Strategy | Expected Win Rate (vs trained net) |
|-------|----------|-----------------------------------|
| 1 | Random moves | ~95%+ |
| 2 | Win if possible, else random | ~80% |
| 3 | Win if possible, block if possible, else random | ~60% |
| 4 | Minimax depth 2 | ~40% |
| 5 | Perfect play (full minimax) | ~10% (mostly draws) |

This mirrors Pentobi's 9 levels enough to test the full pipeline: benchmark runner, per-level results storage, heatmap generation, composite score computation, Elo tracking.

### Three approaches

**Approach A: Report modernization only (this plan)**
- Add charts for existing data, fix HTML
- No external benchmark reporting
- Pentobi/benchmark charts built later when Blokus is ready

**Approach B: This plan + TicTacToe solver + benchmark reporting (all at once)**
- Everything in one shot
- Risk: scope creep, large PR, mixing concerns

**Approach C: Two-phase (recommended)**
- **Phase 1 (this plan):** Modernize report with existing data. ~3 hours.
- **Phase 2 (separate plan):** TicTacToe solver + benchmark runner + BayesElo + heatmap/composite reporting. ~1 day.

### Recommendation: Phase it (Approach C)

Phase 1 is self-contained and immediately useful — every future training run gets a better report. Phase 2 is the higher-value work (the heatmap and Elo curves are the plots we'll actually show people), but it needs architectural decisions about the benchmark runner interface, result storage format, and how to abstract over "Pentobi" vs "TicTacToe solver."

Doing Phase 2 separately means we can:
1. Design the benchmark runner interface properly (Protocol-based, game-agnostic)
2. Test it against the TicTacToe solver end-to-end
3. Swap in Pentobi (via GTP) when Blokus move generation is done
4. Keep PRs focused and reviewable

The TicTacToe solver itself is ~30 lines. The real work in Phase 2 is the benchmark runner, result storage, and reporting — and that's exactly the code we want battle-tested before Blokus.
