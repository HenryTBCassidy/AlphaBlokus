# Run Reporting Refactor

Rebuild both run-reporting surfaces around one question — **is this run improving, or fooling
itself?** — and stop burying the honest signals. The requirements come from the measurement
post-mortems: during the `blokus_paired_gate_rerun` regression (L4 → L3, −44 pool Elo) every
prominently-displayed metric looked healthy (loss ↓, acceptance 100%, eval top-1 0.99) while the
only externally-anchored warnings — policy symmetry KL and value symmetry MAE — flashed red from
generation 5 in a parquet nobody was looking at
([`regression-and-next-steps.md`](../../research/regression-and-next-steps.md) §1.5,
[`plateau-investigation.md`](../../research/plateau-investigation.md) R8,
[`post-regression-recovery.md`](post-regression-recovery.md)).

Two surfaces: the live W&B dashboard (`scripts/setup_wandb_workspace.py`) and the end-of-run HTML
report (`src/alphablokus/reporting/`). Prior reporting passes:
[`report-modernization.md`](report-modernization.md), [`report-redesign.md`](report-redesign.md),
[`reporting-overhaul.md`](reporting-overhaul.md) — this plan replaces their Plotly presentation
layer but keeps their data model (hive parquet tables) and their headline interactive feature
(the arena replay browser) intact.

**Archived 2026-08-03 — outcome.** All items landed. Both reference runs render correctly:
`blokus_paired_gate_rerun` front-loads its regression (pool-Elo alert, both symmetry trends
alerting, gen-17 entropy collapse, 91% colour pinning) and `blokus_cloud_v3` reads as
"externally mixed" (keep-best gen-40 by ladder, pool Elo peaked at gen 32) — exactly the two
verdicts the post-mortems reached by hand.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| R1 | Payload layer (`reporting/data.py`): every table → JSON-ready dicts, all optional, signal statuses + verdict computed with documented thresholds | 4 h | High | ✅ |
| R2 | Compact replay payloads (`reporting/arena_replays.py`): board-diffed move cells + alternatives instead of pre-rendered per-turn HTML | 2 h | High | ✅ |
| R3 | Client-side report shell (`reporting/report.py` + `assets/report.{css,js}`): one self-contained HTML file, no CDN, hand-rolled SVG charts, light/dark | 6 h | High | ✅ |
| R4 | Delete the Plotly chart layer (`reporting/charts.py`) + trim `pentobi_ladder.py` to persistence only | 30 min | High | ✅ |
| R5 | Signal calibration against both reference runs (rerun must alert; v3 must not) | 1 h | High | ✅ |
| R6 | Tests: red flags, trend statuses, ladder keep-best/drift, replay payloads, end-to-end render, empty-run degradation | 2 h | High | ✅ |
| R7 | W&B workspace cut-down (`scripts/setup_wandb_workspace.py`): externally-anchored section first, ~30 panels → 17 | 1 h | High | ✅ |
| R8 | Docs: update layout references (AGENTS/README/STYLE-GUIDE/05/07), archive this plan | 30 min | Medium | ✅ |

---

## R1. Payload layer

**Current state:** `report.py` read ~14 parquet tables and handed DataFrames to 1,300 lines of
Plotly figure builders; four tables were loaded unconditionally (crashing on partially-synced run
directories), the Pentobi ladder was a bare table, and the mini-ladder keep-best / drift state
(post-regression-recovery P3/P4 — the run's actual selection instrument) appeared nowhere.

**Fix:** `reporting/data.py::build_report_payload(config)` reduces every table to plain
JSON-serialisable dicts. Rules:

- **Every table optional.** Absence renders as an explicit "not recorded" placeholder and is
  listed in the config section — absence of evidence must be visible, never silent.
- **Signals + verdict computed in Python** (testable), thresholds as module constants, each citing
  the run that motivated it: symmetry-trend ratios (rerun rose 1.9×/2.5×), target-entropy collapse
  (gen 17: 0.506 vs median ~0.85), sub-binomial variance and exact-0.500 checks (plateau R8c),
  colour pinning (≥85% white), pool-Elo slippage (final below gen-0 anchor = worse than donor).
- **Ladder section** merges `PentobiLadder/*.json` with `MiniLadder/history.json`, recomputing
  keep-best + drift via `evaluation/ladder_selection.py` so the report always shows the same
  verdict the box runbook acts on, and surfaces a `DRIFT_ALARM` flag file directly.

## R2. Compact replay payloads

**Current state:** each replay turn was pre-rendered server-side as two full HTML board tables
(~126k rows → megabytes of markup), navigated by scrolling a long list of turn cards.

**Fix:** per move, emit only what changed: `cells` (diffed from `IBoard.as_2d`, so game-agnostic),
a decoded caption, the MCTS visit share, and up to three alternative moves as ghost-cell overlays
(simulated via `game.get_next_state` on the pre-move board). The browser replays deltas onto a
single SVG board with a scrubber, ◀/▶ steppers and arrow-key navigation; selecting an alternative
previews it (striped) on the pre-move board. Sampling caps unchanged (≤16 generations × 6
games/gen, hive-partition-pruned reads — oom-hardening O7).

## R3. Client-side report shell

**Fix:** `report.html` = HTML shell + inline CSS + inline JS + one `<script type="application/json">`
payload. No Plotly CDN (the old report silently broke offline), no frameworks, no build step —
`uv run alphablokus --config <cfg> --report-only` and double-click the file. Page order:

1. **Verdict banner** — the one-line answer, derived only from externally-anchored signals.
2. **Signal tiles** — ladder, pool Elo, symmetry KL, value MAE, target entropy, instrument health.
3. **External evidence** (badged "cannot be gamed by the loop") — ladder heat-table with
   keep-best/drift, pooled BayesElo, symmetry KL, value MAE, target entropy band.
4. **Arena instrument** (badged "measurement health") — red-flag banner, score-vs-threshold,
   white/black decisive split, rolling Elo (labelled chained/self-referential).
5. **Training telemetry** (badged "self-referential", with an explicit warning note) — loss,
   eval agreement, PVC, calibration, net entropy, LR.
6. **Arena replays** — the game browser (R2).
7. **Operations** (timings, throughput, game length, memory) and collapsed config.

Charts are ~450 lines of hand-rolled SVG (line/band/stacked-bar, shared tooltip, legend toggles),
themed from CSS variables and re-rendered on theme toggle/resize. Assets ship as package data
(`importlib.resources`), matching the `pieces.json` pattern.

## R4. Delete the Plotly chart layer

`reporting/charts.py` and its four chart tests deleted (style guide: delete dead code);
`pentobi_ladder.py` keeps `parse_levels` / `write_ladder_result` / `load_ladder_results` — the
schema contract with `scripts/pentobi_benchmark.py` and `scripts/mini_ladder.py` — and loses its
HTML section builder. `plotly` stays a dependency (`reporting/mcts_profiling.py` still uses it).

## R5. Signal calibration against both reference runs

Pre-registered acceptance: rendered against real run directories, the rerun must alert and v3 must
not false-alarm. Result: rerun → "Regression signals present" (pool Elo −44 below anchor, both
symmetry trends alerting, gen-17 collapse flagged, 91% white pinning); v3 → "External signals
mixed" (keep-best `accepted_40` by ladder 0.344/L4; pool Elo warns the final net is not the best
net — peak +286 at gen 32; instrument red-flags its 16 exact-0.500 clone-split generations).
The pool-Elo rule needed one iteration: alert on *final below the gen-0 anchor* (worse than donor)
or a catastrophic slide, warn on ending well below peak.

## R6. Tests

`tests/reporting/test_report_data.py` + `test_arena_replays.py`: R8c red flags (exact-0.500,
sub-binomial, colour pinning), trend statuses replayed on both runs' trajectories, target-entropy
collapse, ladder keep-best + drift alarm (rerun-shaped history trips at 2 consecutive drops),
aux-loss series parity with the old chart tests, replay cell-diff/alternative filtering on
TicTacToe, an end-to-end `create_html_report` render, and the all-tables-missing payload.

## R7. W&B workspace cut-down

`build_sections()` rewritten: "Is it improving? — externally anchored" (symmetry KL mean+max,
value MAE, target entropy, white-win share) is the first open section; then self-referential
strength/gate telemetry (rolling Elo, score rate, acceptance), training (per-gen losses, smoothed
per-batch, LR), progress (generation, ETA), and a collapsed operational section. Cut: per-episode
duplicates, epoch/episode/batch sawtooths, top-5 agreement, minimax (TTT-only) panels, and the
retired `elo/rating` frozen-baseline keys. ~30 panels → 17. Re-run against `alphablokus-poc` to
upsert the saved view.

## R8. Docs

Layout one-liners updated in `AGENTS.md`, `README.md`, `docs/guides/STYLE-GUIDE.md`,
`docs/05-EVALUATION.md`, `docs/07-DATA-STORAGE.md`; this plan archived.
