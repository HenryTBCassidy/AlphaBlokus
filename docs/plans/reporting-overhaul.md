# Reporting Overhaul

Bring the AlphaBlokus reporting layer up to AlphaZero-grade quality. Three concerns in one branch:

1. **Fix the bugs in the existing HTML report** — Accept Rate KPI is mathematically wrong, generation x-axis sorts alphabetically because of pandas `category` dtype, annotation overlaps, etc. The dashboard is currently lying about whether training worked.
2. **Add the diagnostic metrics that `docs/05-EVALUATION.md` says we should be measuring** — policy entropy, policy accuracy (top-1, top-5), value calibration, Elo vs frozen gen-0 baseline, minimax-vs-network for TTT.
3. **Make both reporting surfaces (live W&B + offline HTML) actually informative** — committed W&B Workspace layouts, new charts in the HTML report, ad-hoc CLI scripts (`play_ttt.py`, `arena_run.py`).

This is the largest scope option from yesterday's discussion (Option C). Single PR, single branch (`feat/reporting-overhaul`), expected ~2000-2500 lines across ~15-20 files. The plan is intentionally a starting point — we extend it as we go.

Companion docs: `docs/05-EVALUATION.md` (what to measure), `docs/07-DATA-STORAGE.md` (how parquet + W&B coexist), `docs/plans/archive/{report-modernization,report-redesign}.md` (prior reporting passes — completed; we build on that work).

---

## Checklist

| # | Item | Phase | Effort | Priority | Done |
|---|------|-------|--------|----------|------|
| R1 | Fix Accept Rate KPI: use `wins / (wins+losses)` (matching `Coach._should_accept_new_network`) and `config.update_threshold` rather than hardcoded 0.5 | Bug fixes | 15 min | High | ✅ |
| R2 | Fix Arena chart `is_accepted` annotations: same logic as R1 | Bug fixes | 10 min | High | ✅ |
| R3 | Fix `generation` `category` dtype sort: cast to `int` at read time in `reporting/training.py` (or in a new `core/storage.py` reader helper) so all sorts are numeric | Bug fixes | 30 min | High | ✅ |
| R4 | Fix profiling plot's explicit `astype(str)` cast (`training.py:357`); replace with proper numeric/sorted x-axis | Bug fixes | 15 min | High | ✅ |
| R5 | Fix arena annotation overlap: only annotate accepted gens (more legible) or rotate/reposition labels | Bug fixes | 20 min | Medium | ✅ |
| R6 | Replace Process Memory grouped-bar chart with a line chart over generations (3 stages = 3 lines), much easier to read | Bug fixes | 25 min | Medium | ✅ |
| R7 | Re-render the `ttt_full` HTML report from existing parquets and visually confirm all 6 bugs are gone | Bug fixes | 10 min | High | ✅ |
| R8 | Log **policy entropy** in two complementary ways: (a) per self-play episode, mean entropy of the MCTS visit distribution (joint MCTS+net signal); (b) per training epoch, mean entropy of the *network alone* on a frozen held-out eval set sampled from gen-1 self-play (clean isolated-net signal — the AlphaZero-paper headline curve) | New metrics | 2 hr | High | ✅ |
| R9 | Add **policy top-1 / top-5 accuracy** logging: at the end of each training epoch, compare network's argmax / top-5 to MCTS's argmax on the frozen eval set. New parquet `PolicyAccuracy/`, W&B keys `training/network_top{1,5}_accuracy`, dedicated chart in Network Diagnostics section. | New metrics | 1.5 hr | High | ✅ |
| R10 | Add **value calibration** metric: bin predicted v ∈ [-1,1] into 10 buckets, log mean(actual outcome) per bucket per epoch. New parquet `ValueCalibration/`, W&B scalar `training/value_calibration_error`, reliability-diagram chart in Network Diagnostics section. | New metrics | 1.5 hr | High | ✅ |
| R11 | Drop the `average_pi_loss` / `average_v_loss` running-mean fields from `log_training` payloads — replaced by smoothed visualisations of raw losses | New metrics | 30 min | High | |
| R12 | Generalise `Arena` to accept arbitrary `Player` callables (function from board → action); current Coach-coupled Arena becomes one caller | Elo + scripts | 1 hr | High | |
| R13 | Add **Elo vs frozen gen-0** tracking: at the end of each generation, play `elo_games_per_gen` games (default 50) of current net vs gen-0, derive single-anchor Elo `400 * log10(wr / (1-wr))` with draws as 0.5 | Elo + scripts | 1.5 hr | High | |
| R14 | Add **minimax baseline** for TTT (perfect-play opponent); log `tictactoe/draw_rate_vs_minimax` per generation as the "is this near-optimal?" signal | Elo + scripts | 1 hr | Medium | |
| R15 | Add `scripts/arena_run.py` CLI: `--player1 <ckpt|random|minimax> --player2 <...> --num-games <N>` — generic head-to-head runner. Used by R13/R14 internally and as a user-facing tool. | Elo + scripts | 1 hr | Medium | |
| R16 | Add `scripts/play_ttt.py --checkpoint <path>`: human-vs-model TTT CLI. Renders 3×3 board, reads moves 0-8 from stdin, model plays back via MCTS with temp=0. | Elo + scripts | 30 min | Medium | |
| R17 | Configure a saved W&B Workspace via the SDK: "**Training health**" view with headline cards (loss curves, arena win rate, Elo curve, policy entropy, top-1 accuracy, value calibration). Checked into repo as `wandb/workspaces/training_health.json`. | W&B + HTML | 1 hr | High | |
| R18 | Configure a second saved W&B Workspace: "**Performance deep-dive**" — timing, throughput, GPU memory, MCTS profiling. `wandb/workspaces/performance.json`. | W&B + HTML | 30 min | Medium | |
| R19 | Add new charts to HTML report for: policy entropy (line over gens), top-1/top-5 accuracy, value calibration (reliability diagram), Elo curve (with confidence interval if computable), minimax draw rate | W&B + HTML | 2 hr | High | |
| R20 | Polish HTML report: new "Diagnostics" section between KPI cards and existing performance charts; collapse less-important sections by default; ensure consistent palette with W&B | W&B + HTML | 1 hr | Medium | |
| R21 | Re-run a short TicTacToe training run with all the new metrics live and verify: a) report has zero bugs, b) all new metrics show sensible values, c) W&B workspaces load with the right charts, d) Elo curve trends up, e) minimax draw rate climbs to ~100% by convergence | Verification | 30 min | High | |
| R22 | Refactor `reporting/display.py` → game-specific renderers: rename existing module to `reporting/display_blokusduo.py`, add new `reporting/display_tictactoe.py`, introduce a small `IBoardRenderer` protocol in `reporting/display.py` that both implement. Each renderer exposes `render_board_html`, `render_policy_overlay_html`, `render_game_replay_html`. | Arena replays | 1.5 hr | High | |
| R23 | Generalise `Arena.play_game` to return the full move sequence: per move, capture `(player, action, top_k_actions, top_k_probs)`. Threaded through `play_games`. Default `top_k=5` (we show top-3 in the report but record a couple extra for future drill-downs and to absorb ties). | Arena replays | 1.5 hr | High | |
| R24 | Persist arena replays: new `MetricsCollector.log_arena_game(...)` method + parquet schema `ArenaReplays/generation=N/games.parquet` with one row per move: `(game_idx, move_idx, player, action, board_serialised, top_k_actions, top_k_probs, outcome_for_new_net)` | Arena replays | 1.5 hr | High | |
| R25 | Render top-3 candidate moves as **concrete previews** per game state. Each renderer decides how — these are the kind of things a recruiter could read at a glance with no project background: <br>• **TTT** (`display_tictactoe.py`): a mirror 3×3 board where the top-3 cells are highlighted with rank badges `①②③` and MCTS probability percentages overlaid. Reader sees "model thought (1,1) was best at 47%, (0,2) second at 28%, (2,2) third at 11%" instantly. <br>• **Blokus** (`display_blokusduo.py`): three thumbnail board previews stacked vertically. Each thumbnail shows the actual piece (in its chosen orientation) overlaid on the 14×14 grid at the proposed anchor cell, with the move described as `Piece P5 (orientation 3) at (7,8) — 23%`. No "policy vector" exposed — only human-readable move descriptions. | Arena replays | 2.5 hr | High | |
| R26 | Add interactive "Arena Replays" section to the HTML report: a) generation dropdown selector at the top, b) below it a grid of game tiles for the selected gen — green if new net won, grey if drawn, red if lost, c) clicking a tile lazy-loads that game's full move sequence into a replay panel below with current board + top-3 candidate-move previews (from R25) + ←/→ stepper showing per-move outcome metadata. All game data embedded as JSON in the single HTML file; interactivity via vanilla JS (no external deps). | Arena replays | 3 hr | High | |
| R27 | Add `scripts/replay.py --run <name> --gen <N> --game <i>`: standalone CLI to play back any recorded arena game from any gen, terminal output with ASCII board + policy hints. Useful for digging into specific games beyond what fits in the HTML report. | Arena replays | 45 min | Medium | |
| R28 | Re-run a short TicTacToe training run *with arena replay recording on* and verify: a) arena replays parquet directory populates, b) HTML report's new replays section loads correctly, c) policy heatmaps make visual sense (sane probabilities on legal moves only), d) standalone replay script works. | Verification | 30 min | High | |
| R29 | `git mv docs/plans/reporting-overhaul.md docs/plans/archive/reporting-overhaul.md` in the final commit before PR | Wrap-up | 2 min | Low | |

Total effort estimate: ~15-18 hours of focused work. Spread across multiple sessions.

---

## Phase 1 — Bug fixes (R1-R7)

Goal: existing HTML report stops lying. No new functionality, no new dependencies. Done in a single session.

### R1. Accept Rate KPI math

`reporting/training.py:82-86` currently:

```python
total_arena = arena_data["wins"] + arena_data["losses"] + arena_data["draws"]
accepted = (arena_data["wins"] / total_arena) > 0.5
```

Two bugs: includes draws in denominator, and hardcodes 0.5 threshold. Fix:

```python
decided = arena_data["wins"] + arena_data["losses"]
win_rate = arena_data["wins"].where(decided > 0, 0) / decided.where(decided > 0, 1)
accepted = (decided > 0) & (win_rate >= config.update_threshold)
```

`config.update_threshold` needs to be threaded into `_make_kpi_cards`. Either pass `config` through or compute the boolean column upstream and pass that in.

### R2. Arena chart `is_accepted`

Same fix at `reporting/training.py:230`.

### R3. Generation dtype

Hive-partitioned parquet loads `generation` as `category`. Cast to `int` at the boundary. Cleanest: add a small `_load_metrics(directory: Path) -> DataFrame` helper that reads + casts. Centralises the conversion in one place. Apply to all 6 metric parquet directories.

### R4. Drop the explicit `astype(str)` in profiling plot

`training.py:357` deliberately strings the generation for the Box plot. Replace with proper int sort. If Plotly Box plot needs discrete x, use `xaxis_type="category"` with the categories explicitly sorted numerically.

### R5. Arena annotation overlap

Two options: (a) only annotate accepted gens, since rejected is the default expectation (cleaner; less visual noise), or (b) rotate annotations and stagger heights. Try (a) first.

### R6. Process Memory chart

Replace grouped bar chart (90 bars) with a line chart: x = generation, y = RSS MB, one line per stage (SelfPlay/Training/Arena). If GPU memory is present, overlay dashed lines for GPU per stage. Much easier to scan.

### R7. Re-render

```bash
uv run python -c "
from core.config import load_args
from reporting.training import create_html_report
cfg = load_args('run_configurations/ttt_full.json')
create_html_report(cfg)
"
open temp/ttt_full/Reporting/report.html
```

Visually confirm: Accept Rate `2/30`, gens in numeric order on every chart, gens 2 and 11 labelled accepted, memory chart readable.

---

## Phase 2 — New metrics collection (R8-R11)

Goal: log everything `docs/05-EVALUATION.md` asks for. New `wandb.log` keys; new parquet schemas; backward-compatibility maintained for existing run data.

(Detail TBD when we start Phase 2.)

---

## Phase 3 — Elo + scripts (R12-R16)

Goal: produce the monotonic Elo-over-time curve that's the headline chart in every AlphaZero paper. Plus the head-to-head CLI tools that the Elo work needs internally.

(Detail TBD when we start Phase 3.)

---

## Phase 4 — W&B Workspaces + HTML polish (R17-R20)

Goal: two committed, curated W&B Workspaces in the repo + corresponding HTML report layout. Reader (or recruiter) lands on either surface and immediately sees the "is this working?" signal.

(Detail TBD when we start Phase 4.)

---

## Phase 5 — Verification (R21)

Short TTT run end-to-end with new metrics live. Visual review of both W&B workspaces + HTML report.

---

## Phase 6 — Arena replays (R22-R27)

Goal: make individual arena games inspectable in the HTML report. The reader (recruiter, future-you) selects a generation, sees a colour-coded grid of game outcomes for that gen, clicks a game, sees a step-through with the actual board next to a heatmap mirror board showing the MCTS policy over legal moves. Catches "did the model miss obvious wins?" with one click.

### Renderer abstraction (R22)

Each game gets its own `display_<game>.py` module exposing a small contract:

```python
class IBoardRenderer(Protocol):
    def render_board_html(self, board: IBoard, last_move: int | None = None) -> str: ...
    def render_top_k_moves_html(
        self,
        board: IBoard,
        actions: list[int],
        probs: list[float],
    ) -> str:
        """Render the top-K candidate moves as a human-readable HTML fragment.
        TTT highlights the top cells on a mirror board with rank badges and %.
        Blokus renders thumbnail boards with the piece in its orientation at
        the proposed anchor cell, plus a textual "Piece P3 orientation 2 at (7,8)" caption.
        """
    def render_game_replay_html(self, game: ArenaGame) -> str: ...
```

`display.py` becomes a thin module exposing the protocol + a `get_renderer(game_name)` factory. Adding a future game = adding a `display_othello.py` with the protocol implementation.

### Data shape (R23-R24)

Per arena game, we now save:
- Outcome for the candidate (new) net: +1, 0, -1
- Per move: `(player, action, top_k_actions[5], top_k_probs[5], serialised_board)`

Top-K default of 5 (showing 3 in the report, 2 extra recorded for tie-breaking and future drill-downs). Tiny storage for both games. **We never store a raw "policy vector" of 17,837 floats** — only the indexed top-K, decoded into human-readable previews at render time.

### Lazy-loaded UI (R26)

Self-contained HTML file with all replay data embedded as JSON. JavaScript:
1. Generation dropdown changes the list of game tiles
2. Tile colour reflects outcome (green=W, grey=D, red=L from candidate's perspective)
3. Clicking a tile pulls that game's JSON from the embedded data, renders it into the replay panel
4. Replay panel = real board (left), top-3 candidate move previews (right), step controls, per-move metadata

Vanilla JS only — no React, no Vue, no fetch(). Keeps the file portable and `file://`-openable.

---

## Phase 7 — Verification + wrap-up (R28-R29)

Final TTT run with arena replays on. Visual review. Archive plan.
