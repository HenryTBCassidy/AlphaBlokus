# Web Play — Browser App + Downloadable Local Play

Two ways for people to play Blokus Duo against the trained net, following the Pentobi
delivery model (browser version + desktop download):

1. **Web tier** — a free, static, shareable browser app. The net (ONNX Runtime Web) and a
   TypeScript port of the rules + play-time MCTS run entirely in the visitor's browser.
   No backend; deployable to GitHub/Cloudflare Pages. Strength is fixed by difficulty
   level (sim counts), independent of visitor hardware.
2. **Download tier** — the same built frontend served from a local Python server
   (`alphablokus-play`) that answers `bestMove` with the REAL torch + MCTS stack at full
   strength. The source of truth.

**One shared frontend, two engine backends** behind a single TypeScript `Engine`
interface. The board, tray, and interaction are written once; only where `bestMove`
comes from differs.

**The rules port is not a rewrite.** `games/blokusduo/jax/tables.py` already reduces the
whole rules engine to three static geometry tables over the action space (`cover`,
`edge`, `corner` — built from `movegen/tables.build_move_tables`) plus a 4-field game
state, and `jax/kernels.py` expresses legality/step/game-end as a handful of comparisons
against those tables. We export the tables from Python as a binary blob and the TS engine
evaluates the same conditions against the same bytes — geometric parity by construction,
verified by a fixture battery. Companion context: `docs/06-INTERFACES.md` §3 (prior UI
scoping), `docs/research/jax-pipeline-ab.md` (kernel parity precedent).

Prerequisites: a Blokus checkpoint to export (default:
`temp/runs/blokus/blokus_run3_overnight/Nets/accepted_82.pth.tar`, 128f×8b conv head —
swappable by design; re-export + redeploy when stronger nets land).

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| W1 | Scaffold `web/` Vite + TypeScript app with lint/format/test tooling | 45 min | High | ✅ |
| W2 | `scripts/export_web_assets.py` — rules blob + pieces JSON + manifest (`--rules-only`) | 1.5 h | High | ✅ |
| W3 | TS rules engine + 44-channel encoder (port of the jax kernel semantics) | 2 h | High | ✅ |
| W4 | Python parity-fixture generator + vitest rules/encoding parity suite | 1.5 h | High | ✅ |
| W5 | ONNX export (+ optional fp16/int8) in the export script + manifest net entry | 1.5 h | High | |
| W6 | TS net predictor (onnxruntime-web, WebGPU → WASM fallback) + net-output parity test | 1.5 h | High | |
| W7 | TS play-time MCTS (PUCT + Gumbel root) + difficulty levels | 2 h | High | |
| W8 | Frontend UI: board, tray, rotate/flip, legal-move highlighting, controls | 3 h | High | |
| W9 | Local Python play server (`alphablokus/play/`, FastAPI `play` extra, `alphablokus-play`) | 2 h | High | |
| W10 | `ServerEngine` (HTTP client) + engine selection in the frontend | 1 h | High | |
| W11 | Scripted browser-engine vs Python-engine agreement game | 1 h | High | |
| W12 | End-to-end browser game exercised headless (Playwright) | 1 h | High | |
| W13 | Difficulty calibration (sims-ladder round robin) + fidelity caveats note | 1.5 h | Medium | |
| W14 | README section, web CI job, plan wrap-up | 1 h | High | |

---

## W1. Scaffold `web/` Vite + TypeScript app

New top-level `web/` directory (app code stays out of the framework core). Vite +
TypeScript, **no runtime CDN dependencies** — everything bundles, including the
onnxruntime-web WASM binaries (copied into the bundle, not fetched from a CDN).
Tooling: `tsc --noEmit` for typing, Prettier for format, Vitest for tests. Vanilla
TypeScript + SVG rendering (no framework — the interaction model is small enough, and it
keeps the bundle lean). `npm run build` → `web/dist/`, servable from any static host and
from the local Python server.

## W2. Export script — rules blob + pieces JSON + manifest

`scripts/export_web_assets.py`. Builds `MoveTables` via
`movegen.tables.build_move_tables` (the same source the jax tables scatter from) and
writes `web/public/assets/`:

- `rules.bin` — little-endian concatenation of the per-move arrays (`piece` u8,
  `action_id` u32, `cells` u8×5, `adj_cells` u8×16, `attach_cells` u8×16, NULL_CELL
  padded) plus `piece_sizes`, start cells, pass index. Layout described by the manifest.
- `pieces.json` — the 91 orientation grids (per orientation id: piece id, name, grid)
  for tray rendering, straight from `PieceManager` (resolved via
  `default_pieces_path()`).
- `manifest.json` — encoding version, action space size, board size, channel count,
  array offsets/dtypes/shapes, sha256 per file. The net entry is added by W5.

`--rules-only` skips the checkpoint so CI can regenerate deterministic rules assets
without a model. Checkpoint/config/output paths are CLI args.

## W3. TS rules engine + encoder

`web/src/engine/rules.ts` + `encoding.ts`. State mirrors the jax `GameState`: `ppb`
(Int8Array 196, +piece_id White / −piece_id Black), `remaining` (2×22 bools),
`lastPiece` (2), `currentPlayer`. Legality per action = the kernel's exact conditions
(placeable ∧ available ∧ no overlap ∧ no own-edge ∧ (own-corner ∨ first-move-covers-start)),
evaluated over the sparse cell lists from `rules.bin` instead of dense matmuls — same
semantics, browser-friendly cost (~13.7k moves × ≤37 cell probes, sub-millisecond).
`step`, `gameResult` (draw = 1e-4), and `score` (−remaining, +15 all-placed, +5
monomino-last) mirror `jax/kernels.py` line for line. Encoder ports
`encode_planes_from_placement`: canonicalise by sign-flip, 21+21 per-piece planes + 2
aggregates → Float32Array(44·196).

## W4. Parity fixtures + vitest suite

`scripts/generate_web_parity_fixtures.py` plays seeded random games with the **reference**
`BlokusDuoGame` and dumps `web/tests/fixtures/rules_parity.json`: per position the ppb
(base64), remaining sets, current player, the sorted legal action ids from
`valid_move_masking`, and a sha256 of `as_multi_channel(1)`. Battery spans opening,
midgame, endgame, and pass/terminal states. Vitest replays every position through the TS
engine and asserts **identical** legal-move sets and byte-identical encodings (the planes
are 0/1 floats — no tolerance needed). Fixtures are committed (small, deterministic).

## W5. ONNX export

Same script, full mode: load the checkpoint (`state_dict` key) into `AlphaBlokusDuo` via
the run-config's net settings, export with dynamic batch axis. Outputs stay
`(log_softmax policy, tanh value)` — exactly the torch forward. Optional `--fp16` /
`--int8` quantisation flags (guarded imports; fp32 is the parity baseline). Manifest
gains the net entry: filters/blocks/head, source checkpoint, param count, file hashes.
Requires the new `web` optional extra (`onnx`, `onnxruntime`) — kept out of the core
install.

## W6. TS net predictor + net-output parity

`web/src/engine/net.ts`: onnxruntime-web session, WebGPU execution provider preferred,
WASM (SIMD) fallback; batch-capable `predict(encodings)` returning priors (exp of
log-softmax) + values. The fixture generator also dumps torch outputs (top-K policy
entries + value) for a subset of positions; a vitest (node ORT) run asserts ONNX outputs
match within float tolerance and records the measured max deviation for the W13 note.

## W7. TS play-time MCTS + difficulty levels

`web/src/engine/mcts.ts`: PUCT mirroring `search/mcts.py` — same selection formula
(`q + cpuct·prior·√N/(1+n)`, EPS at the unexpanded root), same expansion
(mask → renormalise), same backprop sign-flip. Self-play-only machinery (Dirichlet noise,
virtual loss) dropped; K=1 sequential sims with async inference. `gumbel.ts`: Gumbel
root — sample Gumbel noise, top-m by `g + logits`, Sequential Halving with completed-Q,
non-root selection by improved-policy argmax (algorithmic reimplementation of the mctx
approach, not a port — noted in W13). Difficulty levels map to fixed
`{searchPolicy, sims}` so strength is hardware-independent; hardware only affects
latency (UI shows a thinking indicator).

## W8. Frontend UI

SVG 14×14 board (Blokus notation labels), two 21-piece trays, click-to-select +
rotate (R / scroll) + flip (F / double-click), ghost preview snapped to the cursor,
legal-anchor highlighting for the selected orientation, move/pass/resign controls,
difficulty selector, new game, score panel, thinking indicator, and a "download for full
strength" link. Human plays either colour. All engine calls go through the `Engine`
interface (`init`, `legalMoves`, `applyMove`, `bestMove`, `gameStatus`).

## W9. Local Python play server

New subpackage `src/alphablokus/play/` (framework-clean: protocols + registry only, no
`games.*` imports — the wire format is action ints + move history, so the server is
game-agnostic). FastAPI app (optional `play` extra: `fastapi`, `uvicorn`):

- `GET /api/meta` — game id, action size, engine description, difficulty table.
- `POST /api/best-move` — `{history: [action ids], difficulty}` → `{action, value,
  legal}` (the returned legal set doubles as a runtime cross-check for the TS rules).
- Static mount of `web/dist` at `/`.

Backed by `registry.instantiate_game_and_network` + the real `MCTS`; device via the
existing `NetConfig.cuda` flag. Entry point `alphablokus-play --config <run json>
--checkpoint <path> [--port]`. Stateless: each request replays the (≤~40-move) history.

## W10. ServerEngine + engine selection

`web/src/engine/serverEngine.ts` implements `Engine` over the HTTP API, delegating
`bestMove` to the server and cross-checking the returned legal sets against the local TS
rules (mismatch → loud console error; the Python side is authoritative). Frontend
auto-detects: served from `alphablokus-play` (`/api/meta` responds) → server engine
("full strength" badge); static hosting → browser engine.

## W11. Scripted agreement game

`web/scripts/agreement_game.mjs` (node): plays full games with the TS engine choosing
moves (raw-policy argmax through the ONNX net), recording per-ply chosen action + legal
sets + values. `scripts/verify_web_agreement.py` replays the record through
`BlokusDuoGame` + torch and asserts every move legal, every legal set identical, and net
outputs within tolerance. This is the "browser engine vs Python engine" agreement gate.

## W12. End-to-end browser game

Playwright (dev-dependency) drives the **built** app in headless Chromium: new game,
place pieces via a test hook (`window.__alphablokus`) choosing random legal moves for
the human, let the engine reply, play to completion, assert a final score is shown and
every ply was legal. Run via `npm run e2e` against `vite preview`. This is the "don't
claim it works without an actual browser game" gate.

## W13. Difficulty calibration + fidelity caveats

Calibrate level → sims with a Python sims-ladder round robin (arena machinery, e.g.
0/32/128/400 sims at fixed net) — strength is search-budget-determined, so Python
numbers transfer to the browser given parity. Record results + the measured ONNX/torch
deviation + known caveats (Gumbel reimplementation, quantised-variant deviation) in
`docs/research/web-play-calibration.md`.

## W14. README + CI + wrap-up

README "Play against the net" section: browser build/deploy (static), local
full-strength play (`uv sync --extra play && alphablokus-play`), checkpoint re-export
flow. New CI job: install web deps, regenerate rules assets (`--rules-only`), `tsc`,
Prettier check, Vitest (rules parity), `vite build`. Tick remaining rows, archive the
plan.
