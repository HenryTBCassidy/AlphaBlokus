# Web Play — Difficulty Calibration & Fidelity Notes

Companion to [`docs/plans/web-play.md`](../plans/web-play.md) (W13). Two questions answered
here: **how strong is each difficulty level relative to the others**, and **how faithful is
the browser stack to the Python reference** it was ported from.

All numbers measured 2026-07-05 with the run3 checkpoint
(`blokus_run3_overnight/Nets/accepted_82.pth.tar`, 128f×8b conv head) — the same net the
first deployment ships. Re-run the scripts below when the checkpoint is swapped.

---

## 1. Difficulty ladder calibration

Difficulty levels are fixed `{search policy, sims}` pairs (`web/src/engine/difficulty.ts`,
mirrored by the server in `alphablokus/play/service.py`), so playing strength is
hardware-independent — a phone and a workstation play identical moves, only think time
differs. Strength therefore calibrates in Python (`scripts/calibrate_web_difficulty.py`:
arena round robin between sim budgets with the same net, opening-temperature diversified,
16 games per pair, colours alternated) and transfers to the browser given the parity
suites (§2).

| Match (same net) | W-L-D (higher budget) | Δ Elo (higher − lower) |
|---|---|---|
| 32 sims vs raw policy | 8-7-1 | +22 |
| 128 sims vs 32 sims | 10-5-1 | +112 |
| 400 sims vs 128 sims | 15-1-0 | +470 |

Reading of the ladder (16-game matches ⇒ wide CIs on the middle rows; the 15-1-0 row is
unambiguous):

- **Raw policy ≈ 32 sims.** The net's argmax policy is nearly as strong as a tiny
  search; Level 1 → Level 2 is a gentle step, not a cliff.
- **The ladder steepens with budget.** 128 sims is ~+112 Elo over 32; 400 sims is a
  further ~+470 over 128 (15-1-0). Self-play-relative strength keeps climbing with sims
  even though absolute strength vs Pentobi plateaued in the jaxg30 study — the two
  measures answer different questions.
- Level 4 (Gumbel n=256) was not separately arena'd: the Gumbel root is a
  reimplementation (§2.4) and its budget sits between the measured 128/400 PUCT points.
- Level 5 (800 sims) matches the training-arena budget and tops the ladder; by the
  measured trend it is clearly the strongest setting.

Absolute strength context: this net is **below Pentobi Level 1** (6% at L1 — see the
jaxg30 Pentobi baseline). The point of the web tiers is the delivery mechanism with a
swappable checkpoint, not today's strength.

## 2. Fidelity: browser stack vs Python reference

The download tier (local Python server) is the source of truth. The browser stack is
pinned to it by four gates, all run against the same checkpoint:

### 2.1 Rules + encoding (exact)

`scripts/generate_web_parity_fixtures.py` → `web/tests/rules_parity.test.ts`:
8 seeded random reference games, 236 plies. The TS engine reproduces **identical
legal-move sets, byte-identical 44-channel encodings (sha256), identical game-ended
values and board evolution** on every ply. The rules tables are exported from the same
`build_move_tables` the JAX kernels scatter from, so geometric parity is by construction;
the fixtures pin the condition logic.

### 2.2 ONNX net outputs (float tolerance)

`web/tests/net_parity.test.ts`, 32 positions spread over the fixture games:

| Variant | max policy diff vs torch | max value diff | top-1 flips |
|---|---|---|---|
| fp32 (`model.onnx`, 10 MB) | 1.1e-6 | 4.0e-6 | 0/32 |
| fp16 (`model.fp16.onnx`, 5.1 MB) | 7.1e-4 | 1.9e-3 | 0/32 |
| int8 dynamic (`model.int8.onnx`, 2.7 MB) | 2.3e-1 | 2.8e-1 | **6/32** |

**fp32 is the default; fp16 is a safe half-size option** (top-1 stable, deviations far
below move-decision noise). **int8 dynamic quantisation is NOT recommended** — it flips
the argmax on 19% of positions and materially distorts values; it is exported for
experimentation only (`?net=int8`).

### 2.3 Full-game agreement (decision-level)

`npm run agreement` + `scripts/verify_web_agreement.py`: the TS rules + ONNX stack plays
complete games (raw-policy argmax both sides); Python replays each ply and independently
recomputes the decision. Result: 3 games, 93 plies — **identical legal sets, the same
argmax move on every ply (0 near-tie flips), values within 1e-4, matching final scores.**
Separately, a headless end-to-end browser game (`npm run e2e`, Playwright against the
built app at Level 2) replays cleanly through the Python engine: all plies legal, final
scores match.

### 2.4 Known caveats

- **Gumbel is an algorithmic reimplementation, not an mctx port.** The browser's Level 4
  follows the Gumbel-top-m + Sequential Halving procedure (fixed `g + log π` logits,
  mctx's σ(q̂) ranking with `c_visit=50, c_scale=1`), but interior descents reuse the
  PUCT tree rather than mctx's deterministic improved-policy selection, and completed-Q
  mixing is replaced by the tree's running-mean Q. Expect similar—not bit-identical—move
  choices vs the jax training searcher.
- **Play-time MCTS drops self-play machinery.** No Dirichlet root noise, no virtual loss
  (sims are sequential, K=1); the PUCT arithmetic itself mirrors `search/mcts.py`.
- **The server tier's PUCT uses `mcts_batch_size` from the run config** (batched leaf
  evaluation under virtual loss — the production-validated path), so its search differs
  slightly from the browser's K=1 at equal sims. Both are PUCT at the stated budget.
- **fp16 top-1 stability was measured on 32 positions**, not exhaustively; if a future
  checkpoint shows near-tie policies, re-run `web/tests/artifacts`-style measurement
  before defaulting to fp16.

## 3. Reproducing

```bash
# ladder
uv run python scripts/calibrate_web_difficulty.py \
    --config run_configurations/blokus_run3_overnight.json \
    --checkpoint <checkpoint> --pairs 0:32 32:128 128:400 --games 16

# fidelity
uv run python scripts/generate_web_parity_fixtures.py --config <cfg> --checkpoint <ckpt>
(cd web && npm test && npm run agreement)
uv run python scripts/verify_web_agreement.py --config <cfg> --checkpoint <ckpt>
(cd web && npm run build && npm run e2e)
```
