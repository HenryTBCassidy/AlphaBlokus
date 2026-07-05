# Next-run training improvements — from the blokus_cloud_60 analysis

**What this covers.** Turn the findings of
[`../research/blokus-cloud-60-analysis.md`](../research/blokus-cloud-60-analysis.md) into concrete
changes for the next scaled run. The headline finding: the `blokus_cloud_60` run **did not saturate** —
the cosine LR schedule (`T_max=60`, `eta_min=0`) annealed the learning rate below 1e-4 by ~gen 48 and to
2.7e-6 by gen 58, so the **last ~quarter of the run trained at a near-dead LR**. That, not capacity,
caused the late arena rejections (8/10 in gens 49–58 at ~coin-flip scores) and the value-loss rise.
Policy loss was *still falling* (2.53 → 0.40, monotone) and Pentobi went L3→L4 — the net had more to
give and the schedule strangled it. The Top-1 policy-agreement "decline" that triggered the analysis is
**benign** — a stale gen-1 eval-set artifact (Top-5 stayed 96–100%, strength kept rising).

Ordered by leverage. **N1 (LR floor) is the single highest-value change and it's nearly free.**

**Prerequisite:** PR #40 (the DataLoader/JAX-fork crash fix) must be merged first — the next run needs
it or it re-crashes at ~gen 59.

**Ground truth (from the analysis + code):**
- `src/alphablokus/games/base_wrapper.py` — `_create_scheduler` (~`:153-162`) builds
  `CosineAnnealingLR(T_max = num_generations × epochs, eta_min=0)`, stepped once per generation. The
  `eta_min=0` default is the footgun.
- `run_configurations/blokus_cloud.json` — the base config to fork: `num_mcts_sims=64`,
  `gumbel_max_considered=16`, `num_arena_matches=40`, `replay_buffer_games=40000`, net `large` (192×12),
  `num_eps=10000`.
- `src/alphablokus/training/eval_set.py` — freezes the held-out eval set at **generation 1** (positions +
  gen-1's 64-sim MCTS targets), persisted forever; scored by `evaluate_on_eval_set` in `base_wrapper.py`.
  This is why Top-1-vs-MCTS decays as the net surpasses gen-1's search.
- `src/alphablokus/evaluation/elo.py:~19` — the score-rate clamp `[0.001, 0.999]` → the ±1200 Elo cap
  (vs frozen gen-0), saturated from ~gen 18. The real fix is the merged **pool-Elo tournament**
  (`scripts/tournament_elo.py`).
- Warm-start already exists: `load_model: true` (`cli.py:111,120`) loads `best.pth.tar` and starts a
  fresh run (empty buffer that fills, new Elo baseline = the loaded net). Use it to seed from gen-57.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| N1 | Floor the cosine LR — add `lr_eta_min` config knob, used by `_create_scheduler` | 45 min | High | ✅ |
| N2 | `blokus_cloud_v2.json`: LR floor + Gumbel n128/considered32 + arena 100 + buffer 60k + warm-start from gen-57 | 1 h | High | ✅ |
| N3 | Fix the eval-set diagnostic (score vs current-net MCTS + relabel; optional periodic rebuild) | 2.5 h | Medium | |
| N4 | Run the (already-merged) pool-Elo tournament on the checkpoints; optionally add the deferred live sliding-reference Elo | 1 h | Low | |
| N5 | Validate: defaults unchanged; short run exercises the v2 config; CI green | 1 h | High | |

> Do **N1 + N2** before the next run (they're the ones that make it learn further). N3/N4 are
> metric-quality follow-ups that make the *next* run's diagnostics trustworthy — nice-to-have, not
> blocking the launch.

---

## N1. Floor the cosine LR

**Current state.** `_create_scheduler` (`base_wrapper.py:~153-162`) uses `CosineAnnealingLR(T_max=…,
eta_min=0)`. LR reaches ~2.7e-6 by the run's end; the analysis shows ~15 of 58 generations trained at
LR ≤ 1e-4 — mechanically dead. Acceptance collapsed exactly where LR crossed ~1e-4 (gen ~48).

**Fix.** Add a config knob and pass it through:
- `NetConfig.lr_eta_min: float = 0.0` (default preserves today's behaviour → TicTacToe and existing
  configs are byte-for-byte unchanged). In `_create_scheduler`, pass `eta_min=self.net_config.lr_eta_min`
  to `CosineAnnealingLR`.
- Recommended production value: **`1e-4`** (10% of the 1e-3 peak) — set in the v2 config (N2), not the
  default. (Rationale in the analysis §4: AlphaZero annealed 0.2 → 2e-4, a 1000× range but never to
  zero.)
- Consider a one-line startup warning if `lr_scheduler=="cosine"` and `lr_eta_min==0` on a
  multi-generation run, since 0 is a known footgun — optional.

**Test:** with `lr_eta_min=1e-4`, stepping the scheduler `num_generations` times never drops LR below
1e-4; with the default `0.0`, the schedule is identical to before (lock current behaviour).

**Effort:** 45 min.

---

## N2. `blokus_cloud_v2.json` — the next-run config

**Fix.** Fork `blokus_cloud.json` → `run_configurations/blokus_cloud_v2.json` with these deltas (from the
analysis §4 recommended config), keeping everything else (net `large`, `num_eps=10000`, α=0.03, gate
0.55, epochs 1, batch 1024, K=64, `wave_plies` 32):
```jsonc
{
  "run_name": "blokus_cloud_v2",
  "load_model": true,                    // warm-start from best.pth.tar (seed = gen-57 net)
  "replay_buffer_games": 60000,          // was 40000 (staleness 4→6; steadies the value head)
  "num_arena_matches": 100,              // was 40 (gate noise SE ~8→5 pp — fewer false rejections)
  "mcts_config": {
    "num_mcts_sims": 128,                // was 64  (stronger improvement operator)
    "gumbel_max_considered": 32          // was 16
  },
  "net_config": { "preset": "large", "learning_rate": 0.001,
                  "lr_scheduler": "cosine", "lr_eta_min": 0.0001 },  // N1 floor
  "wandb": { "mode": "online" },         // never offline for a real run (protocol rule 8)
  "object_store": { ... }                // REQUIRED per data-safety protocol — S3 sync on
}
```
- **Warm-start recipe:** `load_model` loads `best.pth.tar` from the run's `net_directory`, so seed the
  run by placing the **gen-57 net** there as `best.pth.tar` before launch (we have gen-57's weights on
  the box; the full checkpoint is on the network volume). This gives a **fresh cosine schedule from 1e-3
  + the strong starting net + a fresh, informative Elo baseline** — strictly better than `--resume
  blokus_cloud_60`, which would inherit the dead LR tail.
- **Cost:** ~21 min/gen ⇒ ~21 h for 60 gens on a 5090 (the n=64→128 doubling is most of it).
- **Follow the data-safety protocol** ([`../guides/CLOUD-TRAINING.md`](../guides/CLOUD-TRAINING.md)):
  `object_store` on + verified after gen 1, W&B online, pull-before-stop.
- **Fill in the object-store placeholders before launch.** JSON has no comments, so the shipped
  `object_store` block uses literal `REPLACE_ME-…` `bucket` / `endpoint_url` strings — replace them with
  the real bucket + endpoint. Credentials stay env-only (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`),
  never in the JSON.

**Effort:** 1 h.

---

## N3. Fix the eval-set diagnostic

**Current state.** `eval_set.py` freezes the eval set at gen 1, so "Top-1 vs MCTS" measures agreement
against gen-1's noisy 64-sim targets forever — it *falls* as the net surpasses gen-1's search, which
looks alarming but is benign (this analysis). The value-calibration plot shares the confound (current
values vs gen-1 outcomes).

**Fix (make the diagnostic mean something again):**
- **Primary:** additionally score policy agreement against the **current net's own MCTS** on the fixed
  eval positions — this measures the *net-vs-its-own-search* gap (the genuinely interesting "is the raw
  policy keeping up with search?" quantity), which should trend *up* or stay flat as training works.
- **Relabel** the existing chart to "Top-k agreement vs gen-1 MCTS targets" so it's not misread as a
  strength signal (`reporting/charts.py`).
- **Optional:** rebuild the eval set every N generations (versioned) and plot per-version, so absolute
  agreement is comparable within a version.
- Same change rehabilitates the value-calibration diagnostic.

**Test:** current-net-MCTS agreement is computed and logged; relabelled chart renders; the frozen-gen-1
series still available for continuity.

**Effort:** 2.5 h.

---

## N4. Pool-Elo strength curve — already built; run it (+ optional live version)

**Current state — mostly done.** The non-saturating strength metric is **already merged** (PR #36,
pool-based-elo E1–E7): `scripts/tournament_elo.py` runs a round-robin over a run's saved checkpoints and
fits **BayesElo** (`src/alphablokus/evaluation/rating.py`, Bradley–Terry MM), and the report renders the
curve. So there is **no tool to build here.** What's outstanding:
- The tournament has **not been run** on the `blokus_cloud_60` checkpoints — that needs a pod mounted on
  the network volume that holds the 39 accepted checkpoints.
- The **live, per-generation** Elo during training is still the saturating vs-frozen-gen-0 number (the
  ±1200 clamp at `elo.py:~19`). The *live* non-saturating version was **deferred** (pool-Elo plan E8/E9).

**Fix.**
- **Run the tournament** on the `blokus_cloud_60` (and future runs') checkpoints — this is the analysis
  that answers "did it saturate?", not new code. Treat it + the Pentobi ladder as the strength measures.
- **Optional (the deferred E8):** add a *live* non-saturating Elo — rate each generation against a
  **recent** frozen net instead of gen-0, and/or lift the ±1200 clamp — so the in-training curve is
  informative too. Low priority; the post-hoc tournament + Pentobi ladder already cover strength.

**Effort:** ~1 h to run the tournament; +2–3 h only if implementing the live sliding-reference Elo.

---

## N5. Validate

- **Defaults unchanged:** `lr_eta_min=0.0` default reproduces the exact prior schedule; TicTacToe +
  existing configs behave identically. Mac/CPU path untouched.
- Short run on the `blokus_cloud_v2.json` recipe (few gens, scaled down) exercises: warm-start loads the
  seed net, LR floor holds, bigger buffer/arena/sims run, W&B online + object-store sync work.
- Full CI green (ruff, format, mypy, base + jax tests).

**Effort:** 1 h.

---

## Notes for the executing agent

- **Style contract:** full type annotations (mypy --strict), ruff lint+format, frozen dataclasses,
  loguru (`{}`; no print), Google docstrings, `from __future__ import annotations`, real objects in
  tests. Keep CI green.
- **N1 is the point of this plan** — it's the free, highest-leverage fix; do it first. N2 packages it +
  the analysis's other config bumps into a launchable run. N3/N4 make the diagnostics honest but don't
  block the run.
- **Prereq:** merge PR #40 (crash fix) first.
- One commit per row; tick Done as each lands. Archive to `docs/plans/archive/` when complete.
