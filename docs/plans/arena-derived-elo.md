# Arena-Derived Rolling Elo (live) + Pooled Elo (end-of-run)

Replace the broken per-generation "Elo vs frozen gen-0 baseline" metric with a **rolling, arena-derived Elo** that reuses the arena games we already play, never saturates, and is cheap enough to stream every generation. The rigorous non-saturating curve stays where it belongs — the **post-hoc pool BayesElo tournament** (`scripts/tournament_elo.py`, already built) — which this plan wires to run automatically at end-of-run.

**Why.** The current live metric plays the candidate against a *frozen gen-0 net* over `elo_games_per_gen` games each generation (`coach.py:_evaluate_elo_vs_baseline`). Once the net ≫ gen-0 it either sweeps (→ +1200 cap) or ties on colour (→ +0), so the number is bimodal noise — see `docs/research/blokus-cloud-60-analysis.md` (§ internal Elo saturates from ~gen 18) and the `blokus_cloud_v3` run, where warm-starting from gen-57 made it dead on arrival.

**The design (Henry's).** The arena *already* plays candidate-vs-current-incumbent, and on acceptance the candidate *becomes* the incumbent. So the incumbent **is** a rolling benchmark. Anchor the starting net at Elo 400; each generation derive the candidate's Elo from its arena score against the incumbent (`candidate_elo = incumbent_elo + 400·log10(s/(1−s))`); on acceptance, roll the benchmark forward. Zero extra games — in fact it *deletes* the separate `elo_games_per_gen` eval, saving those games per generation. The result is a gradually climbing, non-saturating curve. Its one weakness (it's a *chained* estimate, so it can drift and the high-score steps are noisy on ~100 games) is exactly what the end-of-run pooled fit corrects.

**Philosophy constraint (Henry):** streamed metrics run **per-generation** or not at all; anything needing multiple generations runs in the **reporting/standalone end-of-run step**. No mid-run periodic jobs.

Companion docs: [`docs/guides/PLAN-FORMAT.md`](../guides/PLAN-FORMAT.md), [`docs/guides/STYLE-GUIDE.md`](../guides/STYLE-GUIDE.md), [`docs/research/pool-elo-methodology.md`](../research/pool-elo-methodology.md), [`docs/plans/archive/pool-based-elo.md`](archive/pool-based-elo.md).

---

## Checklist

| # | Item | Effort | Priority | Files | Done |
|---|------|--------|----------|-------|------|
| S1 | Compute + roll arena-derived Elo in the Coach | 1.5 h | High | `training/coach.py` | ✅ |
| S2 | Reconstruct the rolling benchmark on `--resume` | 1 h | High | `training/coach.py` | ✅ |
| S3 | Record the anchor net's provenance (cross-run splicing) | 45 min | Medium | `training/coach.py`, `storage/metrics.py` | ✅ |
| S4 | Delete the gen-0 per-gen Elo eval (keep the anchor checkpoint) | 1 h | High | `training/coach.py`, `config.py` | ✅ |
| S5 | Metrics schema: rolling-Elo fields + `accepted` | 45 min | High | `storage/metrics.py` | ✅ |
| S6 | Report chart: render the rolling non-saturating Elo | 1 h | High | `reporting/charts.py` | ✅ |
| S7 | Config cleanup: retire `elo_games_per_gen`, keep anchor rating | 30 min | Medium | `config.py`, `run_configurations/*.json` | |
| S8 | Auto-run the pooled tournament at end-of-run | 1 h | Medium | `cli.py`, `training/coach.py` | |
| S9 | Tests: chain, resume reconstruction, clamp, reject-no-advance | 1.5 h | High | `tests/training/`, `tests/evaluation/` | |
| S10 | Docs: EVALUATION + methodology + CLAUDE gotchas | 45 min | Medium | `docs/05-EVALUATION.md`, `docs/research/pool-elo-methodology.md`, `CLAUDE.md` | |

Execution order matters: S1→S2→S3 build the mechanism, S4 removes the old path (do after S1 so the report never has zero Elo data), S5/S6 surface it, S7 cleans config, S8 is independent, S9/S10 finalise.

---

## S1. Compute + roll arena-derived Elo in the Coach

**Current state.** `_learn_loop` (`coach.py:176`) runs arena at `:255-266`, gets `nwins, pwins, draws`, computes `accepted = self._should_accept_new_network(...)` (`:266`), logs the arena result (`:267`), then accepts/rejects (`:288-301`). Strength eval happens separately at `:303-307`.

**Target.** Maintain a running `self._benchmark_elo: float`, initialised to `config.elo_baseline_rating` (400) in `__init__`. Right after the accept/reject block (`:301`), compute and log the candidate's rolling Elo from the arena result, and roll the benchmark forward on acceptance:

```python
from alphablokus.evaluation.elo import compute_elo  # already imported for the old path

# arena already played candidate vs incumbent → derive the candidate's Elo
elo_delta, score_rate = compute_elo(nwins, pwins, draws)   # clamps s to [0.001, 0.999]
candidate_elo = self._benchmark_elo + elo_delta
self.metrics.log_rolling_elo(
    generation=generation,
    rolling_elo=candidate_elo,          # the candidate's absolute Elo this gen
    incumbent_elo=self._benchmark_elo,  # the benchmark it was measured against
    elo_delta=elo_delta,                # arena-derived delta vs incumbent
    score_rate=score_rate,
    wins=nwins, losses=pwins, draws=draws,
    accepted=accepted,
)
if accepted:
    self._benchmark_elo = candidate_elo  # accepted net is the new benchmark
```

Notes for the implementer:
- `compute_elo(wins, losses, draws)` returns `(elo, score_rate)` where `elo = 400·log10(s/(1−s))`, `s = (wins+0.5·draws)/total` clamped to `[0.001, 0.999]` (`evaluation/elo.py:8`). A 100-0 sweep clamps to ~+1200/step — fine and rare for candidate-vs-incumbent (gated scores sit 55–95%).
- `nwins`/`pwins` are new-vs-previous, exactly the orientation we want (candidate first).
- Rejected gens still log a rolling-Elo point (their provisional `candidate_elo`) but do **not** advance `_benchmark_elo` — this gives the report a "rejected attempts" scatter below the accepted line.
- The benchmark net is always the current `best.pth.tar` (= arena incumbent = `pnet`), so no separate net object is needed — only the scalar `_benchmark_elo`.
- Arena games already carry enough diversity (Dirichlet noise / non-degenerate outcomes — see the varied 84/0, 50/50 results in real runs) to make the score a meaningful Elo sample. Do **not** force temp=0 determinism.

---

## S2. Reconstruct the rolling benchmark on `--resume`

**Current state.** Resume reuses the on-disk `elo_baseline.pth.tar` so the gen-0 anchor is stable (`coach.py:150-156`). There is no rolling state today.

**Target.** On resume, `self._benchmark_elo` must equal the Elo of the last *accepted* net at or before `start_generation - 1`, or the anchor rating if none accepted yet. Reconstruct it from the persisted rolling-Elo history rather than adding new checkpoint state (self-healing, no schema coupling to checkpoints):

- Read the rolling-Elo parquet (written by S5) for the run; take the `rolling_elo` of the highest generation `≤ last_completed` with `accepted == True`. If the table is missing/empty, fall back to `config.elo_baseline_rating`.
- Do this in `__init__` (guarded by `self.resume`), mirroring the existing baseline-reuse branch.
- Add a `logger.info("Resume: rolling Elo benchmark = {:.0f} (from gen {})", ...)` line so resumes are auditable (the v3 saga showed how valuable resume-path logging is).

Edge case: a resume that lands mid-run where the last few gens were all rejected must still pick the last *accepted* net's Elo, not the last logged point.

---

## S3. Record the anchor net's provenance (cross-run splicing)

**Why.** The anchor (Elo 400) is run-specific. To compare across runs we must know *which net* is the anchor. For a warm-start run the anchor is the donor net (v3's 400 = `blokus_cloud_60` gen-57), so if we know that net's pooled Elo we can splice the curves.

**Target.** At run start, write the anchor's provenance once:
- A small `Nets/elo_anchor.json` (or a metrics record) with: `anchor_rating` (400), `source` (`"scratch"` or `"warm_start"`), and for warm-start the donor `run_name`, `generation`, checkpoint filename, and a SHA-256 of the weights.
- The warm-start source is known at CLI time (`cli.py:main:115` logs "Warm-starting weights from best.pth.tar"); thread that provenance into the Coach or write it in `cli.py`.
- Surface it in the report subtitle/caption (S6) so a reader knows what "0 Elo" means for this run.

---

## S4. Delete the gen-0 per-gen Elo eval (keep the anchor checkpoint)

**Current state.** `_evaluate_strength_vs_baselines` (`coach.py:303-307`, def near `:475`) calls `_evaluate_elo_vs_baseline` (`:488`), which plays `elo_games_per_gen` games vs `elo_baseline_net` via `_run_elo_serial`/`_run_elo_parallel` (`:525-572`).

**Target.**
- Remove `_evaluate_elo_vs_baseline`, `_run_elo_serial`, `_run_elo_parallel`, the `elo_baseline_net` instance field, and the `elo_games_per_gen > 0` gate in `_evaluate_strength_vs_baselines` (leave the oracle + symmetry branches intact).
- **KEEP saving `elo_baseline.pth.tar` at gen 0.** The pooled tournament depends on it as the gen-0 pool anchor (`scripts/tournament_elo.py:45`, `_ANCHOR_FILENAME`). Retain the `__init__` save (the branch at `:148-154`) but drop the `elo_baseline_net` load — we only need the *file* on disk now, not an in-memory opponent.
- This is the change that saves ~`elo_games_per_gen` games/generation (default 50; v2/v3 set 20).
- Sequence after S1 so the rolling metric already exists before the old one is removed — the report should never regress to an empty Elo panel.

---

## S5. Metrics schema: rolling-Elo fields + `accepted`

**Current state.** `metrics.log_elo` (`storage/metrics.py:695`) writes `elo_rating = baseline_rating + elo_diff`, `elo_diff`, `elo/baseline_rating`, `elo/score_rate`, wins/losses to `_elo_records` + W&B `elo/*` (step_metric `generation`, defined at `:282`).

**Target.** Add `log_rolling_elo` (or repurpose `log_elo`) writing per generation:
- parquet columns: `generation`, `rolling_elo`, `incumbent_elo`, `elo_delta`, `score_rate`, `wins`, `losses`, `draws`, `accepted` (bool).
- W&B: `elo/rolling`, `elo/incumbent`, `elo/delta_vs_incumbent`, `elo/score_rate`, `elo/accepted`.
- `accepted` is essential — the report (S6) splits accepted (line) vs rejected (scatter) on it. Today the arena's `accepted` lives only in the arena records; include it here so the Elo chart is self-contained.
- Remove the now-dead `elo/baseline_rating` / `elo/diff_vs_baseline` keys (or keep as aliases for one release if W&B dashboard continuity matters — implementer's call, note it in the PR).

---

## S6. Report chart: render the rolling non-saturating Elo

**Current state.** `make_elo_plot` (`reporting/charts.py:586`) reads `elo_data` with `elo_rating`/`elo_diff`/`score_rate`/`accepted`, plots accepted vs rejected, titled *"Elo vs Frozen Gen-0 Baseline (saturates once net ≫ gen-0)"* (`:694`).

**Target.**
- Point it at the new rolling-Elo table; plot `rolling_elo` as a climbing line over accepted gens + rejected as a scatter below (the accepted/rejected split logic largely survives — swap column names to `rolling_elo`, `elo_delta`).
- Retitle: *"Rolling Arena-Derived Elo (non-saturating; anchored at start = 400)"*. Add the S3 anchor provenance to the subtitle/hover.
- Hover: show `elo_delta` (vs incumbent), `score_rate`, W/L/D per point.
- Leave `make_tournament_elo_plot` (`:700`) unchanged — that's the pooled curve, still the rigorous one, rendered when `tournament_ratings.parquet` exists.
- Update the report page so both charts sit together: live rolling Elo (always present) + pooled Elo (present after S8 / a tournament run), with a one-line caption explaining live-rough vs pooled-rigorous.

---

## S7. Config cleanup: retire `elo_games_per_gen`, keep anchor rating

**Current state.** `config.py:399` `elo_games_per_gen: int = 50`; `:417` `elo_baseline_rating: int = 400`. Cloud configs set `elo_games_per_gen` (v2/v3 = 20).

**Target.**
- Remove `elo_games_per_gen` from `RunConfig` (and every `run_configurations/*.json`). If backward-compat loading of old configs matters, accept-and-ignore it with a deprecation `logger.warning` rather than erroring.
- Keep `elo_baseline_rating` (now the *anchor start* for the rolling curve) — update its docstring accordingly.
- Add a docstring note on `num_arena_matches` (`config.py:347`): it now doubles as the rolling-Elo sample size, so very low values make the Elo noisier (100 is comfortable; ≤40 is coarse).

---

## S8. Auto-run the pooled tournament at end-of-run

**Current state.** `scripts/tournament_elo.py` is a standalone post-hoc script; nothing runs it automatically. The report renders the pooled curve only if `tournament_ratings.parquet` exists.

**Target.** After the generation loop completes normally (not on crash), optionally invoke the pooled tournament so the rigorous curve is produced without a manual step:
- Add a `RunConfig.tournament.run_at_end: bool` (default False to preserve current behaviour; enable in cloud configs) and, when set, call the tournament entrypoint from `cli.py` (or a `Coach.finalize` hook) after `learn()` returns, before the final report render.
- Reuse `TournamentConfig` (already in `config.py:181`); default the tournament's `num_mcts_sims` low (e.g. 32) since ranking is robust to weak play and this keeps it to ~30–60 min (see the Q&A in `pool-elo-methodology.md` § sparse schedule). Make the sims a `TournamentConfig` field so it's explicit, not inherited from the heavy training `mcts_config`.
- Must be crash-safe: guard so a tournament failure never loses the run's training artifacts (they're already on disk); log and continue to report render.

---

## S9. Tests

- **Chain correctness** (`tests/training/`): given a scripted sequence of arena results + accept/reject decisions, assert `_benchmark_elo` rolls only on acceptance and `rolling_elo` per gen equals `incumbent_elo + compute_elo(...)`. Use real `compute_elo` (no mocks — game-logic test rule).
- **Clamp**: a 100-0 arena sweep yields the clamped ~+1200 delta, not `inf`.
- **Reject-no-advance**: a rejected gen logs a point but leaves `_benchmark_elo` unchanged; the next accepted gen measures against the *unchanged* benchmark.
- **Resume reconstruction** (S2): write a partial rolling-Elo parquet with a mix of accept/reject, then assert reconstruction picks the last *accepted* net's Elo (not the last logged point), and falls back to `elo_baseline_rating` when empty.
- **Pairing/pool unaffected**: existing `tests/evaluation/test_tournament.py` still passes (S4 keeps `elo_baseline.pth.tar`).

---

## S10. Docs

- `docs/05-EVALUATION.md`: replace the gen-0-Elo description with the two-tier scheme — live rolling arena-derived Elo (per-gen, non-saturating, chained/rough) + end-of-run pooled BayesElo (rigorous). Note cross-run splicing via the recorded anchor (S3).
- `docs/research/pool-elo-methodology.md`: add a short "live companion metric" note pointing back here; reaffirm the pooled fit is the rigorous curve.
- `CLAUDE.md` gotchas: update the Elo bullet — the live metric no longer saturates; `elo_baseline.pth.tar` is retained solely as the pooled-tournament anchor; `elo_games_per_gen` is gone.

---

## Scope notes

- **Does not touch the running `blokus_cloud_v3`.** This lands for the *next* run; v3 gets its strength curve from the end-of-run pooled tournament regardless.
- **Cross-run comparability** remains anchor-dependent (same as the pooled metric): only meaningful when runs share an anchor or fold in an external reference (Pentobi at a known level — the deferred E9 extension in the pool-Elo methodology).
