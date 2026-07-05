# LR schedule for self-play training — options, semantics fixes, and A/B

**What this covers.** `blokus_cloud_60` stalled over its last ~10 generations (8/10 arena rejections,
scores 45–50%) while policy loss was still falling briskly (0.53 → 0.40 over gens 45–58), and a
warm-start run from the strongest net trained for one generation at the peak LR (1e-3) beat that net
~20-0. The stopgap was flooring the cosine schedule (`lr_eta_min = 1e-4`). This plan (1) decides what
LR schedule is actually right for a gated self-play regime with a non-stationary objective, (2) fixes
three LR-semantics defects found while grounding that decision (none of which the `eta_min` stopgap
addresses), and (3) defines the A/B that picks the production default. Companion evidence:
[`../research/blokus-cloud-60-analysis.md`](../research/blokus-cloud-60-analysis.md) §3.

---

## Ground truth (verified on current `main`, 2026-07-05)

- `src/alphablokus/games/base_wrapper.py:280-299` — `_create_scheduler`: `match net_config.lr_scheduler`
  with cases `"cosine"` (`CosineAnnealingLR`, `T_max = num_generations × epochs`, `eta_min =
  net_config.lr_eta_min`) and `None`. Stepped once per epoch at `:488-489` (epochs=1 ⇒ once per
  generation). Optimizer is **Adam** at `net_config.learning_rate` (`:215`).
- `src/alphablokus/config.py:224-241` — `NetConfig.learning_rate`, `lr_scheduler: str | None = None`,
  `lr_eta_min: float = 0.0`. Configs load via `dataclass_wizard.fromdict` (`:632`), so new fields with
  defaults are backwards-compatible with every existing JSON.
- `base_wrapper.py:740-775` — `save_checkpoint` always embeds `optimizer_state_dict` +
  `scheduler_state_dict`; `load_checkpoint` **always restores both** when present.
- `src/alphablokus/training/coach.py:223,297` — every generation saves `temp.pth.tar` *before*
  training; on rejection it reloads `temp.pth.tar`.
- `src/alphablokus/cli.py:111-113` — warm start (`load_model: true`) is a plain
  `nnet.load_checkpoint("best.pth.tar")` — full restore, including the donor run's optimizer LR and
  scheduler position.
- The optimizer's actual LR is **not logged anywhere** (no metrics/W&B/report trace).
- `run_configurations/blokus_cloud_v2.json` — the queued next run: `load_model: true`, cosine,
  `lr_eta_min: 1e-4`, `learning_rate: 1e-3`.

### Finding: the run's real LR trajectory was never measured — and never reached the tail

Because `temp.pth.tar` embeds scheduler + optimizer state and the rejection path reloads it, **every
arena rejection rewinds the LR schedule by one step**. The scheduler's effective clock is *cumulative
accepted generations*, not generation number. Simulating the Coach's exact save→step→reject-reload
cycle with `blokus_cloud_60`'s acceptance pattern (39/58 accepted; 37/48 by gen 48; gens 49–54
rejected):

| Gen | LR per the §3 formula table (no rewind) | Actual LR under reject-rewind |
|---|---|---|
| 20 | 7.7e-4 | ~8.5e-4 |
| 40 | 2.7e-4 | ~4.5e-4 |
| 48 | 1.1e-4 | ~3.5e-4 |
| 49–54 (rejection streak) | 9.5e-5 → 3.3e-5 | **pinned at ~3.2e-4** |
| 58 | 6.2e-6 | ~2.7e-4 |

Three consequences:

1. **The §3 LR table overstates the anneal by ~10–50× late-run.** The run never trained below
   ~2.7e-4. The LR-tail *story* survives in weaker form — acceptance was robust at 3.5–4.5e-4
   (gens 34–48: 12/15 accepted at 55–76%) and collapsed once one-generation training deltas at
   ~3.2e-4 stopped clearing a noisy 55%-of-40-games gate — but the mechanism is composite:
   **falling LR raised the odds of entering a gate-hysteresis trap, and the trap then sustains
   itself.** Each rejection rewinds weights + Adam moments + LR, so the next generation retrains
   from the *same* start, at the *same* LR, on an increasingly same-incumbent buffer (after 4
   rejections the 40k buffer is 100% one net's games) — producing a near-identical candidate that
   scores ~50% again. Six consecutive generations of self-play compute (gens 49–54) were fully
   discarded. The stochastic escapes (gens 55, 57 accepted at 73.75%/67.5%) and Pentobi L3→L4 show
   headroom existed throughout.
2. **The `eta_min = 1e-4` stopgap is nearly inert** under these semantics: with a realistic
   acceptance rate a 60-generation run's schedule position stays well above where the floor binds.
3. **`blokus_cloud_v2` as configured will not start at 1e-3.** `load_model` restores the donor
   checkpoint's optimizer LR (~2.7e-4) and scheduler position (~39, with the old `T_max=60`), so the
   warm start silently begins at ~27% of peak — the opposite of what the 20-0 warm-start evidence
   argues for. (Whatever ad-hoc path produced that 20-0 test net, the in-repo warm-start path does
   not reproduce "first generation at peak LR".)

**Adjacent observation (out of scope, park in IDEAS if wanted):** AGZ's gate never reverted the
candidate's optimizer trajectory — training continued through failed evaluations. Our revert-on-reject
is what makes rejection streaks self-trapping and rejected compute a total loss. Worth a separate
idea; this plan only stops the *LR clock* from rewinding.

---

## Options analysis — which schedule family fits this regime

The regime: non-stationary objective (data distribution and MCTS targets improve every generation),
~60 discrete scheduler steps per run, single GPU, gated acceptance, warm-startable across runs.
Annealing-to-converge presumes a fixed loss surface being approached; here the surface moves every
generation, so "the run is ending" is not evidence that smaller steps are appropriate — only "the
target has stopped moving" would be, and this run's target demonstrably hadn't (policy loss still
falling at gen 58, warm-restart 20-0, Pentobi L3→L4 in the final 13 gens).

| Option | Verdict | Reasoning for this regime |
|---|---|---|
| **Constant LR (`lr_scheduler: null`, 1e-3)** | **Recommended default** | The only scheme whose premise matches a moving target: hold step size while the target moves; reduce it between runs, by hand, when externally-anchored strength stalls. AlphaZero's 0.2→0.02→0.002→0.0002 was near-constant per epoch of progress and never approached 0 (`../research/deepmind-run-configs.md`). Empirical support here: acceptance was strongest while effective LR sat at 3.5–8.5e-4, and one generation at 1e-3 from the strongest net won ~20-0. Warm-start friendly (no clock to inherit). Zero config surface. Known failure mode to watch: if 1e-3 is over the stability edge (esp. for a future `xl` net), late-run acceptance collapses **with oscillating/rising policy loss** — distinguishable from cloud_60's stall (falling loss), and the cue to step down. |
| **Stepped / milestone decay (`MultiStepLR`)** | Keep as the escalation path (arm C) | AZ-faithful and bounded-downside, but within one 60-gen run milestone placement is a guess, and gen-indexed milestones re-import cosine's core flaw (schedule keyed to run budget, not progress). Its natural use is **across runs**: warm-start the next run at a lower constant peak once a constant-LR run shows policy-loss flattening or a Pentobi-ladder stall over two runs. |
| **Current floored cosine (`eta_min=1e-4`)** | A/B control arm, not default | Shape is motivated by fixed-objective convergence within a fixed budget — neither holds. `T_max = num_generations` makes per-generation training dynamics a function of an arbitrary run length (a 30-gen run anneals 2× faster than a 60-gen run for no learning-related reason). And under the reject-rewind semantics its realised shape was accidental anyway. Retained because it's the incumbent: the A/B should beat it, not assume it away. |
| **ReduceLROnPlateau** | Reject | Needs a monitor, and every candidate is wrong here. *Arena accept-rate*: gate noise is huge (40 games ⇒ SE ~8 pp) and the sign is backwards — this run's stall demanded LR held or raised; plateau→drop would have deepened the trap (rejections → "plateau" → lower LR → smaller deltas → more rejections: positive feedback). *Value loss*: its late rise came from buffer staleness, not too-high LR — dropping LR treats the wrong cause. *Policy loss*: never plateaued (0.53→0.40 during the stall), so the trigger never fires and the scheduler degenerates to constant with extra machinery. |
| **Cyclical / SGDR warm restarts** | Reject (over-engineering) | The validated half of SGDR is the *restart at peak* — which constant LR gives continuously. The anneal-between-restarts half re-imports the annealing incoherence. Dominated by constant if constant is stable; if constant proves too hot, stepped is the simpler fix. Revisit only after both fail. |
| **"Raise LR when learning fast" (floated)** | Reject as automation | Causally backwards: fast learning at the current LR is evidence the current LR is adequate, not that a higher one would be better. As an automatic rule it's a positive-feedback hazard (accept streak → LR up → instability → rejection streak, each rejection discarding a full generation of self-play). The kernel of truth — *don't anneal while external strength is still climbing* — is exactly what the constant default encodes. Its manual form is already validated practice: warm-start the next run at peak. |

**What metric could legitimately drive an adaptive schedule?** Only externally-anchored ones: the
Pentobi ladder and post-hoc pool Elo. Nothing available per-generation is trustworthy — policy loss is
self-referential (it fell throughout the stall as candidates converged onto a static buffer), arena
results are gate-noise-corrupted, value loss is staleness-confounded, internal Elo is capped. That is
the deepest argument for a **non-adaptive constant within a run**, with adaptation happening *between*
runs by a human reading pool-Elo slope and the Pentobi ladder.

**Recommendation:** default `lr_scheduler: null` at `learning_rate: 1e-3` (the dataclass default is
already `None`, so this is a production-config change, not a code-default change), backed by the
semantics fixes below and confirmed by the A/B (L7). Cosine and step remain available; plateau and
SGDR are deliberately not implemented.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| L1 | Correct the record: addendum to `blokus-cloud-60-analysis.md` §3 (reject-rewind ⇒ actual late LR ~2.7–3.2e-4) | 30 min | High | ✅ |
| L2 | Log the actual optimizer LR per generation (parquet + W&B + report chart) | 1.5 h | High | ✅ |
| L3 | Scheduler clock follows generations: arena rejection no longer rewinds LR state | 1.5 h | High | ✅ |
| L4 | Warm start = weights-only: `load_model` gets a fresh optimizer + scheduler at the config LR (`--resume` keeps full restore) | 1.5 h | High | |
| L5 | Pluggable `lr_scheduler`: `"constant"`/`null`, `"cosine"`, `"step"` (+ `lr_milestones`, `lr_gamma`) in `_create_scheduler` | 1.5 h | High | |
| L6 | A/B run configs: identical warm-start recipe, scheduler-only delta (constant vs floored cosine, optional step arm) | 30 min | Medium | |
| L7 | Run the A/B; judge by head-to-head arena + pool-Elo slope + Pentobi ladder; write `docs/research/lr-schedule-ab.md`; set the production default | ~13 h GPU + 1 h analysis | Medium | |

Execution order is dependency order: L2–L4 make any schedule's behaviour observable and its warm-start
semantics correct *before* the comparison, otherwise the A/B measures accidents again. L1 is
independent and first because the current §3 table is actively misleading.

---

## L1. Correct the record in `blokus-cloud-60-analysis.md` §3

**Current state.** §3's LR table (9.5e-5 at gen 48 → 2.7e-6 at gen 58) is formula-derived
(`0.5·lr₀·(1+cos(πt/60))` indexed by generation), presented as ground truth. Under the
reject-rewind semantics that were live during the run (scheduler state in checkpoints since
`e08a0a1`, 2026-06-23), the actual trajectory was indexed by *accepted* generations and never went
below ~2.7e-4.

**Fix.** Add a dated addendum to §3 (do not rewrite the original — append, per the
preserve-the-record convention): the corrected LR table, the rewind mechanism
(`coach.py:223/297` + `base_wrapper.py:774-775`), and the revised reading — LR decline raised the
probability of entering the gate-hysteresis trap; the trap (weights + Adam + LR all rewound onto an
increasingly single-incumbent buffer) sustained the stall. Note that the `eta_min` floor does not
bind under these semantics and that recommendation #1's mechanism is superseded by this plan.

**Effort:** 30 min.

---

## L2. Log the actual optimizer LR per generation

**Current state.** No LR is recorded anywhere — the reason a ~50× misreading of the run's actual LR
went unnoticed. Any schedule experiment is unreviewable without this.

**Fix.**
- `MetricsCollector.log_learning_rate(generation, epoch, lr)` in `src/alphablokus/storage/metrics.py`
  (hive parquet alongside the existing tables + W&B scalar `train/learning_rate` keyed on
  `generation`, same pattern as `log_training_throughput` at `:864`).
- `RunConfig` directory property (e.g. `learning_rate_directory` → `LearningRate/`) next to its
  siblings in `config.py:540-614`.
- Call site: `base_wrapper.py` `train()`, once per epoch — read
  `self.optimizer.param_groups[0]["lr"]` *before* the `scheduler.step()` at `:488` (that is the LR
  the epoch actually trained at).
- Report: one log-y LR-vs-generation chart in `reporting/report.py` (pattern of the existing
  `_chart` figures, ~`:428`); section omitted when the directory is absent so old runs re-render
  unchanged.

**Effort:** 1.5 h.

---

## L3. Stop the arena rejection from rewinding the LR schedule

**Current state.** `coach.py:297` reloads `temp.pth.tar` on rejection; `load_checkpoint`
(`base_wrapper.py:774-775`) restores `scheduler_state_dict`, and the optimizer restore brings back
the pre-step `lr` in its param groups. Net effect: the schedule clock = accepted-generation count —
undocumented, unmeasured, and it silently freezes the LR during rejection streaks.

**Fix.** The schedule's clock should be *generations of training performed*, independent of gating
(weights and Adam moments **should** still revert — that is the gate's job; only the LR clock must
not). Implementation:
- `load_checkpoint(self, filename: str, *, restore_lr_schedule: bool = True)` in
  `base_wrapper.py`; when `False`, skip the scheduler restore and, after the optimizer restore,
  re-sync `param_groups[...]["lr"]` from `self.scheduler.get_last_lr()` (no-op when
  `self.scheduler is None` — then also skip re-syncing so a scheduler-less run reverts fully,
  bit-for-bit as today).
- Update the `INeuralNetWrapper.load_checkpoint` protocol signature (`interfaces.py:352`) to match.
  Other call sites (`coach.py:158,224`, `parallel/pool.py`, `cli.py`) are keyword-default-compatible
  and unchanged.
- Coach rejection path (`coach.py:297`) passes `restore_lr_schedule=False`.
- Test: reject-then-accept sequence asserts the scheduler's `last_epoch` advances once per
  generation regardless of gate outcome, and that weights/optimizer moments do revert.

**Behaviour change note:** only affects runs with a scheduler *and* rejections — i.e. exactly the
configuration whose current behaviour is the accident being fixed. TicTacToe/Mac configs
(`lr_scheduler: null`) are bit-for-bit unchanged.

**Effort:** 1.5 h.

---

## L4. Warm start loads weights only

**Current state.** `cli.py:111-113` (`load_model: true`) full-restores `best.pth.tar`, inheriting the
donor run's optimizer LR and scheduler position — so `blokus_cloud_v2` would start at ~2.7e-4 with a
39/60-spent cosine clock, not the intended 1e-3. `--resume` (`cli.py:106`) *should* full-restore —
that is a continuation, not a new run.

**Fix.**
- Add `load_weights(self, filename: str)` to `base_wrapper.py` (+ the protocol): loads
  `state_dict` only; optimizer and scheduler stay as freshly constructed from *this* run's config.
- `cli.py:113` uses `load_weights("best.pth.tar")` for the `load_model` path; the `--resume` path
  keeps `load_checkpoint("latest.pth.tar")` unchanged.
- Test: warm start from a checkpoint saved mid-anneal trains its first generation at
  `net_config.learning_rate`.

With L2 in place, the first generation's logged LR verifies this end-to-end on the next real run.

**Effort:** 1.5 h.

---

## L5. Pluggable scheduler in `_create_scheduler`

**Current state.** `config.py:231` `lr_scheduler: str | None` with one real value; the `match` in
`base_wrapper.py:282-299` has cases `"cosine"` / `None` / error.

**Fix.** Keep `str | None` (dataclass_wizard-friendly), extend the `match`:

```python
# config.py — NetConfig additions (defaults keep every existing JSON valid)
lr_milestones: tuple[int, ...] = ()  # "step": generations at which LR multiplies by lr_gamma
lr_gamma: float = 0.1                # "step": decay factor per milestone

# base_wrapper.py — _create_scheduler
match self.net_config.lr_scheduler:
    case None | "constant":
        return None                                    # constant at learning_rate
    case "cosine":
        ...existing branch, unchanged (incl. the eta_min warning)...
    case "step":
        if not self.net_config.lr_milestones:
            raise ValueError('lr_scheduler "step" requires non-empty lr_milestones')
        milestones = [m * self.net_config.epochs for m in self.net_config.lr_milestones]
        return MultiStepLR(self.optimizer, milestones=milestones, gamma=self.net_config.lr_gamma)
    case unknown:
        raise ValueError(f"Unknown lr_scheduler: {unknown!r}")
```

Milestones are in generations (converted to scheduler steps via `epochs`, mirroring the cosine
`T_max` convention). `"constant"` is accepted as an explicit alias for `null` so A/B configs can
state their arm. Docstrings on the fields per the style guide; unit tests for each branch + the
empty-milestones error. Deliberately **not** implemented: `ReduceLROnPlateau`, SGDR (rejected in
the options analysis — adding them would be over-engineering).

**Effort:** 1.5 h.

---

## L6. A/B run configs

Two (optionally three) configs in `run_configurations/`, byte-identical apart from the
`net_config` scheduler block and `run_name`:

- Common: `blokus_cloud.json` recipe (10k games/gen, jax Gumbel n=64, large net, batch 1024,
  `num_arena_matches: 40`), `num_generations: 30`, `load_model: true` warm-started from the **same
  seed checkpoint** — `blokus_cloud_60`'s `best.pth.tar` (the gen-57 accepted net, Pentobi L4) —
  copied into each run's `Nets/` (weights-only load via L4), same `seed`.
- `lr_ab_constant.json`: `"lr_scheduler": "constant"`, `learning_rate: 1e-3`.
- `lr_ab_cosine.json`: `"lr_scheduler": "cosine"`, `lr_eta_min: 1e-4` (the incumbent stopgap;
  `T_max` auto = 30).
- Optional `lr_ab_step.json`: `"lr_scheduler": "step"`, `lr_milestones: [20]`, `lr_gamma: 0.3`
  (1e-3 → 3e-4) — only run if the constant arm shows the too-hot signature (see L7).

Not bumped here (isolate the variable): arena games, sims, buffer — the cloud-60 analysis's other
recommendations go in a later run once the schedule question is settled.

**Effort:** 30 min.

---

## L7. Run the A/B and pick the default

**Compute.** ~12.5 min/gen at this recipe on a 5090 ⇒ ~6.5 h/arm; two arms sequentially on one pod
≈ 13 h (≈ one cloud_60-sized budget), +6.5 h if the step arm is triggered. Evaluation (arena
head-to-head + tournament + ladder) runs on the home box afterwards — do not use the per-generation
internal Elo for anything (capped/saturating).

**Why 30 generations is enough.** Cosine at `T_max=30` from a 1e-3 peak crosses 5e-4 by gen ~15 and
sits at 1.3e-4–1e-4 over gens 25–30, so the two arms' LRs differ by ≥2× across the entire back half
— the exact regime where cloud_60 stalled. Expect near-identical arms over gens 1–10 (cosine still
≥ 8.7e-4); a null read there is expected, not a failed experiment. The read is gens 15–30.

**Judging, in order of weight:**
1. **Head-to-head arena** — `scripts/arena_two_checkpoints.py`, final-A vs final-B, **200 games**
   (SE ≈ 3.5 pp): decisive if either side ≥ 55%. Also A₁₅ vs B₁₅ (the mid-run accepted nets
   nearest gen 15) as a secondary point.
2. **Pool-Elo slope** — `scripts/tournament_elo.py` per arm. Both pools anchor to the *same*
   warm-start gen-0 net, so ratings are roughly comparable across arms; the primary statistic is
   each arm's Elo gain over its final 10 generations (is improvement still compounding late — the
   question cosine failed in cloud_60), secondarily final rating vs the shared anchor.
3. **Pentobi ladder** — `scripts/pentobi_benchmark.py --levels 3-6 --games 100` on each arm's
   final net plus the seed net (baseline), on the box (`--workers 4`): external absolute anchor;
   read = weighted-score delta over the seed.

**Decision rule.** Prefer the arm that wins (1); if (1) is inside noise (<55% both ways), prefer the
arm with the higher late-run pool-Elo slope that is not behind on the ladder; ties → **constant**
(simpler, zero config). **Too-hot escape hatch:** if the constant arm shows sustained acceptance
collapse *with oscillating or rising policy loss* (unlike cloud_60's falling-loss stall — check the
L2 LR chart and loss chart together), constant is over the stability edge: run the step arm and
judge it against cosine by the same rule.

**Write-up:** `docs/research/lr-schedule-ab.md` (pattern of `jax-pipeline-ab.md`), then set the
winner in the production cloud config and update `CLAUDE.md`/`AGENTS.md`'s current-focus line.

**Effort:** ~13 h GPU wall-clock (unattended) + ~1 h analysis/write-up.

---

## Recommended first experiment

Land L1–L5 (one working day), then L6/L7 as a **two-arm** A/B — constant 1e-3 vs floored cosine —
30 generations each, warm-started weights-only from `blokus_cloud_60`'s `best.pth.tar`, judged by
head-to-head arena + pool-Elo slope + Pentobi ladder as above. The step arm stays on the shelf
unless the constant arm shows the too-hot signature. Prior expectation from the evidence (stall at
~3.2e-4, 20-0 at 1e-3, DeepMind's never-near-zero schedules): constant wins or draws; either way
the run finally measures LR for real (L2) with semantics that mean what they say (L3/L4).
