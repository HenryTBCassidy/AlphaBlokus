# Regression post-mortem and next steps — an adversarial re-review

Independent review of the L4 plateau and the `blokus_paired_gate_rerun` regression, written 2026-07-20.
Brief: challenge the prior conclusions (`plateau-investigation.md`, `xl-training-scaleup.md`) rather than
extend them — in particular the demotion of the `xl` net and heavier search. Every claim below is parsed
from the local run artifacts or cites code on the working tree; provenance in §6.

**Verdicts up front:**

1. **The regression is a training pathology the acceptance change unmasked, not "just" a random walk.**
   The regression-guard floor (0.45) was mathematically incapable of rejecting anything — the paired
   instrument compresses even a ~+100-Elo-class real gap to a 0.525 score, so 0.45 corresponds to a
   catastrophe-sized deficit. With the brake removed, a systematically *degrading* training step
   (2 epochs @ constant 1e-3, Adam with **no weight decay**, on single-lineage data) compounded freely:
   value-head overfit (v-loss 0.24 vs v3's healthy 0.36–0.42), near-monotonic decay of every
   internal-consistency metric, and a policy-collapse event at gen 17. Final net: −44 pool Elo, ladder
   0.344 → 0.298.
2. **The `xl` demotion was not justified by the evidence cited for it.** The "no underfitting signature"
   argument is circular — every diagnostic it rests on is measured against the net's *own* self-play
   targets, which a capacity-bound net also fits comfortably. Capacity is **untested**, not refuted.
   A free, same-day probe on the box settles it (§3.4, R3).
3. **"More sims" (n) at the current search width is weakly refuted; "wider search" was never tested.**
   The jax backend truncates to the prior's top-64 actions at *every* node
   (`src/alphablokus/games/blokusduo/jax/search.py:151-198`), so self-play search structurally cannot
   discover moves the stuck policy doesn't already rank — more sims inside that cone bought targets
   that are near-identical (entropy 0.905 vs 0.96 nats) and a one-epoch delta of ~+8 Elo.
4. **Next paid run: `xl` from scratch (config in §5), gated by the free capacity probe; build Pentobi
   distillation in parallel.** Three warm-start continuations of the `large` net through the current
   operator (~$130 total) have produced ≤0 external gain; the only run that ever climbed external levels
   from a plateau was a from-scratch run, and the only external level-jump we have coincided with the
   last capacity step (medium → large). `xl` is config-only and its cost now equals what the failed
   continuations already spent.

---

## 1. Why the paired_gate rerun regressed

### 1.1 The external facts

Final ladder (100 games/level, 400 sims, `temp/benchmarks/rerun_final_ladder.html`) vs the gen-40
starting net (`v3_final_ladder_L1-9.html`):

| Net | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | Weighted |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 gen-40 (start) | 77 | 71 | 61 | 53 | 47 | 40 | 16 | 20 | 21 | **0.344** (L4) |
| rerun gen-20 (final) | 79 | 62 | 58 | 49 | 45 | 35 | 19 | 14 | 10 | **0.298** (L3) |

Down at 7 of 9 levels, −35 pp summed over 900 games. The end-of-run pooled BayesElo tournament
(`blokus_paired_gate_rerun/Tournament/tournament_ratings.parquet`, anchor gen-0 = the gen-40 donor)
agrees in direction and shape: **no generation ever exceeded +5.5** (gen 10), and the final net sits at
**−44.1**, with the trough gens 12–13 at −41/−46 and gens 17–20 at −25/−18/−4/−44. The regression is not
a ladder-noise artifact — two independent instruments say the run went down.

The rolling Elo (`RollingElo/`) tells the same story live: 400 → 352 over 20 generations, biggest single
drop at gen 17 (score 0.4575, −29.6). All 20 gens accepted.

### 1.2 The guard floor was inert — quantify the instrument before trusting the gate

The paired-arena controls (`temp/benchmarks/s5_validation/`) measure the instrument's transfer function:

- **Null** (gen-40 vs itself, paired, 100 games): 0.49, White won 97% of decisive games.
- **Known gap** (gen-40 vs gen-5 — a large, real strength difference): **0.525** paired
  (unpaired with opening diversity: 0.64, `box_20260710/arena_gen40_vs_gen5.json`).
- gen-40 vs gen-57 donor (ladder gap 0.344 vs 0.205!): 51–49 unpaired — indistinguishable.

So the paired score moves ~0.025 per ~100-Elo-class of *true* gap. For the guard floor of 0.45
(`acceptance.py:64-65`, config `guard_floor: 0.45`) to trigger, the candidate would need to be roughly
a **−200-Elo-class** regression — several times the entire v3 run's gain — in a single generation. The
observed per-gen scores span 0.4575–0.5275 (σ ≈ 0.015 across the 20 gens); the floor was never within
2σ of any generation. **`regression_guard` at 0.45 was `always` with extra steps.** Pairing worked
exactly as designed (White won 341–369 of every 400-game arena, `ArenaData.white_wins`, and the split
cancels in scoring) — the problem is that in a game where ~93–97% of games are decided by colour, the
residual "wins as Black" channel carries too few bits at 400 games to detect anything but disasters.

### 1.3 Not a pure random walk: the training step was systematically degrading

A neutral random walk (accept near-equal candidates, some up, some down) predicts a *directionless*
drift and stable internal quality. That is not what the parquets show. Over gens 1 → 20, near-monotonic:

| Metric (source) | gen 1 | gen 20 | v3 for comparison |
|---|---|---|---|
| Policy symmetry KL (`SymmetryDiagnostic`) | 0.639 | **1.236** | fluctuated 0.58–0.98, *ended 0.65* over 40 gens |
| Value symmetry MAE (`PolicyValueConsistency.value_symmetry_mae`) | 0.101 | **0.249** | — |
| Policy–value Spearman (`pvc_spearman`) | 0.400 | **0.278** | — |
| Training value loss, end-of-gen (`rerun.log`) | 0.296 | **0.241** | 0.36–0.42 (healthy) |
| Agreement with frozen gen-1 eval targets (`PolicyAccuracy.mcts_top1`) | 0.90–0.93 | 0.86–0.88 | 0.89–1.00 |

The net was becoming *less equivariant*, *less self-consistent*, and *better at predicting its own
buffer* simultaneously — the signature of overfitting to lineage-specific, non-generalising features,
not of a fair coin-flip over strength. Three ingredients, none present in v3:

- **`epochs: 2`** (v3 ran 1). Emergent reuse doubled to ~12 passes per position
  (`coach.py:_log_training_dynamics`, logged "reuse ≈12.0"). The value head — 10k outcome labels per
  gen spread over ~630k positions — is exactly the head that overfits under reuse, and its loss
  (0.24) undercut v3's floor by ~40%.
- **Constant LR 1e-3 for fine-tuning a converged net.** v3's early gains came from a donor far from
  the operator's fixed point; at the fixed point the same LR is all diffusion, no signal. AlphaZero
  runs its endgame at 2e-5–2e-4. (`plateau-investigation.md` R4 recommended keeping 1e-3 — that was
  written from v1's *frozen* loop, before any data on what 1e-3 does when every candidate is adopted.
  This run is that data. R4 should be considered superseded for warm-start continuation runs.)
- **Adam with no weight decay** (`base_wrapper.py:223` — `optim.Adam(..., lr=...)`, no
  `weight_decay`). AGZ/AZ used L2 c=1e-4 precisely to stop late-stage drift. Nothing in this loop
  penalises the weight growth that the rising symmetry-KL curve is showing.

### 1.4 The gen-17 collapse event

Self-play policy-target entropy (`SelfPlayProfiling.mean_policy_entropy`, per-gen mean) is stable at
0.78–0.91 nats for gens 1–16, then **drops to 0.506 at gen 17** — the sharpest distribution shift in
any run we have — recovering to 0.79–0.84 after. The same generation: the worst arena score of the run
(0.4575 — accepted, floor is 0.45), the biggest rolling-Elo drop (−29.6), and the start of the terminal
loss drop (policy 0.63 → 0.56, value → 0.24 across gens 17–20 as the collapsed games flooded the 6-gen
buffer window). Symmetry KL and value-MAE hit their maxima in gens 17–20. Mechanistically: an accepted
net whose search sharpened pathologically → sharper (lower-entropy) targets → the trainer chased them →
the pool tournament scores gens 17–20 at −25/−18/−4/−44. The guard floor sat 0.0075 below the worst
score and never fired.

### 1.5 Why "loss decreasing, 20/20 accepted" was false comfort

In a gated loop, falling loss on the buffer tracks progress because the incumbent only advances on
external (arena) evidence. With the gate off, the loss is measured against a **moving, self-generated
target**: the loop can make its own data easier to predict (sharper targets, colour-exploiting value
shortcuts, lineage idioms) while external strength falls. Every "healthy" internal signal in this run —
loss ↓, acceptance 100%, eval-set top-1 ~0.99 — is self-referential. The only externally-anchored
per-gen signals (symmetry KL, value symmetry MAE — computed against the game's *ground-truth*
invariances) were flashing red from ~gen 5 onward.

### 1.6 Bug hunt

Re-checked against the rerun (the plateau doc's B1–B10 covered v1/v3): paired scoring arithmetic is
colour-symmetric and pair-linear (`arena.py:306-395`, flip logic at 375–386 correct); the parallel path
maps A=prev/B=new consistently (`coach.py:564-619`, unit-tested); `white_wins/black_wins` logging
(`coach.py:49-69`) matches the per-gen replays; accept/revert paths clean (`coach.py:367-378`);
LR parquet flat at 0.001 as configured; warm start verified by the v1 md5 chain (same load path).
The pooled tournament and the ladder were run by independent code paths and agree. **No code bug found;
the loop did what it was configured to do.** The defect was the configuration's model of the
instrument, plus the training-step settings above.

---

## 2. Hypothesis adjudication

Ranked by what the data actually supports:

**(d) Training dynamics / acceptance policy — CONFIRMED as the cause of the regression.**
Evidence: §1.2–1.4 in full. Note the two failure modes bracket each other: the strict gate (0.55, v1)
froze the loop because the instrument can't *see* small gains; the guard (0.45, rerun) let degradation
through because the instrument can't *see* small losses either. Both are the same root cause — the
arena between near-equal nets carries almost no information in this colour-dominated game — expressed
at opposite thresholds.

**(a) Curriculum / operator ceiling — SUPPORTED as the cause of the plateau (with a sharper mechanism).**
v3-late, sh_v1 (0/17), and sh_v2 (1/9, then scores 0.486–0.511) all show the `large` net sitting at the
fixed point of the current improvement operator: one-epoch candidate delta ≈ +8 Elo internal
(`plateau-investigation.md` §2 B8(iii)), and the rerun shows that *freely compounding* that same
operator for 20 generations yields ≈ **−2 Elo/gen external** once the §1.3 pathology is loaded on top.
The refinement this review adds: the ceiling is partly *structural*, not just distributional — see (c).

**(c) Search too thin — SPLIT VERDICT.** "More simulations at the same width" is weakly refuted:
n 128→256 with considered 32→64 produced targets of nearly identical entropy (0.905 vs 0.96 nats,
B6) and the same fixed point, at 3× the self-play cost. But "search too *narrow*" was never tested:
`topk_legal` truncates to the prior's top-64 of ~17,837 actions **at the root and at every child node**
(`search.py:151-153, 177, 198`), and root exploration noise is applied inside that compact space. A
policy blind spot below rank 64 can never be searched, never appear in a target, and never be fixed —
a closed loop. The ladder/arena evaluations use the full-action-space python MCTS, so this truncation
is exclusively a *training-signal* limitation. (Corroborating hint: in the backend A/B, jax-PUCT
self-play — same truncation but different selection — trained the stronger net head-to-head, 60.5%;
`xl-training-scaleup.md` A6 step 2.)

**(b) Capacity — UNTESTED; the demotion was unsound (§3).** Not confirmed either: no positive
underfitting signal exists. Prior: moderate, and cheap to resolve.

**(e) Code bug — RULED OUT** for the training loop and both instruments (§1.6, plus the plateau doc's
B1–B10 which this review re-checked rather than assumed: within-epoch loss descent, warm-start md5s,
revert semantics, seed handling all verified there and consistent with the rerun's logs).

---

## 3. The bigger-net / more-sims question, re-opened

### 3.1 The demotion argument doesn't hold

`xl-training-scaleup.md` A4 demoted `xl` because "v3 shows no underfitting signature anywhere: value
loss at its best-ever (0.36), top-1 eval agreement 89–100%". Three problems:

1. **It's circular.** All of those diagnostics measure fit to the net's *own* search output: value loss
   against its own games' outcomes, top-1 agreement against a 200-position eval set whose targets are
   the *same lineage's* gen-1 search policies (`training/eval_set.py`, `kind=selfplay_v1`). A
   capacity-bound net produces weaker targets and then fits them comfortably — training loss on
   self-generated data is low in *both* worlds and therefore cannot distinguish them. There is no
   external-quality dataset anywhere in the diagnostics stack.
2. **"A capacity-bound net fails to improve internally too" is asserted, not shown.** In-lineage Elo
   gains via exploitation/cycling are fully available to a saturated net; internal progress with zero
   transfer is consistent with *both* the curriculum story and the capacity story. It cannot be used to
   pick between them, and A4 used it to pick.
3. **"Bigger nets are better at memorising lineage quirks" cuts both ways** and is speculative. The
   standard result in AlphaZero-family systems is the opposite direction: bigger nets produce better
   priors → better search targets → a *different, higher* operator fixed point. The fixed point the
   `large` net is stuck at is a function of net capacity as much as of the operator.

What positive evidence exists is thin but points mildly *toward* capacity mattering: the only external
level-jump in the project's history (L3 → L4 in `blokus_cloud_60`) came with the medium → large step
(2.45M → 8.10M params; `NET_PRESETS`, `config.py:26-29`, counts measured by instantiating
`AlphaBlokusDuo`), and residual policy KL against the net's own n=256 targets never falls below ~0.56
even after two epochs (rerun log) — some unknown share of which is stochastic-target noise, and some of
which may be capacity. 8.1M params for a 14×14 board with a 17,837-action policy is not obviously
saturated (`xl` is 19.07M; AGZ used ~24M for 19×19/362 actions).

Against capacity: v3 *did* transfer real gains at L6–L9 with the `large` net (17→40 / 6→16 / 3→20 /
1→21 after the donor re-baseline, A7.2) — the net was still learning things Pentobi punishes less at
high levels. Capacity is not *proven* binding. It is simply **live**, and it was declared dead on
inadmissible evidence.

### 3.2 The honest cost-of-error ledger

The steer away from `xl`/search was supposed to save ~$100. The subsequent recipe-lever runs cost:
sh_v1 ~$28 (0/17, frozen), sh_v2 ~$60+ (1/9, ladder 0.335 ≈ no change), rerun ~$35 (regression, plus
the strongest net now needs re-verifying) — **≈$125+ and three weeks for ≤0 external progress.** The
"cheap, diagnostic-first" strategy was only cheap per-run. This doesn't make `xl` right, but it resets
the EV comparison: an `xl` run is no longer the expensive option.

### 3.3 Sims vs width

Committing to a position: **do not buy more simulations at top_k=64.** Measured: n=256/64 targets ≈
n=128/32 targets (entropy, fixed point, +8-Elo one-epoch delta), 3× the cost, and the rerun compounded
n=256 targets for 20 gens with nothing to show externally. The untested search lever is **width**
(top_k), which bounds what self-play can ever discover (§2c). The known feasibility cliff was measured
at n=512 + top_k=128 *combined* (`blokus_search_harder` original config stalling); top_k=128 at n≤128
is uncalibrated — a 30-minute box/pod test (`blokus_search_harder_calibration.json` pattern) settles
whether width is even purchasable in the jax backend.

### 3.4 Verdict

- **Capacity: reopen, resolve empirically this week, for free.** The decisive experiment needs no GPU
  rental: fit `large`-fresh vs `xl`-fresh (and `large`-from-gen-40) *supervised* on the same frozen
  buffer of gen-40 self-play games with a held-out split, train to asymptote, compare held-out policy
  CE + value MSE. If `xl` clearly beats `large` on held-out fit of identical data (≥ ~0.03 nats), the
  targets contain structure `large` cannot absorb → capacity binding → the `xl` run is justified on
  evidence for the first time. If they tie, the demotion finally stands on solid ground.
- **Sims: closed at current width. Width: run the calibration probe; if top_k=128/n=128 is feasible,
  it becomes an arm in the next self-play run.**

---

## 4. What the acceptance policy should be

All three arena-gated modes are now empirically characterised: `threshold` freezes (v1, v3-late),
`regression_guard` at any floor the instrument can distinguish from 0.5 is `always` (rerun), and
`always` random-walks-with-drift (rerun). The conclusion is not "pick a better floor" — it is that
**weight-flow decisions should never again be made by candidate-vs-incumbent arena in this game.**
The only instrument that has repeatedly resolved differences the arena calls a tie (gen-57 vs gen-40:
arena 51–49, ladder 0.205 vs 0.344) is the Pentobi ladder.

Recommended policy — **train continuously, select externally**:

1. `gate_mode: "always"` for weight flow (no reverts — reverts caused v1's stationarity; the buffer
   self-corrects), every generation checkpointed (already the case: `accepted_*.pth.tar`).
2. **External keep-best**: an async mini-ladder (L3–L6, 50 games/level, 400 sims, ~2–3 h on the box
   per checkpoint via `scripts/pentobi_benchmark.py`) every 3–5 generations, run on the box in
   parallel with the cloud run. Best-by-mini-ladder is the run's product, not `best.pth.tar`.
3. **Drift circuit-breaker**: two consecutive mini-ladder drops ≥5 pp weighted → stop/resume from the
   best checkpoint. (The rerun would have tripped this by ~gen 8–10, saving ~$20 and the regression.)
4. Keep the paired arena + colour split + rolling Elo as *telemetry only* — the per-gen symmetry-KL and
   value-symmetry-MAE trends earn a place on the report's front page next to them; they were the
   earliest honest warnings this run produced.
5. An EMA teacher / averaged incumbent is *not* recommended yet — it adds a new dynamic to debug and
   the selection problem is already solved more directly by (2)+(3).

---

## 5. Recommendations, prioritised by EV/cost

| # | Action | Cost | Evidence | Success criterion |
|---|---|---|---|---|
| R0 | **Re-crown v3 gen-40** (`blokus_cloud_v3/Nets/accepted_40.pth.tar`) as the project's best net; the rerun produced nothing above +5.5 pool Elo. | free | §1.1 | — |
| R1 | **External keep-best + circuit-breaker** (§4): async box mini-ladder keyed to run checkpoints; arena demoted to telemetry. | ~1 day code | §1.2, §4 | Next run's product is chosen by ladder, not gate. |
| R2 | **Fine-tuning hygiene for any warm-start continuation**: `epochs: 1`, LR `2.5e-4`, switch `optim.Adam` → `optim.AdamW(weight_decay=1e-4)` (`base_wrapper.py:223`), keep batch 1024. | 1-line code + config | §1.3 (v-loss 0.24; monotonic symmetry decay; AZ practice) | Symmetry KL and value-MAE flat-or-falling across a continuation run. |
| R3 | **Capacity probe on the box** (free, same-day): supervised `large` vs `xl` on a frozen gen-40 buffer (regenerate ~10k games at n=256 if the parquets aren't local, ~1 h), 5% held-out, train to asymptote. | ~½ day, $0 | §3.1, §3.4 | Held-out policy CE gap ≥0.03 nats → capacity binding. |
| R4 | **Next paid run — `xl` from scratch** (config below), launched if R3 fires *or* is ambiguous; redirect to R5's run only if R3 shows a clear tie. ~100 gens ≈ $100–130, 4–5 days on a 5090, zero engineering. | ~$100–130 | §3.1–3.2; from-scratch is the only regime that has ever climbed levels; config-only | Weighted ladder > 0.344 (outside CI) or L5 > 50% by run end; stop rule: ladder flat across two mini-ladder windows. |
| R5 | **Pentobi distillation, built in parallel** (1–2 weeks eng while R4 trains): generate 20–50k Pentobi L7–L9 games via the box GTP harness (free CPU), SL fine-tune the best net (policy → Pentobi moves, value → outcomes, LR 1e-4, AdamW), verify by ladder, then resume RL with an opponent pool. The L7–L9 nets that beat us 80–90% are an information source self-play cannot synthesise. | eng-days, ~$0 compute | v3's transfer concentrated at exactly the levels Pentobi teaches; curriculum ceiling (§2a) | +10 pp at any of L5–L7 after SL alone. |
| R6 | **Width calibration**: top_k=128 @ n=128 feasibility test (30 min). If feasible, it's an arm for the run after next; if not, python-PUCT-style full-space exploration goes to IDEAS. | ~30 min | §3.3 | Games/s within 2× of top_k=64. |
| R7 | Measurement hygiene leftover: colour-conditional value calibration in the eval diagnostics (plateau R8a, still undone) — the 0.24 value loss is likely part colour-shortcut and we still can't see it. | ~½ day | §1.3 | — |

**Next-run config** — diff from `run_configurations/blokus_cloud_v2.json` (the validated 60-gen
from-scratch recipe), as `run_configurations/blokus_xl_scratch.json`:

```diff
 {
   "game": "blokusduo",
-  "run_name": "blokus_cloud_v2",
+  "run_name": "blokus_xl_scratch",
   "seed": 42,
   "num_generations": 60,
   ...
-  "num_arena_matches": 100,
+  "num_arena_matches": 200,
   "arena_opening_temp": 1.0,
   "arena_opening_moves": 4,
+  "paired_arena": true,
+  "gate_mode": "always",
   "mcts_config": {
     "num_mcts_sims": 128,
     ...
     "search_policy": "gumbel",
     "gumbel_max_considered": 32
   },
   "net_config": {
-    "preset": "large",
+    "preset": "xl",
     "learning_rate": 0.001,
     "dropout": 0.3,
     "epochs": 1,
     "batch_size": 1024,
     "cuda": true,
     "lr_scheduler": "cosine",
     "lr_eta_min": 0.0001,
+    "weight_decay": 0.0001,        // requires the R2 AdamW change
     ...
   },
+  "tournament": { "run_at_end": true, "opening_temp": 1.0, "opening_moves": 4,
+                  "num_mcts_sims": 32, "games_per_pairing": 30 },
   "selfplay_backend": "jax",
   "jax_selfplay": { "batch_size": 1024, "top_k": 64, "dtype": "bfloat16",
                     "wave_plies": 32, "xla_mem_fraction": 0.8 }
 }
```

Notes: n=128/32 (not 256/64 — §3.3: heavier sims at this width are measured dead weight; from-scratch
gens don't need them and the saving funds ~40 extra generations); cosine+floor is correct for scratch
(constant-LR was a *continuation* fix); `gate_mode: always` + R1's external keep-best replaces the gate;
mini-ladder every 5 gens from gen 20 (before that the net is below L1 and the ladder is uninformative).
Extend past 60 gens via `--resume` while the mini-ladder still climbs.

**What I am explicitly recommending against:** another warm-start self-play continuation of the
`large` net with any variant of the current operator (any n, any gate) expecting L5 — three runs and
one 20-gen free-running experiment all located the same fixed point at or below the starting strength.

---

## 6. Provenance

- Rerun: `temp/runs/blokus/blokus_paired_gate_rerun/` — `ArenaData` (incl. `white_wins`/`black_wins`),
  `RollingElo`, `SymmetryDiagnostic`, `PolicyValueConsistency`, `PolicyAccuracy`, `TrainingEntropy`,
  `ValueCalibration`, `SelfPlayProfiling` (target entropy; gen-17 = 0.506 nats),
  `Tournament/tournament_ratings.parquet`, `rerun.log` (per-gen first/last `Loss_pi`/`Loss_v` parsed by
  splitting on "Starting Training For Generation"; data-regime lines "reuse ≈12.0").
- v3: `temp/runs/blokus/blokus_cloud_v3/` (same tables; `Tournament` gen-40 = +240 vs donor anchor).
- sh_v1/v2: `temp/runs/blokus/blokus_search_harder_v1/`, `_v2/` (`RollingElo`, `ArenaData`).
- Ladders: `temp/benchmarks/rerun_final_ladder.html` (79/62/58/49/45/35/19/14/10, weighted 0.298),
  `v3_final_ladder_L1-9.html`, `sh_v2_ladder.json` (0.335; note 19% at L9).
- Instrument controls: `temp/benchmarks/s5_validation/s5_null.json`, `s5_knowngap.json` (paired),
  `temp/benchmarks/box_20260710/*.json` (unpaired + gen-57 comparison).
- Code: `evaluation/acceptance.py:56-66`; `training/coach.py:49-69, 307-400, 564-619, 770-813`;
  `games/base_wrapper.py:223, 288-323, 1013-1020`; `evaluation/arena.py:306-395`;
  `games/blokusduo/jax/search.py:151-198`; `config.py:26-29` (presets; param counts by instantiation:
  small 0.34M / medium 2.45M / large 8.10M / xl 19.07M).
- Prior analyses challenged: `docs/research/plateau-investigation.md`,
  `docs/research/xl-training-scaleup.md` (incl. A4, A6, A7), `docs/research/blokus-cloud-60-analysis.md`
  (via citations).
