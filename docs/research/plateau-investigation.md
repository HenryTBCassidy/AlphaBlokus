# Plateau investigation — why `blokus_search_harder` rejected 0/17

Post-mortem of the `blokus_search_harder` (v1) acceptance collapse, with a re-assessment of
`blokus_cloud_60`, `blokus_cloud_v3`, and the live `blokus_search_harder_v2`. Written 2026-07-15.
Everything below is parsed programmatically from the metric parquets, run logs, and per-game arena
replay records (data provenance in §6); code claims cite file:line on `main` (working tree clean of
functional diffs — the only uncommitted changes are docstring path renames).

**Verdict up front: there is no training-loop code bug.** The LR is applied, the optimizer moves the
net every generation, the revert-on-reject is clean, and the warm start is byte-exact. What the data
exposes instead is a **measurement failure that became a training failure**: between near-equal
nets, **96.3% of decisive arena games are won by White (the first mover)** — opening diversity on —
so the gate score is structurally pinned at ~0.50 ± 0.02 and a 0.55 (or even 0.52) threshold is
unreachable by any candidate that is merely *somewhat* better. The gate then freezes the incumbent,
which makes the training problem literally stationary, which caps the candidate at
"incumbent + one epoch" forever. The per-generation candidates *were* measurably better (they
doubled their win rate as Black vs the incumbent's, 40 vs 22 over 850 games each) — improvement the
instrument cannot convert into acceptance.

---

## 1. The runs, as they actually ran

Config deltas verified against the parquets and logs, not the committed JSONs. Two
committed-vs-ran mismatches to be aware of:

- The **v3** config committed on the box says cosine/60-gen; v3 actually ran **constant 1e-3 over
  40 gens** (all 40 rows of `blokus_cloud_v3/LearningRate` = 0.001).
- The committed `run_configurations/blokus_search_harder.json` says **n=512 / top_k=128**; the run
  actually used `blokus_search_harder_volume.json` with **n=256 / top_k=64** — confirmed from data:
  `SelfPlayProfiling.total_sims / num_moves` = exactly 256.0 for every v1 episode (128.0 for v3).

| | cloud_60 | cloud_v3 | search_harder v1 | search_harder v2 (live) |
|---|---|---|---|---|
| Start | scratch | warm start, cloud_60 gen-57 | warm start, **v3 gen-40** | warm start, v3 gen-40 |
| Gens (done) | 58/60 | 40/40 | 17/40 (killed) | 40 planned (gen 1 as of 2026-07-15 22:19) |
| LR | cosine→~2.7e-4 (reject-rewind) | constant 1e-3 | constant 1e-3 ✅ verified | constant 1e-3 |
| Gumbel n / considered | 64 / 16 | 128 / 32 | **256 / 64** ✅ verified | 256 / 64 |
| jax top_k | 64 | 64 | 64 | 64 |
| Buffer | 40k | 60k | 60k (full @ gen 6, staleness 6) | 60k |
| Gate / arena games | 0.55 / 40 | 0.55 / 100, deterministic | 0.55 / 100, **openings temp 1.0 / 4 plies** | **0.52 / 400**, openings on |
| Accepted | 39/58 | 21/40 | **0/17** | — |
| Wall-clock/gen | ~12.5 min | ~37 min | **~99 min** (self-play 5,329 s = 90%, training 506 s, arena 92 s — `Timings`) | ~103 min (est.) |

Warm-start integrity (bug-hunt item, ruled out): the donor staged as v1's `Nets/best.pth.tar` is
md5-identical to v3's `Nets/accepted_40.pth.tar` (`6f9c6ed0…`), and v1/v2's saved
`elo_baseline.pth.tar` anchors are md5-identical to each other (`7714cf63…` — same weights + fresh
optimizer state, serialized identically). The load path is weights-only
(`cli.py:111-116` → `load_weights`, `base_wrapper.py:1081-1099`), so no donor optimizer/scheduler
state leaks in (the L4 fix, working as designed).

---

## 2. Bug hunt — every check, with evidence

| # | Check | Verdict | Evidence |
|---|---|---|---|
| B1 | LR actually applied at 1e-3? | ✅ **clean** | `blokus_search_harder/LearningRate`: 0.001 in all 17 generations, logged pre-scheduler-step (`base_wrapper.py:539-544`). `lr_scheduler: "constant"` → scheduler is `None` (`base_wrapper.py:299-301`), so nothing can decay or rewind it. |
| B2 | Does loss fall *within* an epoch? | ✅ **training works** | `TrainingData` (window means, 25 batches): every generation traverses policy-KL **0.72 → 0.62** and value-MSE **0.51 → 0.39** across its ~3,700 batches. The optimizer is not stuck; one epoch just isn't enough to close the KL, and the next generation starts over (B3). |
| B3 | Candidate trained from the incumbent? Rejection reverts cleanly? State leaks (optimizer / scheduler / compiled graph / buffer)? | ✅ **clean — provably** | Code: `temp.pth.tar` saved pre-training (`coach.py:274`), reject reloads weights + Adam moments, LR clock untouched (`coach.py:352`, `base_wrapper.py:1041-1079`); self-play always runs `self.nnet` = the reverted incumbent. Data: the within-epoch loss trajectory is **identical to ~3 decimal places in all 17 generations** (pi_first 0.713–0.729, pi_last 0.613–0.647 for gens 2–17). Any leaked optimizer state, scheduler position, or stale compiled graph would make successive generations drift; they don't. This same fact *proves the loop is stationary*: same start weights + same Adam state (reverted to step-0 moments) + same data distribution → the same candidate, every generation. |
| B4 | Right buffer? | ✅ **clean** | Log (`alpha.log`): fills 10k→60k games by gen 6, then 60,000/60,000 with ~3.795M positions and staleness ≈6.0 every gen; `TrainingThroughput.num_examples` matches. All games are by the (frozen) incumbent — correct gated-loop behaviour, and the second ingredient of the stationarity. |
| B5 | Warm start loaded correctly (no re-init heads / partial load)? | ✅ **clean** | md5 chain in §1; `load_state_dict` is strict by default, so a mismatched head would have thrown. Gen-1 eval-set top-1 vs its own gen-1 targets = 0.985 (`PolicyAccuracy`) — a re-initialised head would score ~0. |
| B6 | Are the n=256/considered-64 policy targets sane (not diffused)? | ✅ **sane — slightly *sharper* than v3's** | Target entropy (`SelfPlayProfiling.mean_policy_entropy` — this is the entropy of the stored Gumbel `action_weights` target, computed at harvest, `harvest.py:129-132`): v1 ≈ **0.905 nats, dead constant** across gens 1–17 (frozen incumbent ⇒ frozen target distribution — corroborates B3). v3's final gens: 0.963–0.966. Per-ply structure from `self_play_16.parquet` (gen 17): support = the full 64 considered slots until the endgame, top-1 target mass mean 0.68, opening-ply entropies 0.46–1.25 nats. These are healthy completed-Q softmax targets (`search.py:204-217`); "heavier search diffused the targets" is **ruled out**. |
| B7 | Opening diversity actually active, and applied fairly? | ✅ active, ✅ symmetric — ⚠️ but see B8 | Plumbing: `pool.py:407-423` applies `(temp 1.0, 4 plies)` to *both* players; per-game seeds are unique (`pool.py:651-652` — episode indices 0–99, generation folded in, so no cross-generation seed reuse). Data: games vary (draws 0–3/gen, White win rate varies 93–100% by gen — impossible under deterministic cloned games, cf. v3's 22/44/50-draw generations of clones). |
| B8 | Does the diversified gate systematically suppress acceptance? | 🔴 **YES — root cause, but not how we guessed** | See §3. Not extra variance, not colour/seed asymmetry — the opposite: **variance collapse via first-mover pinning**. |
| B9 | Gen-9 end-of-epoch loss spike (pi 1.88 at batch 3707) | ✅ benign logging artifact | The epoch-end partial-window flush (`base_wrapper.py:535`) logs a mean over the ~8 leftover batches — tiny window, high variance. Interior windows of gen 9 are normal (0.616–0.625). |
| B10 | Elo/acceptance arithmetic | ✅ clean | `acceptance.py::is_accepted_score_rule` (draws = 0.5, `score >= threshold`), `elo.py::compute_elo` — both recomputed from raw W/L/D and match the logged values. Arena play is noise-free (`add_root_noise` defaults False, `mcts.py:134-156`; `players.py:106` doesn't override it). |

### B8 in full: the arena is colour-pinned

Three independent lines of evidence, all from `blokus_search_harder/ArenaReplays` +
`RollingElo`:

**(i) The score variance is impossibly small.** The 17 per-generation scores span 0.485–0.530 with
std = **0.0113**. Under 100 independent games at p ≈ 0.51, the score std must be ≈ 0.050. A
variance-ratio test gives χ² = 0.82 on 16 dof, **p ≈ 1.3 × 10⁻⁸** — the game outcomes are not
independent Bernoulli draws in net strength; something systematic decides nearly every game.

**(ii) That something is colour.** Reducing the replay records per game (`outcome` ×
`player1_was_white`): **White won 96.3% of the 1,700 decisive arena games** (per-generation range
93–100%, never below 93%). The candidate wins 96.5% of its games as White and 4.7% as Black; the
incumbent symmetric (≈97% / 2.6%). So each 100-game arena is ~50 near-guaranteed points for each
side plus a handful of Black upsets — the score is pinned to 0.50 ± the upset differential, and the
0.55 gate demands the candidate win ~12–14% of its Black games against an equal opponent. No
one-epoch candidate can do that.

**(iii) The candidates were genuinely (slightly) better — the gate just can't see it.** Summed over
17 generations: candidate 40/850 wins as Black vs incumbent 22/850 (≈1.8×, one-sided binomial
p ≈ 0.02). The mean score 0.5115 over 1,700 games ≈ **+8 Elo per candidate** — real, positive,
invisible to a 0.55 gate.

**Why v3's deterministic gate had the same disease in a different form:** with temp = 0 everywhere,
all 50 same-colour-arrangement games are byte-identical clones, so a near-equal pairing collapses
to two distinct games and the mirrored pair splits by colour → **16 of v3's 19 rejections scored
exactly 0.500** (recomputed from `blokus_cloud_v3/ArenaData`; the A1 addendum's "14 of 19"
undercounts slightly). Across all 4,000 v3 arena games, White won 77% of decisive games — lower
than v1's 96% only because v3's *accepted* candidates were strong enough to win as Black (scores
0.75–1.0). Opening diversification (trustworthy-measurements S1–S3) fixed the *clone degeneracy*
but not the *colour pinning*: the S3 null test's 49–51 result is exactly what a colour-pinned null
produces, and the raw JSONs (`temp/benchmarks/box_20260710/arena_g40a_vs_g40b.json`) record no
per-game colour split, so the test was structurally blind to this failure mode. **A7.1's
conclusion "effects below ~50 Elo are now resolvable" is wrong and should be read as retracted**:
the instrument resolves gaps roughly at the "wins-as-Black" threshold (the S3 known-gap pair,
gen-40 vs gen-5 at 64%, implies gen-40 won ~28% of its Black games — a ~+100-Elo-class gap).
Everything below that is squashed into 0.48–0.53.

Whether the 96% first-mover conversion is pure Blokus Duo game property vs amplified by
deterministic 256-sim play is left open (both plausible; self-play at temp = 1 for 12 plies shows
73% White / 18.5% Black / 8.2% draws from the same net — `self_play_16.parquet` first-position
values — so the advantage is real but determinism amplifies it from ~73% to ~96%).

---

## 3. Why v3 climbed (21/40) and v1 froze (0/17) — separating the confounders

The candidate each generation is `incumbent + one epoch @1e-3 on the buffer`. Acceptance requires
that one-epoch delta to clear the gate's *resolution floor* (≈ "can it win as Black"). The four
suspected confounders, adjudicated:

1. **Warm-start-from-converged-net — CONFIRMED, half the story.** v3's donor (cloud_60 gen-57) was
   far from the fixed point of the n=128/32 improvement operator: its gen-1 candidate scored
   **0.85** and 9 of the first 12 generations cleared 0.55 outright — deltas big enough to win as
   Black. v1's donor (v3 gen-40) is already ≈ the fixed point of a nearly identical operator
   (n=256/64 targets have almost the same entropy, B6), so its one-epoch delta is ~+8 Elo. As v3
   itself converged, it developed exactly v1's signature: gens 27–39 include 10 rejections at
   *exactly* 0.500 — v1 is v3's endgame, entered at generation 1.
2. **Opening-diversity gate — CONFIRMED, the other half, with the mechanism inverted.** The
   diversified gate is fair and non-degenerate but colour-pinned (§2 B8). It did not *suppress*
   good candidates below their true score; it compressed the whole score scale so that "somewhat
   better" reads as 0.51. Under v3's deterministic gate the same near-equality read as exactly
   0.50 — both instruments fail near equality, so the switch to diversity is **not** what changed
   v1's outcome vs v3. What changed is (1): v1's candidates never had the big deltas that made
   v3's gate results legible.
3. **Heavier search (n=256/64) — RULED OUT as the cause of collapse.** Targets are sane and
   slightly sharper (B6); the within-epoch KL descent is healthy. What heavier search *did* do is
   triple the price of the loop (self-play 1,353 → 5,329 s/gen) so v1 spent **~28 h / ~$28 of 5090
   time re-deriving the same rejected candidate 17 times**. Whether n=256 improves the *ceiling*
   remains untested — the gate froze the loop before the operator could compound even once.
4. **An actual code bug — RULED OUT** (B1–B5, B9, B10).

Mechanistically the freeze is a closed loop: gate rejects → weights + Adam revert
(`coach.py:352`) → self-play re-runs the same net at the same search settings → buffer distribution
unchanged → next epoch reproduces the same candidate (identical loss traces, B3) → gate rejects.
Nothing in the loop injects novelty except RNG, and ±1 arena game of RNG is 20× smaller than the
gate's margin.

---

## 4. Per-run assessment

**blokus_cloud_60** (58 gens, scratch, cosine LR): the productive scratch run — policy loss
2.53 → 0.40 and ladder L3 → L4; late-run stall (2/10 accepted in gens 49–58) previously attributed
to the LR tail, refined to LR + gate-hysteresis (reject-rewind froze the schedule near 2.7e-4;
[`blokus-cloud-60-analysis.md`](blokus-cloud-60-analysis.md) §3 + addendum — not re-parsed here;
only its `Reporting/` HTML is retained locally). In hindsight its rejection streaks at 45–50%
scores were the first appearance of the colour-pinned/clone-degenerate gate, misread at the time as
pure binomial noise on 40 games.

**blokus_cloud_v3** (40 gens, warm start gen-57, constant 1e-3): real work early, asymptote late.
Accepted 21/40 with big early scores (0.85/0.75/0.95/0.79 in gens 1–6), pool Elo +240, and a real
but modest external gain concentrated at Pentobi L5–L9 (the corrected read after the S5 donor
re-baseline: donor is L3 at 100 g/level; v3 gen-40 clearly stronger at L6–L9 —
[`xl-training-scaleup.md`](xl-training-scaleup.md) A7.2). From ~gen 27 it entered the near-equality
regime: 10 of its last 13 rejections are exact-0.500 clone splits. Its loss curves moved
(0.49→0.67→0.48 policy) because incumbents actually changed 21 times — each acceptance shifted the
buffer distribution.

**blokus_search_harder v1** (17 gens, warm start v3 gen-40, n=256/64, gate 0.55/100 diversified):
zero accepted; provably stationary (identical per-gen loss traces; constant 0.905-nat target
entropy; rolling Elo +0..+21). The run's one useful product is this diagnosis: candidates were
consistently ~+8 Elo (Black wins 40 vs 22) and the instrument cannot pass that. ~99 min/gen, 90% of
it self-play. No new ladder needed: with 0 accepts the final net *is* the donor.

**blokus_search_harder v2** (live; gate 0.52, arena 400, same everything else): as of 2026-07-15
22:19 UTC it is in gen-1 self-play (7%→17% in the log tail; no arena data yet). **Prediction from
v1's distribution**: per-generation score ≈ N(0.5115, ~0.0056 at 400 games) → P(score ≥ 0.52) ≈
5–10% per generation, so expect **~2–4 acceptances in 40 generations** — a partial, slow unfreeze
at best, for ~68 h / ~$67 of pod time. (v1's own scores would have passed 0.52 six times out of 17
*at 100 games*; quadrupling the games halves the spread and cuts that roughly in half.) Each
acceptance does shift the buffer and could compound, so it is not worthless — but it is a weak test
of the search-harder hypothesis through a still-broken instrument. Worth considering whether the
remaining ~$60 buys more as a rerun through a fixed gate (R1/R2 below). Decision is Henry's; this
investigation did not touch the pod beyond reads.

---

## 5. Recommendations, prioritised by expected value / cost

The unifying principle: **v1 failed at the instrument and the acceptance policy, not at the
optimiser.** Fix measurement first, then acceptance, then make each generation's delta bigger;
judge search depth and curriculum through the fixed instrument only.

| # | Change | Cost | Why / spec |
|---|---|---|---|
| **R1** | **Paired colour-swapped openings in the arena gate + pool tournament** (the S3 fallback that was pre-registered in trustworthy-measurements §S3 — its trigger has now fired). Sample a k-ply opening prefix once per pair (from the incumbent's visit distribution, temp 1.0, k=4), then play it out **twice, colours swapped**, deterministically. Score per pair; colour advantage cancels exactly. Also log `white_wins`/`black_wins` per gen in `ArenaData` permanently — one groupby on data we already record (`ArenaReplays` has `player1_was_white`) would have caught this 3 runs ago. | ~½–1 day code | Converts the gate's information content from "colour coin with 4% leakage" to "net win differential per opening". 50 pairs ≫ 100 unpaired games in resolution. Re-run the S3 known-gap/null controls **with colour split reported**. |
| **R2** | **Change the gate's role from improvement-filter to regression-guard**: accept unless the candidate is clearly worse (paired score < 0.48–0.50); or drop gating entirely, AlphaZero-style, and always adopt. Keep `accepted_*.pth.tar` checkpointing + the pool tournament as the offline strength record. | config/semantics, ~hours | The 0.55 gate was inherited from AGZ (400 games), where it was cheap insurance; here it is the *direct cause* of a 28-hour stationary loop. Every DeepMind successor dropped it (deepmind-run-configs §AlphaZero). With a rolling 60k buffer, one mediocre accepted net self-corrects within ~6 gens; a frozen incumbent never does. R1 makes even the conservative 0.50-guard trustworthy. |
| **R3** | **Train the candidate harder per generation: `epochs` 1 → 2** (consider 3 later). | +506 s/gen ≈ **+8.5%** wall-clock | The within-epoch KL is still falling briskly at epoch end (0.72 → 0.62, B2) — the candidate is undertrained relative to the signal available. Training is 8.5% of the generation while self-play is 90%; doubling the cheap phase directly grows the per-gen delta that R1/R2 let through. Emergent reuse goes 6× → 12× — watch value loss for overfit (it's currently comfortable at 0.39–0.44). |
| **R4** | **LR: keep constant 1e-3. No warmup, no cyclical schedule, no acceptance-coupled LR.** For long from-scratch runs only: cosine with `lr_eta_min: 1e-4`. | free | The data removes the LR from the suspect list entirely: it was applied exactly (B1), descent is stable from the first window (no warmup pathology), and v3 already showed constant-LR climbing when the operator had headroom. Adding schedule complexity now would confound the R1–R3 experiment. Revisit (step decay to ~3e-4) only if, *through the fixed instrument*, accepted-gen quality visibly oscillates. |
| **R5** | **Batch size: keep 1024** (train and jax self-play). | free | The binding quantity is optimizer steps per generation on fresh targets, not gradient noise: batch 2048 would *halve* steps/epoch — the wrong direction while the KL is descent-limited (B2). The 5090's idle VRAM is not evidence for bigger batches; it's evidence the net is small. Spend headroom, if anywhere, on `xla_mem_fraction`/wave sizing or R3. |
| **R6** | **Hold n=256/considered=64 through exactly one R1+R2+R3 run, then judge it** against an n=128/32 arm by rolling-Elo slope + ladder. Feasibility envelope stands: top_k=64, n ≤ 256 (n=512/top_k=128 stalls the mctx backend). | config | v1 said nothing about whether searching harder helps — the gate froze the loop before the operator could compound once. Targets at n=256/64 are healthy (B6); the cost is 3.9× v3's self-play. If the slope matches n=128, drop back and reinvest the ~63 min/gen into 2× games or R7. |
| **R7** | **Curriculum levers, unchanged from A6 steps 3–4**: fraction of self-play vs a pool of past checkpoints (break the single-lineage loop), then Pentobi seeding if L5–L6 still don't move. | days | Still the right medium-term ladder — but they were queued *behind* a gate that would have frozen them too. They now sit behind R1–R3. |
| **R8** | **Measurement hygiene**: (a) colour-conditional value calibration in the eval-set diagnostics (73% of self-play outcomes are White wins — the value head's targets are strongly colour-skewed and we currently can't see whether it's exploiting that); (b) persist the *staged* config into the run directory at launch (two of three runs have committed-vs-ran drift, §1); (c) treat exact-0.500 arena scores and sub-binomial score variance as automatic red flags in the report. | ~½ day | Cheap, prevents the next three-run detour. |
| **R9** | (Registered idea, not this cycle) **Playout-cap randomisation** (KataGo): full n on a fraction of moves (those emit policy targets), small n elsewhere (value data only). At 90%-self-play generations this is the biggest cost lever available if n=256 survives R6. | ~days | IDEAS-queue material. |

**Recommended immediate sequence:** R1 + R2 + R3 (+R8a/b) → rerun the search-harder continuation
from v3 gen-40 (~40 gens ≈ $70 at n=256, or ~$35 at n=128 if R6's A/B is folded in) → read
rolling-Elo slope and a 100-game ladder through the fixed instrument. That run, unlike v1 and
(probably) v2, cannot mechanically freeze.

---

## 6. Data provenance / methods

- **v1**: fetched read-only from the pod 2026-07-15 (`/workspace/runs/blokus/blokus_search_harder/`
  → scratchpad): `RollingElo`, `ArenaData`, `ArenaReplays` (58,763 move rows → 1,700 games),
  `TrainingData` (2,161 window rows), `LearningRate`, `TrainingEntropy`, `PolicyAccuracy`,
  `PolicyValueConsistency`, `SelfPlayProfiling` (170k episodes), `Timings`, `Logs/alpha.log`, and
  `SelfPlayHistory/self_play_16.parquet` (gen-17 games, 632,776 positions) for target-sharpness and
  colour-of-winner analysis.
- **v3**: local `temp/runs/blokus/blokus_cloud_v3/` (same tables; `ArenaReplays` 126,534 rows →
  4,000 games; `v3_run.log` for within-epoch tqdm averages).
- **cloud_60**: not re-parsed (only `Reporting/` retained locally); claims cite
  [`blokus-cloud-60-analysis.md`](blokus-cloud-60-analysis.md).
- White-win reduction: a game is a White win iff (`outcome > 0` ∧ `player1_was_white`) ∨
  (`outcome < 0` ∧ ¬`player1_was_white`); `outcome` is from A = prev's perspective
  (`pool.py:444-449`).
- Variance test: χ² = (n−1)s²/σ₀² with σ₀² = p̄(1−p̄)/100 against the 17 per-gen scores.
- Checkpoint identity: md5 over the `.pth.tar` files (pod `md5sum`, local `md5`).
