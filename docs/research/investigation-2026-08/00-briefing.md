# AlphaBlokus — state of the project, for an independent investigation (2026-08-03)

You are being asked to work out why this project has plateaued and what to do next. This
document is the factual briefing. **Sections 1–6 are measurements and code facts.
Section 7 is the current team interpretation and you are explicitly invited to attack it.**

Everything here is either measured on-machine or read out of the code. Where something is
an inference it says so.

---

## 1. The project in one page

**Goal:** an AlphaZero-style engine that beats **Pentobi** (the strongest open-source Blokus
engine) at its maximum difficulty, **level 9**, at Blokus Duo (2-player, 14×14 board,
21 pieces each, ~17,837 discrete actions = (square, orientation) pairs).

**Architecture:** ResNet policy/value net; MCTS; self-play RL. Production self-play runs a
**JAX/mctx Gumbel-MuZero** search (not PUCT), trained on the *completed-Q improved policy*
rather than visit counts. Evaluation uses a separate Python full-action-space MCTS.

**Current best net:** `v3_gen40` — 40 generations of self-play, `large` preset
(192 filters × 12 residual blocks, 8.10M params), 44 input planes.

**Its measured strength:** weighted **0.539** on Pentobi levels 1–5, 50 games/level,
400 sims: 78% / 72% / 58% / 48% / 44%. It beats level 3; level 4 is 48%, just under.
On a wider L1–L9 ladder at 100 games/level the same net scores **0.344** with
77/71/61/53/47/40/16/20/21.

⚠ **Those two numbers are not comparable** (different level ranges and game counts). Mixing
them has already caused confusion. `weighted = Σ(level × wins) / Σ(level × games)`.

**Hardware and budget:**
- Free: personal box, **RTX 3060 Ti (8 GB)**, 20 CPU threads, always available.
- Paid: rented **RTX 5090** on RunPod, ~$0.99/hr. Historic runs cost $28–130 each.
- The owner's stated hope: it should not require a large budget — worst case a ~2-week run
  on the personal box ought to reach level 7–9. **Whether that is realistic is one of the
  questions you are being asked.**

**Total spent on failed runs so far: ~$125 for ≤0 external progress.**

---

## 2. Run ledger — what was actually run, and what happened

⚠ **Do not trust the committed JSON configs as a record of what ran.** Two of three recent
runs had committed-vs-ran drift, and configs were retro-edited afterwards. This table is
reconstructed from run logs and metric parquets.

| | cloud_60 | cloud_v3 | search_harder v1 | search_harder v2 | paired_gate_rerun |
|---|---|---|---|---|---|
| start | scratch | warm from cloud_60 gen-57 | warm v3 gen-40 | warm v3 gen-40 | warm v3 gen-40 |
| generations | 58 | 40 | 17 (killed) | 9 (killed) | 20 |
| learning rate | cosine → 2.7e-4 | constant 1e-3 | constant 1e-3 | constant 1e-3 | constant 1e-3 |
| weight decay | **none** | **none** | **none** | **none** | **none** |
| epochs/gen | 1 | 1 | 1 | 1 | **2** |
| sims / gumbel considered | 64 / 16 | 128 / 32 | 256 / 64 | 256 / 64 | 256 / 64 |
| jax `top_k` | 64 | 64 | 64 | 64 | 64 |
| games/generation | — | 10,000 | 10,000 | 10,000 | 10,000 |
| replay buffer | 40k games | 60k games | 60k (staleness ~6 gens) | 60k | 60k |
| net | medium→large | large | large | large | large |
| gate | 0.55 / 40 games | 0.55 / 100, deterministic | 0.55 / 100, diversified | 0.52 / 400 | regression_guard 0.45, paired, 400 |
| accepted | 39/58 | 21/40 | **0/17** | 1/9 | **20/20** |
| outcome | ladder L3 → L4 | pool Elo +240, ladder 0.344 | frozen, no change | +19 Elo once then flat | **ladder 0.344 → 0.298, pool Elo −44** |
| cost | — | — | ~$28 | ~$60 | ~$35 |

**Auxiliary heads (score / ownership / opponent-reply) were OFF in every run above.** They
are merged into main but default to `False` and have never run in self-play.

---

## 3. Measured facts about why it is stuck

**3.1 The improvement operator has a fixed point, and the current net sits on it.**
A candidate each generation is `incumbent + one epoch on the buffer`. Measured delta at
gen-40: **~+8 Elo**, i.e. the candidate wins 40/850 games as Black vs the incumbent's
22/850. v3 climbed only because *its* donor (cloud_60 gen-57) started far below the
operator's fixed point — its gen-1 candidate scored 0.85 and 9 of its first 12 generations
cleared the gate outright. v3 gen-40 is at the fixed point of a nearly identical operator.

**3.2 The candidate-vs-incumbent arena is almost information-free in this game.**
**96.3% of decisive arena games are won by White (the first mover)** under deterministic
play; 93–97% per generation, never below 93%. Consequences, all measured:
- Per-generation scores span 0.485–0.530, std **0.0113**, where 100 independent Bernoulli
  games at p≈0.51 would give std ≈0.050. Variance-ratio test: χ²=0.82 on 16 dof,
  **p ≈ 1.3e-8**. Outcomes are not independent draws in net strength.
- A paired colour-swapped arena moves only **~0.025 of score per ~100 Elo of true gap**
  (null: 0.49; a known large gap: 0.525).
- Therefore: a 0.55 threshold gate freezes the loop (v1: 0/17), and any regression floor the
  instrument can distinguish from 0.5 is effectively "always accept" (the rerun's 0.45
  floor would have needed a ~−200 Elo single-generation catastrophe to fire).
- **The Pentobi ladder is the only instrument that has ever resolved a difference the arena
  called a tie** (gen-57 vs gen-40: arena 51–49, ladder 0.205 vs 0.344).

**3.3 The one run that free-ran (gate off) actively degraded, and it was diagnosable.**
20/20 accepted, ladder 0.344 → 0.298, pool BayesElo −44. Internal signals looked *healthy*
throughout (loss falling, acceptance 100%, eval top-1 ~0.99) — all measured against
self-generated data. The honest signals, computed against the game's ground-truth
invariances, were flashing from ~gen 5: policy symmetry KL 0.639 → **1.236**, value symmetry
MAE 0.101 → **0.249**, policy-value Spearman 0.400 → 0.278, training value loss 0.296 →
**0.241** (v3's healthy range was 0.36–0.42, so this is overfitting, not improvement).
At gen 17 self-play target entropy collapsed 0.79 → **0.506** in one generation.
Three ingredients present here and absent in v3: `epochs: 2` (reuse ~12 passes/position),
constant 1e-3 while at the fixed point, and **Adam with no weight decay**.

**3.4 More simulations at the current search width does nothing.**
n=128→256 with considered 32→64 produced targets of near-identical entropy (0.905 vs 0.963
nats) and the same fixed point, at **3× the self-play cost**.

**3.5 Search is structurally truncated, and this has never been tested.**
The JAX backend truncates to the prior's **top-`k` of ~17,837 actions at the root AND at
every child node** (`top_k=64` in every run to date). A policy blind spot below rank 64 can
never be searched, never enter a training target, and never be corrected — a closed loop.
The evaluation path (Python MCTS) uses the full action space, so this is exclusively a
*training-signal* limitation.
- Measured 2026-08-03: `top_k` 64→128 at n=128 costs **1.24× per self-play game** (1.081 →
  0.869 games/s on the 3060 Ti), no stall. Affordable.
- ⚠ Gumbel considers `max_num_considered_actions` (32 or 64) root actions, sampled by
  Gumbel-top-k over the whole window. With `considered=64` and `top_k=64` the root
  considered *everything*. So raising `top_k` to 128 while leaving `considered` at 32 could
  make the root **narrower in practice** while widening only child nodes. Target top-1 mass
  is ~0.68 and entropy ~0.9 nats, so rank-65+ actions rarely win a Gumbel draw.
- A prior attempt at n=512 **with** `top_k=128` stalled: 28 minutes, GPU pegged, zero games.

**3.6 Net capacity: four tests, all ties — but the last one shows the tests can't resolve it.**
`large` (8.10M) vs `xl` (256×16, 19.07M):
- July v1 sizing sweep: a 14× parameter range moved weighted 0.088 → 0.102.
- P8 probe, supervised on a frozen buffer of the net's own self-play games: tie.
- Distillation, 128×8 → 192×12 on the Pentobi corpus: **+0.006 weighted**.
- 2026-08-03, `large` vs `xl` on the Pentobi corpus, held-out policy CE:
  `large` **2.9195**, `xl` **2.8802**, and a **replicate of `large` with only the weight
  init rerolled: 2.8698**. The replicate beat the 19M net on CE, top-1 and value MSE.
  **Noise floor 0.0497 > treatment effect 0.0393.**
- ⚠ All four are *supervised* tests, where the ceiling is the teacher. In AlphaZero-family
  RL capacity sets the asymptotic ceiling, not the current fit. **No `xl` net has ever been
  run through self-play here.**

**3.7 Measured noise floors — these invalidate several older readings.**
- Held-out policy CE: **~0.050 nats** (same config, weights rerolled).
- Weighted ladder L1–5 @ 50 games/level: **~0.022** (same config re-run: 0.361 vs 0.339).
- Ladder binomial noise at 50 games/level: **~±7 pp per level.**

**3.8 The value head may be broken, and in the RL regime it is unmeasured.**
Supervised on the Pentobi corpus, **value skill was negative in every arm**
(`1 − mse/colour_only_mse` between −0.16 and −0.28): predicting the outcome from nothing but
whose turn it is beat what the net learned, by 15–28%, and it did not improve with data.
Contributing measured facts: per-game outcomes are **71% / 7% / 22%** (White/Black/draw) in
the corpus, only ~700 Black wins in the whole corpus, and **every position in a game carries
the same label**. In self-play, 73% of outcomes are White wins.
- **Colour-conditional value calibration has never been implemented** (proposed twice, still
  undone), so nobody can see whether the value head is just exploiting the colour prior.
- Mechanistically relevant: Gumbel's training target is
  `softmax(logits + σ(completed Q))`, and completed-Q **substitutes the value net for every
  unvisited action**. A value head no better than a colour prior therefore corrupts the
  target the policy trains on.

**3.9 Pentobi distillation: built, measured, failed its gate.**
A 10,000-game corpus of Pentobi level-9 games with full move distributions and harvested
strong openings. Best net trained from scratch on it: **0.419** weighted (vs v3's 0.539).
Fine-tuning v3 on it: **0.537** — no change, despite the fine-tuned net being far better at
*imitating* Pentobi (top-1 0.355 vs 0.306, policy CE 2.379 vs 2.960, the largest imitation
gain measured). It overfits within two passes. Data-volume scaling decelerates: 2.5k → 5k →
10k games gave 0.252 → 0.321 → 0.361. The v2 corpus does beat the older v1 corpus
(+0.060), and mixing 20% v1 into v2 is worth +0.058 and saturates at 20%.

**3.10 Auxiliary heads: built, merged, dormant, inconclusive.**
`score_head` (predict final margin), `ownership_head` (per-cell 3-way final-board owner),
`reply_head` (predict the opponent's next move) — all `False` by default. Only ever
evaluated *supervised on the corpus*, at 96×6 with 3,565 games:
- Ownership works as a head (74.9% per-cell accuracy) but no policy gain above noise.
- Reply gave the best CE delta (−0.021) but **damaged the value head** (skill −0.083).
- The floor used to judge them was a **single** replicate estimated at ±0.015. §3.7 measures
  ~0.05 in a comparable pipeline, so **the reply head's apparent win should be treated as
  retracted**.
- KataGo ablates ownership+score **jointly at 1.65×** (its largest single factor) and reply
  at 1.30× — but those are **self-play RL** results, and every measurement here is imitation.
- ⚠ The aux *targets* are plumbed through the corpus loader. Threading them through the
  self-play replay buffer looks like real work, not a flag (unverified).

---

## 4. Conclusions that have been RETRACTED or corrected — do not inherit these

1. ~~"Effects below ~50 Elo are now resolvable."~~ **Retracted.** The instrument resolves
   roughly at the "can it win as Black" threshold, ~100 Elo. Everything below is squashed
   into 0.48–0.53.
2. ~~"Capacity is dead. Stop testing net size."~~ **Scoped.** True for imitation, where the
   ceiling is the teacher; untested for RL. Also, +0.006 is inside the noise floor, so
   "unresolvable" not "zero".
3. ~~"The remaining route to level 9 is RL from v3."~~ **Wrong as written.** RL from v3 with
   the *current operator* has been tried three times for ≤0 gain. The corrected claim: RL
   with a *changed* operator — the fixed point is a property of the operator, not of the
   starting net.
4. ~~"The reply head clears the noise floor."~~ **Retracted** — see §3.10.
5. ~~"The v2 corpus redesign bought nothing."~~ **Wrong** — it beats v1 by +0.060 when both
   are measured through a working recipe.
6. ~~"Playout cap randomisation is the key efficiency lever."~~ Still plausible, but it cuts
   cost per game; it cannot raise the ceiling. It is a multiplier on whatever works.

---

## 5. Code and architecture facts that constrain the options

- **Gumbel, not PUCT**, on the production path. No Dirichlet noise at the root (verified:
  `root_log_pi = log_pi if config.policy == "gumbel"` bypasses it entirely, so
  `dirichlet_epsilon` in Gumbel configs is inert). KataGo's forced-playouts + policy-target
  pruning therefore does **not** apply — Gumbel solves the same contamination problem.
- **The JAX search runs all games in jit-compiled lockstep with a static simulation count.**
  Playout cap randomisation is structural engineering (two compiled search functions plus a
  per-step Bernoulli mask), not a config flag.
- **No global pooling / squeeze-excitation anywhere in the trunk.** Every layer is a 3×3
  conv. KataGo ablates global pooling at **1.60×**; Lc0 ships SE in every ResNet. The only
  pooling is an AdaptiveAvgPool in the pass-logit head.
- **No game-specific input features.** 44 planes of raw placement/inventory state. There are
  no placeable-corner planes (the Blokus analogue of liberties) and inventory is only
  implicitly readable as all-zero planes. KataGo ablates game-specific features at 1.55×
  (bundled with two search optimisations).
- **The net exists in three places** — the torch module, a JAX bridge that re-implements the
  forward pass numerically for self-play, and an ONNX web export. Any trunk change lands in
  all three plus parity tests, and **breaks warm-starting from existing checkpoints.**
- Net presets: small 64×4 (0.34M), medium 128×8 (2.45M), large 192×12 (8.10M),
  xl 256×16 (19.07M).
- Symmetry augmentation is order-2 (Blokus Duo's actual symmetry group), correctly.
- Machinery that exists but has never run in an RL run: keep-best-by-ladder selection, a
  drift circuit-breaker (two consecutive ≥5pp ladder drops → stop), AdamW with weight decay
  1e-4 (now default-on), and a paired colour-swapped arena.

---

## 6. Things on the table that nobody has tested

Neutral list, deliberately unranked:

- Search **width** (`top_k` > 64) — cost measured, benefit untested.
- Raising Gumbel `considered` in step with width.
- **Global pooling** in the trunk (1.60× in KataGo).
- **Blokus input features** — placeable corners, explicit inventory, phase scalar.
- **Auxiliary heads inside RL** (1.65× / 1.30× in KataGo, never tried in self-play here).
- **Value-target repair** — outcome-balanced sampling (ELF's fix for exactly this skew),
  a WDL win/draw/loss head (22% of corpus games are draws), teacher-eval λ blending.
- **Colour-conditional value diagnostics** — to find out whether the value head is a colour
  prior in disguise.
- **Self-play against a pool of past checkpoints** instead of a single lineage.
- **Opening seeding of self-play from the corpus's labelled opening DAG.**
- **Playout cap randomisation** (1.37× in KataGo) — cost lever.
- **MuZero-style Reanalyze** — refresh targets on old games instead of generating new ones.
- **`xl` through actual RL** (never done).
- **Games per generation / generation count trade-off** — 10,000 games/gen has never been
  varied. Nor has "many more, much shorter generations", nor a very large accumulating
  buffer with progressive net growth (KataGo's actual practice).
- **A long single run on the free box** rather than short rented runs.

---

## 7. The current team interpretation — ATTACK THIS

Stated so you can disagree with it. It is not established fact.

1. The plateau is a **fixed point of the improvement operator**, so the lever must change
   the operator (search, targets, capacity, curriculum), not the starting net.
2. The **measurement instruments** have been the dominant failure mode so far — three runs
   were lost to a gate that cannot see improvement, and one to a floor that cannot see
   degradation.
3. **Search width** is the leading untested structural candidate, because it is the only
   mechanism that explains a fixed point immune to more sims, more capacity *and* more data.
4. Distillation is a **dead end for strength** but the corpus retains value as an
   architecture-independent asset and as an opening book.
5. The **value head is the most under-investigated component** given its measured negative
   skill and its role in Gumbel's target.
6. Capacity is a **bad buy today** but is not a closed question for RL.

**Known weaknesses in this interpretation:** (a) the width mechanism may be thin in practice
because the prior is concentrated; (b) three of the four changes in the currently-proposed
next run are guardrails that prevent past failures rather than levers that raise the ceiling;
(c) nobody has seriously hunted for a *bug* since the two post-mortems, both of which were
scoped to specific runs; (d) the possibility that this hardware/budget simply cannot reach
level 9 has never been costed out.

---

## 8. What we want from you

Produce a report with these sections. Be concrete, cite evidence, and mark inference as
inference.

1. **Ranked hypotheses for what is actually wrong**, with the evidence for and against each,
   and what would distinguish them.
2. **What conclusions in §1–7 you think are wrong**, including anything in §7.
3. **A specific hunt for bugs or silent defects** — is anything in the pipeline degrading
   the net or corrupting targets? Name where you would look and why.
4. **Ranked recommendations** with estimated cost (GPU-hours and £) and expected value.
5. **The single cheapest decisive experiment** you would run next, and the pre-registered
   result that would falsify your own recommendation.
6. **A direct answer:** is Pentobi level 9 reachable on a free 3060 Ti plus a modest rented
   budget? If yes, sketch the path and the total compute. If no, say what it would take —
   and say what level *is* reachable.
7. **Run-parameter analysis specifically:** learning rate and schedule, games per
   generation, generation count, buffer size and staleness, epochs per generation, batch
   size. These have been chosen largely by inheritance and one post-mortem, not by
   experiment. Say which are wrong.
8. **What you could not determine**, and what you would need.

---

## 9. Raw data pointers (if you have repo access)

- `docs/research/` — plateau-investigation.md, regression-and-next-steps.md,
  distillation-recipe-findings.md, width-and-capacity-probes.md,
  alphazero-technique-review.md, corpus-quality-principles.md,
  corpus-generation-literature.md, xl-training-scaleup.md, blokus-cloud-60-analysis.md,
  deepmind-run-configs.md, jax-pipeline-ab.md.
- `docs/plans/` — supervised-network-improvements.md, pentobi-corpus-v2.md,
  score-auxiliary-target.md; `docs/plans/archive/` for completed work including
  post-regression-recovery.md.
- `temp/runs/blokus/<run_name>/` — per-run metric parquets: `RollingElo`, `ArenaData`,
  `ArenaReplays`, `TrainingData`, `SymmetryDiagnostic`, `PolicyValueConsistency`,
  `PolicyAccuracy`, `TrainingEntropy`, `ValueCalibration`, `SelfPlayProfiling`, `Timings`,
  `Tournament`, `SelfPlayHistory`, plus `Reporting/report.html`.
- `temp/benchmarks/` — ladder results, capacity probe JSONs, width calibration JSONs.
- Code: `src/alphablokus/` — `games/blokusduo/jax/search.py` (Gumbel + top_k truncation),
  `training/coach.py` (the generation loop), `evaluation/` (arena, acceptance, Elo),
  `config.py` (presets and all config dataclasses).
- Git history is complete and commit messages are descriptive; PRs #60–#68 cover the corpus,
  score head, aux targets, technique review and reporting work.
