# AlphaBlokus — Independent Investigation (Fable, 2026-08-04)

Status: COMPLETE. Sections 1–8 follow the briefing's §8 structure; sections 0/0b are the
audit of the briefing itself, reported first because several findings feed everything
below. Convention: **[M]** = measurement (I ran it or read it out of raw data), **[C]** =
code fact (read from source at the cited path), **[I]** = inference.

**Headline findings:** (1) the briefing's numbers reproduce almost perfectly, but its
central §3.2 measurement rests on deleted raw data; (2) the never-implemented
colour-conditional value diagnostic, implemented and run today, shows the production value
head is at best marginally better than a colour prior in the RL regime and **degraded
across the v3 run itself** — I rank this above search width as the leading structural
suspect; (3) the "plateau" has never been tested against a healthy operator — every
stalled run had a since-diagnosed defect (blind gate, or constant 1e-3 + no weight decay +
data reuse); (4) v3's own pool Elo peaked at gen 32, and nobody ever laddered gen-32 —
every experiment may be warm-starting from the wrong checkpoint; (5) level 9 on this
budget is a ~25–40% proposition, not a plan — level 6 is plannable, and the two free
measurements in R1 (re-ladder gen-32/36; eval-sims scaling at L7–9) materially change the
odds either way.

## 0. Briefing audit — errors, omissions, motivated framing found in the briefing itself

I recomputed the briefing's headline numbers from the raw parquets/JSONs wherever the raw
data still exists. Verdict up front: **the briefing's numbers are substantially accurate —
I reproduced nearly every figure I could test to the digit — but it has three defects: one
load-bearing claim whose raw data has been deleted, one run miscounted, and a handful of
framing choices that flatter the "instrumentation was the villain" story.**

### 0.1 Confirmed to the digit [M]

| Briefing claim | My recomputation | Source |
|---|---|---|
| v3_gen40 L1–9 ladder 0.344; 77/71/61/53/47/40/16/20/21 | 0.3444; identical per-level | `temp/runs/blokus/blokus_cloud_v3/PentobiLadder/ladder_accepted_40...json` |
| rerun ladder 0.298 | 0.2984 (79/62/58/49/45/35/19/14/10) | `blokus_paired_gate_rerun/PentobiLadder/...json` |
| search_harder_v1 0/17 accepted; scores 0.485–0.530; std 0.0113; χ²=0.82, p≈1.3e-8 | 0/17; 0.485–0.530; std 0.01129; χ²=0.816/16 dof; p=1.32e-8 | `blokus_search_harder_v1/ArenaData` |
| rerun 20/20 accepted, 400 games/gen | 20/20, 400 | `blokus_paired_gate_rerun/ArenaData` |
| cloud_v3 21/40 accepted | 21/40 | `blokus_cloud_v3/ArenaData` |
| 16 of v3's 19 rejections scored exactly 0.500 | 16/19 | same |
| rerun policy symmetry KL 0.639 → 1.236 | gen1 0.639 → gen20 1.236 | `SymmetryDiagnostic` |
| rerun value symmetry MAE 0.101 → 0.249 | gen1-e0 0.101 → gen20 0.237 (mean) / 0.249 (e0) | `PolicyValueConsistency` |
| rerun Spearman 0.400 → 0.278 | gen1-e0 0.3998 → gen20-e1 0.2777 | same |
| rerun training value loss 0.296 → 0.241 | 0.296 (gen1 final) → 0.241 (gen20 final) | `rerun.log` tqdm epoch-final values |
| gen-17 self-play target entropy collapse to 0.506 | gen17 0.506 (gen16 0.836) | `SelfPlayProfiling` |
| rerun pool Elo −44 | gen20 = −44.1 | `Tournament/tournament_raw.json` |
| v1 target entropy 0.905 vs v3-era 0.963 (§3.4) | v1 gens ≈0.90, dead flat | `SelfPlayProfiling` |
| cloud_v3 arena: White won 77% of decisive games | 77.1% (3747 decisive of 4000) | `blokus_cloud_v3/ArenaReplays` |
| sh_v2 "best" ladder | 0.3349 — note this is *below* v3_gen40's 0.3444 | `temp/benchmarks/sh_v2_ladder.json` |

### 0.2 Problems found

1. **The raw data behind the headline 96.3% white-win figure has been deleted.** [M]
   The figure is computed in `docs/research/plateau-investigation.md` from
   `blokus_search_harder/ArenaReplays` (1,700 games, 17 gens). That directory now contains
   only `Nets/` — no ArenaReplays, and `blokus_search_harder_v1/` has ArenaData but no
   ArenaReplays either. The number is *plausible* — I independently reproduced the
   methodology on cloud_v3's surviving ArenaReplays and got exactly the 77% the same doc
   quotes for v3 — but the single most load-bearing measurement in the briefing (§3.2) is
   currently **unreproducible from raw data**. [I] I believe it, because the v3 cross-check
   validates the pipeline, but it should be re-derived if the replays exist in a backup.

2. **search_harder_v2 is miscounted.** [M] Briefing says "9 generations (killed), 1/9
   accepted". `blokus_search_harder_v2/ArenaData` contains 8 generations, 1/8 accepted
   (gen 1 at 210/188/2 = 0.5275 under the 0.52 gate; gens 2–8 all 0.48–0.51). Trivial,
   but the ledger warned "do not trust committed configs" and then itself miscounted.

3. **Briefing narrows "93–100%" to "93–97%".** [M-doc] plateau-investigation.md says the
   per-gen white-win range was 93–100%; the briefing says "93–97%, never below 93". Minor,
   direction-neutral.

4. **Omitted: the gen-17 entropy collapse recovered.** [M] Gen 18 is back to 0.789, gen 20
   is 0.820. The briefing presents the collapse as the terminal event of a degradation
   ("At gen 17 ... collapsed 0.79 → 0.506 in one generation") without noting it was a
   one-generation transient. The KL/MAE degradation is real and monotone-ish; the entropy
   collapse as narrated overstates it.

5. **Omitted: the pool-Elo trajectory is itself nearly noise.** [M] The rerun's pool Elo
   by gen: −15, −10, +2, −6, −14, −18, −8, −5, −38, +6, −6, −41, −46, −8, −6, −20, −25,
   −18, **−4 (gen 19)**, −44 (gen 20). Quoting the endpoint "−44" implies a trend; gen 19
   was −4. Given the same colour-pinning affects the pool tournament, the ladder drop
   (0.344 → 0.298, which I confirm) is the real evidence of degradation; the pool Elo
   figure is decoration. This matters because the briefing elsewhere (correctly!) teaches
   that the arena instrument squashes everything into 0.48–0.53 — and then quotes a pool
   Elo endpoint from the same family of instrument.

6. **"pool Elo +240" for cloud_v3 verifies (gen40 = +240.3) but omits that v3's own pool
   Elo peaked at gen 32 (+286) and declined over the last 8 generations** (gen32 +286 →
   gen36 +271 → gen40 +240). [M] `blokus_cloud_v3/Tournament/tournament_ratings.parquet`.
   Given pool-Elo noise this is not proof of decline, but it means (a) "v3 climbed then
   plateaued" is the generous reading — "v3 peaked around gen 30 and drifted down" fits the
   same data; (b) **gen-40 was never shown to be the best net of its own run** — gen-32
   was never laddered, and the keep-best-by-ladder machinery existed but never ran. The
   project may be warm-starting every experiment from the wrong checkpoint. Also, the
   ledger's "+19 Elo once then flat" for v2 is the arena-score conversion of a single
   0.5275 gate pass — i.e. a reading from the very instrument §3.2 discredits.

7. **Framing: §3.9's "fine-tuning moved strength by 0.002"** (0.539 → 0.537) is stated
   with three digits of precision. The briefing's own §3.7 measures the L1–5@50 ladder
   noise floor at ~0.022. "Moved by 0.002" should read "unchanged within a noise floor an
   order of magnitude larger than the quoted delta". The direction of the conclusion
   (no strength change) is right; the precision is fake. Same for "+0.006" in §3.6.

### 0.3 Where the two-sided evidence was presented fairly

Sections 3.5's warning that raising `top_k` without raising `considered` could make the
root *narrower*, §3.6's admission that all capacity tests are supervised, §3.7's noise
floors and §4's retraction list are accurate and, if anything, the retraction discipline
is above average. The briefing is not a whitewash; it is ~95% checkable and checks out.

## 0b. Verification of the briefing's mechanistic code claims

All paths under `/Users/henrycassidy/code/personal projects/AlphaBlokus/`.

1. **"Search is truncated to the prior's top-k at the root AND every child node" — TRUE.** [C]
   `src/alphablokus/games/blokusduo/jax/search.py`: `topk_legal()` (line 151) is applied to
   the root logits (line 198) and inside `recurrent_fn` to every expanded child (line 177).
   Compact slots carry global ids; anything below rank `top_k` of the prior does not exist
   in the tree. The Python eval MCTS (`search/mcts.py`) is full-action-space, so this is
   training-only, as claimed.

2. **"Completed-Q substitutes the value net for unvisited actions" — TRUE with a nuance.** [C]
   `.venv/.../mctx/_src/policies.py` (`gumbel_muzero_policy`) defaults to
   `qtransform_completed_by_mix_value`; unvisited actions receive the **mixed value** — an
   interpolation of the raw value-net estimate at the node and the visit-weighted mean of
   visited children's Qs (`qtransforms.py::_compute_mixed_value`) — not the raw value alone.
   The training target really is `softmax(prior_logits + scale·completed_Q)` over the
   64-slot compact window (`policies.py:217-227`), with `scale = (50 + max_visits) × 0.1`
   after min-max rescaling of Q to [0,1]. The briefing's mechanistic conclusion survives:
   a value head that only encodes "whose turn is it" feeds a systematic bias into every
   unvisited action's slot of the policy target, and with `considered=32` of `top_k=64`,
   roughly half the root window per move is priced by the value head. If anything the
   briefing *understates* the coupling: visited actions' Qs are also value-net backups in
   a 128–256-sim search that expands each considered action only a handful of times.

3. **"Aux targets cannot reach the self-play trainer without real plumbing" — TRUE.** [C]
   `training/coach.py:307` calls `self.nnet.train(train_examples, generation, metrics=...,
   eval_set=...)` — no aux arguments. `games/base_wrapper.py::train` (line 638) takes
   `score_margins`/`ownership_targets`/`reply_targets` as separate, index-aligned optional
   arguments, and `ProcessedExample` is a hard 3-tuple (board, sparse-policy, value)
   through `selfplay/episode.py`, `jax/harvest.py`, and the replay buffer. Turning a head
   on in an RL config today would train it on nothing (there is at least a loud warning:
   `_resolve_aux_targets`). Threading targets through `jax/harvest.py` (final board +
   margin + next-ply policy at game end) plus the buffer is genuine work, as claimed.

4. **"Playout cap randomisation is structural work under jit" — TRUE.** [C]
   `SearchConfig.num_simulations` is a compile-time constant of the jitted search closure
   (`make_search` → `jax.jit(search)`), and `actors.py::run_wave` is one jitted `lax.scan`
   over fixed `wave_plies` calling that single search. Per-move random sim budgets need ≥2
   compiled search variants + masking, exactly as described.

5. **"No Dirichlet noise at the root under Gumbel; `dirichlet_epsilon` inert" — TRUE.** [C]
   `search.py:194`: `root_log_pi = log_pi if config.policy == "gumbel" else
   masked_root_logits(...)` — the Dirichlet mix lives only in `masked_root_logits`. The
   rerun's resolved config carries `dirichlet_epsilon: 0.25`, confirming an inert knob
   sitting in a production config (cosmetic hazard, no effect).

6. **"No global pooling / SE anywhere in the trunk" — TRUE.** [C]
   `games/blokusduo/nn/net.py`: `ResNetBlock` is conv3×3–BN–ReLU–conv3×3–BN + skip; the
   only pooling is `AdaptiveAvgPool2d` inside the pass-logit head. No game-specific input
   features beyond the 44 raw planes (21/21 piece planes + 2 aggregate).

7. **Gumbel self-play plays the Sequential-Halving winner at every ply; `temp_threshold`
   is ignored under Gumbel.** [C] `backend.py:96` sets `use_search_action = (policy ==
   "gumbel")`; `actors.py:103-106` then plays `chosen_global` unconditionally. Move
   diversity comes only from root Gumbel noise. This is per the Gumbel MuZero paper, but
   worth knowing: the `temp_threshold: 12` in the configs is a second inert knob, and
   self-play opening diversity is entirely at the mercy of Gumbel noise scale.

8. **Torch→JAX parity, action encoding, symmetry, harvest equivalence are test-covered and
   the tests pass.** [M] I ran the full suite: **911 passed, 1 skipped** (the skip is a
   movegen gauntlet needing an unbuilt 50k-position cache fixture, not a parity test).
   Includes `tests/games/blokusduo/jax/test_parity.py`, `test_step_parity.py`,
   `test_gumbel.py`, `test_net.py`, harvest/backend tests, and
   `games/blokusduo/test_symmetry.py`. On CPU JAX. GPU/bfloat16 behaviour is untested here.

9. **Run-ledger config drift — the briefing's warning is real, and its ledger is the
   correct side.** [M] `blokus_paired_gate_rerun/config.resolved.json` (written at launch,
   2026-07-17) shows `learning_rate 1e-3, epochs 2, constant schedule`, **no
   weight_decay key** — the run predates the AdamW-default commit (`c155047`, 2026-07-22).
   The *committed* `run_configurations/blokus_paired_gate_rerun.json` was edited on
   2026-07-22 ("Set continuation config to epochs 1 and LR 2.5e-4") and now shows
   `weight_decay: 0.0001` — i.e. the committed file describes a hypothetical future run,
   not the one that happened. Anyone reading configs instead of resolved snapshots would
   conclude the rerun had weight decay. It did not.

## 1. Ranked hypotheses for what is actually wrong

**H1 (highest confidence): The "plateau" has never been demonstrated against a healthy
operator — every stalled run had an identifiable, since-diagnosed defect, and the loop as
run was stationary by construction.** [I, from M]
The three post-v3 runs were: (a) v1 — frozen by a gate that demands the candidate win
~12–14% of its Black games against an equal opponent (measured impossible); (b) v2 —
same, softer; (c) rerun — un-gated but with a training step (2 epochs, constant 1e-3,
Adam, **no weight decay** — verified from the resolved config, §0b.9) that the project's
own post-mortem identifies as overfitting toxic. When the gate rejects everything, the
incumbent is frozen, the buffer refills with the frozen incumbent's own games, and the
candidate construction is deterministic — the same candidate is produced every generation
(their B3 finding: loss trajectories identical to 3 decimals across 17 gens). That is not
"RL has hit a ceiling"; that is a loop configured so it cannot move. What HAS been shown:
`incumbent + 1 epoch @ 1e-3` gains only ~+8 Elo — the *training step* is at a fixed point,
under this LR/decay/epoch regime. Whether the *system* (with functioning selection, decayed
LR, weight decay, ladder-based keep-best) is at a ceiling is untested.
Distinguishing test: any free-running run with the guardrails on and ladder-based
selection. If it also goes flat, H1 falls and the ceiling is real.

**H2: The value head is the binding structural defect — it is largely a colour prior, in
the RL regime, and Gumbel's target construction gives it a direct channel into every
policy target.** [M + C]
This was the briefing's "most under-investigated component"; I measured it (§3, bug hunt,
item 2 — the colour-conditional diagnostic proposed twice and never implemented). On v3's
own frozen eval set, **v3_gen40's value skill is −0.10** (worse than predicting the
per-colour mean outcome), and its predictions correlate with mover colour at **0.80**
while the actual outcomes correlate at only 0.52. On matched-vintage data the skill is
+0.06 — barely above a colour prior. Most damning: **v3_gen1 scored +0.38 on the same
positions** — the value head *degraded* over the very run that is celebrated as the
successful one, while the policy improved. Mechanism (verified, §0b.2): with
`considered=32` of `top_k=64`, ~half the root window's completed-Q values per move come
from the value function, and visited actions' Qs are value-net backups at 128–256 sims.
A colour-prior value head biases every one of those numbers the same way. This also
explains the fixed point's immunity to sims (more sims = more queries of the same broken
value function), capacity (a bigger net fits the same corrupted targets), and data (more
games from the same skewed distribution: self-play is 73% White wins, so ~73% of
mover-perspective labels are "colour tells you the answer").
Distinguishing test: outcome-balanced value sampling or a WDL head in a short run;
success = value skill going clearly positive AND ladder moving where previous runs went
flat. Cheap pre-check: my diagnostic run on any new checkpoint (minutes, CPU).

**H3: Search-width truncation (top_k=64 at every node) — the briefing's leading
candidate — is real but second-order.** [C + I]
The mechanism is verified in code (§0b.1) and is a genuine closed loop. But the case for
it being *the* binding constraint is weak: target top-1 mass ~0.68 / entropy ~0.9 nats
means the prior is concentrated, so rank-65+ actions essentially never win a Gumbel draw
into the considered set even when present (the briefing itself flags this); and none of
the three failed runs failed *because* of width — they failed for H1 reasons. Width
uniquely explains nothing that H2 does not also explain. It is worth the 1.24× cost as a
hedge in the next run (with `considered` raised in step — the briefing's own caveat), but
I would not bet the run on it.

**H4: Self-play data skew — 73% White wins, argmax-only Gumbel play, no opening
diversity mechanism — starves the Black-side and endgame learning signal.** [M + C]
Related to H2 but distinct: even a repaired value head trains on whatever distribution
self-play produces. Under Gumbel the engine plays the Sequential-Halving winner at every
ply (§0b.7); diversity comes only from root Gumbel noise. I verified there are no
duplicate games (§3 item 3), so it is not degenerate — but outcome balance and opening
coverage have never been measured or managed (ELF's outcome-balanced sampling and the
corpus's opening DAG are both on the shelf).

**H5: Missing architecture/features (global pooling, corner planes, aux heads in RL) cap
sample efficiency by a multiplicative factor.** [C + literature]
All verified absent (§0b.6, §0b.3). These are KataGo's biggest levers (1.65×, 1.60×,
1.55×) but they are efficiency multipliers, not fixed-point escapes; and every trunk
change breaks warm-starting and lands in three implementations (torch, JAX bridge, ONNX).
Right answer on this budget: bundle them into the one net rebuild you can afford, don't
A/B them individually.

**H6: Instrumentation.** The team's #2 — three runs lost to blind instruments. True,
verified, and the single largest *money* sink so far. But it explains wasted spend, not
what is wrong with the net. With the paired arena + ladder-selection machinery now built,
it is largely a solved problem *if the next run actually uses ladder-based selection as
its acceptance signal* rather than any arena derivative.

## 2. Conclusions in briefing §1–7 that I think are wrong

1. **§7.3 — "Search width is the leading untested structural candidate, because it is the
   only mechanism that explains a fixed point immune to more sims, more capacity and more
   data." The uniqueness claim is false.** The value-head defect (H2) explains all three
   immunities at least as well (see §1), now has *direct* adverse measurement in the RL
   regime (which width lacks), and its own §3.5 caveats (prior concentration; considered-32
   bottleneck) undercut width's mechanism. §7.5 ("value head most under-investigated") and
   §7.3 should swap places.

2. **§3.1 / §7.1 — "v3 gen-40 is at the fixed point of the improvement operator" is
   over-generalised.** What is proven: it is at the fixed point of *this specific training
   step* (1–2 epochs, constant 1e-3, no weight decay) under *this acceptance regime*. The
   phrase invites "RL from v3 is exhausted"; the correct reading is that no healthy
   operator has ever been pointed at v3. The briefing's own §4.3 correction gets this
   right; §3.1's framing partially takes it back.

3. **§3.9 / §7.4 — "Distillation is a dead end for strength" overclaims on a flawed
   experiment.** Three specific flaws: (a) the fine-tune trained the **value head on the
   corpus's outcome labels** — labels the same briefing (§3.8) shows produce
   worse-than-colour-prior value heads — with no flag to mask the value loss
   (`scripts/distill_sl.py` trains policy KL + value MSE unconditionally; verified). A
   policy-better/value-worse cancellation is fully consistent with "imitation way up,
   strength flat", and no policy-only fine-tune arm was ever run. (b) "moved strength by
   0.002" is quoted against a noise floor of ±0.022 — the experiment could not have seen a
   ±40-Elo effect either way. (c) The verdict quietly closes uses of the corpus that were
   never tested: value-label repair (teacher-eval blending), opening seeding, and Reanalyze
   targets. Supportable verdict: "policy imitation through this recipe does not raise
   ladder strength; the corpus's other uses are untested."

4. **§3.2 — the 96.3% headline is currently unfalsifiable from raw data** (ArenaReplays
   for `blokus_search_harder` deleted, §0.2.1). The claim is probably true (v3's 77%
   cross-check reproduces), but a briefing that (rightly) demands nothing be trusted
   without raw data should not rest its central §3.2 measurement on a deleted store.

5. **The run ledger's implicit story that v3 "climbed 40 generations then stopped"** —
   v3's own pool Elo peaked at gen 32 (+286) and fell to +240 by gen 40, and my value-skill
   measurement shows the value head degrading *within* v3 (gen-1 +0.38 → gen-40 −0.10 on
   the same eval set, vintage-confounded but directionally hard to dismiss). The operator
   was plausibly already past its peak *during* v3. Consequence: gen-40 may not even be the
   best warm-start checkpoint on disk; gen-32/36 were never laddered. Cheap to check.

6. **§1 — "worst case a ~2-week run on the personal box ought to reach level 7–9"** (the
   owner's stated hope, flagged as a question): my costing (§6) says a 2-week box run is
   roughly ONE v3-scale run (40–90 generations depending on config). The hope is not
   arithmetic-crazy, but it assumes the operator problem is fully solved on the first try.

7. **Minor factual corrections:** search_harder_v2 is 8 gens / 1-of-8 accepted, not 9 /
   1-of-9 (§0.2.2); the rerun's gen-17 entropy collapse recovered at gen 18 (§0.2.4); the
   rerun's pool-Elo "−44" endpoint sits in a ±40-noise trajectory whose gen-19 value was −4
   (§0.2.5); "93–97%" should read "93–100%" (§0.2.3).

Where I checked and **agree** with the briefing: the arena variance-collapse analysis
(reproduced to the digit); the retractions in §4 (all justified, including retracting the
reply head's win); §3.6's capacity scepticism and the replicate methodology; §3.7's noise
floors; the §5 code facts (all nine verified, §0b); the decision not to buy more Pentobi
games for imitation.

## 3. Bug hunt — silent defects across all runs

Approach: prefer executed checks over read code. Everything below ran on this machine
today unless marked otherwise.

### Executed checks

1. **Full test suite: 911 passed, 1 skipped** (the skip is a movegen gauntlet fixture,
   not correctness). [M] Covers torch↔JAX net parity, env step parity, Gumbel search,
   harvest↔episode equivalence, symmetry, sparse-policy round-trip, checkpoint
   revert/reload semantics, arena orientation conventions, acceptance arithmetic.

2. **Colour-conditional value diagnostic — implemented and run (it had been proposed twice
   and never built). This is the biggest single finding of the investigation.** [M]
   Script: `scratchpad/colour_value_diag.py`; CPU, minutes; mover colour inferred from
   piece-count parity on the canonical compact boards (198–200/200 unambiguous).

   | net | eval set | value skill | corr(colour, pred) | corr(pred, target &#124; colour) |
   |---|---|---|---|---|
   | v3 gen-1 | v3 (own vintage) | **+0.38** | 0.76 | 0.63 |
   | v3 gen-40 | v3 (gen-1 vintage) | **−0.10** | 0.80 | 0.17 |
   | v3 gen-40 | rerun (own vintage) | **+0.06** | 0.81 | 0.35 |
   | rerun gen-20 | rerun (own vintage) | **−0.09** | 0.81 | 0.27 |

   Readings: (i) the production value head is, at best, marginally better than "whose turn
   is it" on its own self-play distribution — the corpus finding (§3.8) replicates in the
   RL regime; (ii) all four checkpoints' value outputs track colour (0.76–0.81) harder
   than the ground truth warrants (0.52–0.62); (iii) the value head **got worse across
   v3** while the policy got stronger, and worse again across the rerun (gen-40 +0.06 →
   gen-20 −0.09 on the identical eval set — vintage-matched, so this pair is clean).
   Caveat marked as inference: the gen-1 vs gen-40 comparison on v3's eval set is
   confounded by target vintage (outcomes recorded under gen-1 play); the rerun-eval-set
   column is not.

3. **Self-play clone check: clean.** [M] Per-generation exact-duplicate detection on
   (num_moves, mean_policy_entropy) float pairs across all 200k rerun episodes, 400k v3
   episodes, 170k v1 episodes: zero duplicates. Self-play is not producing repeated games;
   the data volume is real.

4. **Value-label pipeline read-through + tests: no defect found.** [C] JAX harvest
   (`jax/harvest.py`) and python episode (`selfplay/episode.py`) agree on: mover-perspective
   sign convention (positive = mover wins), the ±1e-4 draw sentinel to the terminal
   player-to-move, and canonical-frame boards (`ppb × mover`). Equivalence is test-pinned.
   v3 gen-1's **positive** value skill (+0.38) is itself evidence against a label-sign or
   frame bug: a corrupted pipeline could not have produced a value head that beats the
   colour baseline early in the run.

5. **Symmetry augmentation: no defect signature.** [C+M] The transpose permutation is
   probed from the game's own `transpose_policy` (involution, so direction-safe),
   augmentation is order-2 matching Blokus Duo's true symmetry group, and v3's policy
   symmetry KL holds at 0.58–0.85 across 40 generations (a broken augmentation would trend
   or sit high). The rerun's KL climb to 1.24 tracks its overfitting recipe, not the
   augmentation (v3, same augmentation, stayed flat).

6. **Checkpoint save/load and gate revert: verified clean in code and covered by tests**;
   revert restores weights + Adam moments and deliberately not the LR clock. One dead
   statement (`coach.py:299`: an `MCTS(...)` constructed and discarded) — harmless.
   The load-time `weight_decay` patch-up (`base_wrapper.py:1544-1550`) correctly prevents
   old checkpoints from silently zeroing the new AdamW decay.

### Defects and hazards found (none rises to "your runs were corrupted")

- **D1 — the value head trains toward a colour prior and nothing in the RL loop pushes
  back** (item 2). Not a coding bug — a target-construction/objective defect. It is the
  only defect I found that plausibly degrades the net across ALL runs.
- **D2 — inert config knobs that misdescribe production behaviour**: `dirichlet_epsilon:
  0.25` and `temp_threshold: 12` are both no-ops under Gumbel (§0b.5, §0b.7) yet sit in
  every production config, inviting the belief that root noise/temperature exploration
  exists. Self-play exploration is Gumbel-noise-only. Anyone tuning these tunes nothing.
- **D3 — committed configs are retro-edited** (verified instance: the rerun config,
  §0b.9). The resolved-config snapshot exists and is trustworthy; the repo configs are not
  a record of anything.
- **D4 — raw-data hygiene**: `blokus_search_harder/ArenaReplays` (basis of the 96.3%
  claim) deleted; `blokus_cloud_60` has no local parquets at all (report HTML only), so
  the ledger's cosine-schedule/39-of-58 row is locally unverifiable.
- **D5 — untested numerics surface**: production self-play runs the JAX net in
  **bfloat16** (resolved config), while every parity test runs CPU float32. Top-k
  selection over 17,837 near-tied logits and completed-Q arithmetic under bf16 has never
  been compared against the fp32 path. Cheap to A/B on the box (one config flag); I found
  no evidence of harm, only absence of evidence.
- **D6 — eval-set vintage**: the frozen 200-position eval set is built from *gen-1*
  self-play of each run and never refreshed, so every per-epoch diagnostic (policy top-1
  ~0.99, value calibration) measures fit to the run's weakest data. This is how the rerun
  looked "healthy" internally while degrading — the briefing knows this, but it remains an
  open trap for the next run.

## 4. Ranked recommendations with cost and expected value

Cost anchors, measured [M]: 5090 ≈ £0.79/hr; a v3-config generation (n=128, top_k 64,
10k games, 1 epoch, 100-game arena) = 2,189 s on the 5090 (`blokus_cloud_v3/Timings`) ≈
£0.48/gen; a rerun-config generation (n=256, considered 64, 2 epochs, 400-game paired
arena) = 6,820 s ≈ £1.50/gen. On the 3060 Ti, self-play at n=128/top_k 64 runs 1.08
games/s [M, width probe] → a v3-config generation ≈ 3.2 h; a widened n=256/top_k 128 one
≈ 7 h. Two weeks of box time ≈ 100 v3-config gens or ~48 widened gens, at £0.

**R1 — Re-baseline before anything else (box, £0, ~2–4 days).**
(a) Ladder v3 gen-32 and gen-36 at L1–9 × 100 — pool Elo says gen-32 (+286) outranked
gen-40 (+240) and nobody ever checked [M]; every experiment since may be warm-starting
from the wrong net. (b) Measure the eval-time sims-scaling curve of the best checkpoint:
L5–L9 at 400 / 1,600 / 6,400 sims. The 0.344 headline is a 400-sim number; the goal is
"beat Pentobi L9", not "beat it at 400 sims", and nobody has measured what thinking time
buys this net at the top levels. EV: possibly a large instant strength reclassification in
both directions of the ledger, and it re-anchors the entire §6 costing. This is the
highest information-per-pound action available.

**R2 — Value-target repair + honest value diagnostics (2–4 days of code, £0).**
Land colour-conditional value calibration in the per-gen report (my
`colour_value_diag.py` is a working seed); add outcome-balanced sampling of the value
loss (ELF's fix for exactly this skew) and/or a WDL head (22% draws); optionally λ-blend
the value target with the stored root search value. Directly attacks H2, the
best-evidenced defect (§3 item 2). EV: this is the change with the strongest measured
justification in the whole option list.

**R3 — The pilot, then the run (box £0, then ≤£150 rented).**
Stage 1 (THE decisive experiment, §5): 15 box generations changing ONLY the training
step: LR 2.5e-4, AdamW 1e-4 (now default), 1 epoch, gate off, keep-best-by-ladder every 5
gens + the drift breaker (both built, never used). Stage 2, if the ladder slope is
positive: add R2's value repair and width (top_k 128 **with** considered 64 — the
briefing's own root-narrowing caveat), run 40–60 gens on the box (free, ~2 weeks) or the
5090 (~£75–150). Pre-registered ladders at fixed gens, ±0.022 floor respected.

**R4 — Architecture rebuild (global pooling + corner/inventory planes + aux heads in RL),
only if R3 moves but plateaus again (1–2 weeks of code, then a from-scratch or distilled
restart).** The KataGo evidence (1.65×/1.60×/1.55×) is strong but it is *transfer*
evidence; every trunk change lands in torch + JAX bridge + ONNX + parity tests and breaks
warm-starting [C]. Bundle all of it into one rebuild; use the Pentobi corpus + the
existing replay data to re-initialise the new trunk (the "dead end" corpus is exactly the
right launchpad here — imitation for initialisation is a different claim from imitation
for strength). Do not A/B individual features at this budget.

**R5 — Do not spend on:** more sims at fixed width (measured 3× cost, zero effect);
supervised capacity probes (measured unresolvable — §3.7); more Pentobi games for
imitation (measured decelerating below v3); any arena-threshold or arena-Elo derived
acceptance (measured blind below ~100 Elo); Reanalyze/PCR engineering before a working
operator exists (multipliers on zero are zero).

## 5. The single cheapest decisive experiment

**R3 Stage 1: 15 generations on the box (£0, ~2 days at v3 config), warm from the best
checkpoint R1 identifies, changing only the training step: LR 2.5e-4 constant, AdamW
weight decay 1e-4, epochs 1, no gate, keep-best-by-ladder every 5 generations, drift
breaker armed. Ladder L1–9 × 100 at gens 0/5/10/15.**

Why this one: it isolates H1 (the operator's training step was toxic/stationary) from
"the ceiling is real", using only machinery that already exists, at zero cash cost, in
the shortest time a ±0.022-resolvable answer allows. Every fancier change (value repair,
width, architecture) is only interpretable once this baseline exists.

Pre-registered outcomes:
- **Ladder ≥ 0.375 at gen 15** (baseline 0.344 + noise floor 0.022 + margin): H1
  confirmed; proceed to R3 Stage 2 with money.
- **Ladder within ±0.022 of 0.344 at gen 15, and my colour-value diagnostic on the gen-15
  net still shows skill ≤ +0.1**: H1's optimistic reading is falsified — the training-step
  fix alone does not move the fixed point. Escalate to R2+width in Stage 2, still on the
  box.
- **Ladder within noise AND value skill > +0.2 after R2 is added in a follow-up 15 gens**:
  my central hypothesis H2 is falsified too — at that point the honest conclusion is that
  the remaining levers are architectural (R4) or the budget answer of §6 applies in its
  pessimistic branch. I commit to that reading in advance.

## 5. The single cheapest decisive experiment

*(pending)*

## 6. Is Pentobi level 9 reachable on this budget? (direct answer)

**Direct answer: not as a plannable outcome of the current system plus a few hundred
pounds — the honest probability is roughly 25–40%, and it requires two specific things to
break right. Level 6 is the outcome you can plan on; level 7 is a coin flip. A
comfortable, high-confidence level 9 is a different project phase costing roughly
£400–£1,000 of rented compute (or 2–3 months of box time) on top of an architecture
rebuild.**

The arithmetic [M → I]:

- **The gap.** v3_gen40 scores 16/20/21% vs L7/8/9 (100 games, ±8pp). 20% ≈ −240 Elo.
  Beating L9 solidly (≥55%) needs ≈ **+275 Elo** of true strength (range +200 to +330
  given the binomial noise). Note the shape: L5 47% → L6 40% is ~50 Elo of spacing, but
  L6 40% → L7 16% is a **~220-Elo cliff** — L7 is where Pentobi's search depth jumps. The
  path to L9 is mostly the path past that cliff.
- **What a successful run buys.** The only two healthy runs on record (cloud_60's cosine
  run, v3) each bought on the order of +100–150 ladder-Elo per ~40–60 generations.
  A 2-week box run ≈ 100 v3-config generations (§4 anchors) ≈ **one** run-scale attempt;
  £150 of 5090 ≈ two more. So the raw budget covers ~2–3 run-scale attempts.
- **Therefore:** if the fixed operator sustains historical yield for two consecutive
  run-scales with no new plateau, the training gap closes to ~±50 Elo of L9. That is the
  optimistic branch, and diminishing returns near a stronger opponent argue against
  assuming it.
- **The two things that must break right:** (1) R3's operator fix actually restores
  ~+100 Elo/run-scale yield (decided for £0 by §5's pilot); (2) **eval-time search
  scaling** contributes its usual AlphaZero-family +100–200 Elo — the 0.344 figure is a
  400-sim number, Pentobi L9 spends far more thinking time than our 400 sims, and the
  goal is "beat Pentobi L9", not "beat it at 400 sims". Nobody has measured the net's
  L7–L9 scores at 1,600–6,400 sims (§4 R1b). If that curve is healthy, a large fraction
  of the +275 comes free at match time; if it is flat, that is itself diagnostic (a
  colour-prior value head caps search scaling — H2 again) and the training-only gap is
  probably out of reach on this budget.

**What level IS reachable with confidence:** the operator fix alone, at historical yield,
takes the L5/L6 boundary (currently 47%/40%) solidly — call it **level 6 planned, level 7
plausible** with value repair + width + one rented run.

**What level 9 actually costs if the budget answer is no:** the KataGo-style bundle (R4:
global pooling, Blokus input features, aux heads in RL, PCR) claims a combined 3–4×
efficiency multiplier — that is the difference between "2–3 run-scales" and "8–10
run-scales" of effective compute. Concretely: the rebuild (1–2 weeks of code), a
distilled re-initialisation, then ~200–400 5090-hours (~£160–£320) or ~2–3 months of box
time, with ladder-gated decision points. Total programme: **£400–£1,000 rented, or a
mostly-free box programme measured in months, after the rebuild.** If the §5 pilot shows
the ceiling is real even with a healthy operator, stop spending on runs entirely until
the rebuild is done — multipliers on a stalled operator buy nothing.

## 7. Run-parameter analysis (LR, games/gen, generations, buffer, epochs, batch)

| parameter | as run | verdict |
|---|---|---|
| **Learning rate / schedule** | constant **1e-3** in every post-cloud_60 run (verified: v3 `LearningRate` parquet flat at 0.001; rerun resolved config) | **Wrong — the single worst parameter choice on the books.** The only run with a schedule (cloud_60, cosine → 2.7e-4) accepted 39/58 and produced most of the project's total progress; every constant-1e-3 warm continuation since produced ≤0. A warm continuation of an already-strong net at the *from-scratch peak* LR is how you get a +8-Elo fixed point (each epoch learns and forgets in equal measure) and, un-gated, the rerun's monotone symmetry-KL/value-MAE decay. The committed continuation config's 2.5e-4 is the right direction [M: `49647f1`]. Recommend 2.5e-4 constant for pilots; a cosine to ~1e-4 for the long run. |
| **Weight decay** | **absent in all five ledger runs** (verified for the rerun via resolved config; the AdamW default landed 2026-07-22, after every run) | Wrong as run, already fixed. Keep 1e-4. The rerun is precisely the "Adam, no decay, high LR, data reuse" failure signature. |
| **Epochs/gen** | 1 (all runs) except the rerun's 2 | 1 is right. 2 at constant 1e-3 with buffer reuse ≈12 passes/position was a co-cause of the rerun's degradation. Do not raise epochs; if more reuse is wanted, lower LR first. |
| **Games/generation** | 10,000, never varied | **Untested, and the most interesting free experiment nobody has run.** The operator improves per *iteration*, and its per-iteration gain (+8 Elo) did not grow when target quality rose (sims 128→256 changed nothing). 2,500–5,000 games/gen with a proportionally smaller buffer at LR 2.5e-4 gives 2–4× more improvement iterations per GPU-hour. Worth one arm of the pilot. |
| **Generation count** | 9–20 for every recent run | Too short to resolve success against the ±0.022 ladder floor at realistic yields (~+3–8 Elo/gen ⇒ ~15 gens per resolvable increment). Plan runs ≥30 gens with ladders every 5; kill via the drift breaker, not via impatience. |
| **Buffer size / staleness** | 60k games ≈ 6 gens staleness | Reasonable *when the gate accepts*. Under a rejection streak the buffer degenerates to 100% frozen-incumbent data — a co-mechanism of the stationarity (their own B3/B4). Moot once threshold gating is abandoned. Scale buffer with games/gen (keep staleness ≈4–6). |
| **Batch size** | 1024 | No evidence against it; at LR 2.5e-4 it is conservative. Lowest-priority knob; leave it. |
| **Gate** | 0.55 → 0.52 → guard 0.45 | All three modes measured blind or vacuous (§3.2, reproduced). The correct acceptance signal at this budget is the ladder (keep-best-by-ladder every ~5 gens, built and never used) with the paired arena kept only as a cheap crash detector. |

## 8. What I could not determine

1. **The 96.3% white-win figure** — raw ArenaReplays for `blokus_search_harder` deleted
   (§0.2.1). Methodology validates on v3's surviving replays; the number itself is
   unverifiable. Restore from backup if one exists.
2. **The cloud_60 ledger row** (39/58 accepted, cosine schedule, medium→large switch) —
   no local raw data at all (`temp/runs/blokus/blokus_cloud_60/` holds only
   `Reporting/`). My §7 LR argument leans on it as documented, not as verified.
3. **Whether v3 gen-32 or gen-36 is actually stronger than gen-40** — pool Elo says maybe
   (+286/+271 vs +240), only a ladder can say (R1a). Everything warm-started from gen-40.
4. **How much of the gen-1→gen-40 value-skill collapse is vintage confound** — needs an
   eval set of fresh games played by gen-40 itself; a box job, not a laptop job.
5. **bfloat16 self-play numerics** — production ran bf16; all parity tests are CPU fp32
   (§3 D5). One-flag A/B on the box.
6. **Whether width (top_k 128) buys anything** — cost measured, benefit needs the pilot.
7. **Pentobi's true L7–9 Elo spacing** — 100-game ladders give ±8pp; the L7 16% < L8 20%
   < L9 21% inversion is noise-compatible and worth 400-game samples at L6–L9 before any
   "we beat level N" claim is made.
8. **Eval-time sims scaling at L7–9** — never measured (R1b); my §6 answer's optimistic
   branch depends on it.
9. **Total historical GPU-hours** — no consolidated record; my £/Elo anchors rest on the
   v3 and rerun Timings parquets only.
10. PR #68 (docs-only) was not reviewed; nothing in this report depends on it.

---

*Method note: all measurements in this report were recomputed from the raw stores under
`temp/runs/blokus/` and `temp/benchmarks/`, or executed fresh (test suite; the
colour-value diagnostic at `scratchpad/colour_value_diag.py`; duplicate-game scan) on
2026-08-04, read-only, on the laptop CPU. No training was run, nothing on the box was
touched, and the repo was not modified.*
