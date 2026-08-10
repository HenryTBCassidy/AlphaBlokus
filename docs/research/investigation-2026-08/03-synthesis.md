# AlphaBlokus — synthesis and action list (2026-08-04)

Built from `01-chatgpt.md`, `02-fable.md`, and the briefing audit in `02-fable.md` §0,
with every decision-critical claim re-verified on this machine today. Verification notes
are marked **[V]**; source attribution is **[F]** Fable, **[C]** ChatGPT, **[B]** both,
**[N]** new here.

Four streams, because three don't cover it:

| Stream | What | GPU? | Parallel? |
|---|---|---|---|
| **A** | Code changes to land now — no experiment needed | no | yes, all of it |
| **B** | Box measurements and parameter sweeps | yes | mostly serial |
| **C** | Bug sweep | no | yes, own worktree |
| **D** | Build-and-test — candidate fixes that need code *and* a run to judge | both | after B6 |

Plus **PARKED** — things deliberately not done, with the reason, so they don't get
quietly re-litigated.

**Every item up to and including B6 costs £0.** The first rented pound is spent only
after B6 returns a positive result.

---

## Honest position up front

Nothing in this plan is a route to Pentobi level 9. On the evidence, **level 6 is the
plannable target and level 7 is a stretch**; level 9 needs an architecture rebuild plus
roughly £400–£1,000 of compute, and neither report puts it above a coin flip. Both
reports independently conclude the current pipeline is capped somewhere around levels
4–7 regardless of how much compute goes into it.

That said, the goal here is to maximise strength gained per pound and stop when the
measurements say stop — not to hit a number. The good news is that the highest-value
unknowns are all free: B1 and B2 could reclassify the project's baseline in either
direction for zero cost, and B6 decides whether any money should be spent at all.

**The single largest risk is spending weeks of free box time on a loop that is
fundamentally capped.** B1–B6 exist to find that out cheaply.

---

## Stream A — land these now

No experiment required. Either both reports agree, or it's hygiene that prevents a
repeat of a known failure.

| ID | Change | Why | Cost | Src |
|---|---|---|---|---|
| **A1** | Delete `dirichlet_epsilon` and `temp_threshold` from all Gumbel configs. Add a config validator that errors when a knob is set that the active search path ignores. | Both are verified no-ops under Gumbel, yet sit in every production config implying root-noise and temperature exploration that does not exist. Anyone tuning them tunes nothing. | ~1 h | [F] |
| **A2** | Retire the acceptance gate as the promotion signal. Accept every generation; select with **keep-best-by-ladder** every N generations; arm the **drift circuit-breaker** as the catastrophe stop. Keep the paired arena as a crash detector only. | Three modes of gate (0.55 / 0.52 / guard 0.45) all measured blind or vacuous. Both machinery pieces are already built and have never run. | ~½ day | [B] |
| **A3** | Learning rate 1e-3 → **2.5e-4** for pilots, cosine 2.5e-4 → 1e-4 for long runs. AdamW weight decay 1e-4, verified active in all parameter groups. **Epochs 2 → 1** — an explicit change, not a hold: v3 ran 1 but the most recent run (paired_gate_rerun) ran 2, so this halves gradient passes relative to that run and must be declared, not assumed. Never reset the LR to peak on a warm continuation (verified mechanism: `load_weights` leaves optimizer and scheduler fresh, so every warm continuation restarts the LR at peak by design). | The only run with a decaying schedule produced most of the project's total progress; every constant-1e-3 continuation since produced ≤0. Weight decay was absent from all five runs. Final number pinned by **B3**. | ~1 h | [B] |
| **A4** | Extend the parity test matrix to run in **bfloat16** as well as fp32, so self-play and tests exercise the same numerics. | Production self-play runs bf16 (`jax_selfplay.dtype = bfloat16`, `autocast_dtype = bf16`); every correctness test runs CPU fp32. Top-64 selection over 17,837 near-tied logits under reduced precision has never been checked against the tested path. **[V]** confirmed in the rerun's resolved config. | ~½ day | [F] |
| **A5** | Run provenance: stamp resolved config + code hash + data manifest into every run directory. **Refuse to start a run when the committed config and resolved config differ.** Stop retro-editing committed configs. | The rerun's committed config was edited 5 days *after* the run and now describes a run that never happened — read it today and you'd conclude the rerun had weight decay. It didn't. W&B does log config, so this is mostly a guard. | ~½ day | [C][F] |
| **A6** | Eval set: rebuild it periodically **during** a run, not once from generation 1. Record the source game ID for every position. Compute all eval-set diagnostics with a **game-cluster bootstrap**. | The frozen set is built from each run's weakest-ever data and never refreshed. **[V, corrected 2026-08-04]** an earlier claim here that the 200 positions were only ~47 game lineages was WRONG — that clustering keyed on shared opening prefixes (gen-1 self-play has almost no opening diversity), not same-game siblings. Verified: the big 81-position cluster contains 42 wins / 30 losses / 9 draws, and games are only ~32 plies, so it cannot be one game. The set is 200 positions drawn uniformly from ~320,000 gen-1 positions across 10,000 games, i.e. ~one position per game (expected same-game collisions ≈ 2), so **per-position statistics on the CURRENT eval set are approximately correct**. The cluster bootstrap is still required — but as insurance for the REBUILT eval set, which will draw multiple positions per game once source games are withheld whole (this is part of A6, not a separate item — see the note below). | ~1 day | [F][C][N] |
| **A7** | Land the colour-conditional value diagnostic in the per-generation report: colour-only *and* colour×phase baselines, split by White-to-move / Black-to-move and game phase, with game-cluster CIs. | Proposed twice, never built; Fable's `colour_value_diag.py` is a working seed but computes statistics per position, which is what inflated its degradation claim. This is the primary instrument for the leading hypothesis. | ~1 day | [B] |
| **A8** | Log wall-clock for ladder runs and self-play generations into the metrics store. | **[V]** ladder duration is recorded nowhere on disk. The backbone measurement of this plan has unknown cost, which makes scheduling guesswork. | ~1 h | [N] |
| **A9** | Stop budgeting in generations. Log and plan in **total games, total optimiser steps, and passes-per-position**. | "Generation" bundles games, epochs and buffer staleness into one word, so no two runs' generations are comparable — which is part of why the run ledger is hard to reason about. | ~½ day | [C] |
| **A10** | Write an erratum to `00-briefing.md` (or fix it in place) covering everything in Fable §0 plus the corrections below. | The briefing is what any future session inherits as fact. Errors in it have already propagated into three paid runs' worth of next steps. | ~1 h | [F][N] |

### Note on item numbering — there is no A11 or A12 **[V, corrected 2026-08-05]**

Stream A is **A1–A10**. Earlier drafts of A6 and of `06-handoff.md` referred to an "A12"
(withholding source games whole) as though it were a separate item. It never existed: no
A11 or A12 is defined anywhere in this document or in the two source reports. The
withholding work is the first half of A6, and it is implemented in PR #69 along with the
rest of A6 — the rebuild cadence (`RunConfig.eval_set_rebuild_every`), `source_game_ids.npy`,
whole-game exclusion via `ReplayBuffer.exclude_games`, and the game-cluster bootstrap
(`alphablokus.bootstrap.game_cluster_bootstrap`, consumed by `evaluation/colour_value.py`).

**A6 is therefore complete, not outstanding** — but note `eval_set_rebuild_every` defaults
to `0` (build once, then freeze, i.e. today's behaviour). `blokus_pilot_b6.json` sets it to
`5`. Any other run that wants a refreshed eval set must opt in explicitly.

### A10 in detail — what the briefing gets wrong

- **The 96.3% white-win figure is unreproducible. Stop quoting it.** **[V]** the raw
  replays are gone — `blokus_search_harder/` holds only `Nets/`, and no `ArenaReplays`
  for that run exists anywhere in the local tree. The conclusion it supported (the
  candidate-vs-incumbent arena cannot see strength) survives on other evidence: v3's
  surviving replays give 77%, and the variance-collapse test reproduces exactly.
- `search_harder_v2` ran **8 generations, 1 of 8 accepted**, not 9 and 1 of 9. **[V]**
- **False precision.** "Fine-tuning moved strength by 0.002" is quoted to three digits
  against a noise floor of 0.022 the briefing measures itself eleven lines earlier. Same
  for "+0.006" on capacity. Correct wording: *unchanged, and the experiment could not
  have detected a ±40-Elo effect in either direction.*
- The **gen-17 entropy collapse recovered** — gen 18 was 0.789, gen 20 was 0.820. The
  briefing presents a one-generation blip as a terminal event.
- The rerun's **"pool Elo −44" implies a trend that isn't there**: the trajectory bounces
  within ±40 and gen 19 was −4. It also comes from the same family of instrument §3.2
  correctly discredits. The ladder drop 0.344 → 0.298 is the real evidence, and it holds.
- "93–97%, never below 93" should read **93–100%**.
- **v3's internal rating peaked at gen 32** (+286) and fell to +240 by gen 40. **[V]**
  But the series bounces ±100 between adjacent generations (gen 18 = 223, gen 24 = 81,
  gen 32 = 286), so "peaked then declined" over-reads it. The actionable half stands: see B1.
- **§3.1's "gen-40 is at the fixed point of the improvement operator" is
  over-generalised** (both reports). It is at the fixed point of *one training step under
  one acceptance rule*. No healthy operator has ever been pointed at v3.
- **§7.3 is wrong**: search width is *not* the only mechanism explaining a fixed point
  immune to sims, capacity and data — the value head explains all three at least as well
  and has direct adverse measurement. §7.3 and §7.5 should swap places.

### Correction to Fable's headline, for the record

The value diagnostic reproduces to the digit **[V]**. Correct CIs (per-position bootstrap,
which the clustering correction above now justifies — an earlier version of this section
used an over-conservative cluster bootstrap and drew one wrong conclusion):

| net | eval set | value skill | 95% CI | paired difference |
|---|---|---|---|---|
| v3 gen-1 | v3 | +0.385 | +0.286 → +0.469 | gen-40 − gen-1 = **−0.487**, CI −0.675 → −0.330 → **resolved** |
| v3 gen-40 | v3 | −0.102 | −0.289 → +0.051 | |
| v3 gen-40 | rerun | +0.056 | −0.135 → +0.204 | gen-20 − gen-40 = **−0.145**, CI −0.370 → +0.082 → **not resolved** |
| rerun gen-20 | rerun | −0.089 | −0.320 → +0.096 | |

- **Holds, and it is the important claim:** the production value head has no demonstrable
  skill beyond guessing from whose turn it is. Both production CIs bracket zero.
- **Holds, and label-independent:** every checkpoint's value output tracks mover colour at
  0.76–0.81 while outcomes only track it at 0.52–0.62. The net is more certain that colour
  decides the game than the game is.
- **Degradation across v3 (gen-1 → gen-40) is statistically resolved** but remains
  *causally* confounded by target vintage — gen-1 is graded on labels from its own games,
  gen-40 on a stranger's. Statistically real, causally unproven.
- **Degradation on the vintage-matched pair (gen-40 → rerun gen-20) is NOT resolved.** This
  was the pair Fable relied on as clean, and it does not carry the claim.
- **Consequence for the plan:** Fable's kill criterion "value skill ≤ +0.1" has a
  measurement error of roughly ±0.17 on a 200-position set, so it is a weak gate even with
  correct statistics. Judge the D-stream on the **ladder** primarily, with value skill as a
  mechanism check — and settle the vintage confound with fresh games played by the
  checkpoint under test (which is what the pending self-play job would supply).

---

## Stream B — box runs, in order

Serial unless noted. Ladder jobs are Pentobi (CPU) + net inference (GPU); two may
co-exist if the 8 GB card allows, but assume serial until A8 gives real timings.

### B1 — Ladder gen-32 and gen-36 · £0 · **do this first**

L1–9, 100 games/level, 400 sims, plus 200–400 games at L6–L9 where the ±8pp binomial
noise currently makes the 16/20/21 scores meaningless.

**Why:** only **three nets have ever been measured against Pentobi at all** — cloud_60
gen-57, v3 gen-40, rerun gen-20 **[V]**. Everything else's strength is inferred from an
internal tournament that ranks gen-32 *above* gen-40, and which **[V]** was played at 128
simulations against the ladder's 400 — a weaker search than the one we care about, and a
concrete mechanism for why internal Elo keeps disagreeing with the ladder. Every
experiment since v3 warm-started from gen-40 without anyone checking.

**Decision rule:** if gen-32 beats gen-40 by more than 0.022 weighted, all future runs
start from gen-32 and the ledger's "v3 plateaued at gen-40" story is wrong. If within
noise, gen-40 stands and the question closes permanently.

### B2 — Eval-time search scaling at L5–L9 · £0 · **highest-value unknown in the project**

The best checkpoint at 400 / 1,600 / 6,400 sims, 100 games/level.

**Why:** the headline 0.344 is a **400-simulation** number. Pentobi at level 9 thinks far
longer. The goal is "beat level 9", not "beat it at 400 sims", and nobody has ever measured
what thinking time buys this net at the top levels. In the AlphaZero family this is
typically worth +100–200 Elo at match time, for free.

**Kill condition:** if 400 → 6,400 sims buys less than ~5pp at L7–L9, then (a) the
optimistic budget case evaporates, and (b) it is independent evidence for the value-head
hypothesis, since a colour-prior value head caps search scaling by construction. Either way
it changes the plan, which is what makes it worth doing first.

### B3 — Learning-rate sweep, offline · £0

One frozen replay buffer from existing v3/rerun data, same starting checkpoint, AdamW at
**1e-4 / 3e-4 / 1e-3 × 3 seeds**. Judged on **held-out games** — never training loss, the
metric that lied throughout the rerun. Nine short jobs, no self-play.

**Why:** ChatGPT's design, and better than just picking 2.5e-4 as Fable suggests. The
parameter is load-bearing and the sweep is nearly free.

**Kill condition:** if all three rates land within the measured 0.05-nat held-out CE noise
floor, the learning rate is not the lever. A3 stays as a cheap hedge but stops being
treated as a fix, and the weight shifts to Stream D.

### B4 — Width shadow test · £0 · ~15–30 box hours

ChatGPT's design, which isolates the mechanism the failed 128→256 test confounded. Frozen
v3 positions, identical randomness. Control: `top_k` 64 / considered 64 / 128 sims.
Intervention: `top_k` 128 / considered 64 / 128 sims. Measure, separately:

1. how often the root's chosen move changes;
2. how often moves ranked 65–128 enter consequential child searches;
3. improvement in completed-Q on the 64 moves common to both;
4. whether changed decisions survive an independent deeper full-action-space search.

**Why test at all when Fable says don't bother:** the two reports disagree, the test is
free, and it settles a question the project has carried for weeks. Note that raising
`top_k` alone does not widen the root — `considered` must move with it, which is why this
test holds `considered` fixed at 64 and measures child-node effects specifically.

**Kill condition:** if rank-65+ moves change the root decision in under 2% of positions,
or the changed decisions don't survive the deeper check, width is dead and we stop paying
the measured 1.24× cost per game.

### B5 — bf16 vs fp32 self-play A/B · £0 · hours

One config flag. Compare top-k selection agreement and training-target distributions on
identical positions.

**Kill condition:** >99.5% selection agreement and negligible target KL closes the question
permanently. Anything worse and self-play moves to fp32 or a mixed policy.

### B6 — The pilot · £0 · ~2–3 days · **the decision point for spending money**

15–20 generations on the box, warm from whatever **B1** says is best, changing **only the
training step**: LR from B3, AdamW 1e-4, 1 epoch, gate off, keep-best-by-ladder every 5
generations, drift breaker armed. Buffer held at 60k so it isn't a confound. Ladder L1–9 ×
100 at generations 0 / 5 / 10 / 15, pre-registered.

**Why:** this is the one experiment that separates "the loop was broken" from "the ceiling
is real." Every stalled run had a since-diagnosed defect — v1 and v2 were frozen by a gate
demanding something measurably impossible, and the rerun trained with settings the
project's own post-mortem calls toxic. **A healthy operator has never been pointed at
v3 even once.** Everything fancier is uninterpretable until this baseline exists.

**Pre-registered outcomes:**
- **Ladder ≥ 0.375 at gen 15** (0.344 + noise floor + margin): the loop was broken, the
  ceiling is not real. Proceed to a longer run, and this is where rented compute becomes
  justified.
- **Ladder within ±0.022 of 0.344:** the training-step fix alone does not move the fixed
  point. **Spend no money.** Escalate to Stream D on the box.
- **Ladder falls below 0.322:** something in A1–A9 made it worse. Stop and diagnose.

### B7 — Games-per-generation arm · £0 · optional second arm of B6

2,500–5,000 games/generation at the chosen LR with a proportionally smaller buffer.

**Why:** 10,000 games/generation has never been varied in the project's history. The
operator improves per *iteration*, so smaller generations buy 2–4× more improvement
iterations per GPU-hour. This is the resolvable half of the two reports' disagreement about
data volume — they actually agree that more frequent updates beat fewer, and disagree only
about buffer size, which is parked.

**Kill condition:** if the ladder slope per GPU-hour is no better than B6's, keep 10,000
and close the question.

---

## Stream C — bug sweep (separate worktree, no GPU, fully parallel)

**Do not redo these — Fable already closed them empirically:** value label sign and
perspective conventions, harvest↔episode equivalence, symmetry augmentation correctness,
torch↔JAX parity in fp32, duplicate/degenerate self-play games (zero exact duplicates
across 770k episodes), checkpoint save/load and gate-revert semantics, acceptance
arithmetic. The suite is 911 passing, 1 skipped **[V]**.

That leaves six items from ChatGPT's list genuinely open:

| ID | Audit | Why it matters |
|---|---|---|
| **C1** | **PRNG key uniqueness** in lockstep JAX self-play — unique per game, move, search, generation and device; no Gumbel vector broadcast across games. | Highest priority of the residue. If randomness is shared across games, the data looks voluminous but is far less varied than it appears, and **every metric still looks healthy**. Fable's zero-duplicates result rules out the extreme version, not partial correlation. |
| **C2** | Is **top-k applied before or after illegal-move masking**? Can a legal move be pushed out of the 64 slots by illegal logits? | A silent, systematic reduction of the effective search width — and it would masquerade as the width hypothesis. |
| **C3** | **Completed-Q perspective at every tree depth** — root, parent or player-to-move? Is the sign flipped correctly each ply? | Fable verified the target *formula* and the top-level convention, not the per-depth perspective. A sign error deep in the tree would corrupt targets in a way no current test catches. |
| **C4** | **Replay sampling audit** — actual vs configured buffer size; times each position is sampled over its lifetime; whether uniform-by-position sampling overweights long games; outcome/role/phase composition of a typical minibatch. | Directly feeds the leading hypothesis, and is the measurement D1 needs to be designed properly. |
| **C5** | **Optimizer continuation** — Adam moments preserved or reset across candidates; whether the LR schedule restarts each generation; whether weight decay is live in all parameter groups. | A3 assumes all three behave. Fable verified the revert path deliberately does not restore the LR clock; the rest is unchecked. |
| **C6** | **Small exact tests** — single-legal-move positions, terminal positions, short endgames solved by exhaustive minimax, JAX Gumbel vs an independent reference implementation. | Cheap, permanent, and the class of test most likely to catch something the parity tests structurally cannot. |

---

## Stream D — build-and-test (the candidate fixes)

These need code **and** a box run to judge. They are the actual remedies for the leading
hypothesis, and they are gated behind **B6** — a fix layered on a broken loop is
uninterpretable.

Implementation status verified today, because three of these were believed to be done:

| ID | Fix | Status **[V]** | Cost |
|---|---|---|---|
| **D1** | **Outcome-balanced value sampling** — draw equal numbers of White-win and Black-win games when computing the value loss, so "White is probably winning" stops paying. ELF OpenGo's fix for this exact skew. | **Not implemented anywhere.** Only corpus *diagnostics* measure outcome balance. This is a permanent training-run change to how the replay buffer samples, not a test. | ~1 day + a box run |
| **D2** | **WDL head** — three probabilities (win/draw/loss) instead of one number from −1 to +1. 22% of games are draws and a scalar cannot separate "certain draw" from "coin flip". Lc0 standard since 2019. | **Never implemented.** `13066c8` only *registers the idea* (`docs/IDEAS.md` I8, plan N7), deliberately sequenced after the score-head A/B. No `wdl`/`win_draw_loss`/`draw_logit` anywhere in `src/`. | ~2 days + a box run; touches torch, JAX bridge and ONNX |
| **D3** | **λ-blend the value target with the search's own evaluation** — `target = λ·search_value + (1−λ)·outcome`, so a position 30 moves from the end isn't labelled purely by who eventually won. | **Built for the corpus, never for RL, never run as an experiment.** Corpus rows store Pentobi's `search_value`; opening rows already use a count-shrunk blend; `--opening-value {outcome,search,blend}` exists. The planned arm (N6, λ ∈ {0, 0.3, 0.5}) has an empty status column. **The RL version cannot be built as-is** — self-play stores only (board, policy, value), so the search's root value must first be plumbed through harvest and the buffer. | ~2–3 days (plumbing dominates) + a box run |
| **D4** | **Auxiliary heads inside RL** (score / ownership / reply). | Merged, default off, and would train on **nothing** if switched on today — targets are not threaded through the self-play buffer. The old evaluation was invalid, not negative: it measured whether the heads help *copy Pentobi better*, when the claimed benefit is that they change the net, which changes the search, which changes the next batch of self-play data — the loop that offline copying deletes. A valid test needs ≥2 seeds and ~220–320 GPU-hours. | Real work; late |
| **D5** | **An actual exploration mechanism for self-play diversity** — opening seeding from the corpus's labelled opening DAG, or sampling rather than always playing the search's winner. | None exists. Under Gumbel the engine plays the Sequential-Halving winner every ply and diversity comes from root Gumbel noise alone. The two knobs that imply otherwise are A1. | ~2 days |
| **D6** | **Hybrid corpus use** — a persistent 10–20% policy-only teacher mixture during RL; querying Pentobi on positions the *student* actually reaches rather than learning from Pentobi's own games; a teacher-policy auxiliary head or annealed KL instead of overwriting the policy. | Untested. Both reports say the "distillation is a dead end" verdict is too broad: it tested one recipe, which also trained the value head on the corpus's own outcome labels with no way to disable that loss — the exact labels shown to produce colour-prior value heads. A policy-only arm was never run. | ~2–3 days each; late |

**Sequencing within D:** D1 first (cheapest, strongest evidence, no architecture change),
then D3, then D2. D4–D6 only if the ladder is still moving.

---

## PARKED — deliberately not doing, with reasons

| Item | Why parked |
|---|---|
| **Net capacity / `xl` through real RL** | The only valid test is running the 19M-parameter net through actual self-play, which is expensive and slower per game. Note the reason has **changed**: not "capacity is dead" (that was an imitation-only result where the ceiling is the teacher) but "we cannot afford to test it and it isn't the leading suspect." ChatGPT cites published evidence that larger AlphaZero nets are *more* sample-efficient per game, which contradicts the assumption used to dismiss it — so this stays open, not closed. |
| **Buffer size (60k vs shrink vs 120–240k)** | Genuinely untested, and the two reports point in opposite directions. Hold at 60k through B6 so it isn't a confound, and revisit only if the loop starts moving. |
| **Restoring the 96.3% raw data** | Changes no decision — the conclusion it supported is independently established. One command to check whether it's on the box, at the bottom of the list. |
| **Playout cap randomisation, Reanalyze** | Efficiency multipliers. Multipliers on a loop that doesn't yet improve are worth nothing. |
| **Architecture rebuild** (global pooling, placeable-corner planes, explicit inventory, phase scalar) | KataGo's biggest levers (1.60×, 1.55×), but every trunk change lands in torch + JAX bridge + ONNX + parity tests and **breaks warm-starting**. Only justified if B6 moves and *then* plateaus again — and then bundled into one rebuild, never A/B'd individually at this budget. |
| **Buying more Pentobi games for imitation** | Measured decelerating below v3's strength. Both reports agree. |

---

## What to kick off now

**Immediately, in parallel:**
1. **B1** on the box — free, gates everything downstream, and may reclassify the baseline.
2. **Stream C** in its own worktree — no GPU, no interference.
3. **A1, A8, A10** — an hour each, and A8 is needed to schedule the rest of B.

**Then:** A2–A7 and A9 land while B2 runs. B3 → B4 → B5 as the box frees up. B6 last, as the
gate on all spending.

**Money:** nothing until B6 clears 0.375. If it doesn't, Stream D runs on the box for free
and B6 repeats.

**Stopping rule, stated in advance:** if B6 comes back within noise, and then B6 repeated
with D1+D3 layered on is *also* within noise, the honest reading is that the remaining
levers are architectural. At that point the choice is a rebuild programme measured in
months of box time, or accepting the project lands at level 4–6 and stopping. That is a
legitimate outcome and it should be called plainly if it arrives — not rationalised into
another run.

---

## Open questions for you

1. **Is `gpu-linux` currently reachable?** Everything in Stream B assumes it is. If not,
   B1/B2 are the only items worth renting for (a few 5090 hours, ~£5), because they gate
   everything else.
2. **B7 as a second arm of B6, or after?** Running it concurrently doubles box time but
   answers the games-per-generation question a week earlier.
3. **A6 and A7 are prerequisites for reading D-stream results** (the eval set can't resolve
   value skill to better than ±0.17 today). Confirm they land before D1 rather than
   alongside it.
