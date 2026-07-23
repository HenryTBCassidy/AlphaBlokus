# Fix the colour-pinned arena gate + acceptance policy, then rerun

Implements the recommendations of [`../research/plateau-investigation.md`](../research/plateau-investigation.md)
(Fable, 2026-07-15). That post-mortem found — with no training-loop bug — that the arena gate is
**colour-pinned**: 96.3% of decisive arena games are won by White (the first mover), so between
near-equal nets the gate score is mathematically stuck at ~0.50 ± 0.02 and *no* threshold at 0.55 or
0.52 can be cleared by a candidate that is merely somewhat better. The gate then freezes the
incumbent, which makes training stationary (`blokus_search_harder` v1: 0/17 accepted, dead-flat loss;
v2: 1/9 then flat, and its one accepted net showed **zero** external transfer on the Pentobi ladder —
0.335 vs gen-40's 0.344). The opening-diversification work (trustworthy-measurements) fixed clone
*degeneracy* but not colour *pinning*, and the S3 validation was structurally blind to it.

**Scope:** code + config + one validation run, then a follow-on training rerun. Fixes the measurement
first (R1), then the acceptance policy (R2), then makes each generation's delta bigger (R3), plus
hygiene (R8). Deliberately does **not** touch the optimiser/LR (R4) or batch size (R5) — those were
cleared by the investigation and changing them now would confound the experiment. Curriculum levers
(R7) and playout-cap randomisation (R9) are explicitly deferred to a later phase.

Companion docs: [`../research/plateau-investigation.md`](../research/plateau-investigation.md) (§2 bug
hunt, §3 confounders, §5 recommendations R1–R9), [`../guides/PLAN-FORMAT.md`](../guides/PLAN-FORMAT.md),
[`../guides/STYLE-GUIDE.md`](../guides/STYLE-GUIDE.md).

---

## Checklist

| # | Item | Effort | Priority | Files | Done |
|---|------|--------|----------|-------|------|
| S1 | Paired colour-swapped arena play (`Arena.play_games_paired`) — shared opening prefix, played out twice with colours swapped, scored per pair | 3 h | High | `evaluation/arena.py`, `evaluation/players.py` | ✅ |
| S2 | Route the gate + pool tournament through paired play; per-pair scoring + white/black split | 2.5 h | High | `training/coach.py`, `parallel/pool.py`, `evaluation/tournament_run.py`, `config.py` | ✅ |
| S3 | Acceptance as regression-guard (config-selectable gate mode: `threshold` \| `regression_guard` \| `always`) | 1.5 h | High | `evaluation/acceptance.py`, `config.py`, `training/coach.py` | ✅ |
| S4 | Measurement hygiene — log white/black wins per gen; persist staged config at launch; red-flag exact-0.500 + sub-binomial variance in report | 3 h | Medium | `storage/metrics.py`, `training/coach.py`, `cli.py`, `reporting/` | ✅ |
| S5 | **Validation gate** — re-run the S3 known-gap + null controls through the *paired* instrument, colour split reported | ~1 h run | High | (box run + analysis) | ⛔ GPU run — **must run before the S7 rerun launches** |
| S6 | Tests + docs (paired scoring, regression-guard; update EVALUATION; retract the "sub-50-Elo resolvable" claim) | 2 h | High | `tests/`, `docs/05-EVALUATION.md`, `docs/research/xl-training-scaleup.md` | ✅ |
| S7 | Author the rerun config (warm-start v3 gen-40, paired gate, regression-guard, epochs 2, n=256) — gated on S1–S6 | 30 min | High | `run_configurations/` | ✅ (config authored: `run_configurations/blokus_paired_gate_rerun.json`; **do not launch until S5 passes**) |

Execution order: **S1→S2→S3 build the fixed gate; S5 validates it (hard gate — the rerun must not launch until the paired instrument passes the known-gap + null controls).** S4 is independent and parallelisable. S6 last of the code work. S7 authored once S5 passes. S1 is the load-bearing change; everything else composes onto it.

> **Implemented defaults (refines this plan).** To keep every existing config/run bit-identical, the
> new fields ship with **backward-compatible defaults**: `paired_arena` defaults to `False` and
> `gate_mode` defaults to `"threshold"` (not the `regression_guard`/`True` the prose below assumes).
> Only the S7 rerun config (`run_configurations/blokus_paired_gate_rerun.json`) opts into
> `paired_arena: true` + `gate_mode: "regression_guard"`. `guard_floor` defaults to `0.48`.

> **Deliberate non-changes (from plateau-investigation R4/R5 — do NOT undo these):** LR stays constant
> 1e-3 (it was applied exactly, descent is stable, and v3 climbed on it — schedule complexity would
> only confound this experiment). Train batch and jax self-play batch stay 1024 (the loop is
> optimizer-steps-limited, not gradient-noise-limited; bigger batch halves steps/epoch — the wrong
> direction). The 5090's idle VRAM is evidence the net is small, not evidence for bigger batches.
> **Bigger net (`xl`) and multi-GPU are NOT in scope and are not indicated** — the failure is the
> instrument + acceptance policy, not capacity or throughput; scaling compute would inherit the same
> broken gate at higher cost.

---

## S1. Paired colour-swapped arena play

**Current state.** `Arena.play_games(num)` (`evaluation/arena.py:210`) plays `num/2` games with
player1 as White, swaps, then `num/2` with player2 as White. Colours alternate, **but each game
samples a *fresh* opening** (the `NetworkPlayer` opening-temp schedule fires per game). So the two
halves are unpaired: in each half whoever is White wins ~96%, and the totals collapse to ~50/50
regardless of true strength (plateau-investigation §2 B8).

**Target.** Add `Arena.play_games_paired(num_pairs, ...)`: for each pair,
1. Sample a **shared opening prefix** once — `k` plies (default `arena_opening_moves=4`) drawn from
   the incumbent's MCTS visit distribution at `arena_opening_temp` (default 1.0). Capture the exact
   action sequence.
2. Play that prefix out to completion **twice**, deterministically (temp 0 after the prefix, no root
   noise — `mcts.py:134-156` defaults hold), once with the candidate as White and once with the
   incumbent as White, **both replaying the identical prefix**.
3. Score the pair from the candidate's perspective across the two games (see design decision below).
   The first-mover advantage cancels exactly because both nets play the same opening from both sides.
- Mechanism for a forced prefix: extend `NetworkPlayer` (or the game loop) to accept an optional
  `forced_opening: tuple[int, ...]` that is replayed for the first `len(prefix)` plies before normal
  (deterministic) play resumes. `play_game` already threads per-move actions — thread an optional
  scripted-prefix through it, applied to *whichever* player is to move for those plies.
- Return per-pair records (reuse `GameRecord`, both halves tagged with `player1_was_white`) so S4 can
  log the colour split and the replay viewer still works.

**Design decision to resolve in review (call it out, don't silently pick):** the per-pair score.
Two reasonable rules — (a) **paired win-differential**: pair contributes candidate_wins − incumbent_wins
∈ {−2,−1,0,+1,+2}, aggregated to a [0,1] score; (b) **pair outcome**: +1 if candidate wins both,
0.5 if split, 0 if loses both. (a) is higher-resolution (a candidate that wins as Black from an
opening where the incumbent lost as Black is rewarded); recommend (a). Both make a colour-neutral
score; the plan assumes (a). Flag for Henry before implementing.

---

## S2. Route the gate + pool tournament through paired play

**Current state.** The gate arena runs via `arena.play_games(num_arena_matches)`
(`coach.py:512`), and the parallel path drives arena games through the worker pool
(`parallel/pool.py`, openings applied to *both* players via `_opening_for_phase`, seeds via
`derive_episode_seed`). The pool tournament (`evaluation/tournament_run.py`) has the same unpaired
structure. `_record_rolling_elo` (`coach.py:718`) and `_should_accept_new_network` (`coach.py:763`)
consume `(nwins, pwins, draws)`.

**Target.**
- Add a `paired_arena: bool = True` field to `RunConfig` (`config.py:387` area) so the gate uses
  `play_games_paired`; `num_arena_matches` becomes the number of *pairs × 2* (keep the field name;
  document that with paired play it's split into `num_arena_matches/2` shared-opening pairs).
- Wire the paired path through **both** the serial (`coach.py:512`) and the parallel worker-pool
  arena drivers — the parallel path must seed the *shared prefix* per pair identically across the two
  colour-swapped games (extend `derive_episode_seed` phase handling so a pair shares one prefix seed).
- Apply the same paired construction in `tournament_run.py` so the pool BayesElo curve gets
  colour-cancelled pairings too.
- The acceptance/Elo inputs become the per-pair aggregate (S1 rule (a)); `_record_rolling_elo` and
  `_should_accept_new_network` are unchanged in logic, just fed colour-neutral counts.

---

## S3. Acceptance as regression-guard

**Current state.** `is_accepted_score_rule(new, prev, draws, threshold)` (`evaluation/acceptance.py:21`)
accepts iff `score ≥ threshold`; `threshold = update_threshold` (0.55/0.52). This 0.55 gate — an
AlphaGo-Zero artifact — is the *direct* cause of the stationary loop; every DeepMind successor dropped
it ([`../research/deepmind-run-configs.md`](../research/deepmind-run-configs.md) §AlphaZero).

**Target.** Add a config-selectable **gate mode** (`config.py`): `gate_mode: "threshold" |
"regression_guard" | "always"` (default `regression_guard`).
- `threshold`: today's behaviour (accept iff paired score ≥ `update_threshold`).
- `regression_guard`: accept **unless clearly worse** — reject only if paired score < `guard_floor`
  (new field, default 0.48); otherwise adopt. With a rolling 60k buffer, a mediocre accepted net
  self-corrects within ~6 gens; a frozen incumbent never does.
- `always`: AlphaZero-style, always adopt the candidate (keep `accepted_*.pth.tar` + the pool
  tournament as the offline strength record).
- Implement as a small dispatch in `acceptance.py` (add `is_accepted(mode, score, threshold,
  guard_floor)`); `coach.py:763` calls it. R1's colour-cancelled score is what makes even the
  conservative 0.48-guard trustworthy.

---

## S4. Measurement hygiene

- **(a) Colour split.** Log `white_wins` / `black_wins` (and the derived white-win rate) per
  generation into `ArenaData` (`storage/metrics.py` `log_arena`, `coach.py:318`). One groupby on data
  we already record (`GameRecord.player1_was_white`) — would have caught this three runs ago. Add a
  colour-conditional value-calibration line to the eval-set diagnostics (73% of self-play outcomes are
  White wins; we currently can't see if the value head is exploiting that).
- **(b) Persist the staged config** into the run directory at launch (`cli.py`) — two of three runs
  have committed-vs-ran drift (plateau-investigation §1); write the *resolved* `RunConfig` to
  `<run>/config.resolved.json` so the ground truth is never ambiguous again.
- **(c) Report red-flags** (`reporting/`): treat an exact-0.500 arena score or sub-binomial score
  variance across generations as an automatic warning banner in the report.

---

## S5. Validation gate — prove the paired instrument resolves sub-Elo gaps

**Why.** The original S3 (trustworthy-measurements) passed a null + known-gap test but was blind to
colour pinning. Repeat the controls through the *paired* gate, now reporting the colour split, before
trusting it for a rerun. **Until this passes, S7's rerun must not launch.**

**The controls (box run + analysis, no new code beyond S1–S4):**
- **Known-gap** (v3 gen-40 vs gen-5): the stronger net must win clearly on the paired score, and the
  per-colour split must show it winning *as Black too* (the thing the old gate couldn't see).
- **Null** (a net vs an identical copy): paired score ≈ 0.50 with proper binomial variance *per pair*
  — and critically, near-zero paired win-differential even though each individual game is still
  ~96% White. This is the exact failure the old null test (49–51) masked.
- **Resolution check:** construct a known ~+15–30 Elo pair (e.g. v3 gen-40 vs gen-38) and confirm the
  paired gate now separates them (the old gate squashed everything under ~+100 Elo to 0.48–0.53).

---

## S6. Tests + docs

- **Unit tests** (real objects, no mocks): (a) `play_games_paired` replays an identical prefix in both
  halves of a pair (assert the first `k` plies match); (b) paired scoring rule (a) maps
  {both-win, split, both-lose} to the expected [0,1] values; (c) `regression_guard` accepts a
  0.49 score and rejects a 0.40; `always` accepts anything; `threshold` matches the old behaviour.
- **Docs:** `docs/05-EVALUATION.md` — document paired colour-swapped arena + gate modes.
  `docs/research/xl-training-scaleup.md` — **retract A7.1's "effects below ~50 Elo are now
  resolvable"** (it was colour-pinning-blind) and point to this fix. Update the `CLAUDE.md` gotcha on
  the two-tier Elo scheme if the gate semantics change.

---

## S7. Author the rerun config (gated on S1–S6)

Warm-start from v3 gen-40 again, but through the fixed instrument: `paired_arena: true`,
`gate_mode: "regression_guard"`, `epochs: 2` (S3/R3), `num_mcts_sims: 256 / gumbel_max_considered: 64
/ top_k: 64` (hold one run to judge search depth per R6), constant LR 1e-3, batch 1024. This run,
unlike v1/v2, **cannot mechanically freeze**. Read the rolling-Elo slope + a 100-game paired ladder.
Budget ~40 gens ≈ $70 at n=256 (or ~$35 at n=128 if the R6 A/B is folded in). Requires a topped-up
RunPod balance and durable storage set up first (R2-of-data-safety; consider finally standing up R2/S3
object storage per [`../guides/CLOUD-TRAINING.md`](../guides/CLOUD-TRAINING.md) §0).

---

## Not in scope (deliberately)

- **Curriculum levers (R7)** — self-play vs a pool of past checkpoints, then Pentobi seeding. The
  right *medium-term* ladder, but they sit **behind** this fix (a broken gate would freeze them too).
  Separate plan once a paired-gate rerun confirms the loop can climb again.
- **Playout-cap randomisation (R9)** — KataGo-style; IDEAS-queue material, biggest cost lever if
  n=256 survives R6.
- **LR schedules, bigger batch, bigger net, multi-GPU** — see the "deliberate non-changes" note above.
