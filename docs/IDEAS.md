# Ideas Register

Candidate avenues we have **not** committed to yet. This is deliberately distinct from:

- **`plans/`** — implementation specs for work we *have* agreed to do (in-flight or queued).
- **`research/`** — deep investigations answering a specific question (e.g. how Pentobi's move-gen works).

An entry here is a *maybe*. When we decide to pursue one, it graduates: spawn a plan in `docs/plans/` (or a research doc if it needs investigation first), flip the status to **Promoted**, and point the row at the new doc. Nothing here is a commitment or a schedule — it's the backlog of "worth considering."

Status legend: **Idea** (raw, unexamined) · **Researching** (actively being investigated) · **Promoted** (became a plan/research doc) · **Parked** (considered, deliberately not now).

| # | Avenue | Status | One-liner |
|---|--------|--------|-----------|
| I1 | [Adaptive simulation budget](#i1-adaptive-simulation-budget) | Idea | Scale `num_mcts_sims` with branching factor / moves-left instead of a flat 300 — mostly to stop wasting sims in the low-branching endgame |
| I2 | [Dirichlet root noise](#i2-dirichlet-root-noise) | Idea | Add exploration noise to root priors so good moves with weak priors still get explored — the standard AlphaZero fix for thin early-game coverage |
| I3 | [Evaluation-time search tuning](#i3-evaluation-time-search-tuning) | Idea | Use a stronger/exact search at eval than at train (e.g. K=1 and/or more sims) since eval cares about strength, not throughput |

> Ideas already captured elsewhere (not duplicated here): mixed-precision / fp16 inference, cross-worker inference server, MCTS tree reuse (F4), cached-valid-moves (F6), Cython move-gen (F5) — all live in the [optimisation menu](plans/full-cycle-optimisation.md#optimisation-menu).

---

## I1. Adaptive simulation budget

**The observation.** `num_mcts_sims` is a flat 300 for the whole game. But the Blokus Duo branching factor swings enormously over a game (measured in `temp/move_count_analysis/`):

| Game phase (ply) | Mean legal moves | 300 sims is… |
|---|---|---|
| 0–8 (opening) | ~410–480 | thinner than one visit per legal move |
| 9–12 | ~300–500 | roughly one visit per move |
| 13–16 | ~100–220 | comfortable coverage |
| 17–20 | ~40–130 | generous |
| 21+ (endgame) | <30 (often <10) | gross overkill — re-walks a tiny tree hundreds of times |

So a flat 300 is simultaneously *thin* in the opening and *wasteful* in the endgame.

**The idea.** Make the per-move sim budget a function of branching factor (or moves-left / pieces-remaining), rather than a constant.

**Current thinking (from the F3 discussion, 2026-06-01):**

- **Tapering sims *down* late-game is the clean win.** At turn 21+ with <30 legal moves, 300 sims is absurd; 50–100 would give equal-or-better coverage and reclaim real wall-clock. Low risk, easy to measure.
- **Scaling sims *up* early-game is expensive and uncertain.** The opening is (a) the most expensive place to add sims (high branching → costly move-gen + more inference) and (b) arguably the least decision-critical (many roughly-equivalent developing moves). Worst ROI.
- **Thin early coverage is better addressed by [I2 (Dirichlet noise)](#i2-dirichlet-root-noise)** than by brute-forcing more sims. AlphaZero gets away with flat sims precisely because the policy prior prunes the legal set down to a promising handful — you don't *need* to visit all 414 opening moves, you need the good ones visited enough, plus exploration insurance for when the prior is wrong.
- **Subtlety:** varying sims per position changes how "sharp" each move's visit-count training target is, position to position. Not necessarily bad, but it makes targets less uniform across the game — be deliberate, and measure Elo impact rather than assume.

**If promoted:** likely a small plan — a branching-aware (or pieces-left-aware) sim schedule, defaulting to today's flat 300 so it's opt-in, validated by an Elo-vs-flat-300 comparison. Composes with everything else (it's orthogonal to F1/F2/F3/F4).

**Related:** [I2](#i2-dirichlet-root-noise) (complementary fix for early coverage), `docs/02-ALGORITHMS.md` (MCTS), `temp/move_count_analysis/` (the branching data).

---

## I2. Dirichlet root noise

**The idea.** AlphaGo Zero / AlphaZero add Dirichlet noise to the root node's priors during self-play for exploration:

```
P(s,a) = (1-ε)·p_a + ε·η_a    where η ~ Dir(α)
```

This guarantees that even moves the network's policy head rates poorly still get *some* root visits, so a genuinely good move with a (wrongly) weak prior isn't permanently starved of exploration. It's the standard insurance against a bad early-training prior.

**Why it matters for us.** We don't do this yet (noted in `docs/02-ALGORITHMS.md` as "not yet implemented"). It's the more principled answer to the thin early-game coverage that [I1](#i1-adaptive-simulation-budget) also touches: with ~414 legal opening moves, 300 sims, and an untrained prior, exploration is thin and we currently have no forced-exploration mechanism at the root. Most relevant once real Blokus training starts and early generations have weak priors.

**Open parameters:** ε (mixing weight, AlphaZero used 0.25) and α (Dirichlet concentration, scaled to the action space — AlphaZero used ~0.03 for Go's 362 actions; our 17,837-action space needs its own value). Self-play only — never at evaluation.

**If promoted:** small, well-scoped change in `MCTS` root handling, gated behind a config flag, with the usual K=1-style "off = bit-identical to today" safety. Should land before (or alongside) the first serious Blokus training run.

**Related:** [I1](#i1-adaptive-simulation-budget), `docs/02-ALGORITHMS.md:489`, the F3 plan's out-of-scope note.

---

## I3. Evaluation-time search tuning

**The idea.** MCTS is a search procedure, not something the network learns, so there's no requirement to search the same way at evaluation as during training. Evaluation (vs Pentobi, arena, Elo) cares about *playing strength*, not throughput, and runs far fewer games — so we can afford a stronger, slower search there.

**Concretely:**
- Set the batched-inference knob `K = 1` at evaluation for the exact, strongest search (F3's K>1 is a training-throughput trick that trades a little search quality for GPU efficiency — see [`plans/batched-inference.md`](plans/batched-inference.md)).
- Optionally *raise* `num_mcts_sims` at evaluation beyond the training value for extra strength (a standard AlphaZero move).

**Status note.** This is less a research avenue than a *configuration decision* to lock once F3 lands and the K=1-vs-K=16 strength comparison is in. Captured here so it isn't forgotten. The strength comparison itself is part of the F3 plan, not this register.

**Related:** [`plans/batched-inference.md`](plans/batched-inference.md), `docs/05-EVALUATION.md`.
