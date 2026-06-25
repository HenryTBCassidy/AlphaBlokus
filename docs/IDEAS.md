# Ideas Register

Candidate avenues we have **not** committed to yet. This is deliberately distinct from:

- **`plans/`** — implementation specs for work we *have* agreed to do (in-flight or queued).
- **`research/`** — deep investigations answering a specific question (e.g. how Pentobi's move-gen works).

An entry here is a *maybe*. When we decide to pursue one, it graduates: spawn a plan in `docs/plans/` (or a research doc if it needs investigation first), flip the status to **Promoted**, and point the row at the new doc. Nothing here is a commitment or a schedule — it's the backlog of "worth considering."

Status legend: **Idea** (raw, unexamined) · **Researching** (actively being investigated) · **Promoted** (became a plan/research doc) · **Parked** (considered, deliberately not now).

| # | Avenue | Status | One-liner |
|---|--------|--------|-----------|
| I1 | [Adaptive simulation budget](#i1-adaptive-simulation-budget) | Idea | Scale `num_mcts_sims` with branching factor / moves-left instead of a flat 300 — mostly to stop wasting sims in the low-branching endgame |
| I2 | [Evaluation-time search tuning](#i2-evaluation-time-search-tuning) | Idea | Use a stronger/exact search at eval than at train (e.g. K=1 and/or more sims) since eval cares about strength, not throughput |
| I3 | [Shared-state self-play workers](#i3-shared-state-self-play-workers) | Idea | Cut the ~2.5 GB-per-worker framework duplication so we can use all 20 cores (only 8 used now) — the highest-ceiling speed lever (self-play ≈ 86% of the cycle) |
| I4 | [Continuous (non-gated) training](#i4-continuous-non-gated-training) | Promoted | First step (rolling game-sized replay buffer + compact storage, full-pass training) promoted to [`plans/archive/replay-buffer-refactor.md`](plans/archive/replay-buffer-refactor.md); full async actor/learner stays parked |

> Ideas already captured elsewhere (not duplicated here): the conv policy head (F4) and the cross-worker inference server (F5) are done — see the [optimisation menu](plans/archive/full-cycle-optimisation.md#optimisation-menu); MCTS tree reuse, Cython move-gen and cached-valid-moves are in that plan's [Considered and set aside](plans/archive/full-cycle-optimisation.md#considered-and-set-aside) section; mixed-precision / fp16 inference is in its Out-of-scope list. Dirichlet root noise is **implemented** (`dirichlet_epsilon`/`dirichlet_alpha` in `MCTSConfig`, default-off).

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
- **Thin early coverage is better addressed by Dirichlet root noise** (already implemented — `dirichlet_epsilon`/`dirichlet_alpha` in `MCTSConfig`, default-off) than by brute-forcing more sims. AlphaZero gets away with flat sims precisely because the policy prior prunes the legal set down to a promising handful — you don't *need* to visit all 414 opening moves, you need the good ones visited enough, plus exploration insurance (the noise) for when the prior is wrong.
- **Subtlety:** varying sims per position changes how "sharp" each move's visit-count training target is, position to position. Not necessarily bad, but it makes targets less uniform across the game — be deliberate, and measure Elo impact rather than assume.

**If promoted:** likely a small plan — a branching-aware (or pieces-left-aware) sim schedule, defaulting to today's flat 300 so it's opt-in, validated by an Elo-vs-flat-300 comparison. Composes with everything else (it's orthogonal to F1/F2/F3/F4).

**Related:** Dirichlet root noise (implemented — the complementary fix for thin early coverage), `docs/02-ALGORITHMS.md` (MCTS), `temp/move_count_analysis/` (the branching data).

---

## I2. Evaluation-time search tuning

**The idea.** MCTS is a search procedure, not something the network learns, so there's no requirement to search the same way at evaluation as during training. Evaluation (vs Pentobi, arena, Elo) cares about *playing strength*, not throughput, and runs far fewer games — so we can afford a stronger, slower search there.

**Concretely:**
- Set the batched-inference knob `K = 1` at evaluation for the exact, strongest search (F3's K>1 is a training-throughput trick that trades a little search quality for GPU efficiency — see [`plans/archive/batched-inference.md`](plans/archive/batched-inference.md)).
- Optionally *raise* `num_mcts_sims` at evaluation beyond the training value for extra strength (a standard AlphaZero move).

**Status note.** This is less a research avenue than a *configuration decision* to lock once F3 lands and the K=1-vs-K=16 strength comparison is in. Captured here so it isn't forgotten. The strength comparison itself is part of the F3 plan, not this register.

**Related:** [`plans/archive/batched-inference.md`](plans/archive/batched-inference.md), `docs/05-EVALUATION.md`.

---

## I3. Shared-state self-play workers

**The observation (measured 2026-06-05).** Self-play uses 8 worker *processes*, each independently loading the full PyTorch+CUDA stack (CUDA context + cuDNN/cuBLAS libraries + allocator reservations) ≈ **2.5 GB RAM each**. The net itself is 1.5 MB — the 2.5 GB is pure **per-process framework duplication**. Result: 8 workers ≈ 20 GB, which **caps the worker count at 8 even though the PC has 20 cores** — 12 cores sit idle and the run hovers at the 28 GB WSL cap. Since self-play is **~86% of the training cycle** ([profiling report](research/profiling-report.md)), freeing those cores is the **single highest-ceiling speed lever** — up to ~2–2.5× on the whole run (the path to the ≤3 h stretch target).

**Candidate fixes (cheapest → most ambitious):**

1. **CPU-only / lightweight-inference workers.** Workers don't init CUDA — run the (tiny) net's forward pass on CPU, or via a minimal runtime (ONNX / a hand-rolled NumPy conv). Per-worker RAM drops to ~0.5 GB → run ~20 workers on the current Python 3.12 *now*. Cost: slower per-inference, but ~2.5× more game-parallelism likely nets ~1.5× self-play, and it frees the GPU. **Lowest effort.**
2. **Revisit the cross-worker inference server (F5 — already built).** One process owns the GPU/net; workers are CPU-only and send leaf batches via shared-memory IPC. F5 was parked for *no speed gain* (inference is cheap), but it *does* solve the memory/scaling problem — lightweight workers means more of them. Reuse the existing code for the memory angle.
3. **Free-threaded Python (no-GIL, 3.13t / 3.14).** The structural fix: CPU-bound MCTS runs as parallel *threads* in one process, sharing one CUDA context + net + tables. Kills the RAM floor, scales to all cores, and makes cross-worker inference batching free (F5's goal, for free). **Caveats:** needs a Python upgrade (we're on 3.12) and every C-extension (torch, numpy) must support free-threading — numpy does; torch's support has been landing through 2025 but isn't guaranteed. **Gated on a spike:** does torch+numpy run on a free-threaded build on the PC? Highest ceiling, speculative. (This is the **B5** lever in [`research/self-play-speed-investigation.md`](research/self-play-speed-investigation.md).)
4. **On-device vectorised self-play (JAX / `mctx`-style) — the "rewrite the engine" option.** Instead of N OS processes, vectorise hundreds/thousands of games onto the GPU with `vmap`/`jit` (how the big labs run massive self-play — e.g. DeepMind's `mctx`). It eliminates the multi-process model entirely: the parallelism lives on the accelerator, so there's no per-worker duplication. **But** it requires rewriting the Blokus game logic + MCTS in pure-functional, jittable JAX — a ground-up reimplementation of the imperative move-gen + tree search. Enormous effort, highest ceiling; a strategic "next engine" bet, not an incremental fix.

**What doesn't help:** putting the net/tables in shared memory saves little — the 2.5 GB is the **CUDA context + framework libraries**, not the (tiny) net or tables, and CUDA contexts are per-process. The fix has to be "one GPU/framework owner" (1–2), "one process, many threads" (3), or "no processes, vectorise on-device" (4).

**If promoted:** start with the free-threading **spike** (cheap, decides the whole approach) and/or the CPU-only-workers experiment (quick win on 3.12). Sequenced after the first real training run delivers results.

**Related:** [`research/self-play-speed-investigation.md`](research/self-play-speed-investigation.md) (B5 + Amdahl), [`plans/archive/cross-worker-inference-server.md`](plans/archive/cross-worker-inference-server.md) (F5), [profiling report](research/profiling-report.md).

---

## I4. Continuous (non-gated) training

**The observation.** Our pipeline (alpha-zero-general lineage) is the *AlphaGo Zero* model: a synchronous loop of generate `num_eps` games with the **best** net → train a **candidate** → play a 50-game arena → promote the candidate only if it scores ≥ `update_threshold` (the "gate"). **AlphaZero (2018) deleted this entirely**: one neural network, updated continuously by the learner; self-play actors always pull the latest weights and stream finished games into a rolling **replay buffer** (last ~500k games for AGZ-Go, 1M for MuZero); no best-vs-candidate, no arena gate. They found the gating machinery wasn't worth its cost once the pipeline was stable, and it parallelises perfectly (actors and learner never block each other).

**Why it could help us.** Removes the arena/Elo eval games spent purely on gating (~100 games/gen here), removes the wasted compute of training candidates that get rejected, and keeps self-play always on the freshest weights (continuous improvement vs discrete gated jumps). Decouples "how much data we hold" (buffer size) from "how often we train on each position" (sample rate) — which the current `epochs × window` design welds together.

**Why we're NOT doing it yet (the case against, for us specifically):**

- **The gate is cheap insurance in our noisy small-scale regime.** With few games, a small net, and noisy MCTS/value targets, the net genuinely *can* regress; without the gate a worse net pollutes the self-play data and can spiral. AlphaZero could drop the gate because at their scale learning was stable and the buffer huge. We're in exactly the regime the gate was designed to protect.
- **True async continuous training is hard on one GPU.** It needs self-play inference *and* the training loop contending for the same card, plus a shared cross-process replay buffer and live weight hot-reload in workers — a real rewrite that fights our synchronous `Coach` loop and its per-episode seeded reproducibility.
- **The cheap 90%** of the benefit (rolling buffer + fresh-weight self-play + an independently-tunable reuse rate) can be captured *inside* the existing synchronous loop by switching training from "E epochs over the last L generations" to "draw a fixed number of random mini-batches per generation from a rolling buffer of the last W games" — and that same change is the lazy-encoding fix that lifts the training-step OOM ceiling. So the buffer-sampling refactor is the pragmatic first step; full async continuous is the speculative end-state.

**Promoted (2026-06-23).** The replay-buffer refactor — rolling game-sized buffer + compact board storage + full-pass epoch training (use all the data), keeping the gate — is now a plan: [`plans/archive/replay-buffer-refactor.md`](plans/archive/replay-buffer-refactor.md). (A `target_reuse` sampling variant was considered and reversed on 2026-06-25 — wrong fit for a data-poor regime.) It's also the OOM fix and gives an independent reuse dial. True async continuous generation stays **parked**: it's the wrong move for a game-limited single-GPU project and only worth revisiting on hardware that can run actors and the learner concurrently. Discussed at length 2026-06-21; promoted after the storage/sampling investigation 2026-06-23.

**Related:** the data-reuse / replay-window discussion, [I1](#i1-adaptive-simulation-budget) (sims budget), and the DeepMind run-config comparison (AlphaGo → AlphaGo Zero → AlphaZero → MuZero).
