# Ideas Register

Candidate avenues we have **not** committed to yet. This is deliberately distinct from:

- **`plans/`** — implementation specs for work we *have* agreed to do (in-flight or queued).
- **`research/`** — deep investigations answering a specific question (e.g. how Pentobi's move-gen works).

An entry here is a *maybe*. When we decide to pursue one, it graduates: spawn a plan in `docs/plans/` (or a research doc if it needs investigation first), flip the status to **Promoted**, and point the row at the new doc. Nothing here is a commitment or a schedule — it's the backlog of "worth considering."

Status legend: **Idea** (raw, unexamined) · **Researching** (actively being investigated) · **Promoted** (became a plan/research doc) · **Parked** (considered, deliberately not now).

| # | Avenue | Status | One-liner |
|---|--------|--------|-----------|
| I1 | [Adaptive simulation budget](#i1-adaptive-simulation-budget) | Promoted (partially shipped) | Scale `num_mcts_sims` with branching factor instead of a flat 300 — the taper landed as `MCTSConfig.sim_schedule: "branching"` + `sims_min` ([`plans/archive/adaptive-sim-budget.md`](plans/archive/adaptive-sim-budget.md)) |
| I2 | [Evaluation-time search tuning](#i2-evaluation-time-search-tuning) | Idea | Use a stronger/exact search at eval than at train (e.g. K=1 and/or more sims) since eval cares about strength, not throughput |
| I3 | [Shared-state self-play workers](#i3-shared-state-self-play-workers) | Promoted (partially shipped) | Cut the ~2.5 GB-per-worker framework duplication — CPU workers (`worker_cuda: false`) + forkserver landed; the "vectorise on-device" endgame shipped as the jax backend |
| I4 | [Continuous (non-gated) training](#i4-continuous-non-gated-training) | Promoted | First step (rolling game-sized replay buffer + compact storage, full-pass training) promoted to [`plans/archive/replay-buffer-refactor.md`](plans/archive/replay-buffer-refactor.md); full async actor/learner stays parked |
| I5 | [Parallel Pentobi benchmark](#i5-parallel-pentobi-benchmark) | Promoted (shipped) | The Pentobi benchmark now fans games across a `spawn` worker pool (`--workers N`), each with its own net + engine — the GPU sat ~2% idle serially ([`plans/archive/parallel-pentobi-benchmark.md`](plans/archive/parallel-pentobi-benchmark.md)) |
| I6 | [Sharded multi-GPU self-play](#i6-sharded-multi-gpu-self-play) | Idea | Split the jax self-play phase across N GPUs — one pinned producer process per card streaming games into the coach's `sink`, serial loop unchanged; a wall-clock lever for when single-card runs exceed ~a week ([`research/xl-training-scaleup.md`](research/xl-training-scaleup.md) §4) |
| I7 | [Error-seeking exploration](#i7-error-seeking-exploration) | Idea | Generate data preferentially where a player is **confidently wrong** rather than uniformly — seed self-play from positions Pentobi misjudges, weight training by prior-vs-search disagreement. Measured 2026-07-28: Pentobi puts 92.9% of its ply-2 search on a reply ranked 35th of 315 |

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

**Update (2026-07-04) — partially shipped.** Exactly the "taper down, don't scale up" version above landed via [`plans/archive/adaptive-sim-budget.md`](plans/archive/adaptive-sim-budget.md): `MCTSConfig.sim_schedule: "branching"` scales the per-move budget with the root's legal-move count, clamped to `[sims_min, num_mcts_sims]` via `sim_branching_scale` — opt-in, with `"flat"` (bit-identical to old behaviour) as the default. **Remaining:** scaling sims *up* in the opening stays deliberately unimplemented (worst-ROI, per the analysis above), and the jax/Gumbel backend sidesteps the question on the production path (fixed n≈64 with Sequential Halving).

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

**Update (2026-07-04) — largely shipped, by two different routes.** Fix 1 landed as the `worker_cuda` flag (now **default `false`** — pool workers run the net on CPU, ~0.5 GB each) via [`plans/archive/lean-self-play-workers.md`](plans/archive/lean-self-play-workers.md), plus `forkserver` start-method on Linux; the throughput question it opened was settled in [`plans/archive/self-play-throughput.md`](plans/archive/self-play-throughput.md) (production = 16 all-GPU workers + K=16 batching; cores, not VRAM, are the python-path cap). Fix 4 — the "rewrite the engine" option — actually shipped as the **jax self-play backend** ([`plans/archive/jax-selfplay-pipeline.md`](plans/archive/jax-selfplay-pipeline.md), mctx exactly as sketched), which removes worker processes from the production path entirely. **Remaining parked:** free-threaded Python (fix 3) — moot unless the python engine becomes the bottleneck again.

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

---

## I5. Parallel Pentobi benchmark

**The observation.** The Pentobi benchmark played its games strictly serially against one
`pentobi-gtp` engine, so a full 1–9 ladder (e.g. 20 games × 9 levels) took ~45–70 min with the
**GPU idle at ~2%** — it's bottlenecked by Pentobi's CPU search (which grows sharply with level)
and the per-move GTP round-trip, not by inference.

**Shipped.** `scripts/pentobi_benchmark.py --workers N` fans the requested games across a `spawn`
worker pool, each worker rebuilding its own net + Pentobi engine from the config path (nothing
GPU-touching crosses the process boundary — forking a Torch/CUDA/JAX process deadlocks). Games
split into even per-worker chunks with disjoint Pentobi seeds; one pool serves all levels at once
so fast low-level chunks free their worker for slow level-8/9 chunks. `--workers 1` reproduces
the serial path bit-for-bit; `--cpu-net` scales past the ~4-worker VRAM ceiling on the 8 GB
3060 Ti. Measured ~2.9× at 4 CPU-net workers on the Mac. Plan:
[`plans/archive/parallel-pentobi-benchmark.md`](plans/archive/parallel-pentobi-benchmark.md).

**Related:** [I2](#i2-evaluation-time-search-tuning) (eval-time search), `docs/05-EVALUATION.md`
(benchmark usage), `docs/plans/archive/pentobi-harness.md` (the GTP adapter this parallelises).

---

## I6. Sharded multi-GPU self-play

**The observation.** Self-play is ~69–71% of a generation's wall-clock at `large`/`xl` net sizes
[measured, `blokus_cloud_60`], and the jax backend's compute is cleanly shardable: `run_wave` is a
jitted pure function over a leading batch dim with fully independent game slots, and nothing in
`src/` reads or sets `CUDA_VISIBLE_DEVICES`, so per-process GPU pinning is unobstructed.

**The idea.** Keep the serial generation loop; split only the self-play phase across N GPUs. N
child processes (`spawn` — forking a JAX-loaded parent is the known hazard), each pinned to one
card via `CUDA_VISIBLE_DEVICES`, each running the existing jax backend on `num_eps/N` games with
`xla_mem_fraction` raised, streaming completed games back to the coach's `sink` over an mp queue —
the same contract the CPU worker pool already implements. Coach still owns the buffer, the single
parquet write, training, and the gate; no storage changes.

**Bounds and cost** (from [`research/xl-training-scaleup.md`](research/xl-training-scaleup.md) §4):
Amdahl-capped at ~3.4× (train+arena stay serial) — 2×5090 ≈ 1.55× wall-clock at +29% $/run,
4×5090 ≈ 2.1× at +88%. Strictly a wall-clock lever, never a cost lever. Estimated 2–4 eng-days.
Worth building the first time a committed run exceeds ~a week on one card; also the natural
foundation if [I4](#i4-continuous-non-gated-training)'s parked async actor-learner ever revives.

**Related:** [I4](#i4-continuous-non-gated-training) (the async end-state), [I3](#i3-shared-state-self-play-workers)
(the CPU-era ancestor of the same instinct), `docs/research/xl-training-scaleup.md` (the costing).

---

## I7. Error-seeking exploration

**The observation.** While scoping the [v2 corpus](plans/pentobi-corpus-v2.md) we measured something
directly exploitable: at ply 2, Pentobi L9 commits **92.9% of its search visits** to a reply that an
independent evaluation ranks **35th of 315**, 0.072 below the best available — roughly 5× the
measurement noise (σ ≈ 0.014). A second opening showed the same pattern more mildly (73.9% of visits
on the 7th-best reply). It is *confidently* wrong, not merely uncertain, and the blind spot is
findable with one extra search per candidate. Evidence: `local/probes/reply_eval.txt`.

That suggests a general principle the current design only exploits passively: **spend generation
effort where a player is wrong, not uniformly over the space.** Two applications, in increasing
ambition:

1. **Seed self-play from positions Pentobi misjudges** (Henry, 2026-07-28). Rather than starting RL
   self-play from the standard position, start a fraction of games from the openings where Pentobi's
   own choice is measurably suboptimal. Those are precisely the positions where our net can build an
   advantage the teacher will not defend well — and, since Pentobi is the benchmark, the positions
   worth being strong in.
2. **Weight training by disagreement.** Upweight positions where the policy prior and the search
   result diverge most; these are where the prior has the most to learn.
3. **Steer the opening DAG toward disagreement.** An alternative expansion policy to the
   visit-proportional one in the v2 plan: expand where shallow and deep judgements disagree, rather
   than where Pentobi is confident.

**"How do you know you are confidently wrong?"** You cannot, from the model alone — that is what a
blind spot *is*. You need an external referee, and the useful insight is that several cheap ones
already exist. In increasing cost:

| Referee | Signal | Cost | Prior art |
|---|---|---|---|
| **Prior vs search** | The policy prior's top move differs from the move the search picks after N sims | **Free** — both already computed every move of self-play | KataGo's policy-surprise sample weighting |
| **Value vs outcome** | Predicted +0.8, lost the game | **Free** — known at game end | [Prioritised experience replay](https://arxiv.org/abs/1511.05952) (Schaul et al. 2015) samples by TD error |
| **Seed disagreement** | Two searches of the same position pick different moves | One extra search | Observed here: two L9 ply-1 searches disagreed on the best first move |
| **Shallow vs deep** | Play the move, search the *resulting* position, compare its backed-up value against siblings | One search per candidate (~23 s at L9) | The method used for the measurement above |
| **Cross-agent** | Our net and Pentobi disagree, and the game says who was right | Free at eval time | — |

The distinction that matters: **uncertainty** is a flat distribution or high variance across
referees; **confident wrongness** is a *peaked* distribution plus large error against the referee.
The second is rarer, more valuable, and the thing worth hunting. Uncertainty sampling finds the
former; only a referee finds the latter.

**The risk, and it is a real one.** [Wang et al. 2022](https://arxiv.org/abs/2211.00241) trained
adversarial policies that beat superhuman Go AIs including KataGo — by finding blind spots and
playing objectively *terrible* Go that the victim misevaluated. Optimising against a specific
opponent's errors produces a net that beats *that opponent*, not a net that plays well. Our stated
goal is literally "beat Pentobi level 9", so that trade may be acceptable — but it should be a
conscious choice, and general strength (pooled self-play Elo, and the full L1–L9 ladder rather than
the top rungs alone) needs watching as the tell.

**Status.** Not committed. The v2 corpus already collects the raw material passively — every stored
position carries Pentobi's full distribution *and* an outcome-grounded value label, so the
disagreement is queryable after the fact (the store's `edge_disagreement` view). The natural first
step is the base-rate probe already scheduled as V2 of the v2 plan: thirty nodes across depths 2–6
tell us whether ply-2's blunder is a systematic property or two unlucky positions. If the rate is
high, this graduates to a plan; if it is low, there is nothing to hunt.

**Related:** [`plans/pentobi-corpus-v2.md`](plans/pentobi-corpus-v2.md) (V2's base-rate probe, V16's
net-in-the-loop phase), [`research/corpus-generation-literature.md`](research/corpus-generation-literature.md)
§8 (why visit distributions are not move-quality distributions).
