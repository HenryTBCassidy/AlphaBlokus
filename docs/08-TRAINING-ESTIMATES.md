# Training Time Estimates

Estimated wall-clock time for Blokus Duo self-play training under various configurations. Originally derived from Mac CPU profiling (March 2026); updated 2026-05-18 with `mcts_profiling.py` numbers on a small net; **updated 2026-05-26 with production-net measurements** from `scripts/benchmark_phases.py` on the home PC.

> ⚠️ **These measured numbers are a pre-optimisation baseline.** They were taken before **F2** (precomputed-table move generation), **F3** (batched MCTS inference + virtual loss), and **F4** (conv policy head) landed. All three have since shipped, so the realised per-game cost and the move-gen/inference split below are now stale. The "post F1+F2+F3" columns and the projections are **estimates, not measurements**. Re-run `scripts/benchmark_phases.py` on the PC to get current figures before relying on them — the "Refresh after" note at the bottom lists exactly what to re-measure.

**Headline finding (2026-05-26):** the cost split depends heavily on net size. Earlier estimates assumed move generation was ~70% of MCTS time — true with the 32f×1b profiling net, **wrong** with the production 64f×4b net we actually train with. At production net size, **inference is ~50% of search time and move-gen is ~43%**. The GPU is *not* idle at production scale; the previously reported "Python move-gen dominates" picture only holds for tiny nets.

## Measured baselines

### Production net (64 filters × 4 residual blocks, 300 sims)

Measured 2026-05-26 with `scripts/benchmark_phases.py --config run_configurations/profile_baseline.json` on the RTX 3060 Ti (16 self-play + 10 arena + 10 Elo games):

| Phase | Per-game wall-clock | Per-move (mean) |
|---|---|---|
| Self-play | **54.1s** | ~1.92s (28 moves/game) |
| Arena | **53.9s** | ~1.99s (27 moves/game) |
| Elo | **55.2s** | ~2.04s (27 moves/game) |

Per-game component split (aggregated across all three phases):

| Component | Share | Time |
|---|---|---|
| Neural-net inference | **49.7%** | 970s |
| Move generation | **42.6%** | 831s |
| Other (UCB, tree ops) | 6.0% | 117s |
| Game-ended checks | 1.7% | 33s |

**This is the number to use for any Blokus self-play estimate at the current config.** Older "65–72% move-gen" claims throughout this doc and the wider codebase reflect a smaller net's profile and should be ignored at production scale.

### Small profiling net (32 filters × 1 block, 50 sims) — kept for context

The earlier `mcts_profiling.py` numbers (5 games, 50 sims/move) used a deliberately tiny net to isolate move-gen cost:

| Machine | ms/sim | ms/move | Inference fraction | Move-gen fraction |
|---------|--------|---------|--------------------|-------------------|
| Mac (Apple Silicon, CPU inference) | 2.96 | 74.0 | 26% | 67% |
| Home PC (RTX 3060 Ti, CUDA inference) | 4.95 | 247.4 | 17% | 72% |

Useful for stress-testing the move-gen path. Not representative of training runs.

## Surprising finding (qualified): the home PC vs Mac trade-off depends on net size

On the **small profiling net** the PC was ~1.7× slower per sim than the Mac — Mac's faster CPU outweighed the PC's GPU advantage on a tiny inference workload. At **production net size**, inference becomes ~half of search time, so the PC's GPU genuinely pays for itself and the PC is faster overall. The earlier "PC is slower" framing applies only to small-net profiling, not real training.

---

## Hardware

| Machine | GPU | Measured ms/sim (small profiling net) | Inference fraction | Notes |
|---------|-----|---------------------------------------|--------------------|-------|
| MacBook (M-series) | CPU only | 2.96 | 26% | Faster overall for small/medium nets — single-thread perf wins |
| Personal PC | RTX 3060 Ti | **4.95** | 17% | GPU under-utilised; CPU is the bottleneck for current configs |

Inference cost for **larger** nets has not been directly measured at full training sim counts; the projections below assume inference scales roughly with parameter count (rough, not measured).

---

## Configurations

The configs below are grouped honestly by **what's actually constraining them**:

1. **Single-PC configs** — sized to fit a chosen wall-clock budget on the 3060 Ti. Names are honest about purpose: development / testing tools, not theoretical milestones. The total games here are too small to plausibly beat Pentobi level 9; they exist to let us iterate on the *algorithm*, not the *agent*.
2. **Cloud configs (Pentobi-targeted ladder)** — sized to actually have a chance of reaching Pentobi-level strength. Each step is 10× more games than the last, so we can climb the ladder and stop when the agent first crosses Pentobi level 9 in evaluation.
3. **Reference: AGZ-equivalent** — what AlphaGo Zero / AlphaZero actually used. Included only as a yardstick — *not* a target, since we have no evidence we need that scale for Blokus Duo.

**Why no "Minimal / Moderate / Serious / Paper-scale" labels anymore:** the previous version of this doc used those headers as if they corresponded to known-good training scales. They didn't. They were ad-hoc wall-clock buckets dressed up as theoretical regimes. We have **no calibrated estimate** for how many games are needed to beat Pentobi at any level — the cloud ladder exists so we can find out experimentally.

---

### Per-game cost model (anchor for all projections below)

Wall-clock for a config is dominated by `total_games × per-game_time`. Per-game time has two main slices, measured from the 2026-05-26 benchmark:

- **Move generation** (CPU, Python): ~0.83 s/move at the current implementation. Roughly **independent of net size and sim count** (it's per-move, not per-sim-leaf).
- **Inference** (GPU): ~3.2 ms per leaf eval at 64f×4b on the 3060 Ti. Scales roughly with **net parameters × MCTS sim count** (one leaf eval per simulation that expands a new node).
- **Other** (UCB, tree bookkeeping, game-ended): ~7% of total. Folded into the model below.

Putting these together, for one game of ~28 moves:

```
per_game_time_s ≈ 28 × (move_gen_s_per_move + sims × leaf_eval_s)
               ≈ 28 × (0.83 + sims × leaf_eval_at_net_size)
```

Net scaling for inference is *rough* — we have only one measured net size (64f×4b). Bigger nets shift the inference slice in two ways: more FLOPs per leaf eval (scales with `filters² × blocks`), partially offset by better GPU utilisation at higher work per kernel launch. The projections below assume a ~10–20× inflation for the 256f×10b / 256f×19b nets, with **high uncertainty** flagged.

Hardware scaling from RTX 3060 Ti:

- **RTX 4090 (cloud):** ~2.5× inference throughput, same move-gen. So per-game time drops by roughly the inference fraction × 2.5 reduction.
- **8 parallel workers (F1 alone):** ~3–4× wall-clock improvement bound by GPU contention on inference. Move-gen scales linearly, inference contends.
- **F1 + F3 (batched inference server):** ~6–8× wall-clock improvement combined.

---

### Single-PC configs (compute-budget-bound)

All use 64f×4b (our current production net) on RTX 3060 Ti. Wall-clock includes self-play + arena + Elo + training overhead.

| Name | Total games | Sims | Wall-clock (current) | Wall-clock (post F1+F2+F3) | Purpose |
|---|---|---|---|---|---|
| **Pipeline test** | 10 (5 ep × 2 gens) | 100 | ~3 min | ~30 sec | "Does the loop run end-to-end" — used by `test_run.json` |
| **Dev iteration** | 16 (16 ep × 1 gen) | 300 | ~15 min | ~3 min | Current `profile_baseline.json`. Use for before/after benchmark comparisons. |
| **Single-PC reference** | 400 (80 ep × 5 gens) | 300 | ~11.5 h | ~2 h | Current `blokus_pc_second` shape. Used for the immutable baseline row in the F1 progress tracker. |
| **Single-PC stretch** | 2,000 (80 ep × 25 gens) | 300 | ~2.4 days | ~10 h | Maximum overnight-ish run on the PC. Suitable for algorithm iteration, not for actually trying to beat Pentobi. |
| **Single-PC stretch (bigger net)** | 2,000 (80 ep × 25 gens) | 300 | ~5 days | ~1 day | Same shape, 128f×5b net. Tests whether a bigger net helps before paying for cloud. |

None of these have enough total games to plausibly beat Pentobi level 9 — see the ladder below for that.

---

### Cloud configs (Pentobi-targeted ladder)

All assume **8× RTX 4090** with F1+F3 landed. Cost is RunPod spot pricing ($0.40/hr per 4090 = $3.20/hr for an 8× node). Wall-clock assumes ~1,000 games/hour cluster throughput after parallelism + batched inference.

| Name | Total games | Wall-clock | Cost | Reading on plausibility of beating Pentobi 9 |
|---|---|---|---|---|
| **Ladder rung 1** | 10,000 | ~10 h | **~$30** | Unlikely. ~25× the Single-PC reference. Cheap enough to climb to as a sanity check. |
| **Ladder rung 2** | 100,000 | ~4 days | **~$310** | Optimistic-case scenario: if Blokus Duo is much simpler than Go and Pentobi 9 is weak by AI standards, this might just suffice. |
| **Ladder rung 3** | 1,000,000 | ~6 weeks | **~$3,100** | Plausible-case scenario. ~1/5 of AGZ Go scale on a much simpler game. This is where my honest guess says we reach Pentobi 9. |
| **Ladder rung 4** | 10,000,000 | ~1.1 years on 8× 4090; ~6 weeks on 64× 4090 | **~$31,000** | Pessimistic case. Probably more than needed, but a fallback if rung 3 doesn't get us there. |

**How to use the ladder:** run bottom-up, evaluate the checkpoint against Pentobi at levels 1/3/5/7/9 after each rung, stop when level 9 is first beaten. Most likely outcome: we find out somewhere between rung 2 and rung 3.

The first time we go to cloud, run rung 1 first as a configuration / pipeline test — don't go straight to rung 3 with $3K of GPU time and discover an env issue.

---

### Reference: AGZ-equivalent (yardstick only, NOT a target)

| | AlphaGo Zero (Go, 2017) | AlphaZero (Chess, 2018) |
|---|---|---|
| Total self-play games | ~4.9 million | ~44 million |
| Replay buffer | 500K positions | 1M positions |
| Per-iteration games | ~25,000 | Continuous (no discrete iterations) |
| Self-play hardware | 1,600 first-gen TPUs | 5,000 first-gen TPUs |
| Total compute time | ~3 days | 9 hours |
| Wall-clock on our 8× 4090 cluster | ~7 months | ~5.5 years (single cluster) |
| Cost on our 8× 4090 cluster | ~$15,000 | ~$160,000 |

These are the numbers that "actually mastered the game" looked like. Our top-of-ladder cloud config (rung 4, 10M games) sits at roughly 1/4 of AGZ-Go scale on a much smaller and simpler game. The bet is that **Blokus Duo doesn't need anywhere near AGZ scale to beat Pentobi**: smaller branching factor late-game, shorter games (~28 moves vs ~250), scoring rather than checkmate, and Pentobi is "best open-source amateur" not "world champion class."

If that bet is wrong and we genuinely need AGZ-Go scale, this becomes a fundamentally different project (~$15K of compute, multi-month wall-clock even with optimised code). Worth knowing the gap is there.

---

## Impact of Move Generation Optimisation

Move generation is **~43% of MCTS search time at production net size**, not the ~70% the small-net profiling suggested. The remaining ~50% is inference and ~7% is everything else. Amdahl's law caps how much pure-CPU move-gen work can save you:

| Move gen speedup | Move-gen share after | Per-game wall-clock | Speedup vs baseline |
|---|---|---|---|
| 1× (current measured) | 42.6% | **54.1s** | baseline |
| 2× faster | 27.2% | ~43s | ~1.25× |
| 6× faster | 11.0% | ~35s | ~1.5× |
| 30× faster (bitboard) | 2.4% | ~31s | ~1.7× |

i.e. even a 30× move-gen speedup only gets you ~1.7× wall-clock at the current net size, because inference is the bigger slice. To get further you also need batched inference (F3 in `docs/plans/full-cycle-optimisation.md`) — together they push the ceiling much higher.

Optimisation approaches (deferred to `docs/plans/move-gen-further-optimisation.md`): pre-computed piece corners, numpy vectorisation, bitboard representation. Still worth doing but no longer the only thing — the bigger near-term win comes from parallelising self-play across CPU cores (F1) so the per-move cost is paid in parallel across many games, not improved per-game.

---

## Assumptions

What's measured vs what's projected:

| Assumption | Measured / Projected | Source |
|---|---|---|
| 28 moves per game (mean) | **Measured** | 2026-05-26 benchmark, 36 games |
| 0.83 s/move move-gen cost (3060 Ti) | **Measured** | Same benchmark |
| 3.2 ms per leaf eval at 64f×4b (3060 Ti) | **Measured** | Same benchmark |
| Inference cost scales ~linearly with `filters² × blocks` | **Projected** | Plausible CNN scaling; not benchmarked on bigger nets |
| RTX 4090 is ~2.5× the 3060 Ti for our inference shape | **Projected** | Based on FLOPs ratio; not benchmarked |
| F1 alone gives ~3-4× wall-clock improvement | **Projected** | Amdahl on the 43% move-gen slice, GPU contention on the 50% inference slice |
| F1+F3 gives ~6-8× wall-clock improvement | **Projected** | Combined effect; both pieces still to land |
| Cluster throughput ~1,000 games/hour on 8× 4090 | **Projected** | Compounds the above |
| Pentobi 9 reachable somewhere between 100K and 1M games | **Guess** | Educated; based on Blokus Duo being structurally simpler than Go and Pentobi 9 being weaker than world-champion-class |
| Training step time negligible vs self-play | **Measured** | Training was ~3 min/gen vs ~70 min self-play in `blokus_pc_second` |

**Refresh after:** F1 lands (re-benchmark with `--num-workers 8` once that flag exists), F2 lands (verify move-gen slice has shrunk), F3 lands (verify inference slice has shrunk and cross-worker batching works), and any net-size change.

---

*Originally compiled March 2026 (Mac CPU only, PC numbers estimated). Updated 2026-05-18 with measured 3060 Ti baseline (5 games, 50 sims/move, small profiling net) from `temp/mcts_profiling/pc_3060ti/`. Updated 2026-05-26 with production-net baseline (16+10+10 games, 300 sims, 64f×4b net) from `temp/profile_baseline_benchmark/` and rewritten configs section with honest labels + cloud Pentobi-targeted ladder. Re-run `scripts/benchmark_phases.py` for end-to-end phase numbers; re-run `scripts/mcts_profiling.py` for component-isolation analysis on a small net.*
