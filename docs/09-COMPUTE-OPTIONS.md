# Compute Options

What hardware can we use, what does it cost, and how long will training take on each? This doc compares the realistic options for AlphaBlokus training runs.

Wall-clock estimates per config (Single-PC reference / Single-PC stretch / cloud ladder rungs 1–4 / AGZ-equivalent reference) are the source-of-truth in [`08-TRAINING-ESTIMATES.md`](08-TRAINING-ESTIMATES.md). This doc translates those numbers into **per-option cost** and gives a recommendation for which option to use when.

---

## Workload characteristics

Three facts shape the choice of compute:

1. **Self-play splits the cost roughly evenly between CPU move-gen and GPU inference** at production net size (64f×4b, 300 sims): ~43% move generation, ~50% neural-net inference, ~7% other (see `08-TRAINING-ESTIMATES.md` for the 2026-05-26 measurement). The earlier "65–72% move-gen" claim came from a small profiling net and doesn't apply at training scale.
2. **Training spikes are short.** Per generation: self-play takes the vast majority of wall-clock; the actual neural-net gradient descent is on the order of seconds. The GPU is hit hard during self-play inference but idle during the brief training step.
3. **Spot / interruptible compute is fine.** `Coach` checkpoints every generation, so losing an instance costs at most one generation of self-play games.

The implication: optimal shape is **many CPU cores AND a decent GPU** (not "lots of CPU + cheapest GPU" as the older bottleneck claim implied). Inference is genuinely half the work at production net size. This still rules in commodity 24 GB cloud cards (RTX 3090 / 4090 / A6000) over H100s — the GPU just needs to keep pace with batched MCTS inference, which a 4090 does easily — but it argues against under-provisioning the GPU side.

---

## Local options

| Machine | GPU | VRAM | What it's good for | Validated? |
|---------|-----|------|---------------------|------------|
| **MacBook (Apple Silicon)** | MPS, but used in CPU mode | shared | Smoke tests, HTML report rendering, development iteration | ✅ |
| **Personal PC** | RTX 3060 Ti | 8 GB | TicTacToe runs, early Blokus iteration | ✅ |

Both are zero marginal cost (electricity aside). The PC is **~1.7× slower per simulation than the Mac on a tiny profiling net** (where inference is negligible and WSL2 + slower CPU dominate), but **faster than the Mac at production net size** (64f×4b and up) where inference is half the work and the GPU pays for itself. Use the PC for Blokus training; the Mac is fine for code iteration and TTT-scale runs.

**Remote training on the PC** is set up over Tailscale + WSL2 + systemd-user; see [`docs/guides/REMOTE-TRAINING.md`](guides/REMOTE-TRAINING.md) for the runbook.

---

## Cloud options (researched 2026-03-18)

Prices below are USD per hour for on-demand GPU instances. Spot / preemptible pricing is typically 50–70% of on-demand. All prices subject to drift — refresh before committing to a multi-day run.

### Budget tier ($0.15 – $0.40/hr)

Commodity 24 GB GPUs from marketplace providers. Best $/hr for our workload.

| Provider | GPU | VRAM | Hourly | Notes |
|----------|-----|------|--------|-------|
| **Vast.ai** | RTX 3090 / 4090 | 24 GB | $0.15 – $0.40 | Cheapest option. Peer-to-peer hosts, variable reliability. Interruptible instances even cheaper. |
| **RunPod Community Cloud** | RTX 4090 | 24 GB | ~$0.40 | Per-second billing, SSH access. More reliable than Vast.ai. |

### Mid tier ($0.60 – $2.50/hr)

Professional cloud with pre-configured PyTorch environments. Faster CPUs, better networking, datacenter reliability. Pay for ergonomics, not raw FLOPs.

| Provider | GPU | VRAM | Hourly |
|----------|-----|------|--------|
| **Lambda Labs** | V100 | 16 GB | $0.63 |
| Lambda Labs | A10 | 24 GB | $0.86 |
| Lambda Labs | A6000 | 48 GB | $0.92 |
| Lambda Labs | A100 | 40 GB | $1.48 |
| **RunPod Secure Cloud** | RTX 4090 | 24 GB | $0.69 – $0.74 |
| RunPod Secure Cloud | A100 | 80 GB | $2.19 – $2.49 |
| **AWS Spot** | g5.xlarge A10G | 24 GB | $0.35 – $0.50 spot | Only 4 vCPUs — likely CPU-bottlenecked for our workload. |

### Free / very cheap tiers

| Provider | Pricing | Constraints |
|----------|---------|-------------|
| **Modal** | $30/mo free credits, then per-second | Serverless model needs slight entry-point adaptation. ~50 hrs of T4 or ~12 hrs of A100 per month at zero cost. |
| **Google Colab Pro** | ~$10/mo | Unreliable for multi-hour runs. Not recommended. |
| **Kaggle** | Free, 30 hrs/week T4 | No SSH access. Not viable for self-play training loops. |

### Ruled out

- **Snowflake** — No SSH access to containers, opaque credit-based pricing, designed for data pipelines and Snowpark ML, not custom RL loops.
- **Managed platforms (SageMaker, Vertex AI, Azure ML)** — 15–20% markup over raw compute for MLOps features we don't need. Overkill for a solo project.

---

## Estimated cost per training run

Cost = `total_games × per-game_cost`. Per-game cost compounds GPU-hour rate with cluster throughput. The canonical configs and their wall-clock projections live in [`08-TRAINING-ESTIMATES.md`](08-TRAINING-ESTIMATES.md). This table converts them to cost across providers.

Assumptions: post-F1+F3 throughput on an 8× 4090 node is ~1,000 games/hour (see the per-game cost model in 08). Single-4090 throughput is ~125 games/hour. Costs below assume each provider for either an 8× node (large runs) or a single card (small runs).

| Config | Total games | Single 4090 (RunPod ~$0.40/hr) | 8× 4090 (RunPod ~$3.20/hr) | A100 80GB (Lambda ~$1.99/hr) |
|---|---|---|---|---|
| **Dev iteration** (single-PC) | 16 | ~$0.05 | ~$0.05 | ~$0.10 |
| **Single-PC reference** | 400 | ~$1 | ~$1 | ~$2 |
| **Single-PC stretch** | 2,000 | ~$6 | ~$6 | ~$10 |
| **Ladder rung 1** | 10,000 | ~$30 (80 h on one card) | ~$30 | ~$50 |
| **Ladder rung 2** | 100,000 | ~$320 (~33 days) | ~$310 (~4 days) | ~$510 (~3 days) |
| **Ladder rung 3** | 1,000,000 | ~$3,200 (~333 days, not feasible single-card) | ~$3,100 (~6 weeks) | ~$5,100 (~4 weeks) |
| **Ladder rung 4** | 10,000,000 | (not feasible single-card) | ~$31,000 (~1.1 years on 8×) | ~$51,000 (~9 months) |
| **Reference: AGZ Go** | 4.9M | — | ~$15,000 (~7 months on 8×) | ~$25,000 (~5 months) |

Ladder rungs 3 and 4 want a bigger cluster than 8× cards for wall-clock reasons — 64× 4090s would take rung 3 down to a long weekend and rung 4 to ~6 weeks. The cost stays the same (more cards × fewer hours); only wall-clock changes.

---

## Recommended phasing

1. **TicTacToe iteration (done):** MacBook for test runs, personal PC for full runs. No cloud needed.
2. **Blokus algorithm iteration (now):** Personal PC for the Single-PC reference / stretch configs. Multi-day runs over Tailscale + systemd-user. This is where F1/F2/F3 development and validation happens.
3. **First cloud touch (after F1+F3 land):** Run **Ladder rung 1** (10K games, ~$30) on RunPod Community 4090 to validate the cloud pipeline end-to-end. Treat this as a pipeline test, not an attempt to beat Pentobi.
4. **First serious Pentobi attempt:** **Ladder rung 2** (100K games, ~$310). Evaluate the checkpoint against Pentobi 1/3/5/7/9. If it beats 9 — we're done, sooner than expected. If it beats 5 or 7 — climb to rung 3.
5. **Pentobi-9-targeted run:** **Ladder rung 3** (1M games, ~$3,100). My honest expectation is this is where the agent crosses level 9. If not, climb to rung 4.
6. **Fallback if rung 3 stalls:** Either rung 4 (10M games, ~$31K) or rethink. At rung-4 cost it's worth pausing to check whether net architecture, search depth, or training stability is the real ceiling rather than just throwing more games at it.

We never need to commit to a big cloud run blind. Every step has a fixed cost and a clear go/no-go signal (Pentobi level beaten).

---

## Optimisations that reduce the bill

The optimisations from [`docs/plans/full-cycle-optimisation.md`](plans/full-cycle-optimisation.md) all reduce per-game cost, which directly reduces every cloud config's price:

1. **F1 (parallel self-play, ~3-4× alone)** — multiplies wall-clock throughput by workers. Same total compute, drastically less wall-clock. On hourly cloud rates, the cost is unchanged but the time to result drops 3-4×.
2. **F3 (batched inference)** — combined with F1 gives ~6-8× wall-clock improvement. *And* this one actually saves compute, because batched GPU calls are far more efficient than batch-of-1. Probably ~30% cost reduction on top of the wall-clock win.
3. **F2 (bitboard move-gen)** — reduces the 43% move-gen slice to ~10%. ~1.5× compute reduction, applies to single-PC and cloud equally.

Doing all three before any cloud run > $1K is the obviously right call. Combined they should give roughly 2× cost reduction *and* 6-8× wall-clock reduction, so a rung-3 attempt drops from ~$3K and ~6 weeks to ~$1.5K and ~1 week.
