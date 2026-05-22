# Compute Options

What hardware can we use, what does it cost, and how long will training take on each? This doc compares the realistic options for AlphaBlokus training runs.

Wall-clock estimates per config-scale (Minimal / Moderate / Serious / Paper-scale) are the source-of-truth in [`08-TRAINING-ESTIMATES.md`](08-TRAINING-ESTIMATES.md). This doc translates those numbers into **per-option cost** and gives a recommendation for which option to use when.

---

## Workload characteristics

Three facts shape the choice of compute:

1. **Self-play is CPU-bound, not GPU-bound.** Across measured runs, move generation is **65–72%** of MCTS search time and neural-net inference is only **17–27%**. A faster GPU buys less than you'd expect — the Python-side move generator is the bottleneck. See `08-TRAINING-ESTIMATES.md` for the measured breakdown.
2. **Training spikes are short.** Per generation: self-play takes the vast majority of wall-clock; the actual neural-net gradient descent is on the order of seconds. The GPU is idle most of the time during a generation cycle.
3. **Spot / interruptible compute is fine.** `Coach` checkpoints every generation, so losing an instance costs at most one generation of self-play games.

The implication: optimal shape is **many CPU cores + 1 modest GPU**, not "the biggest GPU we can find." This rules in commodity 24 GB cloud cards (RTX 3090 / 4090 / A6000) and rules out splurging on H100s.

---

## Local options

| Machine | GPU | VRAM | What it's good for | Validated? |
|---------|-----|------|---------------------|------------|
| **MacBook (Apple Silicon)** | MPS, but used in CPU mode | shared | Smoke tests, HTML report rendering, development iteration | ✅ |
| **Personal PC** | RTX 3060 Ti | 8 GB | TicTacToe runs, early Blokus iteration | ✅ |

Both are zero marginal cost (electricity aside). The PC is **~1.7× slower per simulation than the Mac** at small/medium net sizes because of WSL2 syscall overhead and a slower CPU vs the Mac M-series. The PC only outpaces the Mac at Serious-scale net sizes where GPU inference becomes a meaningful fraction of the loop.

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

Combining the wall-clock estimates in [`08-TRAINING-ESTIMATES.md`](08-TRAINING-ESTIMATES.md) with the cloud hourly rates above. Cloud numbers assume 1.5–2× speedup over the home PC (faster CPUs + 24 GB cards), partially offset by the move-gen bottleneck — so the speedup is real but smaller than raw GPU FLOPs would suggest.

| Config scale | Home PC (free) | Vast.ai ~$0.30/hr | RunPod Community ~$0.40/hr | Lambda A10 ~$0.86/hr |
|--------------|----------------|---------------------|----------------------------|----------------------|
| **Minimal** (50 gens, 128f×5b, 100 sims) | ~6.5 hr | ~$1 | ~$2 | ~$3 |
| **Moderate** (100 gens, 128f×5b, 200 sims) | ~2.4 days | ~$13 | ~$17 | ~$36 |
| **Serious** (200 gens, 256f×10b, 400 sims) | ~18 days | ~$95 | ~$125 | ~$270 |
| **Paper-scale** (200 gens, 256f×19b, 800 sims) | ~165 days (not feasible) | ~$870 (still ~80 days) | ~$1150 | ~$2500 |

Paper-scale on a single GPU isn't feasible regardless of provider — it needs parallel self-play across multiple machines, which we don't have yet (deferred work, see archived plans).

---

## Recommended phasing

1. **TicTacToe iteration (current state):** MacBook for smoke tests, personal PC for full runs. Both validated. No cloud needed.
2. **Blokus early iteration:** Personal PC. 8 GB VRAM is enough for the Minimal/Moderate configs. Multi-day runs over Tailscale + systemd-user.
3. **Blokus Moderate-scale exploration:** Switch to RunPod Community RTX 4090 (~$15–20 for a multi-day run). Per-second billing is convenient for iterative tuning.
4. **Blokus Serious-scale (aim to beat Pentobi):** Vast.ai 4090 at $0.30/hr if reliability is acceptable (~$95), or RunPod Secure Cloud 4090 at $0.69/hr if not (~$200). Always-on checkpointing means we can tolerate spot interruptions.
5. **Paper-scale (future):** Would require parallel self-play infrastructure that isn't built yet — covered in `docs/plans/move-gen-further-optimisation.md` and the deferred R30 in the archived reporting overhaul plan.

---

## Optimisations that reduce the bill

Two levers shift the cost-vs-quality frontier without changing provider:

1. **Move generation rewrite** — bitboard representation could be ~30× faster, taking Serious-scale from 18 days to ~2 days (see breakdown in `08-TRAINING-ESTIMATES.md`). Highest-leverage perf work because the GPU is currently bottlenecked on Python. Deferred to [`docs/plans/move-gen-further-optimisation.md`](plans/move-gen-further-optimisation.md).
2. **Parallel self-play across CPU cores** — straightforward worker pool over the `Player` abstraction. Multiplies throughput by the number of cores. On an 8-core 4090 instance, Serious-scale drops from ~9 days to ~1–2 days. Deferred (see R30 in archived reporting overhaul plan).

Both are worth doing before any multi-week cloud commit.
