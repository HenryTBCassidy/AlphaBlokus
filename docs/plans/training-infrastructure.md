# Training Infrastructure

Plan for setting up compute infrastructure for AlphaBlokus training runs —
both TicTacToe validation and eventual BlokusDuo training.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| I1 | Set up SSH access from MacBook to personal GPU PC | ~1 hour | High | |
| I2 | Validate TicTacToe training on GPU | 15 min | High | |
| I3 | Design BlokusDuo run configurations (when move gen lands) | 30 min | Medium | |

Detailed setup steps for I1 are in Claude memory (not in this repo for security
reasons — contains IPs and usernames). Reference: `memory/reference_gpu_setup_plan.md`.
Ask Claude to pull up the "GPU setup plan" when ready.

Setup: Tailscale (encrypted mesh VPN) + WSL2 + OpenSSH + tmux. 10 steps, all
done at home before travelling. Designed for connecting from anywhere (hotel WiFi,
phone hotspot, etc.) with zero port forwarding and key-only auth.

---

## Compute Options Evaluated

### Available Hardware

| Option | GPU | VRAM | Best For | Cost |
|--------|-----|------|----------|------|
| **MacBook (Apple Silicon)** | MPS | Shared | Smoke tests, report generation, development | Free |
| **Personal PC** | RTX 3060 Ti | 8 GB | TicTacToe runs, early Blokus training | Free (electricity) |
| **Cloud GPU (if needed)** | Various | 16-80 GB | Long Blokus runs, scaling up | $0.15-2.00/hr |

### Cloud GPU Providers (researched 2026-03-18)

**Budget tier ($0.15-0.40/hr) — consumer GPUs on marketplace platforms:**
- **Vast.ai** — RTX 3090/4090 (24 GB) at $0.15-0.40/hr. Cheapest option. Peer-to-peer hosts,
  variable reliability. Interruptible instances are even cheaper but can be reclaimed. Need
  robust checkpointing (we already have this).
- **RunPod Community Cloud** — RTX 4090 (24 GB) at ~$0.40/hr. Per-second billing, SSH access,
  more reliable than Vast.ai. A 24-hour run costs ~$10.

**Mid tier ($0.60-1.50/hr) — professional cloud with pre-configured environments:**
- **Lambda Labs** — V100 ($0.63), A10 ($0.86), A6000 48 GB ($0.92), A100 40 GB ($1.48).
  PyTorch pre-installed, SSH, good CPU allocation. Most polished developer experience.
- **RunPod Secure Cloud** — RTX 4090 ($0.69-0.74), A100 80 GB ($2.19-2.49). T3/T4 data
  centres, higher reliability than community cloud.
- **AWS Spot** — g5.xlarge A10G ($0.35-0.50 spot). Reliable infra but only 4 vCPUs, which
  may bottleneck self-play. More setup ceremony than Lambda/RunPod.

**Free/cheap tiers:**
- **Modal** — $30/mo free credits, per-second billing, T4 to A100. ~50 hrs of T4 or ~12 hrs
  of A100 per month at zero cost. Serverless model requires slight adaptation of entry point.
- **Google Colab Pro** (~$10/mo) — T4/V100, unreliable for multi-hour runs. Not recommended.
- **Kaggle** — 30 hrs/week T4, no SSH. Not viable for self-play training loops.

### Ruled Out

- **Snowflake** — No SSH access to containers, opaque credit-based pricing, designed for data
  pipelines and Snowpark ML, not custom RL training loops.
- **Managed platforms (SageMaker, Vertex AI, Azure ML)** — 15-20% markup over raw compute for
  MLOps features we don't need. Overkill for a solo project.

### AlphaZero Workload Characteristics

- **Self-play phase** is CPU-heavy (MCTS tree search). GPU used only for neural net inference
  during MCTS. Wants many CPU cores + 1 GPU.
- **Training phase** is GPU-heavy (batch gradient descent). Wants a fast GPU.
- **Alternating pattern** means the GPU is idle during much of self-play.
- **Spot/interruptible is fine** — Coach checkpoints every generation. Losing a spot instance
  costs at most one generation of self-play games.
- **8 GB VRAM (3060 Ti) is sufficient** for our ResNet on 44x14x14 input at reasonable batch sizes. Cloud 24 GB cards
  give headroom for larger batch sizes if needed.

### Recommended Phasing

1. **Now (TicTacToe validation):** MacBook for smoke tests, personal PC for real runs.
2. **Blokus early training:** Personal PC. 12 GB VRAM is enough.
3. **Blokus scale-up (100+ generations):** Rent RunPod/Vast.ai RTX 4090 ($0.40/hr), ~$10-20
   for a multi-day run.
