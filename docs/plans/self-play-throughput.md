# Self-Play Throughput — Roadmap

A **process plan** (not a single coding task) for getting maximum self-play game
throughput on the home PC. This is the highest-leverage lever for model strength:
more self-play data per unit time → a stronger net, and self-play is ~86% of the
training cycle. It captures the strategy decided after the benchmarking in
[lean-self-play-workers](archive/lean-self-play-workers.md) and
[MCTS memory reduction](archive/mcts-memory-reduction.md); each phase below gets
its own subplan.

> Supersedes the throughput intent of `lean-self-play-workers.md` (L3–L7), which
> assumed "more CPU workers = more speed." Benchmarking disproved that — see below.
> The memory work from that branch is done and banked; this roadmap is the sequel.

---

## What we learned (the evidence behind this plan)

Measured on the PC (RTX 3060 Ti 8 GB; i5-13600KF = **6 P-cores + 8 E-cores = 14C/20T**; 32 GB):

| Config | games/s | Note |
|--------|---------|------|
| gpu8 (8 workers, own GPU net) | **~0.756** | best measured; capped at 8 by GPU VRAM |
| cpu16 (16 workers, CPU inference) | 0.211 | ~3.6× slower — CPU inference is the bottleneck |
| server 16 / 18 / 20 (central GPU) | 0.476 / 0.498 / 0.493 | **plateaus by ~16**, never beats gpu8 (matches F5's old 0.99×) |

Three conclusions:
1. **Memory is solved and no longer the cap.** Sparse MCTS cut the per-worker tree **1.9 GB → 19 MB** ([archive/mcts-memory-reduction.md](archive/mcts-memory-reduction.md)); 20 workers now fit in ~16 GB.
2. **Self-play throughput is GPU-bound at ~gpu8 level.** Neither many CPU workers nor the central server beats 8 GPU workers — the 8 GB GPU + the per-worker CPU MCTS, not worker count, is the ceiling.
3. **Two levers remain, and they compound** (see strategy).

## The strategy

1. **Speed the per-worker CPU MCTS** — the ~⅔ of self-play that *isn't* GPU inference (UCB selection, move-gen, tree work). Faster CPU work → every worker does more, and it compounds across *every* config. → **B1 (vectorise `_select_action`)**.
2. **Use both compute resources at once** — keep the GPU saturated with GPU-fed workers *and* fill the spare cores with CPU-inference workers (GPU workers block while waiting on the GPU, freeing their cores). → **hybrid workers**.
3. Do it on the **right substrate** — native Linux: clean warmup-free measurement, `forkserver`, stable unattended multi-hour runs, PyTorch-first-class. Move *before* tuning so we tune once.

---

## Checklist (phases)

| # | Phase | Subplan | Effort | Done |
|---|-------|---------|--------|------|
| P1 | **Migrate to native Linux** (dual-boot Ubuntu) — clean substrate before tuning | [linux-migration.md](archive/linux-migration.md) | ~half day | ✔ 2026-06-12 |
| P2 | **B1: vectorise `_select_action`** (UCB) — replace the per-move Python loop with one numpy op; compounds across all configs; OS-agnostic | [p2-vectorise-select-action.md](p2-vectorise-select-action.md) | ~0.5 day | ✔ 2026-06-15 |
| P3 | **Hybrid GPU+CPU workers** — per-worker device assignment (`num_gpu_workers`); tune the GPU/CPU split + threads-per-worker | *(subplan TBD)* | ~1–2 days | |
| P4 | **Clean throughput measurement** — warmth-free games/s + minutes-per-1000 for the candidate configs on Linux; pick the production default | *(subplan TBD)* | ~2 hr | |

Subplans for P2–P4 are written *after* P1 (we tune on Linux). P1's subplan is below-linked and ready now.

---

## P1. Migrate to native Linux

The substrate decision, forced by: WSL was unstable (the "Catastrophic failure"
saga), and native Windows is `spawn`-only + warmup-heavy, which muddies every
throughput measurement. Native Linux fixes all three and is the eventual home.
Full runbook + risks + time estimate: **[linux-migration.md](archive/linux-migration.md)**.

## P2. B1 — vectorise `_select_action`

The biggest CPU hot-spot in self-play (~22–35%, per
[self-play-speed-investigation.md](../research/self-play-speed-investigation.md))
is `_select_action`'s per-move Python `for`-loop computing a UCB score with
`math.sqrt` for every legal move. Replace it with a single vectorised numpy
computation over the (now sparse, post-M5) aligned `acts`/`priors` arrays. Cheap,
no toolchain, OS-agnostic, and it speeds the CPU work that limits CPU workers
*and* the tree-search half of GPU workers. Must stay bit-identical (the
parallel-/movegen-determinism tests are the guard). Subplan (complete):
**[p2-vectorise-select-action.md](p2-vectorise-select-action.md)**.

## P3. Hybrid GPU + CPU workers

Make `worker_cuda` per-worker instead of global: add `num_gpu_workers`, give each
worker a stable id (a shared counter, as the inference server already does), and
the first N build a GPU net while the rest build CPU nets. Then tune the split
(e.g. 6–8 GPU + 12–16 CPU) and threads-per-worker (`~20 ÷ num_workers`, since the
chip is 14C/20T and the 8 E-cores are slower). GPU workers block during GPU-wait,
freeing cores for the CPU workers, so slight oversubscription is fine. This is the
one config that can beat gpu8 on this box. Subplan written at P3 start.

## P4. Clean throughput measurement

On Linux (warmup amortised by `forkserver`), measure **games/s and
minutes-per-1000-games** for gpu8 vs the tuned hybrid, pick the production default,
and record it. This is the number the whole roadmap is for.

---

## Not pursuing (and why)

- **More pure-CPU workers** — CPU inference is ~3.6× slower than GPU (cpu16 = 0.21 g/s « gpu8 0.756). Dead end on its own.
- **Central inference server as the primary path** — plateaus by ~16 workers (~0.49 g/s), never beats gpu8 (GPU-bound), matching F5's prior "~0.99×". Kept as a config flag, not the strategy.
- **Free-threading (B5) / JAX-`mctx` rewrite (B4-class)** — highest ceiling but speculative / a ground-up reimplementation. Revisit only if P1–P4 plateau and we still need more.
- **A faster/bigger GPU (cloud)** — the real lever beyond this box's ceiling; out of scope for now but the natural escalation if the 3060 Ti caps us.
