# Self-Play Throughput — Roadmap

A **process plan** (not a single coding task) for getting maximum self-play game
throughput on the home PC. This is the highest-leverage lever for model strength:
more self-play data per unit time → a stronger net, and self-play is ~86% of the
training cycle. It captures the strategy decided after the benchmarking in
[lean-self-play-workers](lean-self-play-workers.md) and
[MCTS memory reduction](mcts-memory-reduction.md); each phase below gets
its own subplan.

> Supersedes the throughput intent of `lean-self-play-workers.md` (L3–L7), which
> assumed "more CPU workers = more speed." Benchmarking disproved that — see below.
> The memory work from that branch is done and banked; this roadmap is the sequel.

---

## What we learned (the evidence behind this plan)

Measured on the PC (RTX 3060 Ti 8 GB; i5-13600KF = **6 P-cores + 8 E-cores = 14C/20T**; 32 GB).

**Original measurement — Windows (`spawn`, warmup-heavy):**

| Config | games/s | Note |
|--------|---------|------|
| gpu8 (8 workers, own GPU net) | **~0.756** | best measured; capped at 8 by GPU VRAM |
| cpu16 (16 workers, CPU inference) | 0.211 | ~3.6× slower — CPU inference is the bottleneck |
| server 16 / 18 / 20 (central GPU) | 0.476 / 0.498 / 0.493 | **plateaus by ~16**, never beats gpu8 (matches F5's old 0.99×) |

> ⚠️ **These numbers are Windows-era and now known to be misleading.** They drove
> the original strategy below, but the **Linux re-baseline (2026-06-15)** tells a
> different story — see the next table. Kept verbatim for the record; not overwritten.

**Linux re-baseline — `forkserver` (2026-06-15, same hardware), pre- and post-P2:**

| Config | Windows | Linux pre-P2 | Linux +P2 | P2 effect |
|--------|---------|--------------|-----------|-----------|
| gpu8 | 0.756 | 1.154 | **1.194** | +3.4% |
| cpu16 | 0.211 | 0.723 | **0.714** | −1.2% (noise) |
| server16 | ~0.49 | 1.189 | **1.195** | +0.5% |

(Single 150-episode run per cell; pre-P2 = commit `79f03f5`, +P2 = `6cfa88e`.)

Revised conclusions:
1. **Memory is solved and no longer the cap.** Sparse MCTS cut the per-worker tree **1.9 GB → 19 MB** ([archive/mcts-memory-reduction.md](mcts-memory-reduction.md)); 20 workers now fit in ~16 GB. *(Unchanged — still holds.)*
2. **The biggest single win was the substrate, not any code lever.** The Windows→Linux migration (P1) ~1.5–3.4×'d every config. On Linux the configs **cluster at 1.15–1.19 g/s** and **`server16` now ties `gpu8`** — the old "GPU-bound at gpu8 / server plateaus below" conclusion was a *Windows artifact*. The real Linux ceiling and bottleneck are **not yet characterised** → that's now P3.
3. **P2 (vectorise `_select_action`) was ~0% end-to-end.** Locally the function is ~2× faster, but its slice of self-play wall-clock is only ~6% (not the suspected 22–35%), so Amdahl flattened it. Bit-identical and free, so it stays — but it was not the lever. *Lesson (again): profile the current substrate before prescribing.*

## The strategy

> **Revised 2026-06-15 after the Linux re-baseline.** The original three-lever
> strategy (below, struck through where outdated) assumed Windows bottlenecks.
> Two of its premises broke: the per-worker-CPU-MCTS lever (P2) was ~0% end-to-end,
> and "GPU-bound at gpu8" was a Windows artifact. **New strategy: re-profile and
> re-baseline on Linux first (P3), then implement whatever the data says wins
> (P4) — which may be more GPU workers, a hybrid split, or all-GPU.**

Original (kept for context):
1. ~~**Speed the per-worker CPU MCTS** — faster CPU work compounds across every config. → **B1 (vectorise `_select_action`)**.~~ *Done (P2); measured ~0% end-to-end — the slice was too small.*
2. **Use both compute resources at once** — keep the GPU saturated with GPU-fed workers *and* fill the spare cores with CPU-inference workers. → **hybrid workers**. *Still a candidate, but now one option among several (incl. all-GPU) to be decided by the P3 sweep, not assumed.*
3. Do it on the **right substrate** — native Linux. *Done (P1); turned out to be the single biggest win.*

---

## Checklist (phases)

| # | Phase | Subplan | Effort | Done |
|---|-------|---------|--------|------|
| P1 | **Migrate to native Linux** (dual-boot Ubuntu) — clean substrate before tuning | [linux-migration.md](linux-migration.md) | ~half day | ✔ 2026-06-12 |
| P2 | **B1: vectorise `_select_action`** (UCB) — done; bit-identical but **~0% end-to-end** (kept as free) | [archive/p2-vectorise-select-action.md](p2-vectorise-select-action.md) | ~0.5 day | ✔ 2026-06-15 |
| P3 | **Linux re-baseline** — sweep games/s vs worker config on Linux (all-GPU-N, CPU-only, server) + K-sweep + verification. The premise-fixing assessment | [p3-linux-rebaseline.md](p3-linux-rebaseline.md) · [research](../../research/linux-throughput-rebaseline.md) | ~0.5 day | ✔ 2026-06-16 |
| P4 | **Implement the winning config** — collapsed to a config change (no `num_gpu_workers` feature needed): all-GPU **16 workers**, K=16 → `run_configurations/blokus_linux16_15.json` | *(no subplan — one config)* | ~0.5 hr | ✔ 2026-06-16 |
| P5 | **Clean production measurement + default** — the all-GPU-N curve + K-sweep in the research doc are the measurement; ~ 16 GPU workers @ K=16 ≈ **1.6 games/s (~10.5 min/1000)** is the recorded default | [research](../../research/linux-throughput-rebaseline.md) | ~2 hr | ✔ 2026-06-16 |

> **Deep time/memory profile (py-spy/Scalene/tracemalloc) deliberately deferred** —
> P3's sweep answered the worker-config question without it. It's only worth doing if
> we later want to hunt *further* per-worker speedups; tracked as a future option, not
> a blocker. The production run itself (launching `blokus_linux16_15`) is the user's call.

Subplans are written *after* P1 (we tune on Linux). P3's subplan is below-linked and ready now; P4/P5 are written at their start since their shape depends on the P3 sweep.

---

## P1. Migrate to native Linux

The substrate decision, forced by: WSL was unstable (the "Catastrophic failure"
saga), and native Windows is `spawn`-only + warmup-heavy, which muddies every
throughput measurement. Native Linux fixes all three and is the eventual home.
Full runbook + risks + time estimate: **[linux-migration.md](linux-migration.md)**.

## P2. B1 — vectorise `_select_action`

`_select_action`'s per-move Python `for`-loop (UCB score with `math.sqrt` for
every legal move) was *suspected* to be the biggest CPU hot-spot (~22–35%, per
[self-play-speed-investigation.md](../../research/self-play-speed-investigation.md)).
Replaced it with a single vectorised numpy computation over the sparse aligned
`acts`/`priors` arrays, kept bit-identical (golden + parallel-determinism tests).

**Outcome (measured on Linux, 2026-06-15):** the function itself is ~2× faster,
but **end-to-end self-play throughput barely moved** (gpu8 +3.4%, cpu16 −1.2%,
server16 +0.5% — within single-run noise). The suspected slice was wrong: it's
~6% of self-play wall, so Amdahl flattened the win. **Kept anyway** — it's
bit-identical and not slower. Lesson: profile the current substrate first. Full
record: **[archive/p2-vectorise-select-action.md](p2-vectorise-select-action.md)**.

## P3. Linux re-baseline (re-profile + worker-split sweep)

The premise-fixing phase. The Windows evidence that shaped this roadmap is stale
(see the re-baseline table above), and P2 proved the time-profile guesses wrong —
so before tuning anything we characterise the *current* Linux reality:

1. **Re-profile** a single Linux self-play run for **time** (py-spy / Scalene /
   line_profiler on the MCTS hot path) and **memory** (tracemalloc on the
   in-episode tree + the replay buffer). Where does self-play wall-clock actually
   go now, on the small net + `forkserver`?
2. **Sweep the worker config** — games/s across **N GPU workers** (8, 10, 12, …,
   until VRAM/throughput caps), **CPU-only**, and a few **hybrid splits**
   (e.g. 8 GPU + 8 CPU). On Linux the net is tiny and `server16` already ties
   `gpu8`, so "everything on the GPU at a higher worker count" is a live
   possibility — let the curve decide, don't assume hybrid.

Output: the real Linux bottleneck breakdown + a games/s-vs-config curve that
picks the P4 target. Subplan (ready): **[p3-linux-rebaseline.md](p3-linux-rebaseline.md)**.

## P4. Implement the winning config

Whatever P3's sweep selects. If a **hybrid or higher-GPU-count split** wins, make
`worker_cuda` per-worker: add `num_gpu_workers`, give each worker a stable id (a
shared counter, as the inference server already does) so the first N build a GPU
net and the rest build CPU nets; tune threads-per-worker (`~20 ÷ num_workers` on
the 14C/20T chip). If **all-GPU-N** wins outright, this collapses to a config
default bump + raising the VRAM-bound worker cap. Subplan written at P4 start,
once the sweep says which.

## P5. Clean production measurement + default

On Linux (warmup amortised by `forkserver`), measure **games/s and
minutes-per-1000-games** for the P4 config against the gpu8 reference, pick the
production default, and record it. This is the number the whole roadmap is for.

---

## Not pursuing (and why)

- **More pure-CPU workers** — *(Windows read: CPU inference ~3.6× slower than GPU, cpu16 = 0.21 « gpu8 0.756.)* On Linux the gap narrowed sharply (cpu16 = 0.72 vs gpu8 1.19), so pure-CPU is less of a dead end than it looked — the P3 sweep re-checks this rather than assuming.
- **Central inference server as the primary path** — *(Windows read: plateaus ~0.49, never beats gpu8.)* On Linux `server16` (1.19) **ties gpu8** — so this is back on the table as a real candidate, not dismissed. P3 re-evaluates it head-to-head.
- **Free-threading (B5) / JAX-`mctx` rewrite (B4-class)** — highest ceiling but speculative / a ground-up reimplementation. Revisit only if P3–P5 plateau and we still need more.
- **A faster/bigger GPU (cloud)** — the real lever beyond this box's ceiling; out of scope for now but the natural escalation if the 3060 Ti caps us.
