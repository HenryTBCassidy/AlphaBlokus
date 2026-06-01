# Full-Cycle Training Optimisation

Master plan for making AlphaBlokus training meaningfully faster — both single-thread (per-game) and parallel (across games). Acts as the umbrella index for smaller sub-plans that get spun off when each optimisation is actually attempted.

The goal isn't "make it fast for fast's sake" — it's to make Blokus training **practical at the scale needed to beat Pentobi**. AlphaGo Zero needed ~ 5M self-play games to reach superhuman Go. At our current pace (~55s per game on the 3060 Ti, single-threaded), even modest training scales of ~100k games would take weeks. A 10-25× wall-clock reduction makes the project viable.

Companion docs:
- [`docs/08-TRAINING-ESTIMATES.md`](../08-TRAINING-ESTIMATES.md) — measured wall-clock per config and the move-gen-as-bottleneck finding this whole plan is built on.
- [`docs/09-COMPUTE-OPTIONS.md`](../09-COMPUTE-OPTIONS.md) — local + cloud hardware comparison; optimisation lowers the bar for which option suffices.
- [`docs/02-ALGORITHMS.md`](../02-ALGORITHMS.md) — MCTS / self-play / arena structure that each optimisation targets.

---

## How this plan works

It's a **master document**, not a row-by-row implementation checklist. Each numbered section in the optimisation menu below has a status:

- **Pending** — not yet started
- **Planned** — a detailed sub-plan exists (linked in the table)
- **In progress** — a sub-plan is being worked
- **Done** — sub-plan archived, results captured in the progress tracker

When we start a section, we:

1. Spawn a sub-plan doc at `docs/plans/<short-name>.md` with detailed checklist + design.
2. Profile to capture baseline numbers (using `scripts/mcts_profiling.py`).
3. Implement, test, measure post-change numbers.
4. Update the [Progress tracker](#progress-tracker) at the bottom of this doc with the measurements.
5. Archive the sub-plan to `docs/plans/archive/`.
6. Tick the row in the menu.

The master plan itself only gets archived once enough of the optimisations are done that further speedup isn't worth chasing (likely when the Single-PC stretch config from `docs/08-TRAINING-ESTIMATES.md` fits comfortably in an overnight run — see the [Targets](#targets) section).

---

## Current baseline

Measured during `blokus_pc_second` (300 sims, 80 eps, RTX 3060 Ti, 64f×4b net):

| Phase | Wall-clock | Notes |
|-------|-----------|-------|
| Self-play, per game | ~55s | Sequential — one game at a time on the single python process |
| Self-play, per gen (80 games) | ~73 min | Dominant cost per generation |
| Arena, per gen (50 games × 2) | ~45 min | Two networks, also sequential |
| Elo eval (20 games) | ~18 min | Same shape as arena |
| Training (5 epochs) | ~3 min | GPU-bound, already efficient |
| Symmetry diagnostic | ~5s | Negligible |
| Total per gen | **~2h 20min** | |
| Total for 5-gen run | **~11h 30min** | |

Bottleneck breakdown (from `scripts/benchmark_phases.py` 2026-05-26 on the same hardware, **production net 64f×4b**):

| Component | % of MCTS search time |
|-----------|----------------------|
| Neural-net inference | **49.7%** |
| Move generation (`_generate_valid_moves`, `_all_cells_valid`) | **42.6%** |
| Other (UCB, tree bookkeeping) | 6.0% |
| Game-ended checks | 1.7% |

**Two-line takeaway:** the cost is split roughly evenly between Python move generation and GPU inference at production net size — *not* dominated by move-gen as the small-net profiling earlier suggested. Self-play games still run one at a time on a single core while 7 others sit idle; parallelising solves that for the move-gen ~43% but workers will contend on the shared GPU for the inference ~50% (this is why F3 — batched inference server — matters as a follow-up to F1).

---

## Targets

Three milestones, each with an associated wall-clock budget for the **Single-PC reference** config from [`docs/08-TRAINING-ESTIMATES.md`](../08-TRAINING-ESTIMATES.md) (5 gens × 80 eps × 300 sims × 50 arena, 64f×4b net — matches `blokus_pc_second` at ~11.5 h baseline):

| Milestone | Wall-clock target | Implies |
|-----------|------------------|---------|
| **M1 — Practical** | 5-gen run in **~3 hours** | A 4× speedup over baseline. Comfortable for daytime iteration. |
| **M2 — Aggressive** | 5-gen run in **~1 hour** | An 11× speedup. Sufficient for tight parameter-tuning loops. |
| **M3 — Stretch** | 5-gen run in **~30 min** | Approaching the practical limit on this hardware; further gains need cloud / distributed compute. |

Optimisations in the menu below have rough speedup estimates that compound, so M1 is reachable from one or two changes, M2 needs three, M3 needs everything.

---

## Optimisation menu

F-numbers reflect execution order. F1 done, F2 done, F3 next, F4 after. Earlier drafts had F3/F4 swapped — we renamed them so the numbers match the order things are actually done in. The rationale for this order is in [Sequencing rationale](#sequencing-rationale) below.

| # | Optimisation | Expected speedup | Effort | Sub-plan | Status |
|---|--------------|------------------|--------|----------|--------|
| F1 | **Parallel self-play & arena across CPU cores** — N concurrent games per phase, using a worker pool over the `Player` abstraction | Projected ~1.8× alone, measured **4.13× at 4 workers**, **5.72× at 8 workers** on the PC (GPU has more headroom than the Amdahl projection assumed) | Medium (landed in ~1.5 days) | [`archive/parallel-self-play.md`](archive/parallel-self-play.md) | **Done** ✓ |
| F2 | **Move-gen — precomputed move-list table** — Pentobi-style; runtime is byte-array reads + a 4D lookup `(anchor, adj_status, piece) → candidate move IDs` + a 5-cell unrolled OR per candidate. | Measured **9.06× per-call**; **1.33× full-pipeline wall-clock** on top of F1-8w. Per-process move-gen share dropped from 52.4% to 8.7%; inference now 77.5% of per-process time — clear next target. | Medium (landed in ~1 day) | [`archive/move-gen-optimisation.md`](archive/move-gen-optimisation.md) | **Done** ✓ |
| F3 | **Batched neural-net inference in MCTS** — collect K leaf evaluations per MCTS step, single GPU call per batch with virtual-loss to keep paths diversified. Now the headline next-step lever because F2 surfaced GPU-contention as the binding constraint. | **2–5×** on the inference slice → projected **~1.5-2×** overall on top of F2 (inference share is ~77% post-F2, so the headroom is big) | Medium-big (~2 days) | [`batched-inference.md`](batched-inference.md) | **Next** |
| F4 | **MCTS tree reuse between moves** — re-root the search tree after each ply instead of rebuilding from scratch | **1.5–2×** | Small-medium (~1 day) | `mcts-tree-reuse.md` | Pending |
| F5 | **Move-gen — Cython** — recompile existing logic with C type annotations. Now superseded by F2; kept here only as a fallback if F2 were ever rolled back. | **2–3×** overall | Medium (~1 day) | (none — F2 is the chosen approach) | Superseded by F2 |
| F6 | **Cached valid moves per placement point** — incremental cache invalidation rather than full re-enumeration | Unclear (~2× on `_valid_moves` but adds cost to `with_piece`) | Medium (~half day) | (only spawn if F1+F2+F3 leave us short of M2) | Deferred |

### Sequencing rationale

Execution order is **F1 → F2 → F3 → F4**, with F5 / F6 as fallbacks. The order has been refined twice as measurements came in:

- **F1 first** — done. Even with the modest Amdahl ceiling on a single axis, it landed the parallel execution infrastructure that everything else depends on. Measured 5.72× at 8 workers.
- **F2 second** — done. Targeted the biggest remaining CPU slice (move-gen, 52% of per-process time post-F1). Measured 7.2× reduction in move-gen time, 1.33× wall-clock on top of F1. **Surfaced the real binding constraint: GPU contention.** Post-F2, inference is 77.5% of per-process time and 8 workers all hammer the GPU simultaneously.
- **F3 third** — promoted again after the F2 measurement. Pre-F2 the original plan called F3 a "polish" step (small gains because inference was a modest share). Post-F2 it's the headline next-step: inference dominates and 8-worker GPU contention is a single-process problem. Batched inference is what unblocks the rest of the F1+F2 win that's currently being eaten by queue contention.
- **F4 last** — tree reuse halves MCTS work per move regardless of search backend, so it compounds with all the above. Less urgent than F3 now that F3 is the rate-limiter. Worth doing after F3 lands as a final compounding step.

**Why F2 before F3** (rather than the reverse): F3's gain scales with how much time workers spend in inference. Pre-F2, workers spent ~half their time in CPU move-gen, masking inference contention. Doing F3 first would have shown smaller wins because move-gen was the binding constraint. F2 stripped out move-gen and made inference unambiguously the bottleneck — now F3 has clear, measurable headroom to attack.

### Compatibility matrix

| | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| F1 (parallel)              | — | ✓ | within-worker batching is straightforward; cross-worker batching needs an inference server | ✓ |
| F2 (move-list table)       | ✓ | — | ✓ | ✓ |
| F3 (batched inference)     | within-worker batching is straightforward; cross-worker needs an inference server | ✓ | — | ✓ |
| F4 (tree reuse)            | ✓ | ✓ | ✓ | — |

F1 + F3 is the trickier combination. Two flavours of F3 are possible:

- **Within-worker batching** (each worker runs K parallel sims per MCTS step, single GPU call per K sims). Simpler, no IPC, each worker independent. Speedup factor capped by batch size K.
- **Cross-worker batching** (separate inference-server process that collects leaf evals from all workers and batches them). Stronger asymptotic gain because it coalesces requests across processes too, but introduces IPC and a separate process to manage.

The F3 sub-plan picks the simpler of the two first; cross-worker batching is a follow-up if within-worker batching doesn't fully saturate the GPU.

---

## Profiling cadence

Each optimisation lands with a paired measurement. Use this recipe:

1. **Before any change in a section**: run `scripts/mcts_profiling.py` on a fixed config (saved at `run_configurations/profile_baseline.json` — to be created when F1 starts) and a fixed seed. Record numbers in the [progress tracker](#progress-tracker).
2. **After the change**: re-run with the *same* config and seed. Diff numbers.
3. **Once per few sections**: kick off a short end-to-end "validation" Blokus run (1–2 gens, small eps) on the PC and compare total wall-clock against the previous validation. Measures real-world impact including inter-phase overhead.

`scripts/mcts_profiling.py` already exists from the MCTS-profiling work archived earlier in `docs/plans/archive/mcts-profiling.md`. Re-using it gives consistent measurement methodology across the whole optimisation sequence.

---

## Progress tracker

Filled in as each section completes. Empty until the first run.

| Section | Date | Measured before | Measured after | Speedup achieved | Notes |
|---------|------|-----------------|----------------|------------------|-------|
| Baseline (full run) | 2026-05-22 | — | 55s/game, 2h 20min/gen | — | `blokus_pc_second`, 300 sims, 80 eps, 50 arena. Headline measurement at production scale; preserved as immutable reference. |
| Baseline (benchmark) | 2026-05-26 | — | Self-play 54.1s/game, Arena 53.9s/game, Elo 55.2s/game, total 32.6min for 16+10+10 games | — | `scripts/benchmark_phases.py --config run_configurations/profile_baseline.json` on RTX 3060 Ti. Component split: 49.7% NN inference, 42.6% move-gen, 6.0% other, 1.7% game-ended. Re-run this exact benchmark after each F-step for clean before/after deltas. |
| F1 — Parallel self-play (4 workers) | 2026-05-27 | Self-play 54.1s/game; Arena 53.9s/game; Elo 55.2s/game; **total 32.6 min** | Self-play 12.0s/game (4.5×); Arena ~14.6s/game (3.75×); Elo ~14.6s/game (3.83×); **total 7.9 min** | **4.13×** at 4 workers | Bested the ~1.8× Amdahl projection by 2.3× — GPU has more headroom than the inference-fraction model predicted. 8-worker run originally OOM'd WSL — backfilled below after raising the memory cap. |
| F1 — Parallel self-play (8 workers backfill) | 2026-05-28 | (same serial baseline as 4-worker row above) | Self-play 8.08s/game (6.7×); Arena ~10.8s/game (5.0×); Elo ~10.5s/game (5.3×); **total 5.7 min** | **5.72×** at 8 workers | Required raising WSL memory cap to 24 GB (`.wslconfig`). Peak memory ~14 GB during run. 4→8 worker gain is 1.39× — sub-linear because GPU contention on inference grows with worker count (per-process inference share grew from 31.8% to 37.3%). This is the "true" F1 ceiling on the 3060 Ti; further gains would need F3 (batched inference). |
| F2 — Move-gen precomputed table | 2026-05-28 | F1 8-worker baseline: Self-play 8.08s/game, total 5.7 min; per-process move-gen 52.4% / inference 37.3% | F2 + F1 8-worker: Self-play 6.49s/game (1.24×), Arena ~7.65s/game (1.41×), Elo ~7.50s/game (1.40×); **total 4.3 min (1.33× vs F1-8w; 7.58× vs serial baseline)**; per-process move-gen **8.7%** / inference **77.5%** | **1.33×** at 8 workers on top of F1 | F2 collapsed the move-gen slice from 52.4% to 8.7% of per-process time. Inference now dominates at 77.5% — F3 (batched inference) is unambiguously the next lever. Per-call move-gen speedup measured at 9.06× via `scripts/benchmark_movegen.py`. Determinism verified bit-identical at 5 seeds. |
| F3 — Batched inference | | | | | |
| F4 — MCTS tree reuse | | | | | |

---

## Out of scope for this plan

- Multi-machine distributed training (single-PC focus for now; revisit when M3 hits)
- Mixed-precision / fp16 GPU inference (would interact with batched inference; consider as part of F3)
- C++ rewrite of move generation (more invasive than Cython, less leverage than bitboard)
- Custom CUDA kernels for the neural net (over-engineering for our scale)
- Network architecture changes (separate concern; tracked in any future `net-scaling.md` plan, not here)

---

## Notes for whoever picks this up next

- The move-gen rewrite (F2) is a touchy piece because it changes `BlokusDuoBoard` internals that the existing test suite covers. The F2 plan explicitly requires equivalence tests on 10,000+ random positions before switch-over — don't skip them. Make the full test suite pass before merging.
- Parallel self-play (F1) needs careful handling of the `nnet` state. Easiest path: each worker re-loads the latest checkpoint at start. Memory pressure on the 8GB 3060 Ti will be the deciding factor — 8 workers × ~50MB net = 400MB, fine.
- Tree reuse (F4) sounds simple but has subtle correctness traps around canonical-form transitions between turns. Will need targeted equivariance tests, similar to the symmetry work.
- Always measure before claiming a speedup. The move-gen plan (now absorbed here) initially proposed 5 different micro-optimisations of which **the benchmarked low-hanging-fruit ones (O5-O8) were ultimately rejected for not actually helping**. Trust the profiler.
