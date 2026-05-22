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

The master plan itself only gets archived once enough of the optimisations are done that further speedup isn't worth chasing (likely when single-PC training time is in the "feasible overnight for Serious-scale runs" range — see the [Targets](#targets) section).

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

Bottleneck breakdown (from `scripts/mcts_profiling.py` on the same hardware, smaller net):

| Component | % of MCTS search time |
|-----------|----------------------|
| Move generation (`_generate_valid_moves`, `_all_cells_valid`) | **65–72%** |
| Neural-net inference | 17–27% |
| Everything else (UCB, game-ended checks, tree bookkeeping) | the rest |

**Two-line takeaway:** ~70% of a run's time goes to Python move generation. Self-play games are run one at a time even though we have 8+ idle CPU cores. Both are addressable.

---

## Targets

Three milestones, each with an associated wall-clock budget for the current "Serious" Blokus config (5 gens × 80 eps × 300 sims × 50 arena games):

| Milestone | Wall-clock target | Implies |
|-----------|------------------|---------|
| **M1 — Practical** | 5-gen run in **~3 hours** | A 4× speedup over baseline. Comfortable for daytime iteration. |
| **M2 — Aggressive** | 5-gen run in **~1 hour** | An 11× speedup. Sufficient for tight parameter-tuning loops. |
| **M3 — Stretch** | 5-gen run in **~30 min** | Approaching the practical limit on this hardware; further gains need cloud / distributed compute. |

Optimisations in the menu below have rough speedup estimates that compound, so M1 is reachable from one or two changes, M2 needs three, M3 needs everything.

---

## Optimisation menu

| # | Optimisation | Expected speedup | Effort | Sub-plan | Status |
|---|--------------|------------------|--------|----------|--------|
| F1 | **Parallel self-play & arena across CPU cores** — N concurrent games per phase, using a worker pool over the `Player` abstraction | **5–7×** (linear in core count, up to ~num_eps) | Medium (~1 day) | [`parallel-self-play.md`](parallel-self-play.md) | **Planned** |
| F2 | **Move-gen — bitboard representation** — 14×14 board as a 196-bit integer; validation becomes 2 bitwise ANDs vs ~20 Python ops | **3–5×** overall | Big (~3–4 days) | `move-gen-bitboard.md` | Pending |
| F3 | **MCTS tree reuse between moves** — re-root the search tree after each ply instead of rebuilding from scratch | **1.5–2×** | Small-medium (~1 day) | `mcts-tree-reuse.md` | Pending |
| F4 | **Batched neural-net inference in MCTS** — collect leaf evals into batches, single GPU call per batch with virtual-loss to keep MCTS correct | **2–5×** on the inference slice → **~1.2–1.5×** overall (more impact when net is bigger) | Medium-big (~2 days) | `batched-inference.md` | Pending |
| F5 | **Move-gen — Cython** — recompile existing logic with C type annotations. Alternative to F2; lower ceiling but lower risk. | **2–3×** overall | Medium (~1 day) | (none yet — only spawn if F2 is rejected for risk) | Pending |
| F6 | **Cached valid moves per placement point** — incremental cache invalidation rather than full re-enumeration | Unclear (~2× on `_valid_moves` but adds cost to `with_piece`) | Medium (~half day) | (only spawn if F1+F2 leave us short of M2) | Pending |

### Sequencing rationale

Recommended order is **F1 → F2 → F3 → F4**, with F5 / F6 as fallbacks. Why:

- **F1 first**: highest ROI (5–7× wall-clock), independent of the rest, gives immediate iteration speedup that makes everything else faster to develop. Touches `Coach` / `Arena` / `Player` plumbing but doesn't perturb the game-logic core.
- **F2 second**: biggest single-thread gain. With F1 already in place, the dev loop is fast enough to test the bitboard rewrite confidently.
- **F3 third**: complements F1/F2 — tree reuse halves MCTS work per move regardless of search backend.
- **F4 last**: batched inference matters most when the network is bigger. Currently with 64f×4b the inference slice is small; gain is real but modest. Save for when we scale up the net for the Pentobi-beating runs.

### Compatibility matrix

| | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| F1 (parallel) | — | ✓ | ✓ | Tricky (each worker needs own GPU/CPU inference queue) |
| F2 (bitboard) | ✓ | — | ✓ | ✓ |
| F3 (tree reuse) | ✓ | ✓ | — | ✓ |
| F4 (batched inference) | Tricky | ✓ | ✓ | — |

F1 + F4 is the only awkward combination — workers in F1 each have their own MCTS; batching across workers requires an inference server. Probably manageable but the design needs care. Defer F4 until both F1 and F2 are landed.

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
| Baseline | 2026-05-22 | — | 55s/game, 2h 20min/gen | — | `blokus_pc_second`, 300 sims, 80 eps, 50 arena |
| F1 — Parallel self-play | | | | | |
| F2 — Move-gen bitboard | | | | | |
| F3 — MCTS tree reuse | | | | | |
| F4 — Batched inference | | | | | |

---

## Out of scope for this plan

- Multi-machine distributed training (single-PC focus for now; revisit when M3 hits)
- Mixed-precision / fp16 GPU inference (would interact with batched inference; consider as part of F4)
- C++ rewrite of move generation (more invasive than Cython, less leverage than bitboard)
- Custom CUDA kernels for the neural net (over-engineering for our scale)
- Network architecture changes (separate concern; tracked in any future `net-scaling.md` plan, not here)

---

## Notes for whoever picks this up next

- The bitboard rewrite (F2) is the highest-risk piece because it touches `BlokusDuoBoard` internals that the existing test suite covers. Don't skip the existing tests — make them pass before merging.
- Parallel self-play (F1) needs careful handling of the `nnet` state. Easiest path: each worker re-loads the latest checkpoint at start. Memory pressure on the 8GB 3060 Ti will be the deciding factor — 8 workers × ~50MB net = 400MB, fine.
- Tree reuse (F3) sounds simple but has subtle correctness traps around canonical-form transitions between turns. Will need targeted equivariance tests, similar to the symmetry work.
- Always measure before claiming a speedup. The move-gen plan (now absorbed here) initially proposed 5 different micro-optimisations of which **the benchmarked low-hanging-fruit ones (O5-O8) were ultimately rejected for not actually helping**. Trust the profiler.
