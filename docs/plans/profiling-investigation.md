# Profiling Investigation — Where Does Self-Play Spend Time and Memory?

The deliberate **measure-before-you-optimise** step for the post-F5 codebase. The first optimisation wave (F1–F5, archived in [`archive/full-cycle-optimisation.md`](archive/full-cycle-optimisation.md)) took self-play training to **~14× vs serial** and shrank the net **21.6×** — which *moved the bottleneck*, so the old profile (inference 77.5%) is stale. F5's failure (built on that stale profile → **0.99×, no speedup**) is the lesson this plan exists to avoid: **nothing gets optimised until a current profile says which slice is worth it.**

**This plan's output decides whether further optimisation happens at all.** It may well conclude we're near the practical limit on this hardware — in which case the right answer is "stop optimising, just train." Any optimisation work (and its own branch) is spawned *only after* P8, and *only if* P8 says so. This is not an optimisation plan; it's the plan that decides whether we need one.

**Prerequisite:** the training-step OOM fix is on `main` (PR #12), so the loop runs to completion and the profile reflects healthy memory. **Companion:** the candidate-technique menu and the Amdahl analysis live in [`../research/self-play-speed-investigation.md`](../research/self-play-speed-investigation.md) — this plan is the actionable checklist; that doc is the reference for *what we'd do* with the results.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| P1 | `scripts/profile_self_play.py` — play **one** self-play game single-process (fixed seed, production net + sims) under a `--profiler {pyspy,scalene,lineprofiler,tracemalloc}` switch | 1 hr | High | |
| P2 | **py-spy** on an 8-worker self-play run (`--subprocesses`, ~30 s sample) → coarse flame-graph split across real parallel execution | 0.5 hr | High | |
| P3 | **Scalene** on the single-game script — per-line CPU, **Python-vs-native split**, and memory | 0.5 hr | High | |
| P4 | **line_profiler** on the 3–4 hot functions P2/P3 flag (`_select_action`, `_descend_to_leaf`, `get_next_state`, `state_key`) | 0.5 hr | High | |
| P5 | **tracemalloc** — in-episode MCTS tree growth over one long game (scaling risk for long games / high sim counts) | 0.5 hr | Medium | |
| P6 | **tracemalloc/Scalene** — self-play→replay-buffer growth across ≥2 generations (RAM held between phases) | 0.5 hr | Medium | |
| P7 | Write the **current profile** into this plan: time + memory tables, each time-slice annotated with its Amdahl ceiling `1/(1-p)` | 0.5 hr | High | |
| P8 | **Decision** — against the targets + stopping criterion below, decide *optimise or train*; if optimise, name the one lever and spawn its plan/branch (not before) | 0.5 hr | High | |

---

## P1. Single-game profiling harness

A `scripts/profile_self_play.py` that plays exactly one self-play game in a single process (fixed seed, production net + sim count), with a `--profiler` switch selecting the tool. Single-process + fixed seed = a reproducible, isolated target with no multiprocessing noise — the clean surface every other step measures against.

## P2. py-spy — coarse split across the real 8-worker run

Sampling profiler, handles multiprocessing (`--subprocesses`), near-zero overhead, no code changes. `cProfile` doesn't work across our `ProcessPoolExecutor` workers; py-spy is the right coarse tool. Output: the honest % split across workers — select / move-gen / `get_next_state` / inference / backprop.

## P3. Scalene — line-level CPU, Python-vs-native, memory

Run on the single-game script. Crucially splits **Python-interpreter time vs native (numpy/torch) time** per line — that's what tells us whether a hot line is optimisable (interpreter-bound → Cython/Numba/vectorise helps) or already native (→ it won't). Also gives per-line memory.

## P4. line_profiler — exact timings on the hot functions

`@profile` the 3–4 functions P2/P3 surface. Deterministic, exact line costs — the precise targets, and the before-numbers any change is measured against.

## P5. tracemalloc — in-episode tree growth

The per-episode MCTS node/edge dicts grow as sims accumulate. Quantify peak + growth over one long game: is tree memory a scaling risk at high sim counts / long games (informs caps / pruning), or a non-issue?

## P6. tracemalloc/Scalene — replay-buffer growth

The RAM held *between* phases as examples accumulate across generations. The training-step OOM is already fixed; this measures whether the resident buffer itself is a concern at the full config, and how it compares to what's on disk.

## P7. The current profile (the deliverable)

Fill these in from P2–P6. This is the evidence base for P8.

**Time (per self-play game):**

| Slice | % of wall-clock | Python vs native | Amdahl ceiling `1/(1-p)` |
|-------|-----------------|-------------------|--------------------------|
| `_select_action` (UCB loop) | TBD | TBD | TBD |
| Move generation | TBD | TBD | TBD |
| `get_next_state` | TBD | TBD | TBD |
| NN inference | TBD | TBD | TBD |
| Backprop / bookkeeping | TBD | TBD | TBD |

**Memory:**

| Component | Peak / growth | Scaling risk? |
|-----------|---------------|---------------|
| In-episode MCTS tree | TBD | TBD |
| Replay buffer (resident) | TBD | TBD |

## P8. Decision — optimise or train?

With P7 in hand, decide against the criteria below. Output is one of: **(a)** "near the limit → stop optimising, run more/longer training," or **(b)** "lever X has a worthwhile ceiling → spawn `optimise-<lever>` plan + branch." Record the decision and reasoning here, then act on it.

### Targets *(yardstick: a 15-gen × 1000-game run)*

| Milestone | Goal | Implies |
|-----------|------|---------|
| **T1 — Comfortable overnight** | ≤ **12 h** on the RTX 3060 Ti | fits one unattended slot with margin |
| **T2 — Tight iteration** | ≤ **6 h** | two runs/day; faster sweeps |
| **T3 — Stretch** | ≤ **3 h** | near the practical limit on this hardware |

### Stopping criterion — when to stop optimising and *run*

Stop and just train when **all** hold:
1. **It fits** — the 15-gen run is projected to complete comfortably overnight (**T1 ≤ 12 h**).
2. **The next lever is small** — the best remaining candidate's **Amdahl ceiling < ~1.2×** (even perfect, it can't move the needle).
3. **It doesn't crash** — RAM stays flat across the run (OOM fix verified).

The end goal is a world-class net, bought with **generations trained**, not seconds shaved per game. Past the point where the run fits overnight and the marginal lever buys < ~20%, more optimisation is worse ROI than more training.

### Candidate levers *(only if P8 says optimise — detail + Amdahl in the [research doc](../research/self-play-speed-investigation.md))*

| Lever | Targets | Effort |
|-------|---------|--------|
| Vectorise `_select_action` | the per-move UCB Python loop (prime suspect) | Low |
| Code-level trims | recomputed `state_key`, hoisting, cached `np.where`, buffer reuse | Low |
| Bitboard board representation | `state_key` + `get_next_state` + move-gen at once (structural) | High |
| Free-threaded Python spike | drop processes → threads, one shared net (needs a torch/numpy free-threading spike first) | High |
| Cython / Numba on the hot kernel | whatever interpreted kernel remains the top slice | Medium |
| Adaptive simulation taper | *do less work* — stop spending 300 sims on <30-move endgames | Low–Med |

Promotion rule: a lever becomes real work only when P7 shows its slice is large enough to clear the stopping criterion's bar. Then it gets its own plan + branch — named at that point, not now.
