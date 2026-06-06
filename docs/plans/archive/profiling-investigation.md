# Profiling Investigation — Where Does the Training Cycle Spend Time and Memory?

**Status: ✅ COMPLETE (2026-06-05).** The deliberate *measure-before-you-optimise* step for the post-F5 codebase (the lesson from F5, which was built on a stale profile and gave 0.99×). 

**📊 Full visual report (charts + tables + raw evidence): [`../../research/profiling-report.md`](../../research/profiling-report.md).**

**Decision: RUN NOW.** The 15-gen run fits comfortably overnight (~5.3 h), so optimisation is *optional*. Training is ~1% of the cycle — not worth touching. The self-play levers are ranked below for when we want tighter iteration. No optimisation branch was spawned — that happens only if/when we decide tighter iteration is worth the effort.

**Companion:** technique menu + Amdahl detail in [`../../research/self-play-speed-investigation.md`](../../research/self-play-speed-investigation.md). Harness: [`../../../scripts/profile_self_play.py`](../../../scripts/profile_self_play.py); report generator: [`../../../scripts/profile_report.py`](../../../scripts/profile_report.py).

---

## Checklist

| # | Item | Done |
|---|------|------|
| P1 | Profiling harness (`scripts/profile_self_play.py`: timing / cProfile / memory / train modes; F2 enabled to match production) | ✅ |
| P2 | Self-play time profile — coarse phase split (built-in MCTS timers) + cProfile function attribution at the production config | ✅ |
| P3 | Training-step profile — `train()` wall + RSS over a realistic 57k-example buffer | ✅ |
| P4 | Memory profile — in-episode tree growth + process RSS (tracemalloc) | ✅ |
| P5 | Full-cycle phase split (self-play / arena / elo / training) + Amdahl ceilings | ✅ |
| P6 | Decision — optimise or train, and (if optimise) rank the levers | ✅ |

> py-spy / Scalene / line_profiler were **deferred** — single-process cProfile + the built-in MCTS timers gave a clear, sufficient attribution, and the Python-vs-native nature is obvious from the call graph. Promote them only if we attack a slice and need per-line detail.

---

## Results & decision (summary — full detail in the [report](../../research/profiling-report.md))

- **Headline:** self-play is **CPU-bound** (GPU idles ~0–30%); no single bottleneck — **inference ~33%** (mostly GPU↔CPU *transfers*, not compute), **move-gen ~24%**, **`_select_action` UCB ~22%**. **Training is ~1%** of the cycle.
- **Full cycle:** ~21 min/gen → **~5.3 h for 15 generations**. Self-play ~90%, arena+elo ~7%, training ~1%, RAM flat (OOM fix verified, training RSS 5.6 GB).
- **Decision: RUN NOW** — it fits overnight; a strong net comes from generations trained, not seconds shaved. **Don't optimise training** (~1%, ceiling 1.01×).
- **Ranked levers (for later):** 1) vectorise `_select_action` (~1.25×, low effort, best ROI); 2) cut inference transfer overhead (~1.2×, medium); 3) bitboard move-gen (~1.3×, high). #1+#2 plausibly ~5.3 h → ~3.5–4 h.

---

## Method (how the numbers were produced)

- **P1 harness** — `scripts/profile_self_play.py` plays game(s) single-process at a fixed seed, F2 enabled to match production. Modes: `timing`, `cprofile`, `memory`, `train`.
- **P2 self-play time** — built-in MCTS timers for the coarse inference-vs-rest split; cProfile (single-process — works fine, unlike across the worker pool) attributed the rest to move-gen / UCB / board-copy / bookkeeping.
- **P3 training** — `train()` timed over a synthetic 57k-example buffer (1 generation at lookback=1) at the production epochs/batch.
- **P4 memory** — tracemalloc + RSS over one game (tree) and the training step (buffer).
- **P5 full-cycle** — per-game self-play × game counts (with the measured gen-1 wall) + training + arena/elo → the phase split and full-cycle Amdahl ceilings.
- **P6 decision** — recorded above and in the [report](../../research/profiling-report.md).
