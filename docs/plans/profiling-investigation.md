# Profiling Investigation — Where Does the Training Cycle Spend Time and Memory?

The deliberate **measure-before-you-optimise** step for the post-F5 codebase. The first optimisation wave (F1–F5, archived in [`archive/full-cycle-optimisation.md`](archive/full-cycle-optimisation.md)) took self-play training to **~14× vs serial** and shrank the net **21.6×** — which *moved the bottleneck*, so the old profile (inference 77.5%) is stale. F5's failure (built on that stale profile → **0.99×, no speedup**) is the lesson this plan exists to avoid.

**RESULT (measured 2026-06-05): profile complete; decision in [P8](#p8-decision--run-now-optimisation-optional) is RUN NOW — the run already fits comfortably overnight (~5.3 h), so optimisation is optional. Training is ~1% of the cycle (not worth touching). The ranked self-play levers are recorded for when we want tighter iteration.**

**Companion:** the candidate-technique menu and the Amdahl analysis live in [`../research/self-play-speed-investigation.md`](../research/self-play-speed-investigation.md). Measurement harness: [`scripts/profile_self_play.py`](../../scripts/profile_self_play.py).

---

## Checklist

| # | Item | Done |
|---|------|------|
| P1 | Profiling harness (`scripts/profile_self_play.py`: timing / cProfile / memory / train modes; **F2 enabled to match production**) | ✅ |
| P2 | Self-play time profile — coarse phase split (built-in MCTS timers) + **cProfile** function attribution at the production config | ✅ |
| P3 | Training-step profile — `train()` wall + RSS over a realistic 57k-example buffer *(added per scope: the phase MCTS work doesn't help)* | ✅ |
| P4 | Memory profile — in-episode tree growth + process RSS (tracemalloc) | ✅ |
| P5 | Full-cycle phase split (self-play / arena / elo / training) + Amdahl ceilings | ✅ |
| P6 | Decision — optimise or train (see [P8](#p8-decision--run-now-optimisation-optional)) | ✅ |

> py-spy / Scalene / line_profiler (the Python-vs-native split) were **deferred** — single-process cProfile + the built-in MCTS timers gave a clear, sufficient attribution, and the Python-vs-native nature is obvious from the call graph (the UCB loop is pure Python; move-gen/inference are native). Promote them only if we attack a slice and need per-line detail.

---

## P7. The current profile — RESULTS

**Method.** Single-process self-play at the production config (`blokus_scaled_15.json`: 300 sims, K=16, conv 64f×4b, CUDA, **F2 move-gen on**) on the RTX 3060 Ti, plus a synthetic-buffer training profile. Time from the built-in MCTS phase timers + cProfile function attribution; memory from tracemalloc + RSS.

**Headline.** Self-play is **CPU-bound** (the GPU idles at ~0–30%). No single slice dominates — **inference, move-gen, and the UCB loop are three roughly co-equal thirds.** Training is a **rounding error** (~1% of the cycle).

### Self-play — 7.93 s/game (28 moves, 55 examples, 6,845 tree states)

| Slice | % of self-play | Nature | Amdahl ceiling (→0) |
|-------|----------------|--------|---------------------|
| **NN inference** (`predict_batch`) | **~33%** | mostly GPU↔CPU **transfers** (`.cpu()` ~14% + `.cuda()` ~3%) **>** compute (conv/bn/linear ~7%) | 1.49× |
| **Move generation** (F2, `valid_move_mask`) | **~24%** | native-ish; `_fill_legal_actions_for_anchor` is the core | 1.32× |
| **`_select_action`** (UCB loop) | **~22%** | **pure Python** `for`-loop, `math.sqrt` ×2.5M/game — vectorisable | 1.28× |
| **Board transitions** (`get_next_state` / `with_piece` copy) | **~10%** | numpy board copies | 1.11× |
| Tree descend / expand / backprop / bookkeeping | **~10%** | Python | 1.11× |
| Episode overhead (symmetries / sampling) | **~1%** | | 1.01× |

The surprise: **inference's cost is transfer overhead, not compute** — `.cpu()`/`.cuda()` moving the K=16 result tensors GPU↔CPU ~888×/game costs ~3× the actual conv math.

### Training — 12.3 s/generation (57k examples × 2 epochs = 6.1 s/epoch), RSS 5.6 GB

Confirms the OOM fix (5.6 GB, **not** 27 GB) and that training is fast + GPU-efficient on the small conv net.

### Full-cycle per generation (15-gen × 1000-game config)

| Phase | Wall / gen | % of cycle | Sped up by MCTS work? |
|-------|-----------|------------|------------------------|
| **Self-play** (1000 games) | ~19.5 min (measured, gen-1) | **~90%** | ✅ |
| **Arena + Elo** (50 + 50 games) | ~2 min | ~7% | ✅ (same MCTS loop) |
| **Training** (2 epochs, 57k) | **0.2 min** (12.3 s) | **~1%** | ❌ separate phase |
| Diagnostics | <0.1 min | ~0% | ❌ |
| **Total** | **~21 min/gen → ~5.3 h for 15 generations** | | |

### Memory

| Component | Measured | Scaling risk? |
|-----------|----------|---------------|
| In-episode MCTS tree | ~6,800 states/game; process RSS ~3.5 GB | Low — bounded per game, resets each episode |
| Training step (resident) | RSS 5.6 GB at the 57k-example buffer | **None** — OOM fix holds; flat, well under the 28 GB cap |

**Caveats.** Single-process numbers; the 8-worker run's GPU contention raises inference's *real* share (so the transfer-overhead lever is **more** valuable in production, not less). cProfile %s are of instrumented runtime (proportions valid). Full-cycle self-play uses the measured gen-1 wall from the crashed run.

## P8. Decision — RUN NOW, optimisation optional

**Decision: stop optimising for now and train.** Against the stopping criterion:

1. **It fits — comfortably.** Projected 15-gen wall ≈ **5.3 h** — well under T1 (≤12 h) and inside T2 (≤6 h). The run doesn't need to be faster to deliver results overnight.
2. **RAM is flat** — OOM fix verified (training RSS 5.6 GB).
3. **Worthwhile levers exist** (move-gen 1.32×, select 1.28×, both > the 1.2× bar) — so optimisation isn't *pointless*, but it isn't *required*. Per the goal (a strong net = generations trained), more training beats shaving seconds.

**Training: do not optimise** — answered directly by the data. It's ~1% of the cycle (12 s vs ~19 min self-play); reducing it to zero buys ~1.01×. The conv net + 2 epochs are already efficient and the memory fix solved its only real problem.

**Ranked attack vectors (for when we want tighter iteration):**

| Rank | Lever | Win | Effort | Why |
|------|-------|-----|--------|-----|
| 1 | **Vectorise `_select_action`** | ~1.25× | **Low** | pure-Python UCB loop, 22% of self-play, numpy-vectorisable, bit-identical-testable — best ROI |
| 2 | **Cut inference transfer overhead** | ~1.15–1.2× | Medium | `.cpu()`/`.cuda()` dominate inference; fewer/larger transfers, pinned memory, keep results on GPU |
| 3 | **Bitboard move-gen** | ~1.3× | **High** | move-gen is still 24% post-F2; structural rewrite of `_fill_legal_actions_for_anchor` — biggest but most expensive |
| — | Training | ~1.01× | — | not worth it (~1% of cycle) |

Combined #1 + #2 (low/medium effort) plausibly takes ~5.3 h → ~3.5–4 h. The bitboard (#3) is the only path to the T3 stretch but is a multi-week piece — defer until training results justify faster iteration.

**Next step: none required for the run.** When we want speed, promote lever #1 (`vectorise-select-action`) to its own plan + branch.

### Targets *(yardstick: a 15-gen × 1000-game run)*

| Milestone | Goal | Status |
|-----------|------|--------|
| **T1 — Comfortable overnight** | ≤ **12 h** | ✅ met (~5.3 h) |
| **T2 — Tight iteration** | ≤ **6 h** | ✅ met (~5.3 h) |
| **T3 — Stretch** | ≤ **3 h** | ✗ (needs bitboard move-gen) |

### Candidate levers — detail + Amdahl in the [research doc](../research/self-play-speed-investigation.md)

Vectorise `_select_action` (low) · cut inference transfers (med) · bitboard move-gen (high) · code-level trims (low) · free-threaded Python spike (high) · adaptive-sim endgame taper (low–med). Promotion rule: a lever becomes real work only when we decide tighter iteration is worth the effort — then it gets its own plan + branch.

---

## P1–P6. Method (how the numbers above were produced)

- **P1 harness** — `scripts/profile_self_play.py` plays game(s) single-process at a fixed seed, F2 enabled to match production. Modes: `timing` (wall + MCTS phase split), `cprofile` (function attribution), `memory` (tracemalloc), `train` (synthetic-buffer training cost).
- **P2 self-play time** — built-in MCTS timers gave the coarse inference-vs-rest split; cProfile (single-process — works fine, unlike across the worker pool) attributed the "rest" to move-gen / UCB / board-copy / bookkeeping.
- **P3 training** — `train()` timed over a synthetic 57k-example buffer (1 generation at lookback=1) at the production epochs/batch.
- **P4 memory** — tracemalloc + RSS over one game (tree) and the training step (buffer).
- **P5 full-cycle** — per-game self-play × game counts (with the measured gen-1 wall) + training + arena/elo, to get the phase split and full-cycle Amdahl ceilings.
- **P6 decision** — [P8](#p8-decision--run-now-optimisation-optional).
