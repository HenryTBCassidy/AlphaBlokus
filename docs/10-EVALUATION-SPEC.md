# Evaluation spec — what "a fair fight against Pentobi" means

The project's goal is to beat Pentobi at level 9. That claim is only meaningful if both sides get
comparable resources, so this document records exactly what each side gets, measured rather than
assumed, and where the comparison is still one-sided.

Companion docs: [`05-EVALUATION.md`](05-EVALUATION.md) (metrics and the ladder),
[`plans/fair-pentobi-benchmark.md`](plans/archive/fair-pentobi-benchmark.md) (the work that produced these
numbers).

Last measured: 2026-08-11, on the home box (RTX 3060 Ti 8 GB, 20 cores, 31 GB RAM).

---

## The two instruments, and which question each answers

| | Longitudinal ladder | Fair fight |
|---|---|---|
| **Question** | "Is this net better than the last one?" | "How strong are we really?" |
| Our budget | fixed **400** simulations at every level | **calibrated to equal thinking time** |
| Pentobi's book | **off** | **on** — the engine as shipped |
| Levels | 1–9 | 6–9 only (see below) |
| Comparable across time? | Yes — this is the whole point | No — recalibrate per net and per machine |

Keep them apart. Our net at a fixed budget on every rung is a *fixed yardstick*, and that is what
lets a score convert to one Elo scale; a per-level budget destroys it. Results are written to
separate directories and carry a `condition` field so the training loop can never read one as the
other.

---

## Resource comparison at level 9, measured

Our net is `blokus_cloud_v3` gen-40 (192 filters × 12 residual blocks, ~8 M parameters) at its
**parity budget of 4,096 simulations**. Pentobi is level 9, 1 thread, book on. Serial, no
contention, warm-up game discarded.

| | Our net (gen-40) | Pentobi level 9 |
|---|---|---|
| **Wall-clock per move — mean** | ~12.5 s | **15.4 s** |
| **Wall-clock per move — median** | ~11 s | **5.4 s** |
| Simulations per move | 4,096 (exact, fixed) | ~3.1 M realised (est.); 5.5 M nominal |
| Simulations per second | ~330 | ~202,000 |
| Cost per simulation | 3.18 GFLOPs (one net forward pass) | ~0 FLOPs — integer/branch-bound |
| Compute per move | **13.0 TFLOPs** | not meaningfully expressible in FLOPs |
| Hardware | 1 × RTX 3060 Ti + 1 CPU thread | 1 CPU thread |
| Peak memory | ~2 GB GPU, ~1.9 GB RAM | 1.9 GB RAM (preallocated at level 9) |
| Learned parameters | ~8 M | 0 — handcrafted heuristics + RAVE |
| **Energy per move (est.)** | **~2.5 kJ** (200 W × 12.5 s) | **~0.23 kJ** (15 W × 15.4 s) |

**We match on time. We do not match on anything else, and mostly not in our favour to claim.**

---

## What each number means, and how much to trust it

**Wall-clock per move is the parity axis, and it is directly measured.** Matching the *mean* is
equivalent to matching total time per game, which is the AlphaZero-vs-Stockfish convention. The
median would have given our net roughly a third of the budget, because Pentobi's per-move budget
grows through the game (`0.7·exp(0.1·ply)`, ×0.6 for duo) — early moves are cheap, late ones
expensive.

**Parity is a moving target.** Giving our net more time lengthens games, which pushes Pentobi's
later, more expensive moves into the average. A single-pass calibration under-budgeted us by ~20%;
the fix is to re-measure *at* the computed budget, which `scripts/measure_move_times.py
--verify-sims` does.

**Pentobi's simulation count is derived, not measured.** Its node counter is not exposed over GTP:
`move_values` reports zero visits (only RAVE counts), and `move_info` needs a move ID. The ~202k
sims/sec figure assumes level 7 reaches its tabled 221,867 simulations in its measured 1.1 CPU
seconds, then scales by time. Treat it as order-of-magnitude. It is consistent with the independent
finding that level 9 gets ~14× level 7's effort rather than the tabled 25×.

**FLOPs is genuinely one-sided.** Pentobi's search is integer comparisons and random playouts with
essentially no floating-point work, so a FLOPs figure for it would be invented. Quoting our 13
TFLOPs/move alongside a blank is the honest presentation.

**Energy is the only unit that spans both**, and it is the one that flatters us least: at equal
thinking time we burn roughly **11× more energy per move**. Estimated from device power draw
(a 3060 Ti under load vs one CPU core), not metered, so ±50% — but the direction is not in doubt.

**Our search is CPU-bound, not GPU-bound.** Measured during a live fair-fight run, GPU utilisation
sat at **0–2%** with 302 MiB per worker. At 4,096 simulations only ~0.8 s per move is actual GPU
work; the remaining ~11 s is tree traversal, move generation and board encoding in Python. Two
consequences: the fair fight parallelises far better than expected (6 workers fit in RAM and use
~5 of 20 cores), and **buying a faster GPU would barely change our per-move cost.**

---

## Why the fair fight only runs at levels 6–9

Below level 6, Pentobi barely thinks — level 1 is **3 simulations**, level 5 is 861. Equal thinking
time would give our net fewer than 16 simulations, and below that
`search/mcts.py` plays a **uniform random legal move** (the whole budget is consumed expanding the
root, so no move ever gets a visit). A "fair fight" at level 1 would measure random play, not
strength. The comparison is only meaningful where the opponent actually searches.

Pentobi's simulation budget per level, from `counts_duo` in `libpentobi_mcts/Player.cpp`:

| Level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Simulations | 3 | 21 | 77 | 213 | 861 | 7,280 | 221,867 | 1,109,339 | 5,546,695 |

Note the level 6→7 step multiplies the budget by **30×** while every other step is 3–8×. The
ladder's levels are not an evenly spaced difficulty scale, and a win-rate cliff at level 7 is the
opponent's budget jumping, not our net falling over.

---

## Things that would make the comparison fairer, and are not done

- **Pentobi's thread count is pinned at 1** and recorded. More threads make it faster, not
  stronger, so time parity against a multi-threaded Pentobi would hand our net *less* search
  against an equally strong opponent. Pinning at 1 is the choice that does not flatter either side,
  but it is a choice and it must be disclosed with any result.
- **Our net gets no opening book**; Pentobi does. This is deliberate — the goal is to beat the
  engine as shipped — but it is a real asymmetry against us, and the Elo it is worth is unmeasured
  (`plans/fair-pentobi-benchmark.md` F11).
- **Energy and hardware are not equalised** and realistically cannot be. Disclose, do not pretend.
- **Contention.** Parity is calibrated with one game running. A multi-worker run inflates both
  sides' wall-clock. This does *not* invalidate the result, because the budget is a fixed
  simulation count and our net therefore plays identically under contention — but the per-move
  times recorded inside a parallel run are not parity times, and the calibration must be quoted
  alongside.

---

## How to reproduce

```bash
# 1. Calibrate: what simulation budget matches Pentobi's thinking time at this level?
uv run python scripts/measure_move_times.py \
    --config run_configurations/blokus_cloud_v3_eval.json \
    --net accepted_40.pth.tar --levels 9 --games 4 --sims 400 --book --verify-sims

# 2. The fair fight, at the budget step 1 reports.
uv run python scripts/pentobi_benchmark.py \
    --config run_configurations/blokus_cloud_v3_eval.json \
    --net accepted_40.pth.tar --level 9 --games 100 --sims 4096 \
    --book --condition fair-fight --workers 6
```

`blokus_cloud_v3_eval.json` exists because the config validator rejects the original run config on
training-only grounds (learning rate, Gumbel knobs) even for a script that does no training. Same
architecture, same checkpoint directory, no training knobs.
