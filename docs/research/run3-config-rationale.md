# Run-3 Config Rationale — Sizing the Overnight Training Run (2026-06-29)

Why `run_configurations/blokus_run3_overnight.json` is configured the way it is. Synthesised
from four independent analyses run in parallel: (1) past-run trajectories from W&B, (2) net
capacity vs data needs, (3) AlphaZero/KataGo scaling references, (4) the 16-hour compute budget.
Companion to `numba-hot-path-results.md` (the speedups that made this run feasible).

---

## TL;DR — the config

Run-3 = the run-2 bignet config with **two changes**: `replay_buffer_games` 5000 → **10000**, and
`num_generations` 60 → **120** (so the cosine LR schedule spans the real multi-night target;
`--resume` continues across nights). Everything else (128f×8b net, 800-cap branching sims,
1000 games/gen, epochs 1) is kept — it was the strongest config we'd run and it was never
changed *because* it was the strongest, only because it crashed.

| Knob | Value | One-line reason |
|---|---|---|
| Net | **128f×8b** (2.45M params) | Capacity matched to the data a night produces; bigger is data-starved (§Net) |
| `num_eps` (games/gen) | **1000** | Total games/night is fixed; 1000 maximises iterations, which is what we're short on (§Games) |
| `replay_buffer_games` | **10000** (~10 gens) | The new lever (buffer bug now fixed); feeds the data-starved value head (§Buffer) |
| `epochs` | **1** | Reuse is already ~10× via the buffer; epochs=2 didn't help in past runs (§Reuse) |
| `num_mcts_sims` | **800**, branching schedule | Thinnest sims/branching ratio in the reference set but it demonstrably learned; keep for comparability, A/B later (§Sims) |
| `num_generations` | **120** | Multi-night target so cosine LR anneals correctly; ~45 gens fit per 16h night (§Budget) |
| cpuct / Dirichlet / lr-cosine / fp16 / conv head | unchanged | Match AlphaZero norms, all worked |

---

## The reality check that frames the whole thing

The four analyses collide on one number, and it sets expectations:

- **Reaching Pentobi-competitive play** needs ~**0.5–3M self-play games** (~30–60M positions),
  interpolated between Connect-4 single-GPU efforts (~75k games) and low-end chess, scaled for
  Blokus's 17,837-action space (scaling analysis).
- **16 hours on the box** produces ~**50k games** (~5.5M training examples) — self-play is
  88–96% of every generation at 0.95 games/s (budget analysis).

So **one overnight run delivers ~1–10% of the way to the goal.** Run-3 is *one increment*, not a
"beat Pentobi" run. Its jobs: **(a) validate 128f×8b end-to-end** (it crashed at gen 14/60 last
time and never showed its ceiling), **(b) extend the still-climbing trajectory**, **(c) confirm
the curves keep rising** so we can `--resume` toward the real target across nights. The actual
unlock for the goal is **more self-play throughput** — the in-process batched-MCTS rewrite (10×+)
turns 0.5–3M games from months into days. Tonight is about banking a clean, resumable increment.

---

## Param-by-param rationale

### Net size — keep 128f×8b, don't go bigger (for 16h)
Exact param counts (instantiated from `net.py`): 64f×4b = **0.34M**, 128f×8b = **2.45M**,
128f×10b = 3.0M, 192f×10b = 6.8M, 256f×10b = 12.0M.

16h yields ~**3.0M distinct positions** (54,700 games × 55 moves; the 2× transpose symmetry
inflates the *training set* to ~6M but adds no new information). Rule-of-thumb (~10 positions/param,
softened by AlphaZero's soft MCTS targets) → 3M positions well-supports a **~1–3M-param** net.

- The old **64f×4b underperformed from lack of capacity, not overfitting** — 0.34M is tiny for a
  17,837-action game. Run data confirms: 128f×8b hit ~**0.38–0.44 top-1 policy accuracy** vs
  **0.15–0.30** for every 64f×4b run (~2× better at fitting the MCTS target), with the strongest
  Elo climb.
- **128f×8b (2.45M)** sits at the upper edge of well-matched — it *wants* ~8–25M positions to
  saturate, so in 16h it's data-limited but not overfit, and the slight over-capacity is the
  correct AlphaZero posture (soft targets + buffer reuse reward it).
- **192f/256f (6.8–12M)** need **5–20× more self-play than one night provides** → undertrained +
  overfit-prone + VRAM-squeezed on 8GB (slower self-play → even less data). Only worth it once
  self-play volume scales with the net.

### Games per generation — 1000, and "more" is not better here
The 16h budget produces ~**50k games total regardless of `num_eps`** (self-play dominates). So
`num_eps` doesn't change total data — it only trades iteration count vs data freshness:

| games/gen | generations in 16h | reuse |
|---|---|---|
| 1000 | ~49 | 5× (at B=5000) |
| 1500 | ~34 | 3.3× |
| 2000 | ~25 | 2.7× |
| 3000 | ~17 | 1.7× |

Every past run was **undertrained and still climbing** (policy loss never flattened; policy entropy
stayed high = underfit, never collapsed). So we're **iteration-limited**, which argues for the
**1000** end (most generations). Going higher buys fewer iterations for the *same* total data.
(Floor: don't go below ~1000 — the 300-games/gen run plateaued by gen ~25.)

### Replay buffer — 10000 (~10 gens): the new lever
Until the rolling-buffer refactor (merged via PR #16), a window-shrink bug pinned every past run
to ~5 generations regardless of config — so `replay_buffer_games` is a *real knob for the first
time*. Scaling references put the sweet spot at **~5–20 generations** of data in the buffer
(AlphaGo Zero ~20×, alpha-zero-general 20 iters); ~10 is the middle. At `num_eps=1000`,
**B=10000 ≈ 10 generations**, giving **reuse = epochs × B/num_eps = 10×** — squarely the
AlphaZero/Leela norm.

This is the main within-run lever for the **value head**, which is the documented weak spot: value
loss floored within ~4 generations and value calibration *never* improved in any run — classic data
starvation (one noisy outcome shared across ~30 positions). A wider buffer gives more game-outcome
diversity per training step. (Fundamentally the value head needs more *total* games, which loops
back to throughput.) **Memory:** 10000 games is safe with the now-merged lazy board encoding (compact
int8 storage, densified per-batch) — but verify in the 1-gen sanity check.

### Reuse / epochs — keep epochs=1
Reuse is already ~10× via the buffer (above), without touching `epochs`. The one past run that used
**epochs=2** got the lowest policy loss but the *worst* outcome — passes weren't the bottleneck,
data was. More epochs would push reuse past ~10 and risk value-head overfitting. Keep **1**.

### MCTS sims / search depth — keep 800 this run; it's the #1 follow-up experiment
This is the sharpest finding. `sims ÷ branching` across the reference set:

| Game | Sims | Branching | ratio |
|---|---|---|---|
| AlphaZero chess | 800 | ~35 | ~23× |
| AlphaGo Zero (Go) | 1600 | ~250 | ~6.4× |
| AlphaZero (Go) | 800 | ~250 | ~3.2× |
| **Blokus (ours)** | **800** | **~400 (opening)** | **~2×** |

**~2× is the thinnest ratio in the whole reference set** — at the opening, MCTS visits each of ~400
children only ~twice, so early-game policy targets are close to "prior + Dirichlet noise." It's
**defensible** because (a) Blokus branching *collapses fast* (only ~400 for the first few plies, far
lower on average over a 55-move game) and our branching-scaled schedule already concentrates sims
where branching is high, and (b) it *demonstrably learned* (128f reached ~0.4 top-1, still climbing).

**Decision:** keep **800-cap branching** for run-3 so the result is comparable to the crashed bignet
run, and because raising sims costs games (hurting the data-starved value head). But this is the
**#1 knob to A/B next**: lift the cap toward **1,200–1,600**, or add KataGo-style playout-cap
randomisation (a code change), if early policy targets look noisy.

### Everything else
- **cpuct 2.5** — within AlphaZero's 1–4 range; fine.
- **Dirichlet ε=0.25 / α=0.03** — ε matches AlphaZero; α=0.03 suits ~400 early legal moves (Go used
  0.03 for 362). Keep.
- **temp_threshold 12** — keep (comparable to past runs; ~first 12 plies sampled, then greedy).
- **update_threshold 0.55 (arena gate)** — matches AlphaGo Zero. **Watch it:** past runs churned
  ~50/50 at the gate (Elo noise rejecting good nets → wasted iterations). AlphaZero dropped gating
  entirely; candidate to loosen/drop if rejections pile up.
- **lr 0.001 + cosine** — `num_generations=120` sets the cosine `T_max`, so it anneals over the real
  run. **This is why num_generations is the multi-night target, not one night** — set it to 45 and
  the LR hits ~0 by morning, making `--resume` pointless.
- **Eval (arena 50, elo 50, symmetry 100)** — kept. Eval is <9% of a generation; the budget analysis
  showed trimming it recovers only 1–3 generations, not worth the loss of the progress curves.
- **fp16 inference + conv policy head** — keep; both proven, and the conv head is what makes the
  17,837-action policy cheap.

---

## What to watch during the run
- **Value loss / calibration** — does the wider buffer finally move it? (the one metric that never improved)
- **train vs val divergence** — the over-capacity check for 128f×8b on limited data
- **gate acceptance rate** — if it rejects most gens, loosen `update_threshold`
- **per-gen wall-clock** — confirms the ~20 min/gen estimate and how far the curve gets per night

## Next experiments (post-run, in priority order)
1. **In-process batched-MCTS rewrite** — the real throughput unlock (10×+), the only path to the
   0.5–3M games the goal needs. (The cross-worker inference *server* is already ruled out — see
   `inference-server-bignet.md`.)
2. **Sims A/B** — 800 → 1,200–1,600 (or playout-cap randomisation) to thicken early policy targets.
3. **Bigger net** — only *after* throughput scales to feed it (192f+ wants 25–70M positions).

## Caveats
Data-requirement and total-games-to-strong-play figures are interpolations anchored to published
AlphaZero/KataGo/Connect-4 numbers, not measured for Blokus — treat them as planning targets to
validate against our own Elo-vs-games curve. The load-bearing measured numbers are: 0.95 games/s at
16 workers, ~55 moves/game, and the per-net param counts (exact).
