# Adaptive Sims (I1) + First Principled Overnight Run

Two folded items, ahead of the first real training run designed from the
run-design priors (see the discussion summarised in
[linux-performance-findings.md](../research/linux-performance-findings.md) and the
AlphaZero-scaling research): implement the **adaptive simulation budget** (IDEAS.md
I1), confirm **Dirichlet root noise** is active, then launch a run that uses the
priors (more generations, fewer games/gen, branching-scaled sims).

---

## Checklist

| # | Item | Effort | Done |
|---|------|--------|------|
| 1 | **Adaptive sim budget (I1)** — `sim_schedule="branching"` scales per-move sims with the root branching factor, clamped `[sims_min, num_mcts_sims]`; `"flat"` default stays bit-identical | ~0.5 day | ✅ |
| 2 | **Confirm Dirichlet noise** — already on in the run configs (`dirichlet_epsilon=0.25`, `α=0.03`); self-play passes `add_root_noise=True` | — | ✅ (already on) |
| 3 | **Launch the first principled overnight run** — new config applying the priors | run | |
| 4 | **(while running) merge this branch → main** | ~0.5 hr | |
| 5 | **(next) plan + build the Pentobi benchmark harness** | TBD | |

---

## 1. Adaptive sim budget (I1) — done

`MCTSConfig` gains `sim_schedule` (`"flat"` default | `"branching"`), `sims_min`,
`sim_branching_scale`. `MCTS._move_sim_budget` returns `num_mcts_sims` for `"flat"`
(bit-identical — golden test unaffected), else
`clamp(round(scale · branching), sims_min, num_mcts_sims)`. Branching is read from
the expanded-root cache (self-play, where Dirichlet expands it) or via one move-gen
call (no network). Rationale + the Blokus branching data (~450 opening → <10 endgame)
in [IDEAS.md I1](../IDEAS.md). Tested in `test_move_sim_budget_schedules`.

> **Design note (from I1):** tapering *down* the endgame is the clean win (300 sims
> for <10 moves is absurd). Scaling *up* the opening is uncertain ROI and better
> served by the prior + Dirichlet noise (already on) than by brute sims — so the cap
> stays modest. The exact curve is empirical: validate Elo vs flat-300.

## 3. First principled overnight run — `run_configurations/blokus_run1_taper.json`

Applies the priors: **64f×4b net, 16 GPU workers, K=16, Dirichlet on**, and:
- `sim_schedule="branching"`, cap `num_mcts_sims=400`, floor `sims_min=60`, scale 1.0
  → ~400 sims in the opening, tapering to 60 in the endgame (≈ cheaper than flat-300
  *and* better-shaped coverage).
- **300 games/gen × 150 generations = 45k games** (favouring many generations over
  large generations, per the iterations-dominate finding).
- Arena 50 (acceptance gate), Elo 20, symmetry-diagnostic 50 (lighter eval overhead
  for a long run).
- Estimated ~10–12 h on the 3060 Ti (overnight).

Open dial to confirm at launch: the **sim cap** (400 balanced; 300 = conservative
endgame-only taper; 600–800 = aggressive deeper opening, ~more compute).
