# Adaptive Simulation Budget (I1)

Implements [IDEAS.md I1](../../IDEAS.md): scale the per-move MCTS sim budget with the root
branching factor instead of a flat count — so the opening (huge branching, ~450 moves) gets
adequate coverage and the endgame (tiny branching, <10 moves) stops re-walking a trivial tree
hundreds of times. Archived: implemented, benchmarked, and deployed.

*(The rest of the original plan — "confirm Dirichlet noise", "launch the first run", "merge to
main", "plan Pentobi" — was process that's since been done or moved to its own plan; trimmed to
this technical record.)*

## What landed

`MCTSConfig` gains `sim_schedule` (`"flat"` default | `"branching"`), `sims_min`,
`sim_branching_scale`. `MCTS._move_sim_budget` returns `num_mcts_sims` unchanged for `"flat"`
(bit-identical — the K=1 golden test is unaffected), else
`clamp(round(sim_branching_scale · branching), sims_min, num_mcts_sims)`, where `branching` is
the root's legal-move count (read from the expanded-root cache, else one move-gen call, no NN).
So `num_mcts_sims` is the **opening cap** and `sims_min` the **endgame floor**. Tested in
`tests/test_core/test_batched_inference.py::test_move_sim_budget_schedules`.

## Result (2026-06-20)

Benchmarked branching (cap 400 / floor 60) vs flat-300 — same code/net (64f×4b), via the
standardised benchmark:

- **~14% faster at 16 workers (1.65 → 1.88 games/s)**, while doing *more* opening search
  (400 vs 300 sims) and far less endgame waste (60 vs 300).
- **Single-worker neutral** (heavier opening offsets the lighter endgame); the multi-worker
  gain comes from ~27% fewer *total* sims/game → fewer GPU round-trips → less contention across
  the 16 workers sharing one GPU.

## Deployed

First production use: `run_configurations/blokus_run1_taper.json` (branching, cap 400 / floor 60),
launched as the first principled overnight run (W&B `ta6t5s6g`, 2026-06-20). Dirichlet root
noise was already active in the run configs (ε=0.25, α=0.03). Measuring that run's strength
against Pentobi is its own plan: [pentobi-harness.md](../pentobi-harness.md).
