# Box results — raw numbers only

Written by the [GPU-runner](gpu-runner.md), read by the [Analyst](analyst.md).

**No interpretation in this file.** Config, raw counts, intervals the harness produced, wall-clock,
and evidence the run actually completed. Conclusions go in the plan documents.

Large artefacts (HTML reports, ladder JSON) live in `results/` or stay on the box; reference them by
path rather than pasting them.

---

## F9 — fair fight: gen-40 vs Pentobi level 9 at equal thinking time
**Completed** 2026-08-11, 7,407 s wall-clock. Queue entry: F9.

**Config**
- net `accepted_40.pth.tar` (blokus_cloud_v3 gen-40, 192f × 12b), **4,096 simulations**, batch 16,
  `sim_schedule=flat`, `dirichlet_epsilon=0`, temp 0, seed 7
- Pentobi level 9, 1 thread, **book ON** — probe confirmed engaged: opening move returned in 0.00 s
  with no search tree (`root_children: 0`)
- 100 games, 6 workers, `--condition fair-fight`
- Payload: `temp/runs/blokus/blokus_cloud_v3/PentobiFairFight/ladder_accepted_40.pth.tar_20260811T123322Z.json`
- Parity basis: F8 measured Pentobi L9 at 15.38 s/move (book on) vs our net 1.50 s/move at 400 sims

**Raw counts**

| | W | L | D | games | score (W+D/2) |
|---|---|---|---|---|---|
| **Total** | 30 | 67 | 3 | 100 | **0.315** |
| as White (first mover) | 30 | 17 | 3 | 50 | 0.630 |
| as Black (second mover) | **0** | 50 | 0 | 50 | **0.000** |

Harness-reported 95% CI on the total: **[0.232, 0.411]**. Headline line:
`level 9: net 30-67-3 (win rate 30%, 95% CI [23%, 41%])`.

**Per-chunk** (6 chunks, `net W-L-D (as white W-L-D)`):
`6-9-1 (6-1-1)` · `4-12-0 (4-4-0)` · `4-12-0 (4-4-0)` · `6-10-0 (6-2-0)` · `5-12-1 (5-3-1)` ·
`5-12-1 (5-3-1)`

**Completion evidence:** all 6 chunks present and summing to 100 games; result JSON written and
parses; 0 engines left running; no OOM in the log; book probe recorded in the payload.

**Runner note, not an interpretation:** every one of the 30 wins is in the White column and the Black
column is 0/50 across all six chunks independently. Flagging for the Analyst as a possible defect
rather than a result — an absolute cell warrants a colour-handling check before anything is built on
it.

---

## F8 — time-parity calibration
**Completed** 2026-08-11. Serial, warm-up game discarded, Pentobi 1 thread.

| Condition | our net @ 400 sims | Pentobi L9 | ratio | implied parity |
|---|---|---|---|---|
| book off | 0.95 s/move | 8.96 s/move (median 3.51) | 9.5× | ~3,800 sims |
| book off, re-measured at 3,788 sims | 12.45 s/move | 14.87 s/move | 1.19× | ~4,500 sims |
| book on | 1.50 s/move | 15.38 s/move (median 5.44) | 10.2× | ~4,100 sims |

Level 8, book off: our net 1.70 s/move, Pentobi 4.02 s/move (median 2.24), ratio 2.37× → ~950 sims.
Re-measured at 946 sims: 2.99 vs 3.69 s/move (ratio 1.24×).

---

## F2 — Pentobi level 7 vs level 9, engine vs engine
**Completed** 2026-08-10. 200 games, `--nobook`, two colour-swapped batches of 100, 6 engines.

| Batch | L9 score |
|---|---|
| A (L7 moves first) | 0.730 |
| B (L9 moves first) | 0.690 |
| **Pooled** | **0.710**, 95% CI [0.647, 0.773] |

CPU seconds per game: L9 195, L7 14 (**13.9×**). Nominal ratio from `counts_duo` is 25×.

**Completion evidence:** both `.dat` files at 100/100 rows; no engines left running.
