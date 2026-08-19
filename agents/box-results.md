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

---

## I3 (queue F11) — what Pentobi's level-9 opening book is worth
**In progress**, launched 2026-08-19 10:39 UTC. tmux session `i3_book` on `gpu-linux`.
Driver `temp/run_i3_book_delta.sh`, log `temp/benchmarks/i3_book_delta.log`,
output `temp/benchmarks/f11/{A_bookfirst,B_nobookfirst}.dat`.

**Config as run** — exactly the queued commands, no parameters altered.
- Batch A: `--black` = L9 book-on, `--white` = L9 `--nobook`, 100 games, `--threads 3`
- Batch B: `--black` = L9 `--nobook`, `--white` = L9 book-on, 100 games, `--threads 3`
- Box free at launch: GPU 41 MiB / 0%, 27 GB RAM available, 0 pentobi processes, 0 orphans

**Book-engagement probe** (run before launch, single engine, empty board, level 9):

| arm | first move | wall-clock | `move_values` |
|---|---|---|---|
| book on | `e8,e9,f9,d10,e10` | **0.50 s** | empty (no search tree) → book hit |
| book off (`--nobook`) | `f8,e9,f9,g9,e10` | **31.69 s** | full tree, 941,555 sims on top move |

`book_duo.blksgf` is present and symlinked beside the binary (dated 2026-08-05, the gotcha-18 fix).
The two arms return different first moves, so the book changes play rather than only caching it.

### Harness facts established while launching — these change how the `.dat` is read
Not an interpretation of the result; a property of `twogtp` read off its source on the box
(`/home/henry/code/pentobi/twogtp/TwoGtp.cpp`).

1. **`twogtp` alternates colours by itself**, every game: `play_game()` line 90 sets
   `player_black = game_number % nu_players`. So each 100-game batch is *already* colour-balanced
   ~50/50, and no flag is needed to get that. The queue entry and
   `docs/plans/evaluation-instruments.md` I3 both state that `twogtp` has no colour-alternation flag
   and that an unswapped run would therefore measure colour rather than the book. The absence of a
   *flag* is correct — `Main.cpp` has no such option — but the alternation happens regardless.
2. **The `Result` column is normalised to the engine passed as `--black`, not to the black colour.**
   `get_result()` takes the score from `Color(0)` and then applies `if (player_black != 0)
   result = 1 - result`. The queue entry describes it as "black's score".
3. **Consequence for combining the two batches.** Batch B is batch A with the roles of the two
   engines exchanged, so B's column measures the *no-book* engine. B's book-side score is
   `1 - Result`. Averaging A's and B's `Result` columns directly returns ~0.5 by construction,
   whatever the book is actually worth.
4. **Per-game colour is recoverable.** The `.dat` header is
   `# Game	Result	Length	PlayerB	CpuB	CpuW	Fast`; `PlayerB` records which engine index played
   black in that game, so a colour split can be computed after the fact.

Filed for the `instruments` Worker, who owns the plan text and the queue entry's wording.

### Batch A — verified complete
`END A_bookfirst rc=0` at 2026-08-19 13:56:00 UTC. Elapsed 3 h 16 m 45 s for 100 games
(0.51 games/min). Queue estimate for *both* batches was ~4 h; batch A alone took 3 h 17 m.

**Completion evidence:** 100 data rows; game ids 0–99 with no duplicates and no gaps; header intact;
every `Result` value in {0, 0.5, 1}; 0 failure signatures (`Killed|out of memory|Error:|is not
running|Segmentation`) in the log; RAM never below 16 GB available during the batch.

`Result` is the score of the **`--black`-slot engine**, which in batch A is the **book-on** engine.

| batch A | W | D | L | n | score for book-on engine |
|---|---|---|---|---|---|
| pooled over both colours | 46 | 7 | 47 | 100 | **0.4950** |
| `PlayerB=0` — book-on engine held the **black colour** | 45 | 3 | 2 | 50 | **0.9300** |
| `PlayerB=1` — book-on engine held the **white colour** | 1 | 4 | 45 | 50 | **0.0600** |

CPU seconds per game: book-on slot **153.4**, no-book slot **195.0**.
Game length in plies: min 9, median 17, max 34.

**Runner note, not an interpretation.** The two colour rows are near-absolute in opposite directions
(0.93 / 0.06) while the pooled number sits at 0.495. Both engines are level 9 and differ only in the
book, so the colour term here is large relative to the treatment. Flagging that the pooled column is
the one the queue entry points at, and that reading the book's value requires the within-colour rows
— the arithmetic is the Analyst's call, not mine. Batch B (roles exchanged) will give the second half.

Raw file fetched to `temp/benchmarks/f11/A_bookfirst.dat` (also on the box). Parser used:
scratchpad `parse_i3.py` — tallies only, no Elo.
