# Roadmap — what is in flight, what it is called, and where it stands

The index for `docs/plans/`. Read this first; it tells you which plans are live, what each
workstream is for, and what the next decision is. Every plan doc is a checklist; this is the
map over them.

**Folder convention.** `docs/plans/` holds only what is **actively being worked**.
`docs/plans/future/` holds plans that are real but parked — deliberately not started, with the
reason recorded in each. `docs/plans/archive/` holds finished work, kept for context. A plan moves
out of the top level the moment nobody is working it; if you find something at the top level that
nobody is touching, move it rather than leaving it to rot.

Last updated: 2026-08-10.

---

## Naming

**Workstreams have names, not letters.** An earlier triage (2026-08-04) split the work into
"Stream A/B/C/D", which made every status update unreadable. Those letters are retired. The
table below decodes them so older documents and commit messages still parse, but **do not use
them in new writing**.

| Old label | Name to use | What it was for |
|---|---|---|
| Stream A | **Instruments** | Repair the measurement tools before running anything else |
| Stream B | **Free measurements** | Box experiments that cost £0 and gate all spending |
| Stream C | **Bug sweep** | Two independent audits of the training/self-play code |
| Stream D | **Value head** | Candidate fixes for the leading hypothesis about why training stalled |
| F1–F11 | **Benchmark integrity** | Make the Pentobi comparison fair and self-describing |

Item IDs *within* a plan (F1, V11, N6, S7…) stay as they are — they are local to their
document and each maps to one commit.

---

## Status at a glance

| Workstream | Plan doc | State |
|---|---|---|
| **Instruments** | archived | ✅ Done — PR #69 merged |
| **Bug sweep** | archived | ✅ Done — PR #70 merged |
| **Benchmark integrity** | [`fair-pentobi-benchmark.md`](fair-pentobi-benchmark.md) | 🔄 **Live.** Code merged (#71); F1–F7 done, **F9 the fair fight is the outstanding one** |
| **Free measurements** | this file, below | 🔄 2 of 7 done; 1 slipped; the pilot is the gate on spending |
| **Value head** | [`supervised-network-improvements.md`](supervised-network-improvements.md) N6/N7 | ⬜ Not started. **Now the most promising direction** |
| Corpus v2 | [`future/pentobi-corpus-v2.md`](future/pentobi-corpus-v2.md) | ⏸ Parked — 9/16 done, needs a 3-day box run |
| Auxiliary heads | [`future/score-auxiliary-target.md`](future/score-auxiliary-target.md) | ⏸ Parked — code built, A/B never run |

---

## What we learned in the 2026-08 investigation

The full record is in [`../research/investigation-2026-08/`](../research/investigation-2026-08/)
— read [`06-handoff.md`](../research/investigation-2026-08/06-handoff.md) then
[`03-synthesis.md`](../research/investigation-2026-08/03-synthesis.md). It lived in gitignored
`temp/` until 2026-08-10, which is why it was unfindable; treat
[`00-briefing.md`](../research/investigation-2026-08/00-briefing.md) as **superseded** by
[`00-briefing-erratum.md`](../research/investigation-2026-08/00-briefing-erratum.md).

The load-bearing conclusions, with what has changed since:

1. **The plateau is not capacity.** A box probe found `xl` no better than `large`. Net size is
   not the constraint.
2. **The value head has no demonstrable skill beyond guessing from whose turn it is.** Its
   output tracks mover colour at 0.76–0.81 while outcomes track it at only 0.52–0.62. Both
   production checkpoints' confidence intervals bracket zero skill. This is the leading
   hypothesis and the reason the **Value head** workstream exists.
3. **A healthy training operator has never been pointed at the current net.** Every stalled run
   since has a since-diagnosed defect. That is what the pilot tests.
4. **Pentobi's difficulty levels are a hardcoded simulation count** —
   `counts_duo = {3, 21, 77, 213, 861, 7280, 221867, 1109339, 5546695}` in
   `libpentobi_mcts/Player.cpp`. The level 6→7 step multiplies its search by **30×** while every
   other step is 3–8×, so the ladder is not an evenly spaced difficulty scale and reading it as
   one was misleading.
5. **Pentobi's opening book had never been active** — its binary looks for book files beside
   itself and the build directory held none, while the engine still reported `use_book 1`. Every
   Pentobi number in the project's history faced a book-free opponent. Fixed 2026-08-05 by
   symlinking the books; the strength this is worth is unmeasured (V11).
6. **Level 9 is genuinely ~156 Elo stronger than level 7** (measured 2026-08-10, 200
   colour-balanced games, engine vs engine, pooled score 0.710, CI [0.647, 0.773]). An earlier
   claim that Pentobi *saturates* above level 7 was wrong — it came from three of our own ladder
   cells whose intervals are ±87 Elo each. Realised effort is 13.9× between those levels, against
   a tabled 25×, so quote realised effort whenever a ratio is load-bearing.
7. **The fair fight is measured (2026-08-11).** At equal thinking time, with Pentobi's book on, our
   best net scores **0.315** at level 9 (CI [0.232, 0.411]) — so we lose, and it is resolved. Two
   refinements to what was previously written here: 10× more search moved us from 0.22 to 0.315
   against a *stronger* opponent, so **"search is not the lever" was too strong** — it is real but
   insufficient. And the split by colour is the actual finding: **0.630 as first mover (we beat level
   9), 0.000 as second mover across 50 games.** A colour-handling defect has not yet been ruled out.
8. **Previously written here, now superseded by (7):** Every net-vs-Pentobi
   number the project owns was taken at **400 simulations for us against Pentobi's own budget**,
   with Pentobi thinking ~12× longer per move at level 9. On that footing the best net is ~280 Elo
   below level 9 once first-mover advantage is corrected for. Estimating from the ~3.6 doublings of
   thinking time involved, equal time would buy ~70–145 Elo — which would leave a gap, and is the
   basis for "search is not the lever; the network is the constraint". **That estimate has never
   been tested. F9 is the experiment that tests it and it has not been run.** Until it has, treat
   the ranking of the value head above further search work as well-founded but not proven.

Conclusion (6) plus (7) is why **Value head** is now ranked above further measurement.

---

## Free measurements — the remaining items

These were "Stream B". All cost £0. Kept here rather than in their own plan because most are
one-command box runs, not multi-commit work.

| ID | Item | State |
|---|---|---|
| M1 | Ladder the checkpoints around the best net, to check we warm-start from the right one | ✅ Done. gen-40 is the best; gen-32 ties it, gen-36 is worse. Question closed |
| M2 | Eval-time search scaling at the top levels | ➡️ Superseded — became F9/F10 in [`fair-pentobi-benchmark.md`](fair-pentobi-benchmark.md) |
| M3 | Learning-rate sweep on a frozen buffer | ⛔ **Blocked** — both historical replay buffers were deleted, and no generate-only entry point exists |
| M4 | Width shadow test (`top_k` 64 vs 128) | Written on `feat/width-shadow-probe`, never run. Deferred until after the pilot by design — 15–30 box hours, informs a later run only |
| M5 | bfloat16 vs float32 self-play A/B | ⚠️ **Slipped.** Harness exists (`scripts/validate_jax_search.py --dtype`), short, free, never run |
| M6 | **The pilot** — 15–20 generations with a healthy training step | ⬜ Not run. Config is on main (`run_configurations/blokus_pilot_b6.json`). **This is the gate on all spending** |
| M7 | Games-per-generation arm | ⬜ Optional second arm of M6 |

### M3's blocker, stated once

There is no way to generate self-play data without also training on it — every path calls
`generate_games` then trains. A generate-only job (~5,000 games, frozen weights, ~80 min, £0)
would unblock M3, supply a fresh held-out eval set, and settle whether the value head *degraded*
or was merely graded against another net's games. It has not been written.

### M6's threshold needs restating before it runs

The pilot's pre-registered bar is "weighted ladder ≥ 0.375 at generation 15" against a 0.344
baseline. **Both numbers counted draws as losses.** PR #71 scores a draw as half a win, which
lifts every historical figure by 0.7–1.0 pp (measured on this project's own ladder files). Until
the threshold is restated on the new convention the pre-registration is void — and
pre-registration is the only thing standing between this run and the post-hoc reasoning that
already cost three paid runs.

---

## Value head — the case for doing this next

Was "Stream D". Its two live items already exist as **N6** and **N7** in
[`supervised-network-improvements.md`](supervised-network-improvements.md); that document is the
home, and the duplicate ideas registered in `IDEAS.md` (I8) should not be worked separately.

| Item | What | Prerequisite |
|---|---|---|
| **N6** | Outcome-balanced value sampling, weighted **conditionally on colour** | Positions do not record whose turn it is |
| **N7** | Win/draw/loss value head instead of one scalar | Follows N6 |

**The trap in N6.** The skew lives in `P(win \| White to move) ≈ 0.73` while the *marginal* label
split is ~50/50. So balancing the marginal does nothing at all — the weighting has to be
conditional on colour. Three auxiliary heads have already been built and measured as doing
nothing; this is the same failure mode waiting to happen.

**The prerequisite.** `ProcessedExample` is `tuple[compact_board, (indices, probs), value]` — no
colour. Two routes:

- Derive it with `infer_mover_colour()`, which exists. Cheap, but its own docstring notes the
  parity it relies on "breaks and the colour is genuinely unrecoverable" once a player has
  passed — and passing happens in the **endgame**, precisely where the value target matters most.
- Thread the true player through `episode.py` → self-play store → replay buffer → dataset. Exact,
  and it also removes the workaround in the colour-value diagnostic.

Take the second. **Both replay buffers being deleted means there is no legacy data to migrate**,
so the schema change is unusually cheap right now, and that window closes the moment new data is
generated. This is a live ordering argument: do the plumbing *before* the generate-only job, or
the one dataset everything depends on will lack the column.

---

## The next decision

Ranked by evidence, not by sequence:

1. **Merge PR #71.** CI green, reviewed twice independently (codex found 6 defects, an Opus pass
   confirmed all 6 and found 4 more), all fixed, 1041 tests passing.
2. **Start N6's plumbing.** No GPU needed, no file overlap with #71, and the schema window is
   open only until new data is generated.
3. **M5** — short, free, and it has slipped twice.
4. **F9, the fair fight** — the one measurement that says where the project really stands.
5. **Then decide on the pilot (M6)**, with its threshold restated first.

An honest option that stays on the table: the evidence now says search will not reach level 9 and
the network is the constraint. If N6 and N7 both come back within noise, the remaining levers are
architectural, and **scaling the goal down to level 6–7 or stopping** is a legitimate outcome
rather than a failure. That was pre-registered in the investigation and should not be quietly
dropped.
