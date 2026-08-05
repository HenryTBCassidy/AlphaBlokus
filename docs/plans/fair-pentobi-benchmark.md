# Fair Pentobi benchmarking

The ladder gives our net a fixed 400 MCTS simulations per move and Pentobi a fixed `--level N`.
Both are search-effort settings and the two efforts are unrelated, so the ladder has never
controlled for how much either side thinks. At level 9 Pentobi searches 5,546,695 simulations
against our 400, and thinks ~12× longer per move. This plan makes the comparison defensible.

It also fixes a discovered defect: **Pentobi's opening book has never been active**, so every
number the project has ever quoted is against a weaker-than-shipped opponent.

Prerequisites: none. Everything through F6 is £0 and most of it needs no GPU.
Companion docs: [`05-EVALUATION.md`](../05-EVALUATION.md),
[`pentobi-corpus-v2.md`](pentobi-corpus-v2.md) (fact 13 — the book), and
[`plan-format`](../guides/PLAN-FORMAT.md).

---

## Design decisions, up front

Two instruments, not one. **Do not replace the ladder.**

1. **The longitudinal ladder** stays exactly as it is: fixed 400 sims, book-free, L1–9,
   100 games/level. Every historical number lives on this scale and it is the only way to
   compare a new net against gen-40. Its one job is tracking progress over time.
2. **The fair-fight condition** is new and separate: the top levels only, our net at a
   calibrated budget, book **on**, run serially so neither side is contended. Its one job is
   answering "how strong are we really".

Why not merge them: our net at a fixed 400 sims on every rung is a *fixed yardstick*, which is
what lets a score against each level convert into a common Elo scale. Make the net's budget
depend on the level and that property is destroyed — cross-level comparison stops meaning
anything, and the saturation question can never be asked again.

**Report simulation counts; do not enforce them.** A Pentobi playout (microseconds, RAVE) and
one of our simulations (a 192f×12b ResNet forward pass) are not the same unit, so a
`min(time, sims)` cap equalises a quantity that is not comparable. It also breaks concretely at
the bottom: `search/mcts.py:226` plays a **uniform random legal move** when
`num_mcts_sims <= mcts_batch_size`, so a 3-simulation budget at L1 (K=16) is random play, not a
weak search.

**Time parity is a claim about one hardware pairing, not about the engines.** Pentobi's thread
count changes its wall-clock but not its strength, so equal-time is only meaningful with threads
pinned and hardware disclosed. State it as a convention (as AlphaZero–Stockfish did), not as an
objective measure.

---

## Checklist

| # | Item | Effort | GPU? | Priority | Done |
|---|---|---|---|---|---|
| F1 | Activate Pentobi's opening book and verify engagement | 1 h | no | **Critical** | ✅ |
| F2 | Pentobi L7 vs L9 head-to-head — is the top of the ladder flat? | 3 h box | no | **Critical** | 🔄 |
| F3 | Measure Pentobi's *realised* effort per move; explain 14.7× vs 25× | 2 h box | no | **Critical** | |
| F4 | Record the full comparison context in every ladder result | 3 h | no | High | ✅ |
| F5 | Seed the net's RNG; audit opening diversity per arm | 3 h | no | High | ✅ |
| F6 | Unify draw handling to score = W + D/2 everywhere | 1 h | no | High | ✅ |
| F7 | Refit ladder Elo with colour + draw terms | ½ day | no | High | ✅ |
| F8 | Calibrate the time-parity budget on non-book moves | 2 h box | yes | High | 🔧 tooling ready |
| F9 | **The fair fight**: net vs L9 at parity budget, 300–400 games | 1–2 d box | yes | **Critical** | |
| F10 | Search-scaling slope: 400/1,600/6,400 at L8 and L9 | 1–2 d box | yes | Medium | |
| F11 | Book-on vs book-off strength delta at L9 (plan item V11) | 4 h box | yes | Medium | |

F1–F3 gate everything else: every later interpretation depends on them.

**Code landed** (PR: fair-pentobi-benchmark): F1's follow-through (explicit `nobook`), F4, F5,
F6, F7 and F8's tooling, plus the condition separation described below. F2 is running on the
box; F3 and F9–F11 are box runs that use the landed code.

### The condition separation, and why it is not merely tidiness

`Coach._check_ladder_and_drift` reads **every** `ladder_*.json` in one directory as a single
series, and `ladder_point_from_payload` looks only at `net` and `metrics.weighted_score` — it
does not inspect sims, book state, level range, or games per level. So a fair-fight result
dropped into `PentobiLadder/` would be absorbed into the series that drives keep-best-by-ladder
and the drift circuit-breaker, comparing a book-on 300-game L9-only score against book-free
100-game L1–9 weighted scores. That corrupts promotion and can trip the catastrophe stop on
nothing.

Two independent defences, because one is a convention and conventions get broken:

1. **Separate directories.** `--condition fair-fight` writes to `PentobiFairFight/`.
2. **`is_longitudinal()`**, which Coach filters through, so even if both land in one directory
   only `condition == "ladder"` (or a pre-2026-08-05 payload with no key) can drive promotion.

---

## F1. Activate Pentobi's opening book and verify engagement ✅

**Current state — the defect.** `pentobi_gtp/Main.cpp:110` sets
`books_dir = application_dir_path`, i.e. the directory holding the binary. That directory
(`~/code/pentobi/build/pentobi_gtp/`) contained no `.blksgf` files, so no book could ever load.
`param_base` reports `use_book 1`, which is the *setting*, not the capability — which is exactly
how this went unnoticed. Already documented at `pentobi-corpus-v2.md:191` and never actioned.

**Consequence.** Every Pentobi measurement in the project's history — the ladder, the corpus,
the distillation teacher — played a Pentobi with no opening book. Numbers are internally
consistent (all book-free), but "beat Pentobi level 9" has never meant the shipped engine.

**Fix, applied.** Symlink the books next to the binary:

```bash
cd ~/code/pentobi/build/pentobi_gtp && ln -sfn ~/code/pentobi/opening_books/*.blksgf .
```

**Verified engaged** at L9 on the opening position:

| | CPU time | `move_values` | move chosen |
|---|---|---|---|
| book on | **0.47 s** | empty (no tree) | `f8,d9,e9,f9,e10` |
| `--nobook` | **25.82 s** | populated | `f8,e9,f9,g9,e10` |

A book hit is unmistakable: instant return, empty `move_values`, different move.

**Follow-through required.** Because the book is now active by default, `PentobiPlayer` must
pass the book state **explicitly** rather than inheriting a build-path accident — otherwise a
future rebuild silently flips it again. Add `nobook: bool` to `PentobiPlayer.__init__`
(`PentobiGtp` already has it), require callers to set it, and record it in every result. The
longitudinal ladder keeps `nobook=True` so its history stays comparable; the fair fight (F9)
uses `nobook=False`.

---

## F2. Pentobi L7 vs L9 head-to-head 🔄

**The question.** Our net scores 0.17 / 0.21 / 0.22 against L7 / L8 / L9 — flat, or even
inverted. Two explanations, and they imply opposite strategies:

- **Pentobi saturates.** Its extra search buys nothing above L7, so "beat level 9" is
  approximately "beat level 7" and the target is much closer than assumed.
- **Our net has a ceiling.** Pentobi keeps improving, and our net is equally outclassed at both
  levels, so our flat score reflects our limit rather than Pentobi's.

Our net cannot distinguish these, because it is the thing in question. Removing it from the
experiment does distinguish them: play Pentobi against itself.

**Design.** L7 vs L9, `--nobook` (book moves are level-independent and would dilute exactly the
difference under test), 200 games, run as **two colour-swapped batches of 100** because
`twogtp` has no colour-alternation option and Duo's first mover wins ~75% of decisive games —
an unswapped run measures the colour advantage, not the level difference.

```bash
P=~/code/pentobi/build/pentobi_gtp/pentobi-gtp
cd ~/code/pentobi/build/twogtp
./twogtp --game duo --nugames 100 --threads 3 \
  --black "$P --game duo --level 7 --quiet --nobook" \
  --white "$P --game duo --level 9 --quiet --nobook" --file A_L7first
# then the same with the levels swapped
```

Memory: each L9 engine preallocates ~1.96 GB, so 3 threads/batch × 2 batches = 12 engines
≈ 15 GB of 31 GB. Do not raise the thread count.

**Decision rule.** Pooled across both colour batches, at 200 games the SE on the score is
~3.5pp:
- **L9 scores ≤ 0.55** → saturation is real and engine-side. "Beat L9" ≈ "beat L7"; retarget
  the goal at L7 and treat L8/L9 as the same rung.
- **L9 scores ≥ 0.64** (~100 Elo) → Pentobi keeps improving; our flat scores are about our net.
  More search will not fix it and the weight shifts to Stream D.
- **In between** → unresolved; do not build on either reading.

---

## F3. Measure Pentobi's realised effort per move

**The hole.** The level table implies L9/L7 = 25× the simulations, but measured wall-clock is
14.7×. Something absorbs the difference, and until it is explained the whole analysis rests on
an unknown.

Candidate causes, in order of likelihood:
1. **Tree/memory truncation.** `get_memory(max_level)` sizes the tree from the level; at 5.5M
   playouts the search may hit a node cap and stop early. If so **L8 and L9 are identical in
   realised effort**, and F2's flatness would be an artefact of this box's build and RAM rather
   than a property of Pentobi.
2. **Per-move weighting.** `0.7·exp(0.1·i)` (×0.6 for duo) means the per-move budget is ~0.42×
   the table value on move 0 and rises ~5× by the late midgame — consistent with the observed
   L9 mean 16.0 s vs median 13.0 s.
3. **Level changes something other than the playout count** — never checked.

**Method.** Per move at L7 / L8 / L9, book off, single engine, one game each: record the
`cputime` delta and the sum of `value_count` over `move_values` entries as a realised-effort
proxy (the `visits` column reads 0 in practice, so it cannot be used). Then read the tree-memory
and max-node handling in `libpentobi_mcts/Player.cpp` and `libboardgame_mcts/SearchBase.h` on
the box, and check whether level touches anything besides the count.

**Output.** A realised-effort number per level to quote instead of the table, or a documented
statement that the table holds.

---

## F4. Record the full comparison context in every ladder result

Today `benchmark_level` returns aggregate wins/losses/draws, and `write_ladder_result` stores
the net name, sims and games/level. That is not enough to reconstruct or trust a comparison.

Add to every ladder result:

- **our side**: sims, `mcts_batch_size` (K), `sim_schedule`, opening temp/moves, all RNG seeds,
  device, torch.compile state, measured s/move (mean **and** median — the per-move weighting
  makes means misleading)
- **Pentobi's side**: level, tabled sim count, realised effort from F3, thread count,
  **book on/off**, binary version, measured s/move
- **per game**: which colour our net played and the result — currently lost to aggregation, and
  required by F7
- **diversity**: count of distinct positions at ply 8 across the games

Schema change to `PentobiLadder` JSON; keep older payloads readable (the reader already skips
unparseable files).

---

## F5. Seed the net's RNG; audit opening diversity per arm

**Two defects.** Pentobi is carefully reseeded per game (`player.py:69`) but our side is not:
`NetworkPlayer.__call__` samples openings through the global `np.random`
(`evaluation/players.py:165`), and temp-0 tie-breaks use it too (`search/mcts.py:240`). Ladder
runs are therefore not reproducible.

**Worse, diversity is confounded with the treatment.** Game variety comes only from sampling the
net's first 4 plies from the *visit* distribution, and that distribution sharpens as simulations
rise. A 6,400-sim arm may play far fewer distinct games than a 400-sim arm, so the ±8pp noise
floor is not constant along the axis F10 varies — it grows exactly where we are measuring.

**Fix.** Thread an explicit seed into `NetworkPlayer`; count distinct ply-8 `state_key`s per arm
and record it (F4); and reuse the existing paired-arena machinery to share opening prefixes
across arms, turning arm comparisons into paired tests. Re-audit the existing gen-40 data too —
its quoted noise floor is a lower bound.

---

## F6. Unify draw handling

`benchmark_level`'s `win_rate`, the Wilson CI, and `compute_headline_metrics`' "highest level
beaten at >50%" all treat draws as losses. Elo analysis uses score = W + D/2. At the observed
1–4% draw rate that is ~20 Elo at the tails, and it silently makes two numbers in the same
report incomparable.

Adopt **score = (wins + 0.5·draws) / games** everywhere: win rate, CIs, weighted score, headline
metric. Recompute gen-40's published numbers on the new definition and note the change, since it
shifts every historical figure slightly.

---

## F7. Refit ladder Elo with BayesElo, colour and draw terms

Converting a pooled half-and-half score with `-400·log10(1/s - 1)` is wrong here. With a colour
advantage `c`, the pooled score is `[σ(d+c) + σ(d−c)]/2`, which is **flatter** than `σ(d)`, so
inverting with a plain logistic *underestimates* the gap everywhere off 0.5. At `c ≈ 190` Elo
(75% of decisive games to the first mover) an observed 0.22 implies a true gap near **−280**,
not the −220 a naive inversion gives.

`scripts/tournament_elo.py` already has BayesElo fitting. Feed it the per-game records from F4
(net + 9 rungs as 10 players) with explicit colour-advantage and draw parameters. Publish every
Elo with a CI: at 100 games/level and a score near 0.2, one point carries **±87 Elo**, so
adjacent-level differences are ±120 Elo and mostly unresolvable. Restate the L7/L8/L9 comparison
on that basis, and drop per-level "Elo per doubling" figures — only the aggregate L2→L7 slope
(~33 Elo/doubling over 13.4 doublings) has usable signal.

---

## F8. Calibrate the time-parity budget on non-book moves

Measure Pentobi's s/move at L8 and L9 with the **book on** (F9's condition), excluding book
plies — a book move returns in ~0.5 s against ~26 s for a searched move, so including them
would badly under-budget our net. Measure our net's s/move at 400 sims after a discarded warmup
game (`scripts/measure_move_times.py`'s L7 figure is inflated by `torch.compile` warmup, so its
parity number is wrong at the source). Then scale.

Two fixes to the probe itself: pin `sim_schedule="flat"` and `dirichlet_epsilon=0.0` as the
benchmark does (`_eval_mcts_config` currently inherits them, so a `"branching"` config would
silently corrupt the calibration), and discard the first game.

Note the residual mismatch to disclose rather than fix: Pentobi's per-move budget rises through
the game while a flat sim count spends evenly, so a mean-matched budget out-thinks Pentobi early
and under-thinks it late — which is where Blokus games are decided. Matching **total per-game
time** is the more defensible convention.

---

## F9. The fair fight: net vs L9 at parity budget

The direct answer to "how strong are we really", with no extrapolation.

- gen-40 vs **L9**, book **on**, Pentobi at 1 thread (recorded), our net at F8's parity budget
- **300–400 games**, colour-balanced
- run **serially or at ≤2 workers**: parity measured uncontended and applied inside a 6-worker
  run is not parity, because Pentobi (CPU) and our net (GPU) degrade differently
- report score, Elo with colour/draw terms (F7), both sides' realised effort and s/move

**Cost.** At parity our net spends ~16 s/move and a game is ~23 plies, so ~6 min/game serial;
300 games ≈ 30 h serial, ~15 h at 2 workers. Check memory before raising workers: L9 Pentobi
needs 1.96 GB and our tree at parity sims is ~12× the 400-sim footprint, which is untested.

At a score near 0.35–0.5 the Elo resolution per game is roughly twice as good as at 0.2, so 300
games resolve about ±45 Elo — enough to tell "closes the gap" from "does not".

---

## F10. Search-scaling slope at L8 and L9

400 / 1,600 / 6,400 sims at **both** L8 and L9, 100–200 games/point, giving the slope and a
level-consistency check for less than the cost of one 25,600-sim arm.

Fitting a line over doublings {0, 2, 4} at ~44 Elo/point gives a slope SE around 10–13
Elo/doubling — enough to separate ~40 from ~0, **provided F5 has fixed diversity**, without
which the arms are not independent.

**Do not extrapolate far from this fit.** The project's leading hypothesis is a value head with
no skill beyond a colour prior, and a colour-prior value head caps search scaling by
construction — so Elo-vs-log-sims is expected to *flatten*, and projecting "5.5 doublings →
18,000 sims" assumes the one functional form our own evidence says is least likely. Use the slope
as a mechanism check; use F9 for the answer.

Add a 25,600 arm only if F9 lands ambiguously. It costs ~85 s/move, ~43 min/game, 70+ GPU-hours
for 100 games, and parallelises poorly on one 8 GB card.

---

## F11. Book-on vs book-off strength delta at L9

Plan item V11 from `pentobi-corpus-v2.md`, now unblocked by F1. Book-on L9 vs book-off L9 via
`twogtp`, colour-swapped, 200 games, no GPU.

Gives the Elo the book is worth, which is what converts every historical book-free number onto
the "as shipped" scale. Without it, F9's result and the entire pre-2026-08 ladder history sit on
two different scales with an unknown offset between them.

---

## Scope additions

Discovered while writing this plan, folded in above rather than left implicit:

- The colour-swap requirement in F2 — `twogtp` has no `--alternate`, and an unswapped run
  measures Duo's first-mover advantage instead of the level difference (gotcha 10, again).
- `PentobiPlayer` needs an explicit `nobook` argument (F1 follow-through). Relying on a
  build-path accident is what caused the original defect.
- Recomputing gen-40's published numbers under F6's draw convention — a change to every
  historical figure, so it needs stating rather than doing quietly.
