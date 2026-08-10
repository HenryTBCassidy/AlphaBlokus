# Erratum to `00-briefing.md` (2026-08-04)

**Status:** this file corrects `00-briefing.md`. Where the two disagree, **this file wins.**

`00-briefing.md` is the document any future session inherits as fact, and errors in it have
already propagated into three paid runs' worth of "next steps". It is *not* a whitewash —
independent recomputation confirmed nearly every headline figure to the digit (see
`02-fable.md` §0.1), and its retraction discipline in §4 is above average. The defects below
are specific: one load-bearing figure whose raw data is gone, one miscounted run, several
statements quoted to more precision than the instrument supports, and two interpretive
over-reaches.

Markers: **[V]** re-verified on this machine 2026-08-03/04 · **[M-doc]** disagreement between
the briefing and the research doc it summarises · **[I]** inference.

---

## 1. Corrections to the measurements

### 1.1 The 96.3% white-win figure is unreproducible. Stop quoting it. **[V]**

§3.2's headline — "96.3% of decisive arena games are won by White" — cannot be re-derived
from raw data. It was computed in `docs/research/plateau-investigation.md` from
`blokus_search_harder/ArenaReplays` (1,700 games, 17 generations). That run directory now
holds only `Nets/`; no `ArenaReplays` for it exists anywhere in the local tree, and
`blokus_search_harder_v1/` has `ArenaData` but no replays either.

The number is plausible — the same methodology reproduces exactly the 77% that doc quotes for
`cloud_v3`, whose replays survive — but the single most load-bearing measurement in the
briefing is currently unverifiable. **[I]**

**What survives:** the *conclusion* it supported — that the candidate-vs-incumbent arena
cannot see strength — does not depend on this figure. It rests independently on v3's
surviving replays (77% White among 3,747 decisive of 4,000) and on the variance-collapse
test, which reproduces exactly (0/17 accepted; scores 0.485–0.530; std 0.01129; χ²=0.816 on
16 dof; p=1.32e-8). Quote those instead.

### 1.2 `search_harder_v2` ran 8 generations, 1 of 8 accepted **[V]**

Not "9 generations, 1/9". `blokus_search_harder_v2/ArenaData` contains 8 generations, 1
accepted (gen 1 at 210/188/2 = 0.5275 under the 0.52 gate; gens 2–8 all 0.48–0.51). Trivial
in itself, but the ledger opens by warning "do not trust the committed JSON configs as a
record of what ran" and then miscounts a run.

Note also that the ledger's outcome cell for that run, "+19 Elo once then flat", is the
arena-score conversion of that single 0.5275 gate pass — i.e. a reading taken from the very
instrument §3.2 correctly discredits.

### 1.3 "93–97%, never below 93" should read **93–100%** **[M-doc]**

`plateau-investigation.md` gives the per-generation white-win range as 93–100%. The briefing
narrows it. Minor and direction-neutral, but it is a narrowing, not a summary.

### 1.4 The gen-17 entropy collapse recovered **[V]**

§3.3 states "At gen 17 self-play target entropy collapsed 0.79 → **0.506** in one
generation" and stops there. Gen 18 was back to **0.789** and gen 20 was **0.820**. It was a
one-generation transient, and the briefing presents it as a terminal event.

The rest of §3.3's degradation evidence is unaffected: the symmetry KL (0.639 → 1.236) and
value symmetry MAE (0.101 → 0.249) trends are real and roughly monotone. The entropy
collapse, as narrated, overstates the case.

### 1.5 The rerun's "pool Elo −44" implies a trend that is not there **[V]**

The rerun's pool Elo by generation: −15, −10, +2, −6, −14, −18, −8, −5, −38, +6, −6, −41,
−46, −8, −6, −20, −25, −18, **−4 (gen 19)**, −44 (gen 20). The trajectory bounces within
roughly ±40 throughout; gen 19 was −4. Quoting the endpoint as a result implies a decline the
series does not show.

It is also the same *family* of instrument as the arena, and is affected by the same
colour-pinning — so the briefing teaches (correctly) that this class of measurement squashes
everything into 0.48–0.53, and then quotes an endpoint from it as evidence.

**The real evidence of degradation in that run is the ladder: 0.344 → 0.298.** That holds
**[V]** and it is sufficient. Drop the pool-Elo figure from the argument.

### 1.6 False precision: "moved strength by 0.002" and "+0.006" **[V]**

§3.9 reports that fine-tuning v3 on the corpus "moved strength by 0.002" (0.539 → 0.537),
quoted to three digits against a ladder noise floor the briefing itself measures at **~0.022**
eleven lines earlier in §3.7. §3.6's "+0.006 weighted" for capacity has the same problem.

Correct wording in both cases:

> **unchanged — and the experiment could not have detected a ±40-Elo effect in either
> direction.**

The *direction* of both conclusions (no strength change) is right. The precision is fake, and
stating it invites the next reader to treat a 0.002 difference as a signal.

### 1.7 v3's own pool Elo peaked before gen 40 — but "peaked then declined" over-reads it **[V]**

§2's "pool Elo +240" for `cloud_v3` verifies (gen 40 = +240.3), but omits that v3's internal
rating peaked at **gen 32 (+286)** and fell to +240 by gen 40
(`blokus_cloud_v3/Tournament/tournament_ratings.parquet`).

Both the briefing's omission *and* the obvious reading of the omitted data are wrong. The
series bounces by ±100 between adjacent generations (gen 18 = 223, gen 24 = 81, gen 32 =
286), so "v3 peaked at 32 and declined" over-reads it just as much as "v3 climbed then
plateaued" does. Neither is established.

**The actionable half stands and is not about the trend:** gen-40 was never shown to be the
best net *of its own run*. Gen 32 was never laddered, the keep-best-by-ladder machinery
existed and never ran, and every experiment since v3 warm-started from gen-40 without anyone
checking. That is exactly what **B1** now measures.

---

## 2. Corrections to the interpretation (§7)

### 2.1 §3.1's fixed-point claim is over-generalised

§3.1 concludes that "v3 gen-40 is at the fixed point of the improvement operator". It is at
the fixed point of **one training step under one acceptance rule** — one epoch on a 60k-game
buffer at constant LR 1e-3 with no weight decay, promoted through a gate that §3.2 shows
cannot see strength.

"The improvement operator" invites the reading that the *class* of operators has been
explored. It has not: **no healthy operator has ever been pointed at v3 even once.** Every
stalled run since has a since-diagnosed defect — v1 and v2 were frozen by a gate demanding
something measurably impossible, and the rerun trained with settings the project's own
post-mortem calls toxic. This is the entire reason **B6** exists.

The correctly scoped claim: *gen-40 is a fixed point of the operator as configured in v3, and
the operator is the thing to change.*

### 2.2 §7.3 is wrong as stated; §7.3 and §7.5 should swap places

§7.3 claims search width "is the only mechanism that explains a fixed point immune to more
sims, more capacity *and* more data".

It is not the only mechanism. **The value head explains all three at least as well**, and
unlike width it has direct adverse measurement behind it (§3.8: value skill negative in every
supervised arm, −0.16 to −0.28). The mechanism is concrete: Gumbel's training target is
`softmax(logits + σ(completed Q))`, and completed-Q substitutes the value net for every
unvisited action. A value head no better than a colour prior therefore corrupts the target the
policy trains on — at every simulation count, at every net size, and on every quantity of
data. That is immunity to all three.

Width remains a live candidate and **B4** tests it. But the ranking is wrong: **§7.5 (the
value head is the most under-investigated component) is the stronger hypothesis and should be
§7.3.** The current ordering is why the value head went two rounds of proposals without a
diagnostic being built.

---

## 3. Correction to the value-head headline itself

Not a briefing error — a correction to the follow-up investigation, recorded here because it
changes how §3.8 should be read and because it gates part of the plan.

The colour-conditional value diagnostic reproduces to the digit **[V]**, but its confidence
intervals were computed over *positions*. Every position in a game carries the same outcome
label, so positions are not independent. Recomputed with a bootstrap over **game lineages**:

| net | eval set | value skill | 95% CI (by game) |
|---|---|---|---|
| v3 gen-1 | v3 | +0.385 | +0.109 → +0.466 |
| v3 gen-40 | v3 | −0.102 | −0.150 → +0.092 |
| v3 gen-40 | rerun | +0.056 | −0.159 → +0.191 |
| rerun gen-20 | rerun | −0.089 | −0.265 → +0.079 |

- **Holds:** the production value head has no demonstrable skill beyond guessing from whose
  turn it is. Both production CIs straddle zero.
- **Holds, and is label-independent:** every checkpoint's value output tracks mover colour at
  0.76–0.81 while outcomes track it at only 0.52–0.62. The net is more certain that colour
  decides the game than the game is.
- **Does NOT hold at the stated precision:** "the value head degraded over the run." The
  clean vintage-matched pair (+0.056 vs −0.089) has almost entirely overlapping intervals.
- **Consequence for the plan:** the proposed kill criterion "value skill ≤ +0.1" carries a
  measurement error of ±0.17 and **cannot be read on the current eval set at all.** Rebuilding
  the eval set (**A6**) is therefore a prerequisite for the D-stream, not a nice-to-have.

> **[V, corrected 2026-08-05] — the conclusion above stands, but not for the reason given.**
>
> The table in this section computed its intervals by bootstrapping over *game lineages*
> inferred from board contents. That inference does not work, and `03-synthesis.md`'s A6 cell
> now records why: the clustering keyed on shared opening prefixes, and generation-1 self-play
> has almost no opening diversity, so it merged unrelated games. The appendix at the bottom of
> this file reached the same negative result independently (counting maximal boards gives 132
> lineages, not ~47). The v3 eval set is 200 positions drawn uniformly from ~320,000
> generation-1 positions across ~10,000 games — about one position per game — so **position-level
> intervals on the current eval set are approximately correct**, and the per-position table in
> `03-synthesis.md` ("Correction to Fable's headline") is the one to use, not the table above.
>
> What that changes: on the per-position intervals, v3 gen-1 → gen-40 degradation **is**
> statistically resolved (−0.487, CI −0.675 → −0.330) though still causally confounded by
> target vintage, while the vintage-matched pair (gen-40 → rerun gen-20) is **not** resolved
> (−0.145, CI −0.370 → +0.082). The two claims that survive both versions are the load-bearing
> ones: production value heads show no skill beyond a colour prior (both CIs bracket zero), and
> value output tracks mover colour at 0.76–0.81 while outcomes track it at only 0.52–0.62.
>
> **Why A6 is still a prerequisite.** Not because of clustering, but because of §3a below: the
> eval set was never held out at all. Its positions were sampled from the training buffer and
> never removed from it, no source game id was recorded, and the draw was not reproducible at a
> fixed seed. A diagnostic measured on data the net trained on cannot resolve a kill criterion
> at any interval width. That defect is independent of the clustering question and is what A6
> (landed in PR #69) actually fixes.
>
> The lesson worth carrying: **an interval that is too wide and a metric that is measured on
> training data are different failures.** This section diagnosed the first and prescribed the
> fix for the second. Do not "simplify" the reasoning by deleting the prerequisite.

---

## 3a. §3.3's "internal signals looked healthy" has a concrete mechanism **[V]**

§3.3 notes that the rerun's internal signals (loss falling, acceptance 100%, eval top-1
~0.99) looked healthy throughout, and attributes this to their being "measured against
self-generated data". That is true but it understates the problem, and the specific
explanation matters because it is fixable.

**The "held-out" eval set was not held out.** The 200 eval positions were sampled *from the
training buffer* and never removed from it, so the net trained on the exact positions being
reported as held-out — along with their symmetry twins and their ~60 same-game siblings
each, every one of which carries the same outcome label. In the rerun those positions stayed
in training from generation 1 through roughly generation 6 at `epochs: 2`, i.e. up to ~12
gradient passes over the reported holdout. Eval top-1 of 0.99 is what memorisation looks
like, not health. Worse, the metric silently changes meaning around generation `B/F` when
those positions age out of the buffer, producing an artefactual kink that is easy to read
as a real change in the net.

Three further defects in the same instrument, all now fixed together:

- it was built once from generation-1 data and never refreshed, so it got easier as the net
  improved — the frozen set measures the run's weakest-ever data forever;
- its positions were treated as independent when they are not (see §3 above);
- it was **not reproducible at a fixed seed**: positions were drawn with a seeded numpy
  generator from a list the replay buffer had shuffled using Python's *global* `random`
  module, which nothing in the run seeds. The indices reproduced; the positions they pointed
  at did not. Two nominally identical arms therefore differed before any treatment applied.

**Consequence for the plan.** Every "internal signals looked healthy" observation in the
briefing should be read as *uninformative* rather than as evidence of health, and any
pre-registered threshold read off these diagnostics — including the D-stream's — is only
meaningful on data produced after the fix. `docs/plans/` and `AGENTS.md` now carry the
invariants; the mechanism was independently confirmed by two auditors with executed probes.

For balance, two related suspicions were checked and **cleared**: the JAX PRNG key chain is
genuinely unique per game, move, search, generation and device (Gumbel noise is per-slot,
not broadcast across games), and illegal moves are masked *before* top-k selection at the
root and at every expanded child. Neither is a defect.

---

## 4. What the briefing gets right, for the record

Worth stating so the corrections above are not mistaken for a general loss of confidence:

- §3.5's warning that raising `top_k` without raising `considered` could make the root
  *narrower* in practice — subtle, correct, and easy to have got wrong.
- §3.6's admission that all four capacity tests are supervised, where the ceiling is the
  teacher.
- §3.7's noise floors, which are what make most of the corrections in this erratum
  computable at all.
- §4's retraction list.

The briefing is roughly 95% checkable and checks out. Its defects are concentrated in
precision and framing, not in fabrication.

---

## Appendix — one claim in `03-synthesis.md` that could not be reproduced **[V]**

`03-synthesis.md` A6 states, marked [V], that the v3 eval set's 200 positions represent "only
~47 game lineages, two of which hold 150 positions". **This could not be reproduced from the
eval set on disk, and the stated method cannot work:** `EvalSet/` contains only
`boards.npy`, `compact_boards.npy`, `target_policies.npy`, `target_values.npy` and
`targets_kind.txt` — **there is no game or episode id stored**, and the v3 run directory
retains no self-play episode store to match positions back against.

Reconstructing lineages from board contents alone is ambiguous in both directions: subset-
closure over placement boards collapses all 200 positions into a single cluster (the
near-empty opening board is a subset of every later board), while counting maximal boards —
positions that are not a strict subset of any other — gives **132**, not ~47. The set holds
168 distinct boards among 200 records, so there is *some* duplication, and target values are
±1 for decisive games with 21 draws in v3 encoded as ±0.0001.

**What this does and does not change.** The direction of A6's argument is unaffected and the
remedy is unchanged: positions within a game share a label, so position-level intervals are
too narrow, and the fix is to record the source game id at build time and bootstrap over
games. But the specific figures "~47 lineages" and "two hold 150 positions" should be treated
as **unverified**, and the "roughly 2×" CI-narrowing factor is an estimate of unknown
accuracy rather than a measurement. Once A6 lands, the real clustering will be recorded and
the factor can be measured directly instead of asserted.
