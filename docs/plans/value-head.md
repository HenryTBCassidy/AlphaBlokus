# The value head, and why the net cannot play from behind

The network's estimate of *who is winning* has no demonstrable skill beyond guessing from whose turn
it is. This plan attacks that, and the concrete failure it appears to produce: **our net beats Pentobi
level 9 as first mover and has never beaten it as second mover.**

**Scope owner:** `value-head`. **H2–H5 are blocked on
[`selfplay-data-and-loop.md`](selfplay-data-and-loop.md) D1** — they cannot start until stored
positions record whose turn it is. **H1 was not**: its answer is read off each game's *first*
position, where the player to move is the first mover by definition, so no turn field is needed. It
is done — see below.

Read [`../research/alphazero-technique-review.md`](../research/alphazero-technique-review.md) before
proposing any approach. Predecessor: [`archive/supervised-network-improvements.md`](archive/supervised-network-improvements.md).

---

## The evidence, in one place

**The value head is worse than useless.** Every supervised arm scored *negative* value skill:
predicting the outcome from nothing but whose turn it is beats what the net learned, by 15–28%. Its
output tracks mover colour at 0.76–0.81 while actual outcomes track colour at only 0.52–0.62 — the net
is more certain that colour decides the game than the game is.

**And it shows up in play, measured 2026-08-11/12:**

| Opponent | as White (first mover) | as Black (second mover) |
|---|---|---|
| Pentobi level 1 | 1.00 (10-0-0) | 0.65 (6-3-1) |
| Pentobi level 2 | 1.00 (10-0-0) | 0.50 (5-5-0) |
| Pentobi level 9 | **0.63** (30-17-3) | **0.00** (0-50-0) |

**Not a colour-handling bug.** That was tested directly, against a rule written before the run, and
refuted: as Black against level 1 the net scores 0.65, 95% CI [0.354, 0.863] — a second-mover score
near zero is excluded — through the same benchmark path, the same `Arena.play_games_by_colour` split
and the same GTP colour mapping that produced the 0/50. Full judgement in
[`archive/fair-pentobi-benchmark.md`](archive/fair-pentobi-benchmark.md) under F9. At level 9 the two
colour cells are 0.630, 95% CI [0.491, 0.750], and 0.000, 95% CI [0.000, 0.071] — disjoint by a wide
margin, so the gap is real and large. **The deficit widens with opponent strength.** The net can
convert an advantage; it cannot create one.

The level-1 and level-2 rows are 20 games each (10 per colour) at 400 simulations, book off; the
level-9 row is 100 games at 4,096 simulations, book on. They establish the *shape*; they are not on
one scale and must not be pooled into a trend.

**Why the mechanism is plausible.** The search consults the value estimate for every unvisited move
when building its training target. An estimate that reads "am I the first mover?" instead of reading
the board gives the search nothing to work with once the answer is no — so from behind it has no
gradient toward a plan, at any simulation count, net size or data volume.

---

## Checklist

| # | Item | Role | Effort | Priority | Done |
|---|---|---|---|---|---|
| H1 | Diagnose: do our self-play games contain second-mover wins at all? | `A` | ½ d | **Critical** — gates H2–H3 | ✅ |
| H2 | Colour-conditional value diagnostic on the fresh eval set | `W→R→A` | ½ d | High | |
| H3 | Outcome-balanced value sampling, behind a flag, **A/B'd immediately** | `W→R→A` | 1 d + box | High | |
| H4 | Teacher λ-blend — measure it, expecting to decline it | `W→R→A` | ½ d + box | Medium | |
| H5 | Win/draw/loss value head | `W→R→A` | 2 d + box | Medium | |

---

## H1. Do our self-play games contain second-mover wins at all?

**Pure analysis, no code, and it gates everything below.** Count, in the self-play data we hold: what
fraction of games the first mover wins, and what fraction of *training positions* come from games the
second mover won.

If second-mover wins are rare in the data, the net has barely been shown what winning from behind
looks like — and the fix is a **data** problem (colour-balanced self-play, opening seeding, deliberately
training the losing side), not an architecture problem. If they are common, the net is seeing them and
failing to learn from them, which points at the target or the trunk instead.

Those two diagnoses imply completely different work. Do not build until this is answered.

### Result (2026-08-18): second-mover wins are common, not scarce — this is a weighting problem ✅

**Answer: second-mover wins are abundant in the data. H2/H3 take the "the net is seeing them and
failing to learn" branch; the colour-balanced-self-play / opening-seeding branch is not indicated.**

**Method — and note it needs no turn field, so it runs on pre-D1 data.** Stored rows are flat in
game order and the first position of every game is the empty board, which canonicalises to an
all-zero tensor and therefore to all-zero bytes; that alone marks the game boundaries with no
decoding. The player to move at that position is always the first mover, and `episode.py` labels
every row from the perspective of the player at that position — so the first row's `value` *is* the
game outcome for the first mover (+1 / −1 / |1e-4| draw).

**The method was validated against an independent instrument before it was believed.**
`SelfPlayProfiling` holds one row per episode, and it agrees exactly: game counts 25/25, 80×5, 10/10
match the segmentation game for game, and mean plies per game (rows ÷ 2 symmetry copies) reproduces
the profiler's `num_moves` to three digits (e.g. 27.52 vs 27.52, 27.59 vs 27.5875).

#### What we hold locally: 470 games, 27,284 stored positions, across 9 generation-files

| Run | gens | games | first-mover wins | second-mover wins | draws |
|---|---|---|---|---|---|
| `blokus_pc_first` | 2 | 50 | 24 | 23 | 3 |
| `blokus_pc_second` | 5 | 400 | 217 | 137 | 46 |
| `blokus_mac_test` | 2 | 20 | 9 | 10 | 1 |
| **Pooled** | 9 | **470** | **250 (0.532)** | **170 (0.362)** | **50 (0.106)** |

| Quantity | Value | 95% interval |
|---|---|---|
| P(first mover wins) | 0.532 | [0.481, 0.574] |
| **P(second mover wins)** | **0.362** | **[0.310, 0.433]** |
| First mover's share of *decisive* games | 0.595 | — |
| **Share of stored positions from second-mover-won games** | **0.366** | — |

Intervals are a bootstrap resampling the nine generation-files, so a generation's shared net counts
as one cluster — the per-generation first-mover rate ranges 0.400–0.613, which a game-level interval
would ignore. Game-level Wilson is [0.487, 0.577] and [0.320, 0.406], up to 3.6 pp narrower, and
would understate the uncertainty.

#### The load-bearing caveat: this is not the production regime, and the gap is large

None of the three runs above is the run behind this plan's evidence. They are early 2026-05/06 runs
at 10–80 games per generation on weak nets; `blokus_cloud_v3` and every later box run kept their
`SelfPlayHistory` on the machine that produced it, and only `Reporting/` was ever fetched. This is a
convenience sample of what happens to sit on the laptop.

The one production-regime measurement we have used exactly this method and disagrees materially:
[`../research/plateau-investigation.md`](../research/plateau-investigation.md) reports **73% White /
18.5% Black / 8.2% draws** from `self_play_16.parquet` first-position values. So the two samples give
second-mover win shares of **36.2%** and **18.5%**.

The obvious explanation — a stronger net converts the first-mover advantage more often, so the skew
grows with strength — is plausible but **is not established here.** Within `blokus_pc_second` the
first-mover rate rises 0.450 → 0.566 from generation 0 to generations 1–4, which is only ~1.9σ and
therefore consistent with noise; and the cross-run comparison confounds net strength with run,
config and search backend all at once. Treat it as the leading hypothesis, not a measurement.

**It does not change the answer, because both ends of the range are far from "rare".** At 18.5% and
10,000 games per generation, self-play produces roughly 1,850 second-mover-won games and ~115,000
stored positions carrying a second-mover-win label *every generation* — and the production replay
buffer measured on `blokus_paired_gate_rerun` holds 60,000 games. The net is shown winning-from-behind play in enormous volume and does not learn it. Whichever
sample is right, the gate resolves the same way.

#### What this rules in and out

- **Ruled out: a data-scarcity diagnosis.** "The net has barely been shown what winning from behind
  looks like" is false at either measured rate. Colour-balanced self-play and opening seeding are not
  the indicated fix, and building them first would have been the wrong work.
- **Ruled in: H3 as specified, and its conditional framing is confirmed correct.** The data is
  *skewed* (~4:1 toward the first mover at production strength) without being *absent*, and skew is a
  weighting problem. This is also exactly why H3's first trap is load-bearing: with a marginal label
  split near 50/50 and the whole skew living in `P(win | White to move)`, balancing the marginal is a
  no-op. Reweight, do not regenerate.
- **Not resolved: why the net fails to learn from data it has in abundance.** H1 was never going to
  answer that; it only chose between two branches. H2 is now the next instrument.

#### Two corrections this turned up

**1. The "22% of our games are draws" figure in H5 is not ours.** 22% is the *Pentobi distillation
corpus* draw rate — "the corpus is 71% White wins, 22% draws, and only 7% Black wins"
([`../research/investigation-2026-08/01-chatgpt.md`](../research/investigation-2026-08/01-chatgpt.md)),
where the same sentence goes on to give self-play separately as 73% White wins. Our self-play draw
rate is **8.2%** (production, plateau-investigation) to **10.6%** (the local sample above). H5's
premise is corrected below; the argument survives at 8–11% but it is a materially smaller prize than
22% implied, and H5 should be re-costed on that basis. The 22% figure remains correct and relevant
for anything trained on the corpus.

**2. `self_play_16.parquet` is now the load-bearing number and we no longer hold it.** The 73 / 18.5 /
8.2 split shapes this plan's whole diagnosis and cannot currently be re-derived — the same shape as
the "96.3% of decisive arena games" failure. Re-running the segmentation above over a production
run's `SelfPlayHistory` is a file read: no GPU, no contention, minutes. It should be requested of the
GPU-runner before H3's reweighting is calibrated, because H3's weights depend on the *value* of the
conditional skew, not merely on its sign.


## H2. Colour-conditional value diagnostic on data that is actually held out

The existing diagnostic reproduces to the digit, but it has only ever run on an eval set that was
**sampled from the training buffer and never removed from it** — so its numbers are memorisation, not
skill. `selfplay-data-and-loop.md` D3 produces a genuinely held-out set for the first time.

Re-run the colour-only and colour×phase baselines on it, split by side-to-move and game phase, with
game-cluster intervals. This is the instrument every row below is judged on, so it has to be sound
first.

**Note the precision limit:** on a 200-position set, value skill carries ~±0.17 measurement error. So
judge the rows below on the **ladder** primarily, with value skill as a mechanism check.

## H3. Outcome-balanced value sampling

Weight the value loss so "the first mover usually wins" stops earning marks.

**Two traps, both easy to hit:**

- **Balance conditionally on colour, not marginally.** The marginal win/loss split is already ~50/50;
  only `P(win | White to move) ≈ 0.73` is skewed. Balancing the marginal is a silent no-op.
- **Reweight, do not resample.** Discarding positions costs ~46% of first-mover data *and* starves the
  policy head, which does not have this problem. Weighting only the value term costs ~21% effective
  sample size for that head alone.

**Honest cost:** the first mover really does win ~73% of the time, so balancing discards true
information and will mis-calibrate absolute win estimates. It trades one bias for another.

**Ship it behind a flag defaulting off, and the A/B is the first thing that uses the flag** — not a
someday. If it wins, it becomes the default immediately and the flag is deleted. Three auxiliary heads
in this project are built, default-off and still unmeasured; an untested flag is worse than an
untested default.

## H4. Teacher λ-blend — measure it, expect to decline it

`target = λ·pentobi_eval + (1−λ)·outcome`, λ ∈ {0, 0.3, 0.5}.

The technique review ranks this first on cost and evidence. **That ranking does not transfer to us.**
Stockfish and Lc0 blend toward their own engine's evaluation because *matching that engine is the
goal*; ours is to **surpass** Pentobi, and the outcome labels are the only signal that can disagree
with it. Blending the teacher's opinion in dilutes the one channel carrying information the teacher
does not already have.

Run it as an arm because it is cheap and the ranking deserves a test, but **expect to decline it.**

The real problem it points at is genuine: **every position in a game carries the same label**, so a
30-ply game gives 30 boards stamped with one result and the value head cannot tell an open opening from
a decided endgame. The margin and ownership targets in
[`network-experiments.md`](network-experiments.md) supply that missing discrimination — and unlike the
teacher's opinion, they are *facts*.

## H5. Win/draw/loss head

Three probabilities instead of one scalar. A scalar cannot separate "a certain draw" from "an even
fight" — both are zero — and **8–11% of our self-play games are draws**. Lc0 standard since 2019.

**Corrected 2026-08-18 (H1).** This row previously said 22%. That is the *Pentobi corpus* draw rate,
not ours; our self-play draws are 8.2% (production) to 10.6% (the local sample H1 counted). The
argument survives, but the prize is roughly half what 22% implied, so re-cost this row before
committing two days to it.

Sequenced after H3: it changes the value head's shape, which would muddy every comparison above if
done first. Touches torch, the jax bridge and the ONNX export.
