# The value head, and why the net cannot play from behind

The network's estimate of *who is winning* has no demonstrable skill beyond guessing from whose turn
it is. This plan attacks that, and the concrete failure it appears to produce: **our net beats Pentobi
level 9 as first mover and has never beaten it as second mover.**

**Scope owner:** `value-head`. **Blocked on [`selfplay-data-and-loop.md`](selfplay-data-and-loop.md)
D1** — nothing here can start until stored positions record whose turn it is.

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

Not a colour-handling bug — the net wins plenty as second mover at low levels, so the path handles
colour correctly. **The deficit widens with opponent strength.** The net can convert an advantage; it
cannot create one.

**Why the mechanism is plausible.** The search consults the value estimate for every unvisited move
when building its training target. An estimate that reads "am I the first mover?" instead of reading
the board gives the search nothing to work with once the answer is no — so from behind it has no
gradient toward a plan, at any simulation count, net size or data volume.

---

## Checklist

| # | Item | Role | Effort | Priority | Done |
|---|---|---|---|---|---|
| H1 | Diagnose: do our self-play games contain second-mover wins at all? | `A` | ½ d | **Critical** — gates H2–H3 | |
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
fight" — both are zero — and **22% of our games are draws**. Lc0 standard since 2019.

Sequenced after H3: it changes the value head's shape, which would muddy every comparison above if
done first. Touches torch, the jax bridge and the ONNX export.
