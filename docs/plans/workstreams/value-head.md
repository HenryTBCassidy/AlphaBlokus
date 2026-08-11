# Session charter — Value head

You own **the network's "who's winning?" estimate**, and the data plumbing it needs. The 2026-08
investigation concluded the network — not search — is what caps this project, and this is that work.

Read first, in order: [`README.md`](README.md) (the inter-session contract),
[`../ROADMAP.md`](../ROADMAP.md), [`../supervised-network-improvements.md`](../supervised-network-improvements.md)
(N6/N7 are yours), and
[`../../research/alphazero-technique-review.md`](../../research/alphazero-technique-review.md) —
**read this one before proposing any approach.** It is a ranked list of nine techniques with
KataGo/Lc0 ablation numbers, and it already answers "what do other projects do about this".

---

## The problem, precisely

Part of the network predicts who is winning. In Blokus Duo the first mover wins ~73% of the time, so
the network learned a shortcut: score well by checking **whose turn it is**, without reading the
board. Measured — its output tracks mover colour at 0.76–0.81 while actual outcomes track colour at
only 0.52–0.62. Both production checkpoints' confidence intervals bracket **zero skill** beyond that
prior.

Why it matters beyond the head itself: the search consults this estimate for every unvisited move
when building its training target. An estimate that ignores the board corrupts the target the policy
learns from — at every simulation count, every net size and every data volume. That is the mechanism
that explains a plateau immune to more compute.

## What you own

- `src/alphablokus/selfplay/**` (`episode.py`, `generate.py`)
- `src/alphablokus/storage/selfplay_store.py`
- `src/alphablokus/training/**` (`replay_buffer.py`, `coach.py`)
- `src/alphablokus/games/base_wrapper.py`
- `src/alphablokus/cli.py`
- `docs/plans/supervised-network-improvements.md`

Do not edit `games/blokusduo/pentobi/**`, `evaluation/ladder_*`, or the benchmark scripts — Benchmark
owns those. `config.py` is shared; say so in your PR title if you add a knob.

**You do not own the box.** Ask before running anything on it.

## Work queue, in this order

**1. Thread the true player through the data path.** ⬅ do this first, the window is closing

`ProcessedExample` is `tuple[compact_board, (indices, probs), value]` — no record of whose turn it
is. Thread it through `episode.py` → `selfplay_store.py` → `replay_buffer.py` → the dataset.

Why first: **both historical replay buffers were deleted, so there is no legacy data to migrate.**
Changing the schema is nearly free right now, and the window shuts the moment new data is generated.

The alternative — inferring colour from piece parity with `infer_mover_colour()` — is already built
but its own docstring notes the parity "breaks and the colour is genuinely unrecoverable" once a
player has passed. Passing happens in the **endgame**, exactly where the value signal is strongest.
Do the real plumbing.

Threading `player` through also removes the piece-parity workaround in the colour-value diagnostic.

**2. A generate-only entry point.** There is currently no way to produce self-play games without
also training on them — every path calls `generate_games` then trains. ~5,000 games with frozen
weights, no gradient updates, ~80 min on the box, £0. This unblocks three separate things: the
learning-rate sweep, a genuinely held-out eval set, and settling whether the value head *degraded*
or was merely graded against another net's games. Must land **after** step 1 so the dataset carries
the colour column from birth.

**3. Teacher-value λ-blend.** Ranked **#1 of nine** in the technique review. Blend the outcome label
with the search's own evaluation of that position: `target = λ·search_value + (1−λ)·outcome`. Today a
position 30 moves from the end is labelled purely by who eventually won — this attacks target
*variance*, which is the better-evidenced lever. Precedent: Stockfish NNUE trains at λ≈0.7, Lc0's
Q-ratio. The corpus side already stores `search_value`; the RL side needs the search's root value
plumbed through harvest and the buffer, which is the same shape of work as step 1.

**4. Outcome-balanced value sampling, behind a flag defaulting off.** Henry's instruction: build the
switch, and **commit to the A/B being its first use, not a someday.** If it wins, it becomes the
default immediately and the flag is deleted.

Two traps:
- **Balance conditionally on colour, not marginally.** The marginal win/loss split is already ~50/50;
  only `P(win | White to move) ≈ 0.73` is skewed. Balancing the marginal is a silent no-op.
- **Reweight, do not resample.** Discarding positions costs ~46% of first-mover data *and* starves
  the policy head, which does not have this problem. Weighting only the value term costs ~21%
  effective sample size for that head alone.

Note the honest cost: the first mover really does win 73% of the time, so balancing discards true
information and will mis-calibrate absolute win estimates. It trades one bias for another. This is
why it is ranked below the blend.

**5. Win/draw/loss head.** Lc0 standard since 2019; 22% of our games are draws and a single scalar
cannot separate "certain draw" from "coin flip". After 3 and 4.

## Hard-won facts you must not rediscover

1. **Three auxiliary heads already exist, default off, and were never measured.** Do not add a
   fourth unmeasured thing. A flag that is never tested is worse than an untested default — this is
   why step 4 must be A/B'd immediately.
2. **The eval set was never actually held out** until PR #69: its positions were sampled from the
   training buffer and never removed, no source game id was recorded, and the draw was not
   reproducible at a fixed seed. So every historical "internal signals looked healthy" observation
   is **uninformative**, not evidence.
3. **The buffer reaches capacity at generation 6 and starts empty on a warm start.** Lifetime reuse
   is ~12 gradient contributions per stored example (24 per underlying position counting its
   symmetry twin); a game's single outcome label is seen ~750 times by the value head.
4. **`config.seed or 0` is a bug** — it makes an explicit seed of 0 indistinguishable from unseeded.
5. **Weight decay is on by default** (`AdamW`, 1e-4); training without it let a converged net drift
   from level 4 to level 3.
6. Read `AGENTS.md`'s gotcha list before touching training — there are 22 and several are yours.

## Reporting contract

Every finding goes into `supervised-network-improvements.md` **in the same commit as the work**, with
the row ticked. If it changes a project-level belief, add an `AGENTS.md` gotcha and flag it for
`ROADMAP.md`. Findings that live only in chat are lost — that is the failure this split exists to
fix.

Judge value-head changes on the **ladder**, with value skill as a mechanism check only. The kill
criterion "value skill ≤ +0.1" carries ±0.17 measurement error on a 200-position eval set, so it is
a weak gate even computed correctly.
