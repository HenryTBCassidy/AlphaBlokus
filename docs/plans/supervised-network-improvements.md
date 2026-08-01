# Supervised-phase network improvements

Turns the shortlist in [`../research/alphazero-technique-review.md`](../research/alphazero-technique-review.md)
§3A/§3B into sequenced work. Everything here acts on the **stage-1 v2 corpus** and can therefore
move the V15 ladder gate; the self-play techniques (§3C) are blocked on that gate and get their own
plan later.

**The organising constraint: one change at a time, measured before the next.** These techniques all
touch the same shared trunk, so stacking them makes the result unreadable — you learn that four
things together did something, and never which. That is not hypothetical: the score-head A/B was
found to be confounded by a *side effect* of adding the head (it shifted the data shuffle), at a
magnitude four times the effect being measured. Each row below is therefore build → measure →
**keep or delete**, and a row that improves nothing is deleted rather than left in.

**Depends on:** [`pentobi-corpus-v2.md`](pentobi-corpus-v2.md) V12 (the corpus, generating now) and
the score head on `feat/score-auxiliary-head` (built, unmeasured). N3 below *is* that plan's S7.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| N1 | The A/B harness: two runs identical but for one thing, fixed seeds, one metric set, one command | ½ day | High | ✅ |
| N2 | Data-fraction curve (25 / 50 / 100% of the corpus) — makes every later result interpretable | 3 h box GPU | High | ✅ |
| N3 | **Score-head A/B** (code already built — this is `score-auxiliary-target.md` S7) | ½ day box GPU | High | ⚠️ |
| N4 | Ownership head: predict the final board, per cell | 2 days + ½ day box | High | ✅ built, ⚠️ inconclusive |
| N5 | Opponent-reply target: predict the reply distribution from the current position | 1 day + ½ day box | High | ✅ built, ⚠️ inconclusive |
| N6 | Value-label arms: teacher blend λ ∈ {0, 0.3, 0.5}, and outcome-balanced sampling | ½ day + ½ day box | Medium | |
| N7 | Win/draw/loss value head (IDEAS I8) | 2 days + ½ day box | Medium | |
| N8 | Global pooling in the trunk — **gated on N4/N5 showing the trunk is the constraint** | 3–4 days + ½ day box | Medium | |

**Every row gets a paired A/B — no change is taken on faith.** Two instruments, chosen for
sensitivity rather than cost:

- **Held-out metrics (N1) run for every single change.** They are sensitive enough to rank small
  effects, and cheap.
- **The Pentobi ladder runs at decision points, not per arm.** At 50 games per level its binomial
  noise is ≈ ±7 pp, so it simply cannot see a 2 pp improvement — running it per arm would measure
  mostly noise, not save time. It answers "is this combination actually stronger", which is the
  V15 gate's question and remains the one that matters.

**These rows can start before the corpus finishes.** Generation is CPU-only (12 of 20 cores) and
leaves the GPU completely idle; the shards already written are complete and valid, so N1–N3 can run
against a partial corpus. Keep training's dataloader workers low so the two do not contend, and
watch that generation's games/hour does not drop.

---

## N1. The A/B harness

Every row below is a two-arm comparison, so the comparison itself should be built once, correctly,
and then reused. Without this each experiment reinvents its own setup and none are comparable.

**What "controlled" has to mean here**, learned the hard way:

- **Identical seeds and identical data order.** Adding a head consumes RNG draws, which shifts the
  shuffle unless the shuffle is seeded independently — already fixed on
  `feat/score-auxiliary-head`, and the property this protocol must preserve.
- **Identical initialisation of everything shared.** New heads are constructed *last* so the trunk
  and existing heads start from the same weights in both arms.
- **A no-op arm where one is available.** The score head supports `score_loss_weight=0`, which is
  mathematically inert; running it quantifies the residual noise floor. Any effect smaller than
  that gap is not an effect. Where a technique has no natural no-op, use a second seed instead.

**The metric set**, read on the opening-subtree holdout:

| Metric | What it answers |
|---|---|
| **value skill** (`1 − mse / colour_only_mse`) | is the value head reading the board, or the colour prior? |
| top-1 and top-3 agreement with Pentobi | did the policy head pay for it? |
| per-colour value calibration | is any gain real or a re-shuffle between colours |
| the technique's own loss | is the new head learning at all |
| train/holdout position leakage | is the holdout honest (already implemented) |

**Deliverable:** one command that takes a list of arms and emits a single comparison table plus the
per-arm JSON. Not a framework — a script that runs `distill_sl.py` twice and diffs the results.

> **As built — `scripts/ab_harness.py`.** The first `--arm` is the control every delta is measured
> against; each later one names the single thing it varies:
>
> ```bash
> uv run python scripts/ab_harness.py \
>     --config run_configurations/blokus_cloud_v2.json \
>     --corpus ~/corpora/pentobi_l9_v2 --out-dir temp/ab/ownership \
>     --arm control \
>     --arm replicate \
>     --arm ownership="--ownership-head" \
>     --noise-floor-arm replicate --noise-floor-seed 8
> ```
>
> It writes `<out-dir>/<arm>.json` (the untouched `distill_sl.py` run JSON), plus
> `comparison.md` and `comparison.json`, and prints the table.
>
> **Three mechanisms make an unfair comparison hard to build**, rather than merely discouraged:
>
> 1. **Data and protocol are harness-level.** Corpus, seed, `--max-games`, holdout fraction,
>    schedule, LR, τ and the mix weights are given *once* and forwarded verbatim. An arm cannot set
>    them because they are not arm flags.
> 2. **Arm flags are allow-listed** to the auxiliary-head switches and their weights. Anything else
>    is refused *before* a GPU-hour is spent, with a message naming `--allow-varying` as the
>    deliberate escape hatch — which is then printed in the comparison, so a reader always knows the
>    control was loosened. (`--allow-varying max-games` — bare, no dashes, since argparse would
>    otherwise read `--max-games` as the next option — is exactly how N2's data-fraction curve runs
>    through this harness.)
> 3. **The corpus is frozen before the first arm starts.** The shards present at launch are
>    symlinked into `<out-dir>/_snapshot` and every arm reads that. Without it, a corpus still being
>    generated grows between arms, `--max-games` samples different games for each, and the arms sit
>    different exams — which is how the first score-head A/B (N3 below) was wasted. Symlinks, so
>    freezing a 30 GB corpus is free. `--no-freeze-corpus` opts out for a finished corpus.
> 4. **It re-checks afterwards.** Each arm's run JSON records the settings it *resolved* plus the
>    **measured** holdout leakage, and those are diffed across arms; so is "did these arms differ in
>    exactly one head". That last check reads each head's *resolved* weight and scale, not just its
>    on/off switch, so a loss weight cannot ride along unnoticed with the head under test. Any
>    disagreement marks the run `comparable: false`, prints a
>    `NOT COMPARABLE` banner above the table, and exits non-zero. The table is still written —
>    useful for diagnosis, impossible to mistake for a result.
>
> **The noise floor is wired in.** `--noise-floor-arm` names a **replicate**: the control's exact
> settings, the same games and the same holdout, re-run from a different roll of the initial
> weights (`--noise-floor-seed`, which reaches `distill_sl.py` as `--init-seed`; reseeding `--seed`
> as well would re-split the holdout, and the floor would then measure a variation no treatment arm
> is ever exposed to). Every other arm's delta is annotated `below noise` when
> it does not exceed the replicate's own movement on that metric. Deltas are signed `(+)`/`(−)` by
> whether the metric is better high or low, so nobody has to remember that CE improves downward.
>
> A head at weight 0 is *not* a usable floor, which is why the example above is not one: the
> auxiliary heads are built after every primary head, so at a shared seed the trunk, policy head
> and value head start from identical weights, and a zero-weighted term contributes no gradient.
> The arm therefore trains bit-identically to the control, its delta is exactly 0 on every metric,
> and `below noise` could never fire for anybody. The harness refuses that configuration up front —
> a floor arm must carry no flags and must be differently seeded.
>
> The metric set is read from each arm's **best** epoch (arms early-stop at different points):
> value skill, top-1 and **top-3** agreement (`ImitationDiagnostics.top3_accuracy`, added here),
> per-colour bias and MSE, each auxiliary head's own loss *against its own baseline*
> (`evaluate_score_head` / `evaluate_ownership_head` / `evaluate_reply_head` — each returns `None`
> rather than a fabricated zero for a head the arm did not build), and the leakage figure.

## N2. Data-fraction curve

Fit at 25%, 50% and 100% of the corpus and plot held-out policy agreement against data volume.

**Why this comes before the techniques.** Our corpus is ~300k positions where every comparable
published effort used millions
([`../research/corpus-quality-principles.md`](../research/corpus-quality-principles.md) §5). If a
later result is marginal, there are two explanations — the technique does not help, or *nothing*
will help at this data volume — and without this curve we cannot tell them apart. That ambiguity
would cost far more than the three hours this takes.

Read it as: still climbing steeply at 100% ⇒ we are data-limited, and "generate more" is the
highest-value action regardless of what any technique does. Flattening ⇒ the corpus is adequate and
technique work is the right lever.

> **Measured 2026-07-31** on the partial stage-1 corpus (96×6 net, 6 epochs, seed 0, holdout 0.1),
> run on the box's idle GPU while generation continued on the CPU:
>
> | games | held-out policy CE | top-1 | value skill |
> | --- | --- | --- | --- |
> | 1,000 | 4.066 | 0.207 | −0.218 |
> | 2,000 | 3.713 | 0.231 | −0.203 |
> | 4,000 | 3.534 | 0.251 | −0.278 |
>
> **We are data-limited, decisively.** Doubling the corpus buys ~0.18–0.35 nats of CE and ~2.4
> points of top-1 each time, with no sign of flattening. Every technique result below has to be
> read against that: a marginal gain is not evidence the technique is weak, and "generate more
> games" outranks all of them. It also means the 10,000-game run is worth finishing and a larger
> corpus after it is worth generating.
>
> **The value head is worse than useless — every arm scores negative value skill.** Predicting the
> outcome from nothing but whose turn it is beats what the net learned, by 15–28%. That is a
> stronger and more specific finding than the data-fraction curve itself, and it does not improve
> with data. It is what N6 (value-label arms) and N7 (win/draw/loss head) exist to attack, and it
> raises their priority above N8.
>
> *Caveat:* the corpus was still being written while these ran, so each arm sampled its games from a
> slightly larger pool than the one before. That adds noise; it cannot manufacture a monotone
> 0.53-nat improvement, so the trend stands. Re-run on the finished corpus for a clean curve.

## N3. Score-head A/B

`score-auxiliary-target.md` S7, run under N1's protocol. The code is built and reviewed.

**This row is the pathfinder for N4 and N5.** It answers a question none of the others can: *do
auxiliary targets help this network on this data at all?* Three arms — no head, head at weight 0
(the noise floor), head at weight 0.15.

**Read it as:** a clear gain ⇒ auxiliary targets work here, proceed to N4 which is the stronger
version of the same idea. No gain but no harm ⇒ the *idea* is unproven, and N4 is the better test
of it before abandoning the family. A loss ⇒ stop, delete the head, skip N4 and N5.

> **Attempted 2026-07-31 — the result must not be read, and the attempt is why N1 exists.** Three
> arms were run directly through `distill_sl.py` rather than through the harness, and they were not
> comparable. The corpus was being written while they ran, so each arm globbed a different number of
> shards and `--max-games 4000` sampled *different games* for each — visible directly in the holdout,
> which held 11,804 scored rows for one arm and 13,165 for another. The arms sat different exams.
> `check_comparable` refuses exactly this (`num_games`, `holdout_leakage`), and the harness now
> freezes a corpus snapshot before running. Re-run through `scripts/ab_harness.py`.
>
> For the record, and *not* as a result: CE 3.493 (off) / 3.528 (weight 0) / 3.509 (weight 0.15).
> The weight-0 arm — which changes nothing that can affect the policy — moved further from the
> control than the treatment did, which is the shape of pure noise.

## N4. Ownership head

Predict, for every one of the 196 cells, who holds it at the end of the game: White, Black, or
neither.

**Why this is the strongest of the auxiliary targets for Blokus.** The final Blokus board *is* an
ownership map — unlike Go, where ownership is a scoring abstraction, here it is literally the
finished position. That gives **~196 labels per position instead of 1**, each anchored to a specific
square, forcing the trunk to learn which regions each player can actually reach. And the score
margin is exactly the sum of that map, so this strictly refines the score head rather than
competing with it.

**Honest evidence note:** KataGo ablates ownership and score *jointly* at 1.65× — its single largest
factor — and never separates them, so "ownership beats score" is **our inference**, not a published
result (review §3.4). N3 is what makes it testable: if the score head helped, ownership should help
more; if it did nothing, ownership is the fairer test of the family before giving up on it.

**Implementation.** A 1×1 convolution to 3 channels over the 14×14 board, cross-entropy per cell,
masked where the corpus lacks a final board. The loader must attach each game's final position —
derivable by replaying the stored actions, no regeneration. Reuses the score head's
checkpoint-compatibility machinery wholesale. Same rules as the score head: **off by default, never
read at play time.**

> **As built.** `NetConfig.ownership_head` / `ownership_loss_weight` (default 0.15) build
> `AlphaBlokusDuo.ownership_head`, a bare `Conv2d(num_filters → 3, 1×1)` — **579 parameters at the
> 192×12 preset, 0.007% of the net**. Deliberately no normalisation and no depth: an auxiliary
> target exists to apply pressure to the *trunk*, and anything deeper spends the capacity in the
> head instead. `BaseNNetWrapper.loss_ownership` is a per-cell cross-entropy averaged over the
> **unmasked cells only**, so it starts at ln 3 ≈ 1.10 — the same O(1) scale as the value loss,
> which is what makes `ownership_loss_weight` comparable to `score_loss_weight`. (KataGo's quoted
> 1.5 is against a differently-normalised loss and is deliberately *not* copied.)
>
> **The label is in the position's own canonical frame**, not the absolute one:
> `distill.final_ownership` returns a White-positive map (replay the stored actions, then multiply
> by `players[0]`), and each row multiplies that by *its own* `player`, so `+1` always means "the
> side to move holds this cell". Class index is `ownership + 1`, so the head's channels read
> opponent / neither / mine. The symmetry twin takes the **transposed** map. Two properties pin
> this down in tests: the map matches an independent replay from the *empty* board with the real
> colours, and every cell already occupied in a stored board carries `sign(stored_board)` as its
> label — an absolute-frame label passes that for White-to-move rows and fails for every
> Black-to-move one.
>
> Rows with no final board are masked, and there are two kinds: v2 opening rows (a DAG node has
> many games through it) and — should it ever happen — a game whose stored rows do not replay to a
> terminal position, which `final_ownership` detects and reports rather than labelling a
> half-played board as final.
>
> **Still to run: the box A/B** (N1's harness, `--ownership-head` against a control).

> **Measured 2026-07-31 — first controlled run, and it does not decide anything yet.** Four arms
> through `scripts/ab_harness.py` on a frozen 400-shard snapshot of the partial corpus (3,565 games,
> 96×6 net, 6 epochs, seed 0, holdout 0.1, leakage 0.0000 in every arm), on the box's idle GPU while
> generation continued. `replicate` is the noise floor: identical settings and holdout, different
> initial weights.
>
> | metric | control | replicate (floor) | ownership | reply |
> | --- | --- | --- | --- | --- |
> | policy CE | 3.4709 | 3.4558 (**±0.0151**) | 3.4598 (−0.0112, *below noise*) | 3.4503 (−0.0207) |
> | top-1 | 0.2519 | 0.2528 (**±0.0009**) | 0.2569 (+0.0050) | 0.2559 (+0.0041) |
> | top-3 | 0.4912 | 0.4950 (**±0.0038**) | 0.4930 (+0.0018, *below noise*) | 0.4893 (−0.0019, *below noise*) |
> | value skill | −0.1589 | −0.0928 (**±0.0660**) | −0.1829 (−0.0240, *below noise*) | −0.2418 (−0.0829) |
> | the head's own skill | — | — | **+0.5457**, 74.9% cell accuracy | top-1 0.0986, CE 4.9035 |
>
> **What is solid.** The ownership head *works as a head*: 74.9% per-cell accuracy against a 1.096-nat
> marginal baseline. The target is learnable and the trunk has the capacity to fit it. The reply head
> also trains, and masks 474 holdout rows that have no opponent reply — the pass-gap masking behaving
> as designed.
>
> **What is not.** Neither head produces a policy gain that clearly beats noise. Ownership is below
> the floor on CE and top-3. Reply clears the CE floor (−0.021 vs ±0.015) but **damages the value
> head**: skill −0.083 and value MSE +0.035, both well clear of the floor and both in the wrong
> direction. Read together with N2 — still climbing steeply at 4,000 games — this is what "the
> experiment cannot resolve it yet" looks like, not "the technique does not work".
>
> **The floor itself is one sample.** A single replicate gives a *lower bound* on run-to-run
> variation, not an estimate of it: its top-1 delta of 0.0009 is implausibly small and makes the
> top-1 row read as significant when it probably is not. Before either head is judged, the re-run
> needs **2–3 replicates** with the floor taken as the largest of their deltas. The harness supports
> one today; making `--noise-floor-arm` repeatable is the concrete next change.
>
> **Verdict: neither adopted, neither rejected.** Re-run on the finished 10,000-game corpus, at more
> epochs, with multiple replicates. Both heads stay off by default meanwhile.

## N5. Opponent-reply target

A second policy-shaped head predicting the *opponent's* next move, from the current position.

**Why it is not what the search already does.** The search predicts replies while playing, using
simulations, and discards the result — nothing about it changes the network. This target asks the
network to answer *without searching*, which forces the trunk to carry features that anticipate what
a position provokes. Those features are then free to the main policy head and the value head on
every subsequent forward pass. The search *uses* reply knowledge; this *installs* it.

**Evidence:** KataGo ablates it at **1.30×** in isolation, and credits the idea to Darkforest where
it improved *supervised* move prediction — so the evidence covers our phase, not only self-play.

**The data is already on disk.** A game row's reply distribution is the *next* row's stored soft
target; the loader attaches it index-shifted, masking the final ply of each game exactly as the
score loss masks its gaps. No regeneration.

> **As built.** `NetConfig.reply_head` / `reply_loss_weight` (default 0.15 — KataGo's own weight for
> this target) build a **second policy head** through the same factory as the first, so the two
> cannot drift apart in architecture and a `policy_head: "fc"` config gets an fc reply head.
> **17,756 parameters at the 192×12 preset, 0.22% of the net** (the score head, for
> comparison, costs 38,211 — 0.47%). `loss_reply` is the same batch-mean
> KL as `loss_pi`, restricted to the unmasked rows, so the two numbers are directly comparable.
>
> **The target is the next row's already-transformed policy target, by reference** — the same object
> the next example carries, after temperature and any legal-set floor — so attaching replies to a
> corpus costs one pointer per row and nothing else. Action indices are colour-free
> (`(square, orientation)`), so no re-indexing is needed across the canonical frame flip; the
> symmetry twin takes the *twin's* next policy, whose support is transposed.
>
> Masking uses an **all-zero** dense row as the sentinel: a real target sums to 1, so it is
> unambiguous, and `loss_reply` excludes those rows from both the numerator and the denominator.
> Rows are built **per game**, which is what stops a game's last ply borrowing the next game's
> first row — the failure mode that would be invisible in every other metric.
>
> **Cost note for the box:** with the head on, the DataLoader densifies a *second*
> full-action-space vector per example (~71 KB), so expect the input pipeline to do roughly twice
> the per-item work. Watch `dataloader_workers` if the GPU starts starving.
>
> **Still to run: the box A/B** (N1's harness, `--reply-head` against a control).

> **Measured 2026-07-31.** Run in the same four-arm comparison as N4 — see the table there. The reply
> head trains and its masking behaves correctly, its policy CE gain (−0.021) is the largest of the
> two heads and does clear the noise floor, but it is the only arm that measurably *hurts* the value
> head. Not adopted, not rejected; re-run on the full corpus with multiple replicates.

## N6. Value-label arms

Two cheap changes to *what the value head is asked to predict*, run as arms rather than as changes:

- **Teacher blend**, λ ∈ {0, 0.3, 0.5}: `λ · pentobi_eval + (1 − λ) · outcome`. λ = 0 is today.
- **Outcome-balanced sampling**: weight the ~700 Black wins up so the value head sees a less skewed
  label distribution (ELF OpenGo's precedent). A sampling weight, not a data change.

**Why the blend is an arm and not a recommendation.** The review ranked it first on cost and
evidence; that ranking was corrected. Stockfish's networks blend toward their own engine's
evaluation because **matching that engine is the goal** — there is nothing to surpass. Our goal is
the opposite, and the outcome labels are the only signal that can disagree with Pentobi. Blending
its opinion in dilutes the one channel carrying information the teacher does not already have.

The real problem the blend points at is genuine and is better solved elsewhere: **every position in
a game carries the same label**, so a 30-ply game gives 30 boards all stamped with one result and
the value head cannot tell the open opening from the decided endgame. The margin (N3) and the
ownership map (N4) supply exactly that missing within-game discrimination — and unlike the teacher's
opinion, they are facts. Expect to measure the blend and decline it.

## N7. Win/draw/loss value head

Replace the single value output with three probabilities. The scalar head cannot distinguish "a
certain draw" from "an even fight" — both are zero — and **22% of our corpus games are draws**. Lc0
moved to WDL for exactly this reason.

Deliberately sequenced after the auxiliary-target rows: it changes the value head's shape, which
would muddy the N3/N4 comparisons if done first. Already registered as IDEAS I8.

## N8. Global pooling

Add layers that summarise the whole board and feed that summary back to every square. A plain
convolution stack only ever sees local neighbourhoods, so board-wide facts — space remaining, piece
inventory, phase — are inferred slowly and badly. KataGo ablates this at **1.60×**, its second
largest factor, and Lc0 ships the equivalent in every network.

**Gated deliberately.** It is the most expensive row (the net exists in three places: torch, the jax
bridge, the ONNX export) and it changes every future network. Do it only once N4 and N5 have shown
that *the trunk's representation* is the binding constraint — if auxiliary targets move nothing,
enriching the trunk further is unlikely to either.

---

## Not in scope

- **Self-play techniques** (§3C: playout cap randomisation, policy-surprise weighting, opening
  seeding from the DAG, reanalyze). Blocked on V15; a separate plan when it fires.
- **Blokus-specific input features** (§3.6). Real but high blast radius, and it competes with N8
  for the same "help the trunk see more" budget. Revisit after N8.
- **Transformer bodies.** Excluded by project rule (`CLAUDE.md`); the numbers are recorded in the
  review for honesty.
