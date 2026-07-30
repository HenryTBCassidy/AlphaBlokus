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
| N1 | The comparison protocol: paired arms, fixed seeds, one metric set, one command | ½ day | High | |
| N2 | Data-fraction curve (25 / 50 / 100% of the corpus) — makes every later result interpretable | 3 h box GPU | High | |
| N3 | **Score-head A/B** (code already built — this is `score-auxiliary-target.md` S7) | ½ day box GPU | High | |
| N4 | Ownership head: predict the final board, per cell | 2 days + ½ day box | High | |
| N5 | Opponent-reply target: predict the reply distribution from the current position | 1 day + ½ day box | High | |
| N6 | Value-label arms: teacher blend λ ∈ {0, 0.3, 0.5}, and outcome-balanced sampling | ½ day + ½ day box | Medium | |
| N7 | Win/draw/loss value head (IDEAS I8) | 2 days + ½ day box | Medium | |
| N8 | Global pooling in the trunk — **gated on N4/N5 showing the trunk is the constraint** | 3–4 days + ½ day box | Medium | |

**The gate that matters is still V15**, the Pentobi ladder. Nothing here replaces it: these rows
choose *which net* goes into that gate. Screening happens on cheap held-out metrics (N1); the
ladder runs once, on the winning combination, not once per arm.

---

## N1. The comparison protocol

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

## N3. Score-head A/B

`score-auxiliary-target.md` S7, run under N1's protocol. The code is built and reviewed.

**This row is the pathfinder for N4 and N5.** It answers a question none of the others can: *do
auxiliary targets help this network on this data at all?* Three arms — no head, head at weight 0
(the noise floor), head at weight 0.15.

**Read it as:** a clear gain ⇒ auxiliary targets work here, proceed to N4 which is the stronger
version of the same idea. No gain but no harm ⇒ the *idea* is unproven, and N4 is the better test
of it before abandoning the family. A loss ⇒ stop, delete the head, skip N4 and N5.

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
