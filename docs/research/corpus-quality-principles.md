# What makes a supervised training corpus good — principles and a scorecard (2026-07-30)

What properties make a supervised corpus good for teaching a policy+value network to play a
two-player perfect-information board game — and how to tell whether ours is deficient. Companion to
[`corpus-generation-literature.md`](corpus-generation-literature.md), which covers generation
*mechanics* (how the strong projects built their corpora); this note covers corpus *quality
properties* — what to measure on a finished corpus, what values are healthy, and how the v2 Pentobi
corpus ([`../plans/pentobi-corpus-v2.md`](../plans/pentobi-corpus-v2.md), generating now) scores on
each. Written for a reader without an ML research background; terms are defined where first used.

**The corpus being scored** — measured on the first 400 games of the v2 stage-1 run:

| Metric | Value |
|---|---|
| Policy targets | mean entropy 0.760, ~3.0 effective moves, ~26 moves with non-zero probability out of ~370 legal |
| Value labels (per position, side to move) | 42.6% win / 41.0% loss / 16.4% draw |
| Game outcomes (per game) | White (first player) 71% / Black 7% / draws 22% |
| Structure | 0 duplicate positions, 1,034 distinct opening lines, ~29.5 plies/game |
| Scale | ~10,000 games ≈ 300k positions, ×2 mirror augmentation ≈ 600k examples |

Background fact that shapes everything below: Blokus Duo has a **severe first-player advantage** —
an earlier measurement put ~96% of decisive deterministic games between near-equal strong players
in the first player's column.

---

## What a good corpus looks like, in one paragraph

A good corpus gives the network **many independent, honestly-labelled decisions** spread over
**the states the trained agent will actually face**. Concretely: policy labels that carry the
expert's full preference ordering rather than a single move; value labels whose variation reflects
*position quality*, not a constant that could be read off whose turn it is; positions drawn from a
state distribution wide enough to cover the student's own future mistakes, not just the expert's
narrow best play; enough *independent* games (not just positions — positions within a game are
nearly copies for value purposes) relative to the network's size; and a train/holdout split that
cannot leak near-identical positions across the boundary, so the numbers you steer by are real.
Every failure mode below is a violation of one of those clauses.

---

## 1. Outcome balance and the first-player problem

**What it is.** In games where moving first is a large advantage, game outcomes are dominated by
*who moved first* rather than *who played better from here*. The value head — the part of the
network that predicts "will the side to move win?" — is trained on those outcomes.

**Why it matters mechanically.** The value head is a function from board → expected outcome. If
outcome is ~predictable from a trivially-readable feature, the network learns that feature in the
first few minutes of training and then receives almost no further gradient signal: most labels are
already explained, and the residual — the part that actually requires *reading the position* — is
a small minority of the data. The result is a head that is well-calibrated *on average per colour*
but nearly flat *within* a colour: it can't tell a winning Black position from a losing one,
because it barely ever saw a winning Black position. In MCTS that flatness is fatal — the value
head is what steers search, and a head that returns "≈0.45, you're Black" for every Black position
steers nothing. (This is a mild, legitimate form of *shortcut learning* — networks preferring an
easy predictive feature over the intended hard one; [Geirhos et al.
2020](https://arxiv.org/abs/2004.07780) is the general reference.)

**How Go and chess handle it.**

- **Go: komi** — the second player receives fixed compensation points (currently 6.5–7.5),
  *chosen to make the game balanced*, so outcomes carry positional signal by construction. KataGo
  goes further and **randomises komi** during training data generation — normal(7, 1), and 5% of
  the time normal(7, 10) — precisely so the value head learns "advantage as a function of the
  position *and* the handicap" across the whole spectrum, rather than memorising one operating
  point ([Wu 2019](https://arxiv.org/abs/1902.10565); [Komi
  overview](https://en.wikipedia.org/wiki/Komi_(Go))).
- **Chess: acceptance + resolution.** Chess has no komi; White's edge (~54–56% score) is simply
  lived with. Two coping mechanisms matter to us. First, at engine level the imbalance mostly
  converts to **draws** — in AlphaZero's 10,000-game self-play match at 1 min/move, 98% of games
  were drawn and White won 86% of the decisive 2%
  ([first-move advantage in chess](https://en.wikipedia.org/wiki/First-move_advantage_in_chess)).
  Second, Leela Chess Zero replaced the scalar value with a **WDL head** — three outputs
  (win/draw/loss probabilities) instead of one number — because a scalar conflates "certain draw"
  with "50:50 gamble" ([lczero.org
  2019](https://lczero.org/blog/2019/06/whats-going-on-with-training/);
  [WDL rescale/contempt, 2023](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/)).
  Evaluation-side, engine chess universally plays **colour-swapped pairs from shared openings** to
  cancel the imbalance — our `paired_arena` already does this.

Blokus Duo has **no komi mechanism** — the score is the score. So the KataGo trick is unavailable
in its literal form; the available analogue is **varying the starting balance of the games
themselves**, which is exactly what the v2 allocation does by playing out unbalanced openings
(V4's waste table: at T=2, much of the game mass starts from positions well below the best first
move — those are Black-favoured or drawish starts).

**Does the side-to-move convention solve the problem, or hide it?** Neither — it's the correct
convention (a value is meaningful only relative to who acts next) but it does **not** balance
anything, and the per-position 42.6/41.0 win/loss split is **mechanical, not evidence of
balance**: positions alternate perspective every ply, so ~half the rows are White-to-move (labelled
mostly "win") and ~half Black-to-move (labelled mostly "loss"), and *any* game-outcome mix produces
a near-symmetric per-position split. The number that describes the corpus's real balance is the
**per-game** distribution: 71/7/22. Read the per-position split as a checksum of alternation, not
as a health metric.

**Can the value head "cheat" by inferring who moves first?** Yes, trivially — and it's important
to be precise about why that is only half a problem. Whose turn it is, is *not hidden*: the
canonical encoding literally reorders planes so the current player comes first, and first-mover
identity is readable from piece-count parity (equal pieces placed → the first player is to move).
This is **legitimate information** — the first-mover advantage is real, and a correct value
function *should* output different priors for the two colours. The problem is not cheating in the
sense of an illegitimate feature; it is that **the colour prior explains so much of the label that
little gradient is left for position-reading**. The corpus contains only ~700 Black wins in 10,000
games — that is the entire dataset from which "what a winning Black position looks like" can be
learned.

**How to measure it.**

1. **Per-game outcome distribution** (not per-position). Ours: 71/7/22.
2. **The colour-only baseline.** Compute the loss of a "predictor" that ignores the board and
   outputs each colour's base rate (White-to-move → +0.42-ish on the tanh scale, Black-to-move →
   −0.42-ish). The trained head's held-out value loss must beat this floor *by a margin*; the gap
   is the amount of actual position-reading learned. This is a few lines against the finished
   corpus and is the single most diagnostic number for this property.
3. **Per-colour calibration curves** (already built — `evaluate_imitation_diagnostics` in
   `training/holdout.py` reports 10-bucket reliability split by side-to-move). A head that learned
   only the prior shows as a *flat cluster* of predictions within each colour: all White-to-move
   predictions piled near one value regardless of the true bucket.
4. **Within-colour prediction spread.** Standard deviation of the head's outputs conditioned on
   colour; near zero means prior-only.

**What good looks like.** There is no literature-blessed threshold; the honest statement is that
Go solves this with komi (unavailable to us), chess mostly converts it to draws, and nobody has a
clean recipe for a game that is simply unbalanced. Directionally: the more *within-colour outcome
variance* the better, and mixed outcomes for *both* colours are required, not optional. A corpus
where one colour wins <10% of games is thin on exactly the labels its value head most needs.

**How ours scores.** Better than every previous attempt, and still the weakest property. The
seed-only v1 probe measured 96% White wins from balanced strong starts — near-constant labels;
v1's random prefixes got to 75/10/15; v2's allocation gets to **71/7/22**. Draw share tripling to
22% is genuine progress (draws are intermediate-value labels — real variance). But among decisive
games White still wins 91% (71 of 78), and **Black wins fell to 7%** — unbalanced starts made more
games drawish without materially teaching Black-win patterns. Two stored assets mitigate this and
should be used: the **margin** (score difference) is stored per row and carries graded signal even
inside the White-win mass (winning by 2 vs winning by 40 are different labels), with KataGo's
score-distribution auxiliary target as production precedent that score targets densify value
learning ([Wu 2019](https://arxiv.org/abs/1902.10565); [KataGo methods
docs](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)); and the 16–22%
draw mass argues for at least considering a **WDL-style 3-way value output** (Lc0 precedent) over
a scalar tanh. Neither requires touching the corpus — both are training-time changes.

## 2. Coverage of the states the trained agent will actually meet

**What it is.** *Distribution shift* (also *covariate shift*): the corpus positions come from one
process (Pentobi L9 playing near-optimally from allocated starts), but the trained network will be
evaluated on positions from a different process (its own imperfect play, plus its opponents').
Supervised learning only guarantees accuracy on the distribution it was trained on.

**Why it matters mechanically.** A policy trained by imitation makes a small error, which takes it
to a state slightly off the expert's distribution, where it was never taught, so it errs more —
errors *compound*. [Ross, Gordon & Bagnell 2011 (DAgger)](https://arxiv.org/abs/1011.0686) proved
the cost is quadratic in game length for naive behavioural cloning, and linear if you can get
expert labels *on the learner's own states*. Search softens this (MCTS re-evaluates at play time
rather than trusting the policy blindly), but the value head is consulted on every node of every
search tree — including the off-corpus ones.

**How to measure it.**

- **Design-side (available now):** fraction of the legal opening space with labels. Measured: 279
  of 414 first moves covered by allocation; 52 of 212 canonical first positions entirely outside
  Pentobi's search (a recorded gap). 1,034 distinct opening lines in the first 400 games.
- **Empirical (the real test, cheap, not yet planned as a metric):** a **DAgger-style probe** —
  play ~50 games of the *distilled net* against Pentobi, harvest Pentobi's `move_values` on the
  *net's own* positions, and measure the net's policy agreement and value error there vs on
  held-out corpus positions. The gap between those two numbers *is* the distribution-shift cost.
  All the machinery exists (harvesting, mini-ladder players); it's one script.

**What good looks like.** The literature doesn't offer a number; it offers a mechanism ranking.
Expert-labels-on-wide-states beats expert-labels-on-expert-states (DAgger; Stockfish's training
mix explicitly includes "self-play data from openings it usually gets wrong" — [nnue-pytorch
wiki](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets)); and states from
a *different* agent labelled by your expert work well (ChessBench labelled 10M *human* games with
Stockfish and reached grandmaster-level play — [Ruoss et al.
2024](https://arxiv.org/abs/2402.04494); Stockfish trains largely on converted *Leela* games).

**How ours scores.** Good by design, unmeasured in the way that counts. The corpus is
deliberately wider than Pentobi's own play (allocation plays lines Pentobi searched but would not
choose — DAgger-shaped by construction), and every continuation is full-strength, so labels on the
widened states are honest. The two known holes: the 52 never-searched first positions (closable
only by V16's net-in-the-loop phase), and — more importantly — **no measurement yet of accuracy on
the student's own state distribution**. The DAgger probe above should be run alongside the V15
gate: if the gate fails, it distinguishes "corpus positions were fine, the net can't use them"
from "the net is fine on-corpus and dies off-corpus" — completely different next moves.

## 3. Label quality and label noise

**What it is.** Each position carries two labels. The **policy target** says which moves the
expert prefers; the **value target** says how the game turned out. Both can be noisy (high
variance — the label would differ if you regenerated it) or biased (systematically wrong).

**Why it matters mechanically.** Networks average away *noise* given enough data, but they
faithfully learn *bias*. And noise isn't free either — it sets how much data "enough" is. The key
accounting: a policy target from a deep search is an average over ~10⁶ simulations (low variance);
a policy target that is just "the one move played" is a single sample (maximum variance — this was
v1's one-hot mistake); a value target that is one game's outcome is **a single coin flip** from
the position's true win probability (a Bernoulli sample — variance p(1−p), i.e. worst at exactly
the interesting positions near 50%); a value target from an engine evaluation is low-variance but
inherits the engine's bias.

**How the strong projects handle it.** Lc0's *rescorer* overwrites game-outcome labels with
tablebase truth wherever the game touches a solved position, and "deblunders" — corrects labels
distorted by deliberately-injected exploration randomness
([TB rescoring](https://lczero.org/blog/2018/09/tb-rescoring/);
[WDL rescale post](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/)).
Stockfish blends and filters engine evaluation against game outcome. Both treat "one game result"
as a noisy estimate to be improved with whatever lower-variance truth is available.

**How to measure it.**

- Policy: target entropy / effective moves (`exp(entropy)` — the "how many moves does this
  distribution effectively spread over" number) per ply bucket; tail mass lost to truncation;
  agreement between the stored argmax and what the engine actually plays (`top_action` vs
  `action`, recorded per row).
- Value: replica disagreement — the corpus plays each start ≥2 times (R=2..32), so the spread of
  outcomes across replicas of the same start *is* the measured label noise; the noise floor of the
  teacher's own evaluation is separately measured at σ≈0.014.
- Bias: the V2 "confidently-wrong base-rate" probe (how often the teacher's visit-mass argmax is
  measurably below its own best candidate) — pending, and the most important open number.

**What good looks like.** Soft full-distribution policy targets are the validated standard: ChessBench
stored Stockfish action-values for *every legal move* and this outperformed one-hot behavioural
cloning at scale ([Ruoss et al. 2024](https://arxiv.org/abs/2402.04494)); KataGo prunes
forced-exploration visits before using visit counts as targets, i.e. even the visit vector gets
cleaned ([Wu 2019](https://arxiv.org/abs/1902.10565)); and visit distributions in general are
"where the search spent effort", not calibrated move quality ([Grill et al.
2020](https://arxiv.org/abs/2007.12509)). For values, single-outcome labels are what AlphaZero
itself uses — acceptable, provided the *correlation* problem (§4) is handled and some
lower-variance signal (margin, engine value, outcome averages) is stored alongside.

**How ours scores.** This is the corpus's strongest property. Policy: full `move_values`
distributions from ~10⁶-simulation searches, top-32 support, tail mass ~2–4%, mean entropy 0.760 ≈
2–3 effective moves — which simply *is* the teacher's measured concentration (per-ply effective
moves 1.1–7.4, corridor-dominated), faithfully harvested; ~26 labelled moves out of ~370 legal
carries the full preference ordering v1 threw away. One caveat worth knowing: the ~344 unlabelled
legal moves get probability zero in the target, and under the KL loss zero-target moves are
suppressed only indirectly (via normalisation), so the under-rated tail moves the teacher never
searched are *unlabelled*, not *condemned* — consistent with the design's intent. Value: single
Bernoulli outcomes per game-row (standard), but with margin + `search_value` stored per row and
replica-averaged `outcome_mean` at opening nodes with a count-shrunk blend — better provisioned
than the precedents require. The open flank is *bias*, not noise: fact 10 of the v2 plan measured
the teacher putting 92.9% of its mass on a reply its own deeper search ranks ~35th of 315; the V2
probe will say whether that is rare or the norm.

## 4. Diversity, redundancy and correlation

**What it is.** Positions within one game are highly correlated: they share almost all their
board content with their neighbours, and — critically — **all ~29.5 of them share one outcome
label**. A corpus of 300k positions from 10k games is not 300k independent value examples; for
value-learning purposes it is closer to 10k.

**Why it matters mechanically — the canonical episode.** AlphaGo's value network, trained on all
positions of complete games, **memorised game outcomes**: train MSE 0.19 vs test 0.37. The fix was
one position per game, from 30M distinct games (0.226/0.234) ([Silver et al.
2016](https://www.nature.com/articles/nature16961)). The mechanism: the network learns to
recognise *which game* a position comes from (openings are distinctive) and recalls that game's
outcome — high accuracy, zero generalisation. The lesson splits by head: **policy targets are
per-position** (within-game correlation is harmless — every ply is a distinct decision with its
own label), **value targets are per-game**.

**How to measure it.** Duplicate-position rate (raw and mirror-collapsed); positions per game;
games per opening; **train-vs-holdout gap on value loss** (the direct memorisation symptom);
per-colour calibration (memorisation shows as excellent train calibration, flat holdout
calibration). LLM-side evidence that deduplication per se improves generalisation:
[Lee et al. 2021](https://arxiv.org/abs/2107.06499).

**What good looks like.** Zero (or measured-and-small) exact duplicates; value-head training that
either samples ~one position per game, or downweights the value loss per position by game size, or
demonstrates via calibration that memorisation isn't happening. Standard practice post-AlphaGo is
simply *many independent games* — AlphaZero generates millions, so each game's outcome is a drop
in the ocean.

**How ours scores.** Structurally strong: 0 duplicate positions in 400 games (will rise as shared
openings replicate — expected and tracked), 1,034 distinct opening lines, opening rows deduplicated
into the DAG by construction. The exposure: 29.5 correlated rows per outcome × only 10k games ×
a 71% predictable-from-colour outcome is a *favourable* setup for value memorisation, and the
decorrelation lever (`1/game_size` value-loss weighting) exists but **defaults off**, gated on
calibration diagnostics. That gate must actually be looked at — with this outcome skew, checking
the train/holdout value-loss gap in the *first* V14 fit, not after something smells wrong, is the
cheap version of AlphaGo's lesson.

## 5. Size, and how it interacts with network capacity

**What it is.** How many examples, against how many network parameters.

**Why it matters mechanically.** A network with far more parameters than examples can fit the
training set exactly without learning transferable structure — regularisation (weight decay),
augmentation and early stopping push against this, but data volume is the fundamental currency.
For the policy head the unit is positions; for the value head (per §4) it is games.

**The reference points, honestly scaled.**

| Corpus | Size | Net | Result |
|---|---|---|---|
| AlphaGo SL ([Silver et al. 2016](https://www.nature.com/articles/nature16961)) | 29.4M positions / 160k games | 192-filter 13-layer conv | 57% top-1, base of the system |
| Maia ([McIlroy-Young et al., KDD 2020](https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf)) | 12M games *per rating bucket* | Lc0-style resnet | 46–52% top-1 human-move prediction |
| ChessBench ([Ruoss et al. 2024](https://arxiv.org/abs/2402.04494)) | 530M states / 10M games, 15B action-values | up to 270M-param transformer | ~2895 Lichess blitz Elo, no search; "strong performance only arises at sufficient scale" |
| **AlphaBlokus v2 stage 1** | **~300k positions / ~10k games (×2 mirror)** | 1–8M params (sizing sweep) | pending |

Our corpus is 2–3 orders of magnitude smaller than every success story, on a game with a larger
action space than chess (17,837 vs ~1,900 typical legal-move encodings), though a smaller board and
shorter games than Go. At the provisional 160×10 net (4.7M params) the ratio is ~8 params per
augmented example — AlphaGo SL sat at ~0.1. The v1 sizing sweep already showed the data-limited
signature: 18× more parameters bought +1.8pp top-1.

**How to measure it.** Held-out top-1/top-3 and CE as a function of *training-set fraction*
(train on 25/50/100% of the corpus, same net): if the curve is still rising steeply at 100%, data
is the binding constraint and a top-up beats every other intervention. This is hours of GPU and is
the cheapest way to answer "would 50k games fix it?" before spending the £100+ to generate them.

**What good looks like.** No universal number — but the data-fraction curve flattening before
100% is the empirical definition of "enough for this net on this game".

**How ours scores.** The honest reading: **size is the corpus's most under-powered dimension and
the design has consciously bet against it.** v1 failed at 13k games ≈ 390k positions; v2 is ~10k
games ≈ 300k positions — *smaller*, with the bet placed entirely on label quality (soft targets)
and opening coverage. That bet has support (junk openings + one-hot targets plausibly explain the
v1 failure, and ChessBench shows target *type* matters enormously), but if V15 comes back marginal
rather than clearly passed/failed, the data-fraction curve should be the first diagnostic run, and
"generate 40k more games" (a re-plan top-up, already engineered to be incremental) is likely the
highest-expected-value spend.

## 6. Phase balance — openings vs midgame vs endgame

**What it is.** Whether the corpus over- or under-represents parts of the game, relative to how
much the network needs to learn about each.

**Why it matters mechanically.** Rows are sampled ~uniformly at training time, so phases with few
rows get few gradient updates. Openings are structurally rare in any game-generated corpus (each
game has one opening and many midgame plies), and rarer still here because games start at median
depth ~4 with the shallow plies stored as deduplicated opening nodes. The measured v2 shape:
~1.6k opening rows vs ~260k game rows — **depths 1–3 are ~0.6% of rows**, while being the phase
this project has explicitly identified as its strategic edge (Pentobi's endgame calculation is the
teacher's strength; opening pattern recognition is meant to be ours).

Human-prediction work sees phases differently and it is worth knowing why: Maia-style benchmarks
*exclude* the first 10 plies as noise (memorised opening theory, not decisions) and find accuracy
dips in the midgame where choice variability peaks ([Maia-2](https://arxiv.org/abs/2409.20553);
[personalized-Maia](https://www.cs.toronto.edu/~ashton/pubs/maia-personalized2021.pdf)). That
exclusion is *not* an argument against our opening focus — their openings are rote human
repetition; ours are 25-second L9 searches, the most expensively-labelled rows in the corpus.
Different data-generating process, opposite conclusion.

**How to measure it.** Rows per ply bucket (built — V7's `analyze` reports the row mix); mean
target entropy per ply bucket (junction plies carry more learnable signal than corridor plies);
held-out policy agreement *per phase* after training (the output-side check: is the net weakest
where the corpus is thinnest?).

**What good looks like.** No literature standard exists; the defensible position is "row share
should not be wildly out of proportion to strategic weight, and where it is, fix it with sampling
weights, not regeneration". That is exactly the design's position (opening rows upweighted to ~5%
of sampled examples, weight is a V14 arm).

**How ours scores.** Known, quantified, and mitigated-on-paper: the 0.6% → ~5% upweighting exists
as a flag but is untested (a V14 arm), and per-phase held-out agreement is not yet a standard
report. Endgame coverage is naturally complete (every game is played to its true final position —
`--noresign` exists precisely for this). Adequate, pending the arm actually being run.

## 7. Train/test hygiene for game data

**What it is.** The holdout set exists to estimate performance on *unseen* positions. Game
positions defeat naive random splits: consecutive positions differ by one piece, many games share
opening prefixes, and different move orders *transpose* into identical positions. A random
position-level split puts near-copies of training rows in the holdout, and every holdout metric
becomes flattery.

**Why it matters mechanically.** Every decision downstream — early stopping, net sizing, τ
sweeps, the pass/fail read on arms — steers by the holdout number. Leakage doesn't just inflate
it; it inflates it *more for bigger/more-memorising nets*, silently corrupting model comparisons.
Empirically, moving from row-level to game-level splits measurably changes validation error and
reduces memorisation of position-specific patterns
([PAWN, 2026](https://arxiv.org/pdf/2604.15585); the general leakage taxonomy:
[Exploring Data Leakage Risks in ML](https://arxiv.org/pdf/2401.13796)).

**What the right unit is.** The unit of splitting must be the unit of correlation. For ordinary
game corpora that is the **game** (AlphaGo's value fix was effectively this). For *this* corpus it
is coarser: v2 deliberately gives many games a shared opening, so game-level splitting would put
near-identical early positions on both sides. The design's answer — split by **canonical ply-1
opening subtree** — is correct and ahead of common practice.

**How to measure the residue.** Duplicate-position rate *across the train/holdout boundary* (raw
and mirror-collapsed): transpositions can still smuggle identical midgame positions across subtree
lines. Report it; if material, dedupe holdout rows against the train set.

**How ours scores.** The split design is right; the verification is **not done** — the v2 plan's
own post-review note flags that the promised train/holdout duplicate-position metric is
unimplemented, and states it must land before the V15 verdict is trusted. It needs only the
finished corpus and seconds of compute. This is the single cheapest outstanding item on the whole
list and it guards the project's main gate.

## 8. Distilling from a fixed engine — ceiling and blind spots

**What it is.** The teacher is frozen. Its mistakes are stationary — every game reiterates them —
and pure imitation can at best converge to the teacher-minus-noise.

**What the record says.** Perfect distillation isn't achieved even at industrial scale:
ChessBench's 270M-parameter transformer on 15B Stockfish annotations remains explicitly short of
its teacher ([Ruoss et al. 2024](https://arxiv.org/abs/2402.04494)). Teacher error is not
hypothetical even for superhuman engines — KataGo-class networks harbour systematic, *stationary*
blind spots exploitable by adversarial policies ([Wang et al.
2023](https://arxiv.org/abs/2211.00241)), and our own measurements found Pentobi placing 92.9% of
its visit mass on a reply its own deeper search ranks ~35th (v2 plan, fact 10). A student trained
on those distributions inherits the misallocations *without* the deep search that lets the teacher
partially recover at play time.

**How anyone escapes.** Three mechanisms, all with precedent: (1) **imitate-then-RL** — AlphaGo's
RL policy beat its SL parent in 80% of games ([Silver et al.
2016](https://www.nature.com/articles/nature16961)); Expert Iteration formalises the loop
([Anthony et al. 2017](https://arxiv.org/abs/1705.08439)); (2) **outcome-grounded correction** —
label states with what *actually happens* when lines are played out, so reality overrules the
teacher's opinion (Lc0 rescorer; our allocation-plays-out-the-alternatives design); (3) **targeted
weakness data** — Stockfish's mix includes self-play "from openings it usually gets wrong"
([nnue-pytorch wiki](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets));
our V16 net-in-the-loop phase is this.

**How to measure teacher inheritance.** Track **top-3 agreement alongside top-1** (a student that
*correctly* disagrees where the teacher is wrong scores lower top-1 — already adopted); measure
the confidently-wrong base rate (V2 probe, pending); and treat the ladder, not agreement, as the
verdict metric (adopted).

**How ours scores.** The design is unusually self-aware here — the correction channels
(allocation breadth, outcome labels, play-time search) match the literature's mechanisms, and the
one number that would quantify the inherited-bias risk (the V2 probe) is scoped and pending. The
ceiling itself is accepted and priced in: SL is not expected to beat L9; it's expected to move the
ladder and hand RL a broad base. Nothing found in this pass contradicts that framing.

---

## Where the current corpus looks weakest — ranked

1. **Value-signal concentration in the colour prior.** 71/7/22 per game; 91% of decisive games to
   White; ~700 Black wins in the whole corpus. Most of the value label is predictable from
   side-to-move identity — legitimate information, but it starves position-reading gradient,
   especially for Black. **Do:** compute the colour-only baseline loss and report the trained
   head's margin over it; make per-colour calibration a headline V14 metric, not a footnote;
   promote the deferred **margin-aware value target** experiment (the margin is already stored;
   KataGo's score targets are the precedent) and consider a WDL-style 3-way head given 22% draws.
2. **Raw size.** ~300k positions / 10k games is smaller than the failed v1 corpus and 2–3 orders
   below every working precedent. The quality bet is reasonable and articulated, but unhedged.
   **Do:** run the data-fraction curve (train on 25/50/100%) in V14 — hours of GPU — so that if
   V15 is marginal, "top up 4×" is a measured decision, not a guess.
3. **Unverified split hygiene.** The subtree holdout design is right, but the promised
   train-vs-holdout duplicate/transposition metric is unimplemented, and the V15 verdict leans on
   the holdout numbers. **Do:** land it before reading any V14/V15 number (seconds of compute;
   already flagged in the plan's post-review note — this is a reminder, not news).
4. **Value memorisation risk left on a manual gate.** 29.5 shared-outcome rows per game × 10k
   games × skewed outcomes is AlphaGo's overfitting setup in miniature; the `1/game_size`
   decorrelation flag defaults off. **Do:** check the train/holdout value-loss gap and per-colour
   holdout calibration in the *first* V14 fit; flip the flag if the gap yawns.
5. **No measurement on the student's own state distribution.** All quality numbers are on-corpus;
   the DAgger lesson says the off-corpus gap is where imitation dies. **Do:** the 50-game
   probe — harvest Pentobi labels on the distilled net's own positions, compare agreement/value
   error vs held-out corpus positions — alongside V15, so a gate failure is diagnosable.
6. **Teacher-bias base rate unknown.** Fact 10's 4–5σ mass misallocation is n=2; the V2 probe
   (n≈30) is scoped but hasn't run. Low urgency only because the design already routes correction
   through outcomes — but V15's diagnosis needs the number.

Items 3, 4 and 1's measurement half cost minutes-to-hours and should all exist before the V15 gate
is read.

## Pre-commit checklist for any future corpus

Run down this list before declaring a corpus ready to train against:

- [ ] **Per-game outcome mix** reported (not per-position); both colours have non-trivial win
  counts; draw share known.
- [ ] **Colour-only baseline loss** computed; recorded as the floor every value head must beat.
- [ ] **Policy target entropy / effective moves / tail mass** per ply bucket; support ⊆ legal
  verified; `action` vs `top_action` mismatch rate known.
- [ ] **Duplicate-position rate** raw + mirror-collapsed, within corpus **and across the
  train/holdout boundary**.
- [ ] **Split unit = the correlation unit** (opening subtree here), stratified, documented.
- [ ] **Games count and positions-per-game** reported separately (value-head sample size = games,
  not positions).
- [ ] **Rows per phase** (ply buckets) vs intended sampling weights.
- [ ] **Params-per-example** for the intended net; data-fraction curve planned if > ~1.
- [ ] **Label-noise estimate** (replica outcome spread; teacher eval test-retest σ).
- [ ] **Teacher-bias probe** rate known or explicitly accepted as open.
- [ ] **Off-corpus probe** (student's own states, teacher-labelled) scheduled against the gate.
- [ ] Validation replay of every row (legality, labels, terminal scores) — already standard here.

## Open questions the literature does not settle

- **How much outcome imbalance is too much.** Go fixed it with komi and chess drowned it in
  draws; no precedent trains a strong value head on a komi-less game where one colour wins >90%
  of decisive games. Whether margin/WDL targets fully compensate is our experiment to run.
- **Minimum viable corpus size for a 14×14 game with soft targets.** ChessBench's "scale is
  necessary" was measured on chess with one-hot-to-action-value comparisons at 10⁸–10⁹ examples;
  nothing calibrates the 10⁵–10⁶ regime for a smaller game with richer targets. The data-fraction
  curve is the only honest answer.
- **Whether breadth beats replication in a tiny, overlapping opening space.** The imitation
  scaling-law evidence ([Lin et al. 2024](https://arxiv.org/abs/2410.18647)) is
  environments-and-objects, not shared-board openings; V13's ablation is the right instrument and
  the literature is a prior, not a verdict.
- **Whether SL agreement metrics predict ladder movement at all.** ELF OpenGo observed a
  superhuman policy agreeing with strong humans only ~46% of the time
  ([Tian et al. 2019](https://arxiv.org/abs/1902.04522)); top-1-vs-teacher is a selection metric,
  not a strength metric, and the mapping between the two is unknown for Blokus.

## Sources

- [Silver et al. 2016, *Mastering the game of Go with deep neural networks and tree search*](https://www.nature.com/articles/nature16961)
- [Wu 2019, *Accelerating Self-Play Learning in Go* (KataGo)](https://arxiv.org/abs/1902.10565) + [KataGo methods docs](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- [Komi (Go) — Wikipedia](https://en.wikipedia.org/wiki/Komi_(Go))
- [First-move advantage in chess — Wikipedia](https://en.wikipedia.org/wiki/First-move_advantage_in_chess)
- [lczero.org 2019, *What's going on with training*](https://lczero.org/blog/2019/06/whats-going-on-with-training/); [2018, *TB Rescoring*](https://lczero.org/blog/2018/09/tb-rescoring/); [2023, *WDL rescale/contempt*](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/)
- [official-stockfish/nnue-pytorch wiki, *Training datasets*](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets)
- [Ruoss et al. 2024, *Amortized Planning with Large-Scale Transformers* (ChessBench)](https://arxiv.org/abs/2402.04494) ([repo](https://github.com/google-deepmind/searchless_chess))
- [McIlroy-Young et al. 2020, *Aligning Superhuman AI with Human Behavior* (Maia, KDD)](https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf) ([blog](http://csslab.cs.toronto.edu/blog/2020/08/24/maia_chess_kdd/)); [Maia-2, NeurIPS 2024](https://arxiv.org/abs/2409.20553); [personalized Maia 2021](https://www.cs.toronto.edu/~ashton/pubs/maia-personalized2021.pdf)
- [Ross, Gordon & Bagnell 2011, *DAgger*](https://arxiv.org/abs/1011.0686)
- [Grill et al. 2020, *MCTS as Regularized Policy Optimization*](https://arxiv.org/abs/2007.12509)
- [Geirhos et al. 2020, *Shortcut learning in deep neural networks*](https://arxiv.org/abs/2004.07780)
- [Wang et al. 2023, *Adversarial Policies Beat Superhuman Go AIs*](https://arxiv.org/abs/2211.00241)
- [Lee et al. 2021, *Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499)
- [Lin et al. 2024, *Data Scaling Laws in Imitation Learning*](https://arxiv.org/abs/2410.18647)
- [Tian et al. 2019, *ELF OpenGo*](https://arxiv.org/abs/1902.04522)
- [Anthony, Tian & Barber 2017, *Expert Iteration*](https://arxiv.org/abs/1705.08439)
- [*PAWN: Piece Value Analysis with Neural Networks* (game-level vs row-level splits)](https://arxiv.org/pdf/2604.15585)
- Internal: [`corpus-generation-literature.md`](corpus-generation-literature.md), [`../plans/pentobi-corpus-v2.md`](../plans/pentobi-corpus-v2.md), [`../plans/archive/pentobi-distillation.md`](../plans/archive/pentobi-distillation.md), [`distillation-net-sizing.md`](distillation-net-sizing.md)
