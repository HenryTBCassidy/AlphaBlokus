# AlphaZero-family techniques — what the leading projects do, and what applies to us (2026-07-30)

A literature review with an engineering verdict. Half one reads the primary sources of the leading
open AlphaZero-family projects — KataGo, Leela Chess Zero, ELF OpenGo, the DeepMind papers,
Stockfish NNUE — and reports what each technique is, why it works, what it costs, and what evidence
exists for its size of effect. Half two judges AlphaBlokus against that list: what we already do
well, what we are missing, and in what order to close the gaps.

Written for a reader without an ML research background; terms are defined where first used.
Companions: [`corpus-quality-principles.md`](corpus-quality-principles.md) (what makes a supervised
corpus good — not repeated here) and
[`corpus-generation-literature.md`](corpus-generation-literature.md) (how the strong projects built
their corpora — not repeated here).

**How to read this document.**

- Every technique carries a **phase tag**: **[SL]** = supervised distillation from Pentobi (the
  current phase — the v2 stage-1 corpus is generating now, and the V15 ladder gate decides what
  happens next); **[RL]** = self-play reinforcement learning (Phase 3, gated on V15 — *no RL
  technique can be evaluated until that phase starts*); **[Arch]** = network architecture (benefits
  both phases); **[Search]** = play-time search (orthogonal to training).
- Claims from papers/docs are cited inline. Where a claim is **our inference rather than a published
  result, it is labelled "our inference"** — the two must never blur, and one section below exists
  specifically to un-blur one such claim (§3.4).
- KataGo's per-technique numbers are **removal factors** from its ablation (see §2): "1.37×" means
  the run *without* the technique needed 1.37× more training compute to reach the same strength.
  They were measured on 2-day shortened runs in Go with self-play RL; treat them as evidence of
  *direction and rough magnitude*, not as predictions for a 14×14 game in an SL phase.

---

## 1. The ranked shortlist

Techniques worth doing, best first. Ranking weighs expected impact on *our measured weak points* ×
strength of evidence ÷ implementation cost, with SL-phase items ranked ahead of RL-phase items
because only they can move the V15 gate.

| # | Technique | Phase | Cost | Evidence | Why us |
|---|-----------|-------|------|----------|--------|
| 1 | Teacher-value λ-blend into game-row value targets (§3.8) | SL | ~½ day (loader flag + one V14 arm) | Stockfish NNUE master nets train at λ = 1.0 → 0.7 eval-vs-outcome blend; Lc0 Q-ratio; our V9 already blends on opening rows | Attacks the corpus's measured weakest property — a value head starved by the 71/7/22 colour-prior outcome skew — using the `search_value` column already stored on every row |
| 2 | Auxiliary opponent-reply policy target (§3.3) | SL | ~1 day (second tiny policy head + loader) | KataGo ablation 1.30×; grounded in a **supervised**-learning precedent (Darkforest) | The reply distribution is already in the corpus (the next row's soft target); pure representation gain, discarded at play time |
| 3 | Ownership head — per-cell final-board prediction (§3.4) | SL | ~2 days (head trivial; loader must attach the final board per game) | KataGo ablates ownership+score jointly at 1.65× — its largest single factor; the Blokus-specific case is our inference | The margin is literally the sum of the ownership map, so this strictly refines the score head already built on `feat/score-auxiliary-head` — ~196 labels per position instead of 1 |
| 4 | Global pooling / squeeze-and-excitation in the trunk (§3.5) | Arch | ~3–4 days (torch net + jax bridge + ONNX export all change) | KataGo ablation 1.60×; Lc0 ships SE in every ResNet since T60 (no isolated Elo published) | Blokus has genuine board-wide state a plain conv stack computes badly: piece inventory, tempo parity, phase; our trunk has no global path at all |
| 5 | WDL (win/draw/loss) value head (§3.7) | SL | ~2 days | Lc0 standard since July 2019 (no isolated Elo figure exists); 22% of our corpus games are draws | Already registered as IDEAS I8; deliberately sequenced *after* the score-head A/B so the experiments stay readable |
| 6 | Outcome-balanced value sampling (§3.12) | SL | hours | ELF OpenGo evenly sampled black-win and white-win games to break a feedback loop | The cheapest possible mitigation of "~700 Black wins in the whole corpus" — a sampling weight, not a data change |
| 7 | Playout cap randomization (§3.1) | RL | ~2–4 days (awkward under jit — see verdict) | KataGo ablation 1.37×; also beat every fixed cap they tried | Our value signal is the scarce resource; more games per GPU-hour = more independent outcome labels |
| 8 | Policy surprise weighting (§3.10) | RL | ~1 day | KataGo: "one of the larger improvements… between its g170 run and earlier runs" (no Elo number) | Already half-registered as IDEAS I7 (prior-vs-search disagreement is its free referee) |
| 9 | Self-play opening seeding from the corpus DAG (§3.9) | RL | ~1 day | KataGo forks 5% of games into engineered openings; Lc0 self-play runs from an opening book (`run2_book.pgn`) | We own a labelled, allocation-weighted opening DAG — a better book than either project had |
| 10 | Reanalyze-style target refresh (§3.11) | RL | ~3+ days | MuZero Reanalyze: 80% of updates from re-searched old positions; 731% median Atari score | Worth it only if RL proves game-generation-limited; park until measured |

Below the line (evaluated in §3, not shortlisted): Blokus-specific input features (§3.6 — real but
high blast radius), forced playouts + policy target pruning (§3.2 — **superseded by Gumbel on our
production path**), moves-left head, transformer bodies, KataGo's search-time value corrections,
resignation machinery.

---

## 2. Where KataGo's numbers come from — the headline vs the ablation

KataGo's core text is David Wu, *Accelerating Self-Play Learning in Go*
([arXiv:1902.10565](https://arxiv.org/abs/1902.10565)). Two different claims live in it, and they
are routinely conflated:

**The headline (whole-system) claim.** The final paper (v5, Nov 2020) claims "a 50x reduction in
computation over comparable methods": KataGo surpassed ELF OpenGo's final model "after only 19 days
on fewer than 30 GPUs" (4.2M self-play games), where ELF used ~2,000 V100s for ~two weeks. Versus
Leela Zero the claim is deliberately weaker — "at least an order of magnitude" — because LZ's
multi-year distributed run isn't cleanly comparable. The earlier v1 (Feb 2019) claimed only ~5× vs
Leela Zero at strong levels and 30–100× at amateur levels. The 50× is a comparison of *entire
systems* (all techniques + engineering + net-size scheduling), **not** the product of the
per-technique factors.

**The ablation (per-technique) claim.** The paper *does* ablate individual techniques — Table 2,
§5.2. Each ablation run is the main run minus one component, shortened to ~2 days; strength is read
at a fixed compute mark (2.5 billion equivalent 20-block queries); the "factor" is how much longer
the ablated run needs to catch up:

Read this table as *the cost of deleting each technique*. The first row is the full system;
every row below is that same system **with one component removed**, so every score is lower
than 1329 and every technique is a gain, not a cost.

| System | Elo it reaches at a fixed compute budget | Extra compute needed to catch up |
|---|---|---|
| **Full system (all techniques)** | **1329** | 1.00× |
| …minus playout cap randomization | 1242 | **1.37×** |
| …minus forced playouts & policy target pruning | 1276 | **1.25×** |
| …minus global pooling | 1153 | **1.60×** |
| …minus auxiliary policy targets (opponent reply) | 1255 | **1.30×** |
| …minus auxiliary ownership and score targets | 1139 | **1.65×** |
| …minus game-specific features and optimizations | 1168 | **1.55×** |

The product of the factors is ~9.1×, and the paper itself flags this as an *underestimate* (several
techniques kept gaining as runs lengthened, and the ablations were short). So the honest attribution
is: **the paper credits the six ablated technique groups with ~9× jointly, individually 1.25–1.65×
each; the remaining gap to 50× is everything else** — net-size scheduling, target weighting, komi
and opening randomization, training engineering — which the paper does **not** ablate separately.
Any finer-grained attribution than Table 2 is invented.

Post-paper techniques (2020–2023) live in
[KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md), most
with weaker or no controlled measurement; the ones that matter to us are folded into §3.10.

---

## 3. Technique by technique

Grouped by **when we could act on it**, because that is the decision the list has to support:

- **§3A — usable now**, during supervised distillation. Only these can move the V15 gate.
- **§3B — the network's architecture.** Usable now in principle, but changes every future net.
- **§3C — self-play only.** Blocked until the RL phase, which is itself gated on V15.
- **§3D — ruled out**, with the reason.

### §3A — Usable now (supervised distillation)

These are the only techniques that can move the V15 gate, because they act on the corpus we
already have.

### 3.3 Auxiliary policy target: predicting the opponent's reply — verdict: **applies now**

**[SL now; also RL later]**

**What it is.** A second, small policy head trained to predict the *opponent's next move* — in
KataGo, with weight 0.15 relative to the main policy loss. The head's output is **never used to
play**; after training it is dead weight (or simply dropped).

**"Isn't MCTS already doing this?"** — the reasonable objection, and the answer is the key
distinction of the whole auxiliary-target family:

- **Search predicts replies at play time, with compute.** MCTS explores the opponent's answers
  every move — but that prediction lives in the *tree*, is rebuilt from scratch position by
  position, and costs simulations each time. Nothing about it changes the network.
- **An auxiliary target predicts replies at training time, into the weights.** Adding "also predict
  the reply" to the loss forces the **shared trunk** — the residual stack both heads read from — to
  carry features that anticipate threats and answers. The trunk cannot minimise the reply loss
  without representing "what does this move provoke?", and those same features are then available
  for free to the main policy head and the value head on every future forward pass.

So the search *uses* reply knowledge; the auxiliary target *installs* it. The literature's framing
is that auxiliary targets are regularisers-plus: extra supervision that shapes the representation
toward game-relevant structure instead of whatever shortcut minimises the primary loss. The measured
effect in KataGo's RL: removal factor **1.30×**. Crucially for us, the technique's provenance is
**supervised**: the paper credits it to Tian & Zhu's Darkforest, where it "improve[d] supervised
move prediction" — so the evidence covers our current phase, not just RL.

**For us — the data is already on disk.** Every v2 game row's *next* row carries Pentobi's full
`move_values` soft distribution for the reply position; the loader (`CorpusGameRows`) can attach it
index-shifted with no regeneration (last ply of each game has no reply — mask it, exactly as the
score loss already masks its gaps). Cost: a second `ConvPolicyHead`-shaped head (1×1 conv — tiny),
one loss term, loader threading; the checkpoint-compat machinery from the score head
(`training/checkpoint_compat.py`, prefix-scoped tolerance) is the template. One subtlety our
encoding introduces (our inference): the trunk sees the board *canonicalised to the side to move*,
so "opponent reply" is a prediction in the opponent's frame; the cleanest formulation is to predict
the reply distribution over the *current* frame's action space of the position after the argmax
move, matching how the corpus stores it. **Verdict: applies now; shortlist #2.**

### 3.4 Ownership and score targets — verdict: **applies now**, and here is the honest audit of our own claim

**[SL now; also RL later]**

**What KataGo actually does and actually credits.** Two auxiliary target families, trained
alongside win/loss:

- **Ownership**: for every board point, predict who will own it at game end (win/lose/shared —
  a per-point classification). Loss weight 1.5/b² per point (b = board width) — i.e. roughly
  weight 1.5 spread across the board.
- **Score**: predict the **full distribution** over final score differences — a probability per
  possible score, trained with both a PDF term (weight 0.02, "rewards guessing the score exactly")
  and a CDF term (weight 0.02, "pushes the overall mass to be near the final score"), plus small
  self-prediction terms.

The paper's rationale is the value-data-starvation argument: the win/loss target is one bit per
game, while "these auxiliary targets provide a dense signal on every point of the board / every
position" that regularises the shared trunk toward reading territory. **The ablation removes
ownership and score together as one group: 1.65×, the single largest factor in Table 2. The paper
never splits them.** KataGo also uses the score prediction at *search* time — a small score-utility
term (c ≈ 0.4–0.5, arctan-shaped, re-centred at the root each search) in the utility MCTS
maximises — which is a separate decision our score-head plan explicitly (and rightly) declined.

**The claim under audit.** It was put to Henry that a per-square ownership target might be *more*
valuable for Blokus than the score target, because a finished Blokus board literally is an
ownership map — every cell ends White, Black or empty — giving ~196 labels per position instead
of 1. **That was an inference by us, not a KataGo result**, and the audit comes out as follows:

1. **Correctly labelled: KataGo provides no evidence for the comparison.** The 1.65× is joint;
   nothing published separates ownership from score. The only whisper in that direction is that
   KataGo's *loss weights* put most of the auxiliary mass on ownership (≈1.5 total vs 0.02 + 0.02
   for score) — a design choice by an author who tuned both, which is suggestive but is not an
   ablation. Anyone stating "KataGo showed ownership is worth more" is wrong.
2. **The mechanism genuinely transfers — arguably better than in Go** (our inference). In Go,
   "ownership" needs rules conventions (life/death, territory scoring) to define. In Blokus Duo the
   final position *is* the label — no judgement call, read it off the terminal board, which every
   corpus game shard contains in full. Moreover the connection to score is exact and linear:
   **margin = (#White cells − #Black cells) of the final map**, so the ownership target is a strict
   refinement of the score target — the score head's `tanh(margin/25)` is a 1-dimensional shadow of
   it. Predicting the map forces the trunk to answer "which regions can each side still reach and
   fill, and which will die empty?" — and region access is plausibly *the* central strategic
   quantity of Blokus. ~196 dense, spatially-grounded gradients per position versus one scalar.
3. **The honest caveats.** (a) Every row of a game shares one final map, so the target is
   per-game-correlated exactly like the value label (the AlphaGo memorisation lesson —
   [`corpus-quality-principles.md`](corpus-quality-principles.md) §4 — applies; the map's high
   dimensionality makes rote memorisation harder, but the correlation is real). (b) A fair
   ownership-vs-score comparison would pit ownership against KataGo's *score distribution*, not
   against our scalar tanh margin — our score head is already the weaker member of that pair.
   (c) Empty is the modal cell class late in games; the head must be a 3-way per-cell softmax, not
   a binary one, and "predict empty" is where much of the signal lives (contested dead zones).

**Verdict: the reasoning survives the audit as a well-founded hypothesis — plausibly yes for
Blokus — but it is a hypothesis.** The economical sequence: read the score-head A/B (S7 — already
built and queued) first; if the margin signal moves value skill at all, the ownership head is the
natural escalation and can reuse the entire S-series machinery (flag-gated head, prefix-scoped
checkpoint compat, masked loss, holdout metric). Loader cost is the real work: attach each game's
final placement board (the last row's board plus its action, or a replay) to every row. Shortlist
#3.

### 3.8 Blending the teacher's evaluation into value targets (the NNUE λ) — verdict: **applies now; shortlist #1**

**[SL — the current phase, directly]**

**What it is.** Stockfish NNUE's training target for a position is not the game result alone: it is
a **λ-interpolation between the engine evaluation and the game outcome**, both mapped into the same
win-probability space — `target = λ · eval + (1 − λ) · result`
([nnue-pytorch docs](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)).
Real master-net recipes anneal λ from 1.0 to ~0.7 over training ([e.g. Stockfish PR
#4782](https://github.com/official-stockfish/Stockfish/pull/4782)) — mostly-eval with a meaningful
outcome component. Lc0's T60-era "Q-ratio" experiments (blending the search's Q into the value
target) are the same instinct ([end-of-era post](https://lczero.org/blog/2019/07/end-of-era/)), and
Lc0's value-repair work attacks the same label-noise problem from another angle.

**The mechanism, in our terms.** A game outcome is one Bernoulli coin-flip per game — and our coin
is loaded: 71/7/22 per-game outcomes mean most of the label is explained by the colour prior, and
the within-colour gradient the value head actually needs is scarce
([`corpus-quality-principles.md`](corpus-quality-principles.md) §1). The teacher's per-position
evaluation is a **low-variance, per-position label that varies within a colour** — precisely the
signal the outcome labels lack — at the price of inheriting the teacher's bias. λ is the dial
between label variance (outcome) and label bias (eval).

**Why this is nearly free for us.** Every v2 game row already stores `search_value` — Pentobi's
backed-up evaluation of that exact position from a ~10⁶-simulation search — and V9's opening-row
blend (`(n·outcome_mean + k·(2·search_value − 1))/(n + k)`) already built the rescaling and the
precedent. Extending a λ-blend to *game rows* is a loader flag and a V14 arm.

**Why the Stockfish precedent transfers worse than it looks (Henry, 2026-07-30).** Stockfish's
NNUE exists to *imitate Stockfish's own search* as cheaply as possible — matching the engine **is**
the objective, so training toward its evaluation is training toward the goal. Our objective is the
opposite: **surpass** the teacher. The outcome labels are the only signal in the pipeline that can
disagree with Pentobi, so blending its evaluation in dilutes the one channel carrying information
the teacher does not already have. Same mechanism, opposite goal — which is a reason to treat the
λ arms as a measurement we may well decline, not as a default. **Ranking note:** #1 was assigned on
"cheap × attacks the weakest measured property × industrial precedent"; on this objective mismatch
it should sit below the two targets that improve within-game discrimination *without* importing the
teacher's opinion at all — the score margin and the ownership map, both of which are ground truth.

**What the outcome label actually lacks — and why margin and ownership fix it more cleanly.** The
problem is not mainly noise: it is that **every position in a game carries the same label**. A
30-ply game yields 30 boards all stamped with one result, so the wide-open opening and the
long-decided endgame are labelled identically and the value head has nothing to tell them apart.
Pentobi's evaluation varies within the game and supplies exactly that missing discrimination — but
so do the **margin** and the **ownership map**, and those are facts rather than opinions. Prefer the
facts.

**The honest tension — this dilutes the design's correction channel.** The v2 plan deliberately
keeps game-row values pure-outcome: outcomes are the channel through which reality *corrects* the
teacher's misjudgements (the plan's "allocation-T diversifies the evidence; outcome labels carry
the correction"). A λ-blend feeds some teacher opinion back into that channel. Both arguments are
right; they trade off, and the trade is measurable: run λ ∈ {0, 0.3, 0.5} as arms and read
**within-colour value skill** (the `1 − mse/colour_only_mse` diagnostic that already exists) and
per-colour calibration on the subtree holdout. Sanity notes: Pentobi's values are
win-probability-*like* but unbounded and only meaningful for visited children (v2 plan fact 5) —
the `2v − 1` rescale is approximate, one more reason to keep λ moderate rather than NNUE's 0.7+.
**Verdict: applies now — the cheapest intervention aimed at the corpus's measured weakest
property, with industrial precedent. Shortlist #1.**

### 3.7 WDL value head — verdict: **applies (already registered as I8), sequenced after the score-head A/B**

**[SL now; RL later]**

Lc0 replaced its scalar value with three softmax outputs — P(win), P(draw), P(loss) — in every
network since July 2019 ([WDL head](https://lczero.org/blog/2020/04/wdl-head/)); the scalar any
consumer needs is recovered as P(win) − P(loss), so downstream search is untouched. The mechanism:
one number in [−1, 1] cannot distinguish "certain draw" from "50:50 gamble" — both are 0 — and
that distinction changes correct play. Honest evidence note (verified): **no isolated Elo figure
for the WDL head exists in Lc0's primary sources** — it arrived in test50 bundled with other
changes; the case rests on mechanism and permanent adoption.

For us the case is quantified locally: **22% of v2 corpus games are draws**, and in Blokus a draw
is an exact score tie — a specific, predictable event, not chess's default outcome. All of this is
already argued in [IDEAS I8](../IDEAS.md); the one thing this review adds is that the literature
side checks out, with the caveat above. **Verdict: applies; keep I8's sequencing (score-head A/B
first, one readable experiment at a time).** Also fold in Lc0's related search-time work only if
this ships: WDL rescale/contempt is inference-time machinery on top of a WDL head and is why their
piece-odds bots work ([v0.30 rescale
post](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/)) — not
relevant to the gate, noted for completeness.

### 3.6 Game-specific input features — verdict: **applies, mid-priority**

**[Arch — benefits both phases]**

**What KataGo feeds in beyond raw stones**: liberties (1/2/3), ko/superko legality, the last five
move locations, ladder status (three history planes plus "a move here catches a ladder"),
pass-alive regions, plus global scalars (komi, rules, pass history). Removal — *bundled with two
minor Go-specific search optimizations* — costs **1.55×**. The paper's framing is worth keeping:
domain-specific features "account for only a small fraction of the total improvement" relative to
the general techniques; ELF OpenGo's ladder failure (§3.12) shows what happens *without* them —
convnets "lack the right inductive bias" for some tactical structures and memorise instead.

**Blokus equivalents, given our 44-channel encoding** (our inference; ordered by expected value):

1. **Placeable-corner planes** — for each side, the cells that are currently legal attachment
   points (diagonally adjacent to own colour, not orthogonally adjacent, not occupied). This is the
   Blokus analogue of liberties: the game-defining resource, and cheap to compute (we already
   compute far more in move-gen).
2. **Explicit inventory** — 21 + 21 binary "piece still in hand" indicators (or a
   remaining-squares count) as global-scalar inputs rather than inferring from all-zero planes.
   Pairs naturally with §3.5, which is the structure that consumes global scalars.
3. **Move number / phase scalar** — trivial.
4. Last-move planes matter less here than in Go (our state is fully Markov given placements +
   inventory, which we encode), and there is no ko/ladder analogue.

**Cost is the blast radius**: the encoding is produced in `as_multi_channel`, replicated in the
compact-board lazy encoder, re-implemented as int8 matmuls in the jax kernels
(`games/blokusduo/jax/kernels.py`), and consumed by the web export — four surfaces per feature.
**Verdict: applies; do it once, deliberately, alongside the §3.5 trunk change (one
checkpoint-compat break instead of two) rather than piecemeal.**

### 3.12 ELF OpenGo's operational lessons — verdict: **one applies now, two are cautions**

**[SL + RL]** From [Tian et al. 2019](https://arxiv.org/abs/1902.04522):

1. **Outcome-balanced sampling — applies now.** ELF hit a feedback loop (early value bias → Black
   resigns early → less diverse buffer) and corrected it "by evenly sampling black-win and
   white-win games". Our 71/7/22 skew is worse than anything ELF faced. A per-outcome sampling
   weight (or value-loss weight) that upweights Black wins toward parity is hours of work on the
   existing loader and composes with §3.8. Not a substitute for richer targets — reweighting ~700
   Black wins cannot invent patterns that aren't there — but the cheapest first-aid available.
   **Shortlist #6.**
2. **Caution: some tactics resist convnets.** Go ladders were "learned slowly… never fully
   mastered"; ELF suspects the net "lack[s] the right inductive bias" and memorises. Our inference
   for Blokus: long forced corridor sequences and multi-step reachability chains are the analogous
   shapes to watch; §3.5/§3.6 are the standard mitigations.
3. **Caution: LR floors and eval honesty.** ELF found strength variance did not shrink with LR, and
   training at 1e-5 went unstable — consistent with our own `lr_eta_min` lesson from
   `blokus_cloud_60`. And their observation that self-play Elo vs consecutive checkpoints inflates
   and misses regressions is exactly why our pooled BayesElo tournament exists. Nothing to do —
   already internalised.

Also for context: even ELF's final model gained ≈200 Elo per doubling of rollouts — search depth
remains a strength lever independent of training (relevant to how we spend eval budget, IDEAS I2;
the systematic version of this trade is [Jones 2021](https://arxiv.org/abs/2104.03113)).

### §3B — Architecture (usable now, but changes every future net)

### 3.5 Global pooling layers — verdict: **applies now**

**[Arch — benefits both phases]**

**What a plain convolutional network cannot do.** Every layer of our net
(`games/blokusduo/nn/net.py`) is a 3×3 convolution: each output cell is a function of a small
neighbourhood. Stack enough layers and the *receptive field* eventually covers the board, but
information still travels one ring of cells per layer, and board-wide **aggregates** — "how many
pieces does each side have left?", "who is ahead overall?", "is this the opening or the endgame?" —
must be laboriously assembled through many layers of local mixing, spending capacity that should be
reading shapes. The correct move preference in Blokus depends on exactly such globals: whether to
grab territory or block depends on inventory and tempo; endgame placements maximise raw cell count
while opening placements maximise reach.

**What global pooling is.** KataGo inserts a "global pooling bias structure" into two to three
residual blocks (and the policy/value heads): pool each channel over the whole board (its mean,
its mean scaled by board width, and its max), pass that compact global summary through a small
fully-connected layer, and **add the result back as a per-channel bias** to the spatial pathway.
Every cell thereafter sees local patterns *conditioned on global context*. Lc0's
**squeeze-and-excitation (SE)** blocks are the same family (global average-pool → bottleneck FC →
sigmoid gate that rescales channels), shipped in every Lc0 ResNet since T60
([network topology](https://lczero.org/dev/backend/nn/)); KataGo's paper itself notes the kinship
with SE (Hu et al. 2018).

**Evidence.** KataGo removal factor **1.60×** — second-largest in Table 2, and the paper notes the
gap widened as training progressed. Lc0 publishes no isolated SE Elo figure (verified — do not
quote one), but adopted it permanently. Two independent lineages converging on the same structure
is about as good as architecture evidence gets in this family.

**For us.** Our trunk has *no* global pathway (the only pooling anywhere is the pass-logit head's
AdaptiveAvgPool). And our encoding makes one global quantity — piece inventory — only *implicitly*
readable (an all-zero plane means the piece is unplayed): a global pool over the 42 piece planes
recovers "what's left in each box" in one step, something the current stack must infer by
convolution over emptiness (our inference; the KataGo/Go analogue global they cite is board-wide
status for ko fights). This is also, note, an *architecture-level sibling* of what the aux targets
(§3.3–3.4) do at the loss level — both push the trunk toward globally-informed features; they
compose.

**The real cost is not the layer, it's the triplication.** The net exists in three places: the
torch module, the jax inference bridge (`games/blokusduo/jax/net.py` re-implements the forward
pass numerically for self-play), and the ONNX web export. A trunk change lands in all three plus
their parity tests, and old checkpoints stop warm-starting across the change (fine for scratch SL
fits; a scheduling constraint for warm arms). ~3–4 days honest effort. **Verdict: applies now,
shortlist #4 — the highest-evidence architecture change available to us, sequenced behind the
pure-loader items only because of the blast radius.**

### §3C — Self-play only (blocked until the RL phase, itself gated on V15)

### 3.1 Playout cap randomization — verdict: **applies later (RL)**, and is *not* our sim taper

**[RL — self-play data generation]**

**What it is.** During self-play, KataGo does **not** give every move the same search. With
probability p = 0.25 a move gets a **full search** (cap N = 600 playouts, raised to 1000 later in
the run) and **only these moves are recorded as training examples**. The other 75% of moves get a
**fast search** (cap n = 100, later 200) that additionally *disables Dirichlet noise and the other
exploration settings* — its only job is to play a decent move cheaply and get on with the game.

**The mechanism — a cost asymmetry between the two training targets.** The policy head trains on
the search's visit distribution: quality scales with playouts *per move*. The value head trains on
the game outcome: **one noisy win/loss label per entire game**, so its data volume scales with
*games*, not playouts. A uniform 600-playout search buys excellent policy targets but very expensive
value labels; a uniform 100-playout search buys cheap games but junk policy targets. Randomizing the
cap gets both: ~4× more games per GPU-hour for the value head, while a quarter of moves still carry
full-quality policy targets. KataGo also checked the obvious alternative — fixed caps at 100 / 150
/ 200 / 250 / 600 — and the randomized scheme "clearly outperforms a wide variety of possible fixed
values" (paper Figure 5). Removal factor: **1.37×**.

**How it relates to our `sim_schedule: "branching"` taper — it doesn't, really.** The taper
(`MCTSConfig.sim_schedule` in `src/alphablokus/config.py`, implemented in
`src/alphablokus/search/mcts.py::_move_sim_budget`) scales the per-move budget with the root's
legal-move count: fewer sims in the endgame where the tree is tiny. Three differences, all
structural:

| | Our taper | Playout cap randomization |
|---|---|---|
| Budget varies by | board position (branching factor) — deterministic | coin flip per move — random |
| Cheap-search moves | **still produce training targets** | **excluded from training** |
| Purpose | wall-clock saving | more *games* per value label, without polluting policy targets |

The taper is a compute optimisation; PCR is a *target-quality* scheme. They compose in principle,
but the load-bearing PCR idea — don't train the policy on cheap searches — is absent from our
pipeline: on the python path every move's visit distribution becomes a target regardless of budget.

**For us.** Directly relevant to Phase 3, and to exactly our weak point (the value head is
data-starved — [`corpus-quality-principles.md`](corpus-quality-principles.md) §1). Two honest
caveats. First, Blokus games are short (~30 plies vs ~200+ in Go), so the value-vs-policy cost
asymmetry is milder here — the gain should be assumed smaller than 1.37×. Second, the jax backend
searches all games in jit-compiled lockstep with a *static* simulation count
(`games/blokusduo/jax/search.py::SearchConfig.num_simulations` — explicitly flat by the fidelity
contract), so a per-move random budget doesn't drop in: the natural implementation is two compiled
search functions (e.g. Gumbel n = 16 and n = 64/128) with a per-step Bernoulli mask choosing which
result each game uses, recording only full-search plies. Real engineering, not a flag. **Verdict:
applies later; queue it for the Phase 3 plan with a measured A/B.**

### 3.9 Self-play opening and temperature practices — verdict: **applies later (RL)**; one live disagreement to note

**[RL — self-play data generation]**

What the projects actually do, from the primary sources:

- **AlphaGo Zero**: τ = 1 for the first 30 moves (sample from visit counts), then τ → 0; Dirichlet
  Dir(0.03), ε = 0.25; resignation auto-thresholded to < 5% false positives with 10% of games
  played out ([Silver et al. 2017](https://www.nature.com/articles/nature24270)).
- **ELF OpenGo**: τ = 1 for **all** moves of training games ([Tian et al.
  2019](https://arxiv.org/abs/1902.04522)).
- **Lc0** (live training-server settings): self-play starts from an **opening book**
  (`openings-pgn=books/run2_book.pgn.gz`; T60 also mixed 2% Chess960), temperature 0.9 held for 20
  plies then decayed over 60, **floored at 0.6 in the endgame — never 0**; and their test53
  experiment ("temperature = 0 in endgames") was "clearly weaker"
  ([training runs](https://training.lczero.org/training_runs); [2019 training
  post](https://lczero.org/blog/2019/06/whats-going-on-with-training/)). Weak root noise
  (ε = 0.1, α = 0.12) *on top of* book + temperature.
- **KataGo**: opening variety is *engineered*, not sampled — raw-policy opening moves, 2.5% in-game
  branches, 5% of games forked into deliberately unusual openings, komi randomization (see
  [`corpus-generation-literature.md`](corpus-generation-literature.md) §3).

**The disagreement worth recording:** AGZ cuts temperature to 0 after move 30; Lc0 measured that
cutting it to 0 late is harmful and never does; ELF never cuts it. There is no clean resolution in
the sources — game length and draw dynamics differ — but the weight of open-project practice is
"keep some late randomness in *training* games". Our python path is AGZ-style
(`temp_threshold: 12` then argmax — `selfplay/episode.py`); our Gumbel path plays the
Sequential-Halving winner with exploration supplied by the Gumbel noise itself (`jax/actors.py`),
which is the modern answer and sidesteps the τ question at the root.

**For us, the actionable item is openings**: when Phase 3 starts, seed a fraction of self-play
games from the v2 corpus DAG (allocation-weighted, already mirror-canonical, already labelled) and
the 44 Pentobi book lines — a strictly better version of Lc0's PGN book, and the mechanism KataGo
paid a forking scheme to get. This also composes with IDEAS I7's "seed from positions Pentobi
misjudges". Resignation machinery: **does not apply** — our games are ~30 plies and always played
out (`--noresign` is load-bearing for margins/ownership targets, same reason KataGo never resigns);
KataGo's alternative of *downweighting* hopeless-position rows (weight 0.1 + 0.9λ) is worth
remembering if RL games ever get lopsided. **Verdict: opening seeding applies later, cheap, high
fit; temperature schedule is a Phase 3 A/B, not a today decision.**

### 3.10 KataGo's post-paper toolbox — mostly **applies later** or **park**

**[Mixed — per item]** All from
[KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md); note
the evidence quality is one tier below the paper (few controlled ablations).

- **Policy surprise weighting** [RL] — resample training positions in proportion (half the weight)
  to the KL-divergence between the policy prior and the search's target: train hardest where the
  search *changed its mind*. KataGo: "one of the larger improvements in KataGo's training between
  its g170 run and earlier runs" (no Elo figure). For us this is IDEAS I7's "prior vs search"
  referee turned into a sampling weight — free signal on the RL path, and an SL analogue exists
  (weight by prior-vs-teacher-target KL after the first epoch). **Applies later (RL); consider a
  small SL arm only if V14 leaves headroom.**
- **Root policy softmax temperature** [RL] — soften the root prior (T ≈ 1.25 early → 1.1) during
  self-play to counteract the self-reinforcing sharpening of one move among near-equals. Our v2
  corpus already hit the same phenomenon from the teacher's side (Pentobi's hyper-concentrated
  visits, plan fact 6); the SL-side answer is the existing τ loader knob. **Applies later.**
- **Subtree value bias correction** [Search] — pattern-bucketed correction of node utilities;
  30–60 Elo in KataGo. Genuinely clever, genuinely complex, and Go-pattern-specific in its
  bucketing. **Park** — our eval-strength budget is better spent on sims (I2).
- **Uncertainty-weighted playouts + short-term value targets** [Search + Arch] — ~75 Elo combined
  (v1.9.0 vs v1.8.0). Needs new heads plus search plumbing. **Park.**
- **Optimistic policy head** [Search + Arch] — 40–90 Elo at inference; needs the error heads.
  **Park.**
- **Auxiliary soft policy target** [SL/RL] — a second policy head on the T=4-flattened target;
  "appears to greatly improve the speed of learning of the policy". Cheap; overlaps our existing
  load-time τ. **Maybe-later; below the aux targets above.**
- **Shaped Dirichlet noise** — KataGo's own doc says it "has *not* been validated as a measurable
  improvement… merely suggestive". **Does not apply** (and we run Gumbel anyway).
- **Nested bottleneck residual nets / fixed-variance init + one batch norm** [Arch] — efficiency
  and BN-mismatch fixes at 19×19 scale (b18c384 ≈ old b60c320 per evaluation). Our nets are small
  and our bottleneck is elsewhere; the one-BN idea is worth remembering only because our BN
  running-stats are one of the fiddlier things the torch→jax bridge must replicate. **Park.**

### 3.11 Reanalyze (MuZero) — verdict: **applies later, if RL is generation-limited**

**[RL — training-data curation]**

MuZero Reanalyze re-runs search over *old buffered games* with the latest network and uses the fresh
results as training targets — 80% of its updates come from re-searched old positions, with sample
reuse raised 20× ([Schrittwieser et al. 2020](https://arxiv.org/abs/1911.08265), Appendix H;
headline: 731% median normalised Atari score vs 192–431% for model-free baselines; EfficientZero's
2-hour-Atari result is the same regime pushed further
([Ye et al. 2021](https://arxiv.org/abs/2111.00210))). No learned model needed for the AlphaZero
version: it is "spend GPU on refreshing targets instead of only on new games". For a single-GPU,
game-limited project this could be the right trade — but it competes head-on with §3.1 (which
spends the same GPU on *more games*), and which wins depends on whether value data (→ PCR) or
policy staleness (→ reanalyze) binds. **Verdict: applies later; measure, don't guess. Park behind
PCR.**

### §3D — Ruled out

### 3.2 Forced playouts and policy target pruning — verdict: **does not apply on our production path** (Gumbel already solves the same problem more cleanly)

**[RL — self-play data generation]**

This needs the clearest explanation in the document, so here it is from first principles.

**The problem: your training target is contaminated by your own exploration.** In AlphaZero-style
self-play the policy head's training target is the root visit distribution — "the search visited
move A 300 times and move B 20 times, so teach the policy 0.9 A / 0.06 B". But to avoid blind
spots, self-play deliberately injects **Dirichlet noise** into the root priors — random extra prior
mass on arbitrary moves, so that occasionally the search is forced to look at something the policy
rated poorly. Here is the trap: when noise inflates a bad move's prior, PUCT dutifully spends
visits on it. Those visits then sit in the visit distribution. And the visit distribution **is the
training target**. So the policy is literally trained to reproduce the noise: "the vast majority of
the time, noise moves are bad moves, and … we would train the policy to predict these extra bad
playouts" (Wu 2019, §3.2). Exploration and target-quality are welded together, and each round of
training bakes a little of the exploration randomness into the policy itself.

**KataGo's fix, part 1 — force the exploration properly (forced playouts).** If you're going to
explore, explore enough to *learn something*. At the root of each full search, every child that has
at least one visit is guaranteed a minimum number of playouts:

```
n_forced(child) = sqrt(k · P(child) · total_root_playouts)      with k = 2
```

so a noise-boosted move gets a real look (a handful of visits that actually read the position)
rather than one token visit that reveals nothing.

**Part 2 — then delete the evidence (policy target pruning).** Before the visit distribution is
written out as a training target, KataGo identifies the most-visited child and, from every *other*
child, **subtracts up to `n_forced` visits — as long as doing so does not make that child's PUCT
score exceed the best child's**. Any child reduced to a single playout is pruned to zero outright.

**Why subtract the very visits you forced?** Because the two purposes of a visit are different:

- Visits spent *finding out* whether a move is good are **exploration** — a cost you pay during
  search. They tell you nothing about the answer, only that you asked the question.
- Visits a move *keeps attracting on its own merits* — because its Q-value held up after the forced
  look — are **evaluation**. They are the answer.

The training target should contain only the second kind. The subtraction rule's PUCT-consistency
test is precisely the filter for "did the search come to like this move on its own?": if the move's
value genuinely held up, its unforced visit count is defensible under plain PUCT and survives the
subtraction; if it only had visits because we forced them, they are removed and the target says
what the search *concluded*, not where it was *told to look*. A useful mental model: forced
playouts are scaffolding you erect to inspect a wall; target pruning is taking the scaffolding down
before you photograph the building.

**What goes wrong if you force but don't prune:** you make the contamination *worse* than vanilla
AlphaZero — you've guaranteed every noise move a visit floor and then taught the policy that floor
as if it were preference. Each generation's policy inherits a smear of probability over arbitrary
weak moves, which the next generation's noise then builds on. Pruning is not an optional cleanup;
it is what makes the forcing safe. Removal factor for the pair: **1.25×** (and the paper notes the
benefit grew in later training, so likely understated).

**Why this does not apply to our production path.** Our production self-play uses **Gumbel search**
(`search_policy: "gumbel"`, mctx `gumbel_muzero_policy` — `games/blokusduo/jax/search.py`), which
dissolves the whole problem at the root:

- Exploration comes from **Gumbel noise inside the action-sampling procedure**, not from Dirichlet
  mass injected into the priors — and our code correctly applies *no* Dirichlet in Gumbel mode
  (raw priors at the root; `dirichlet_fraction=0.0`).
- The training target is **not the visit distribution at all**: it is the *completed-Q improved
  policy* `π′ = softmax(logits + σ(completed Q))` ([Danihelka et al., ICLR
  2022](https://openreview.net/forum?id=bERaNdoegnO)) — built from what the search *concluded*
  (Q-values), with the value net standing in for unvisited actions. Exploration visits never enter
  the target, so there is nothing to prune.

So KataGo and Gumbel are two solutions to the same disease, and we already run the newer one.
**Verdict: does not apply to jax/Gumbel self-play. It *would* apply if we ever ran RL self-play on
the python PUCT path with Dirichlet noise on** — that path trains on raw noised visit counts,
exactly the contaminated target described above — so the practical rule is: python-path RL at
ε > 0 should not be run for production data without either target pruning or a switch to Gumbel.
(Arena/Elo/Pentobi evaluation uses the python path with noise off and produces no targets — no
issue there.)

One config-hygiene footnote found while verifying this: `run_configurations/blokus_cloud_v2.json`
sets `dirichlet_epsilon: 0.25` alongside `search_policy: "gumbel"`. The code ignores it (inert), but
it reads as if noise were on; worth removing from Gumbel configs to keep intent legible.

### 3.13 Transformer / attention bodies (Lc0 BT2–BT4) — verdict: **does not apply (by project rule), recorded for honesty**

**[Arch]** Lc0 has moved past ResNets: encoder-only transformers over 64 square-tokens with
**smolgen** (a learned, position-dependent attention-bias generator compensating for chess's
non-Euclidean adjacency) now define their frontier — BT4's *raw policy* is "270 Elo stronger than
that of T78, our strongest convolution-based model, with fewer parameters and computations"
([Transformer Progress](https://lczero.org/blog/2024/02/transformer-progress/); [Monroe & Chalmers
2024](https://arxiv.org/abs/2409.12272)). This is real and large — and out of scope here by an
explicit standing decision (CLAUDE.md: no architecture switch until the ResNet path is exhausted),
which remains correct at our compute scale: transformer gains materialised for Lc0 at hundreds of
millions of parameters and years of distributed training. Recorded so the decision is made with the
number in view, not in ignorance of it.

---

## 4. What we already do well

Places where current practice matches or exceeds the literature — stated plainly, because half the
value of a review is knowing what *not* to churn:

1. **Gumbel self-play is the state of the art for low-simulation search, and we run it properly.**
   Production self-play uses mctx `gumbel_muzero_policy` (n = 64–128); the code plays the
   Sequential-Halving winner, applies **no** Dirichlet noise at the Gumbel root, and trains on the
   **completed-Q improved policy** rather than raw visit counts
   (`games/blokusduo/jax/search.py`, `actors.py`) — all three of which are exactly what
   [Danihelka et al. 2022](https://openreview.net/forum?id=bERaNdoegnO) prescribe, and the third of
   which quietly gives us the benefit KataGo's target pruning buys (§3.2). Validated by our own A/B
   at strength parity (`docs/research/jax-pipeline-ab.md`).
2. **Soft full-distribution policy targets under a KL loss.** The v2 corpus stores the teacher's
   full move distribution and `BaseNNetWrapper.loss_pi` is KL against arbitrary distributions —
   Hinton-style distillation done right, aligned with ChessBench's finding that target richness
   beats one-hot at every scale, and precisely the fix for v1's one-hot failure.
3. **Evaluation hygiene is ahead of common practice.** Paired colour-swapped arenas from shared
   openings (chess-world standard), a regression-guard gate option, pooled BayesElo instead of
   saturating self-anchored Elo (ELF's own criticism of naive self-play Elo, §3.12), subtree-level
   holdout splits, per-colour value calibration and the colour-only value-skill floor. Most
   hobby-scale projects have none of these.
4. **The canonical hyperparameters are where the papers put them.** AdamW weight decay 1e-4 (AGZ's
   L2 c = 1e-4), Dirichlet α = 0.03 scaled for a ~400-move action space (AZ's per-game α logic),
   virtual loss (AGZ used n_vl = 3, as do we), games-sized replay window (how AZ/MuZero/KataGo all
   describe theirs), symmetry augmentation appropriate to the game's actual symmetry group (AZ
   dropped augmentation *because chess is asymmetric*; Blokus Duo's order-2 group makes ours
   correct, not lazy).
5. **The gate decision is reasoned, not cargo-culted.** AZ dropped AGZ's gating; ELF measured async
   AZ-style training beating gated AGZ-style 100:0 *at their scale*. We keep the gate with an
   explicit written argument for why the small-noisy regime differs (IDEAS I4) plus `gate_mode`
   options — a deliberate, documented deviation, which is what a deviation should be.
6. **The corpus design independently reinvented the field's opening-diversity playbook** — and in
   persisted, labelled form (the DAG) rather than KataGo's disposable forks
   ([`corpus-generation-literature.md`](corpus-generation-literature.md) §3). The score head
   experiment (`docs/plans/score-auxiliary-target.md`) follows the KataGo auxiliary-target playbook
   with a cleaner A/B design (RNG-stream-preserving head construction, prefix-scoped checkpoint
   tolerance, masked loss) than anything the sources document.

## 5. What we are missing that matters most — ranked

1. **Value-target enrichment.** Everything about our current phase says the value pathway is the
   binding constraint: the 0.304-MSE colour-only floor, 71/7/22 outcomes, ~700 Black wins, and the
   v2 plan's own thesis that surpassing the teacher routes through the value head. The literature's
   three answers — teacher-eval blending (§3.8), denser outcome-adjacent targets
   (score/ownership/WDL, §3.4/§3.7), and balanced sampling (§3.12) — are all cheap-to-moderate and
   all currently absent from game-row training. KataGo's 1.65× for the aux-target group is the
   single largest ablated factor in the canon.
2. **Auxiliary supervision generally.** We train two heads on two targets; KataGo trains ~six on
   the same trunk and credits the practice heavily. The opponent-reply target (§3.3) is the
   highest-evidence, lowest-cost member for our SL phase.
3. **Global context in the trunk (§3.5).** 1.60× in KataGo, universal in Lc0, absent here.
4. **RL-phase data-generation economics.** PCR (1.37×), surprise weighting, opening seeding,
   reanalyze — none evaluable until Phase 3, all should be in that phase's plan from day one rather
   than retrofitted. The one *hard* constraint to design around: our jit-compiled lockstep search
   makes per-move budget randomization structural work, not a config flag (§3.1).
5. **Input features (§3.6).** Real but bundled evidence (1.55×), high blast radius; do once,
   with §3.5.

## 6. Open questions and where the sources disagree

- **Do KataGo's RL ablation factors transfer to SL distillation at 14×14?** Unknown. The
  aux-target and pooling results have supervised-era provenance (Darkforest; Wu's pre-KataGo
  supervised experiments; SE from image classification), which is encouraging, but no source
  measures them in our regime. Our A/Bs are the only answer.
- **Ownership vs score:** genuinely unresolved in the literature (§3.4) — KataGo never splits them.
  Our claim that ownership is the stronger half *for Blokus* is a testable inference, not a
  citation.
- **Late-game temperature:** AGZ says cut to zero after move 30; Lc0 measured "clearly weaker" and
  keeps τ ≥ 0.6 forever; ELF keeps τ = 1 throughout. Unresolved; matters only in Phase 3, and
  Gumbel changes the terms of the question.
- **Teacher-eval blending vs outcome-only correction (§3.8):** NNUE's λ ≈ 0.7 says lean on the
  teacher; our v2 design says outcomes are the correction channel and the teacher is measurably
  wrong at specific nodes (plan fact 10). Both have evidence. The λ arm exists to settle it for us.
- **Warm-starting RL from SL:** the AGZ paper contains **no controlled warm-start ablation** —
  the SL-vs-RL comparison is two independent runs, and AlphaGo Master (human-bootstrapped, finished
  327 Elo below the zero run) also differed in input features. The common claim that DeepMind
  "showed warm-starting hurts" over-reads the source. At our compute scale, SL-then-RL has the
  stronger case (AlphaGo 2016 itself; Expert Iteration), and is our plan of record.
- **Does SL agreement predict ladder movement at all?** Still the deepest open question from
  [`corpus-quality-principles.md`](corpus-quality-principles.md) (ELF's 46%-human-agreement
  plateau); none of the newly-read sources resolve it.
- **One unverifiable negative:** the "value function dominance" finding sometimes attributed to ELF
  OpenGo does not appear in the paper (checked against the full text). If we ever cited it, stop.

## Sources

**KataGo**: [Wu 2019/2020, *Accelerating Self-Play Learning in Go*, arXiv:1902.10565](https://arxiv.org/abs/1902.10565)
(ablations: Table 2 §5.2; v1 vs v5 claims from the arXiv version history) ·
[KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)

**Leela Chess Zero**: [WDL head](https://lczero.org/blog/2020/04/wdl-head/) ·
[WDL rescale/contempt](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/) ·
[v0.25 / moves-left head](https://lczero.org/blog/2020/05/lc0-v0.25-has-been-released/) ·
[2019 training status (test5x: WDL, FPU, endgame-τ, books)](https://lczero.org/blog/2019/06/whats-going-on-with-training/) ·
[End of era (T60)](https://lczero.org/blog/2019/07/end-of-era/) ·
[Transformer Progress (BT4, smolgen)](https://lczero.org/blog/2024/02/transformer-progress/) ·
[Monroe & Chalmers 2024, arXiv:2409.12272](https://arxiv.org/abs/2409.12272) ·
[Network topology (SE)](https://lczero.org/dev/backend/nn/) ·
[Training-data formats (policy target, plies_left, orig_q)](https://lczero.org/dev/wiki/training-data-format-versions/) ·
[Live training-run parameters](https://training.lczero.org/training_runs) ·
[Project history](https://lczero.org/dev/wiki/project-history/) ·
[Value repair (jhorthos wiki)](https://github.com/jhorthos/lczero-training/wiki/Value-Repair-method)

**Stockfish NNUE**: [nnue.md (λ interpolation)](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md) ·
[Training datasets wiki](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets) ·
[PR #4782 (start-λ 1.0 / end-λ 0.7)](https://github.com/official-stockfish/Stockfish/pull/4782)

**DeepMind & family**: [Silver et al. 2017, AlphaGo Zero, Nature](https://www.nature.com/articles/nature24270)
([open-access text](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)) ·
[Silver et al. 2018, AlphaZero, arXiv:1712.01815](https://arxiv.org/abs/1712.01815) ·
[Silver et al. 2016, AlphaGo, Nature](https://www.nature.com/articles/nature16961) ·
[Danihelka et al. 2022, *Policy improvement by planning with Gumbel*, ICLR](https://openreview.net/forum?id=bERaNdoegnO) ·
[Schrittwieser et al. 2020, MuZero, arXiv:1911.08265](https://arxiv.org/abs/1911.08265) ·
[Ye et al. 2021, EfficientZero, arXiv:2111.00210](https://arxiv.org/abs/2111.00210) ·
[Tian et al. 2019, ELF OpenGo, arXiv:1902.04522](https://arxiv.org/abs/1902.04522)

**Distillation & scale**: [Hinton et al. 2015, arXiv:1503.02531](https://arxiv.org/abs/1503.02531) ·
[McIlroy-Young et al. 2020, Maia, arXiv:2006.01855](https://arxiv.org/abs/2006.01855) ·
[Ruoss et al. 2024, ChessBench, arXiv:2402.04494](https://arxiv.org/abs/2402.04494) ·
[Norelli & Panconesi, OLIVAW, arXiv:2103.17228](https://arxiv.org/abs/2103.17228) ·
[Jones 2021, *Scaling Scaling Laws with Board Games*, arXiv:2104.03113](https://arxiv.org/abs/2104.03113)

**Internal**: [`corpus-quality-principles.md`](corpus-quality-principles.md) ·
[`corpus-generation-literature.md`](corpus-generation-literature.md) ·
[`jax-pipeline-ab.md`](jax-pipeline-ab.md) · [`../plans/pentobi-corpus-v2.md`](../plans/pentobi-corpus-v2.md) ·
[`../plans/score-auxiliary-target.md`](../plans/score-auxiliary-target.md) · [`../IDEAS.md`](../IDEAS.md)
