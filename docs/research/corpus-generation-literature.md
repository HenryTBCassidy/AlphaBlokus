# Engine-generated training corpora — what the literature says (2026-07-28)

Research supporting the v2 Pentobi-distillation corpus design
([`../plans/pentobi-corpus-v2.md`](../plans/pentobi-corpus-v2.md)) and the search-space store
([`../plans/corpus-search-space-store.md`](../plans/corpus-search-space-store.md)). The question —
how to build a supervised corpus from a strong engine's play, and how to trade opening breadth
against per-opening replication — is decades old; this note collects what the strongest prior
projects actually did, with citations, and flags where our situation differs from each.

---

## 1. AlphaGo (2016): all positions for the policy net, one-per-game for the value net

The SL policy network was trained on **29.4 M positions from 160,000 KGS games** (6–9 dan humans),
using *every* position of every game, and reached 57.0% top-1 move prediction — the base of the
whole system ([Silver et al., *Nature* 2016](https://www.nature.com/articles/nature16961);
[summary](https://blog.acolyer.org/2016/09/20/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/)).

The value network is the sharp lesson. Trained on *complete KGS games* it **memorised outcomes**
— train MSE 0.19 vs test MSE 0.37 — because all ~200 positions of a game share one outcome label
and are strongly correlated. The fix was a new dataset of **30 M distinct positions, each sampled
from a separate self-play game** (train 0.226 / test 0.234). Two rules fall out, and they are
*different* rules for the two heads:

- **Policy targets are per-position** — within-game correlation is harmless, so harvest every ply.
- **Value targets are per-game** — a game is one Bernoulli sample smeared over ~30 rows.

**Where we differ:** we cannot afford one-position-per-game (an L9 game costs ~250 s of search; the
harvest is the point), and we don't need to copy the fix at generation time — decorrelation is a
*training-time* concern. Our existing game-granular holdout split already prevents the
leak-driven part of the overfit; per-game value-loss weighting (or per-game position subsampling
for the value head only) is the training-side lever if calibration diagnostics show outcome
memorisation. The corpus should store everything.

AlphaGo is also the precedent for the whole programme: the RL policy network, initialised from the
SL network, beat the SL network in 80% of head-to-head games — imitate-then-surpass works.
[Expert Iteration (Anthony, Tian & Barber 2017)](https://arxiv.org/abs/1705.08439) formalises the
same imitate/improve loop we plan as Phase 2/3.

## 2. AlphaZero / AlphaGo Zero: diversity from noise, not books

AlphaGo Zero and AlphaZero generate opening diversity with exactly two mechanisms: **Dirichlet
noise at the root** (α = 0.03 for Go) and **temperature 1 sampling for the first 30 moves** of
self-play, τ→0 afterwards ([Silver et al. 2017](https://www.nature.com/articles/nature24270);
[Silver et al. 2018](https://www.science.org/doi/10.1126/science.aar6404)). No opening books, no
position deduplication. That works because the *learner* is stochastic and improving — millions of
games under noise cover the opening space.

**Where we differ:** our expert is a fixed, near-deterministic engine. D3 measured the failure
directly: 24 seed-varied L9 games produced only **3 distinct first moves**. Noise-based diversity
is not available to us without weakening the expert (temperature over Pentobi's move choice is
exactly "imitate worse play"), so diversity has to be *engineered* into the starts — which is what
the opening DAG is.

## 3. KataGo: engineered opening diversity is a design feature, not a hack

KataGo is the strongest precedent for deliberately manufacturing off-distribution starts and then
playing them out on-policy ([Wu 2019, arXiv:1902.10565](https://arxiv.org/abs/1902.10565)):

- The first *r* moves of a game are played **directly from the raw policy** (r ~ Exp, mean
  0.04·b²) — cheap, varied starts.
- Temperature on move selection decays 0.8 → 0.2 over the game.
- **In 2.5% of positions the game is branched** to try an alternative move drawn from the policy
  at temperature 1/2/∞.
- **In 5% of games the game is branched early**, then "between 3 and 10 moves are chosen uniformly
  at random, each given a single neural net evaluation, and the best one is played" — explicitly
  to guarantee "a small percentage of games with highly unusual openings".
- Komi randomisation and 5% handicap games broaden the value head's experience.

The pattern — *impose* a start the engine would not have chosen, then continue at full strength
and label everything — is precisely our leaf-playout scheme. KataGo validates both halves: forced
starts don't poison the data (the continuations are on-policy), and breadth of starts is worth
engineering even at the cost of some off-distribution prefixes.

**Where we differ:** KataGo's branching is *unlabelled* — a branch is just a new game. Because our
expert is external and expensive, we get more from making every branch a *node in a persistent,
labelled DAG* (searched once, reused by every game through it) than from KataGo's in-game
disposable branching. Their 2.5%-branch trick is, in our design, simply "widen the candidate pool"
— same idea, but persisted.

## 4. Leela Chess Zero: temperature openings, book experiments, and endgame noise that helped

Lc0's training games rely on early-ply temperature for opening variety; the project also
considered "training from some kind of opening book (external or self-generated)" and ran a test
(test49.9) on "how fixed opening book affects network strength"
([lczero.org blog, 2019](https://lczero.org/blog/2019/06/whats-going-on-with-training/)). Two
details are useful to us:

- **Removing endgame randomness made the net weaker** (test53, "temperature = 0 in endgames" was
  "clearly weaker") — diversity deep in the game carries real training signal, not just opening
  variety. Our per-game engine seeds (D3: 24/24 unique games, 91.7% distinct positions from one
  start) are doing this job.
- Match/evaluation play in the chess world universally uses **paired games on fixed openings** to
  cancel opening and colour variance — the same logic as our `paired_arena`, and the reason the
  book-strength measurement (V12) must be colour-balanced.

## 5. Stockfish NNUE: off-policy states with strong labels is the state of the art

Modern Stockfish NNUE data practice is the closest industrial analogue to our design, and its
history moved *toward* our pattern:

- Classic `gensfen` self-play data used **random opening plies and `random_multi_pv`** to
  diversify starts, then depth-limited search labels
  ([nnue-pytorch wiki](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets);
  [robotmoon.com](https://robotmoon.com/nnue-training-data/)).
- Since ~2022 the best Stockfish nets train largely on **converted Leela training data** — states
  generated by a *different* engine's games, labelled/filtered by Stockfish search — plus
  "Stockfish self-play data **from openings it usually gets wrong**" (targeted weakness data — our
  Phase-2 net-in-the-loop proposal, already in production use elsewhere). Datasets produced from
  Lc0 games "generally end up producing better networks" than Stockfish-only self-play data.
- The `wld-fen-skipping` filter drops positions "where the evaluation doesn't correlate with the
  end game result". Note carefully: **they discard eval/outcome disagreement; we keep it.** Their
  objective is a low-noise eval regressor; ours is to *correct the teacher*, and the
  outcome-vs-teacher-value disagreement is the correction signal. The mechanism transfers; the
  filter direction does not.
- Lc0's **rescorer** propagates tablebase/outcome information back into position labels — evidence
  that blending "teacher evaluation" with "actual outcome" is standard practice, supporting our
  shrinkage blend for opening-node value targets.

## 6. ChessBench (DeepMind 2024): full per-move engine annotations, and a distillation ceiling

"Grandmaster-Level Chess Without Search" / *Amortized Planning with Large-Scale Transformers*
([Ruoss et al., NeurIPS 2024](https://arxiv.org/abs/2402.04494);
[repo](https://github.com/google-deepmind/searchless_chess)) built **ChessBench: 10 M human
(Lichess) games with Stockfish-16 action-value annotations for every legal move — 15 G data
points** — and trained a 270 M-parameter transformer by pure SL to grandmaster-level blitz play.
This is our design at industrial scale: off-policy states (human games), a single strong engine
labelling **full per-move value distributions** (not one-hot played moves), every position
harvested. Directly validates: soft full-distribution targets, harvesting the engine's evaluation
of moves it did *not* play, and states the expert would not itself have reached.

Their headline caveat is our planning assumption: "a remarkably good approximation of Stockfish's
search-based algorithm can be distilled … via supervised learning, but **perfect distillation is
still beyond reach**." SL alone will not match L9, let alone beat it — consistent with ELF
OpenGo's observation that even a superhuman policy agrees with strong humans only ~46% of the time
([Tian et al. 2019](https://arxiv.org/abs/1902.04522)). The corpus's job is to move the ladder and
hand RL a broad, honestly-valued base — the surpassing is Phase 3's job, per AlphaGo's SL→RL
precedent.

## 7. Diversity vs quantity: the imitation-learning evidence points one way

- **Data Scaling Laws in Imitation Learning for Robotic Manipulation**
  ([Lin et al. 2024, arXiv:2410.18647](https://arxiv.org/abs/2410.18647)): generalisation follows
  a power law in the **number of distinct environments and objects, not the number of
  demonstrations per environment** — "once the number of demonstrations per environment or object
  reaches a certain threshold, additional demonstrations have minimal effect." Mapping
  environments→openings and demonstrations→replicas, this is the strongest direct evidence on our
  breadth-vs-replication question: breadth wins, with a small per-opening threshold (theirs was
  modest, not 1) below which replication still pays.
- **robomimic** ([Mandlekar et al. 2021, arXiv:2108.03298](https://arxiv.org/abs/2108.03298))
  systematically varied demonstrator quality and dataset size: quality and *coverage* of
  demonstrations dominate raw quantity for BC performance.
- **Deduplicating Training Data Makes Language Models Better**
  ([Lee et al. 2021, arXiv:2107.06499](https://arxiv.org/abs/2107.06499)): removing duplicates cut
  verbatim memorisation ~10×, reduced train-test overlap, and matched or improved accuracy with
  fewer steps. For us this supports position-keyed deduplication (the DAG collapses transposition
  and repeated-opening duplicates by construction) and the game-granular holdout.
- **DAgger** ([Ross, Gordon & Bagnell 2011, arXiv:1011.0686](https://arxiv.org/abs/1011.0686)):
  behaviour cloning compounds errors quadratically in horizon because the learner visits states
  the expert never labelled; querying the expert *on off-expert states* restores a linear bound.
  Our corpus is DAgger-shaped by construction — expert labels on a deliberately wider state
  distribution than the expert's own play (the wide candidate pools are states Pentobi searched
  and rejected), and Phase 2's net-proposed openings close the loop on the learner's own states.
  This is the principled argument for **not** restricting the corpus to Pentobi's argmax lines.

**Where we differ from the BC literature:** their diversity axis is usually *task/environment*
diversity with i.i.d.-ish episodes; ours is a tiny, heavily-overlapping opening space where a few
hundred "environments" share one board and most mid-game structure. The power-law-in-environments
result should therefore be read as a strong prior, not a law — hence the ablation rather than a
decision by citation.

## 8. Visit distributions are not move-quality distributions

Added 2026-07-28, after the ply-2 measurement (v2 plan fact 10: 92.9% of L9's root visits on a
reply its own one-ply-deeper search ranks ~35th of 315). The finding has a literature home — the
field already treats raw visit counts as a flawed policy target:

- **[Grill et al. 2020, *Monte-Carlo Tree Search as Regularized Policy Optimization*](https://arxiv.org/abs/2007.12509)**
  (ICML) prove AlphaZero's visit-count distribution is only a coarse approximation to the
  regularized policy-optimization solution the search is implicitly computing, and that acting/
  training on the underlying Q-value-derived target instead "reliably outperforms the original
  algorithm", most strongly at low simulation counts. The visit distribution records *where the
  search spent effort* (exploration dynamics included), not a calibrated posterior over move
  quality — which is exactly the structure of our ply-2 measurement: the good reply is *in* the
  candidate set (visit rank 2) but the mass on it is wrong.
- **KataGo's policy target pruning** ([Wu 2019](https://arxiv.org/abs/1902.10565)) edits the raw
  visit counts before they become a training target — subtracting playouts attributable to forced
  exploration so the target reflects the search's genuine preference. Production precedent that
  the visit vector is a *starting point* for a policy target, not gospel.
- **Lc0/NNUE practice** (§5) manages the same tension from the value side: the rescorer overwrites
  engine evaluations with propagated outcome/tablebase truth, and `wld-fen-skipping` filters
  positions where evaluation and outcome disagree. Both accept that a strong engine's in-search
  opinions are systematically wrong somewhere and correct them with *outcome-grounded* data —
  the same correction channel our design uses (allocation spreads games across the misjudged
  alternatives; outcome labels tell the value head which ones actually win).

The design conclusion drawn in the v2 plan: imitation targets stay Pentobi's distributions (locked
decision), the correction flows through allocation breadth + outcome labels + play-time search
rather than through target surgery, and value-informed target reshaping — the Grill-flavoured
intervention — is a *gated* follow-up, justified only if the measured base rate of
confidently-wrong nodes is high and the ladder gate fails on opening play.

---

## What this changes about our defaults

1. **Breadth-first, small replication floor.** The scaling-law and robomimic evidence, plus
   AlphaGo's value-decorrelation lesson, favour many distinct openings over many replicas — but
   with a threshold ≥ 1, not exactly 1. Default: coverage-driven replicas (every leaf reaches
   replica *r* before any leaf starts *r*+1), target R = 2 at stage 1, ablate (V11) before
   committing the stage-1 split between DAG growth and replication.
2. **Value-target blending is orthodox.** NNUE/Lc0 practice mixes engine evaluation with outcome
   information; a count-shrunk blend of `outcome_mean` toward `search_value` is the literature-
   consistent default for opening nodes, while per-game decorrelation of the value loss is the
   known fix if calibration shows memorisation.
3. **Keep the disagreement, don't train on it directly.** Stockfish *filters out*
   eval-outcome-disagreeing rows for their objective; no precedent trains an auxiliary
   disagreement head. Ours flows through honest outcome labels + wide coverage; disagreement stays
   as queryable data for Phase-2 seeding.
4. **Treat visit distributions as effort maps, not quality posteriors** (§8). Imitate them —
   that is the locked target — but do not expect training temperature to repair their measured
   mass-misallocation (it is order-preserving), and route the correction through allocation
   breadth and outcome-grounded value labels, where Grill/KataGo/NNUE precedent sits.

## Sources

- [Silver et al. 2016, *Mastering the game of Go with deep neural networks and tree search*](https://www.nature.com/articles/nature16961) ([summary](https://blog.acolyer.org/2016/09/20/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/))
- [Silver et al. 2017, *Mastering the game of Go without human knowledge*](https://www.nature.com/articles/nature24270)
- [Silver et al. 2018, *A general reinforcement learning algorithm…* (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)
- [Anthony, Tian & Barber 2017, *Thinking Fast and Slow with Deep Learning and Tree Search* (ExIt)](https://arxiv.org/abs/1705.08439)
- [Wu 2019, *Accelerating Self-Play Learning in Go* (KataGo)](https://arxiv.org/abs/1902.10565)
- [lczero.org 2019, *What's going on with training*](https://lczero.org/blog/2019/06/whats-going-on-with-training/)
- [official-stockfish/nnue-pytorch wiki, *Training datasets*](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets)
- [robotmoon.com, *Stockfish NNUE training data*](https://robotmoon.com/nnue-training-data/)
- [Ruoss et al. 2024, *Amortized Planning with Large-Scale Transformers* (ChessBench)](https://arxiv.org/abs/2402.04494) ([repo](https://github.com/google-deepmind/searchless_chess))
- [Tian et al. 2019, *ELF OpenGo*](https://arxiv.org/abs/1902.04522)
- [Lin et al. 2024, *Data Scaling Laws in Imitation Learning for Robotic Manipulation*](https://arxiv.org/abs/2410.18647)
- [Mandlekar et al. 2021, *What Matters in Learning from Offline Human Demonstrations* (robomimic)](https://arxiv.org/abs/2108.03298)
- [Lee et al. 2021, *Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499)
- [Ross, Gordon & Bagnell 2011, *A Reduction of Imitation Learning…* (DAgger)](https://arxiv.org/abs/1011.0686)
- [Grill et al. 2020, *Monte-Carlo Tree Search as Regularized Policy Optimization*](https://arxiv.org/abs/2007.12509)
