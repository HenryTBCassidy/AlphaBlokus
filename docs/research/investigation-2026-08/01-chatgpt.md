# Independent investigation: AlphaBlokus plateau

## Executive verdict

The plateau is real, but the team has moved too quickly from “the current operator has a fixed point” to “search width is probably the cause.” My ranking is different:

1. The value/target pipeline is the leading suspect, including role imbalance, sign/perspective errors, poor calibration, or replay sampling that lets a colour prior masquerade as value learning.
2. Self-confirming exploration failure is next: hard top-k, absent root-wide exploration, possibly correlated RNG, and deterministic single-lineage self-play.
3. The late-stage optimization regime is demonstrably wrong: constant Adam at `1e-3`, no weight decay, and six-to-twelve lifetime passes per position are capable of turning lower training loss into a weaker player.
4. Missing global information, game-specific features, and auxiliary targets probably lower sample efficiency, but have not yet been causally tested in this project.
5. Capacity is an open question, not a leading one.

The distillation result rules out replacing v3 with the current purely supervised recipe. It does not rule out policy-only regularization, on-policy teacher queries, opening-state seeding, or a small persistent teacher-data mixture during RL.

The auxiliary-head experiment was not a valid test of KataGo’s claim. It tested imitation at one small scale with insufficient replication. KataGo’s claimed gains are online self-play gains produced by changing representation learning and the subsequent data distribution.

My direct compute answer is: **Pentobi level 9 is plausibly reachable within a few hundred pounds, but not by continuing the present loop and not on a credible two-week schedule.** I would assign roughly a 50–65% chance under a staged £300 cap, conditional on the first diagnostics finding and correcting a real signal problem. A reasonable planning range is 2–4 million new games, approximately 640–1,280 RTX 3060 Ti self-play hours at the measured top-k 128 throughput, or four to eight weeks continuously. If no external gain appears after the first corrected 200,000 games, that forecast should be withdrawn.

---

## 1. Ranked hypotheses for what is actually wrong

### 1. Value learning or value-target handling is defective

**Evidence for**

- On the Pentobi corpus, the trained value head is 15–28% worse than a colour-only predictor.
- The corpus is 71% White wins, 22% draws, and only 7% Black wins; RL self-play is similarly skewed at 73% White wins.
- Every position in a game carries the same result label, so the nominal position count greatly exaggerates the number of independent value observations.
- Gumbel’s completed-Q target substitutes the value estimate for unvisited actions. A weak value head therefore affects not only search evaluation but the policy target that trains the next policy.
- ELF OpenGo observed the same type of feedback loop: overestimating White caused premature Black resignations, reduced replay diversity, declining value loss, and overfitting. It corrected this by sampling Black-win and White-win games evenly. [ELF OpenGo, pp. 4–5](https://proceedings.mlr.press/v97/tian19a/tian19a.pdf)
- In the regression run, falling value loss accompanied worsening symmetry, consistency, entropy, Elo, and ladder strength. That is much more consistent with learning a shortcut than with useful value improvement.

**Evidence against**

- The RL value head has never been evaluated against a colour/phase baseline, so the negative supervised result may not transfer.
- v3 did improve substantially from its donor, so the pipeline cannot have been completely non-functional throughout training.

**What distinguishes it**

On a held-out set split by complete RL games, compare v3 with constant, colour-only, and colour×phase baselines using Brier/MSE, log loss, calibration, ranking skill, and spatial symmetry. Bootstrap by game, not position. Report every result separately for White-to-move, Black-to-move, draw/win/loss, and game phase.

Also verify Torch and JAX value equality and exact sign changes over a move. This should precede another paid run.

### 2. The self-play search cannot discover its own blind spots

**Evidence for**

- Top-k 64 is imposed at every node before search over roughly 17,837 actions.
- There is no Dirichlet noise in the Gumbel path. Gumbel randomness explores the candidate set, but cannot recover an action removed before it sees that set.
- Gumbel MuZero was motivated by the observation that AlphaZero may fail to improve when actions are unvisited, and samples actions without replacement to address that failure. A hard pre-search truncation weakens the premise behind that guarantee. [Danihelka et al., ICLR 2022](https://openreview.net/forum?id=bERaNdoegnO)
- KataGo documents precisely the self-confirming mechanism: a tactic is undervalued, its move prior decays, and the whole training window eventually reinforces the mistake. It added policy softening, shaped exploration, and surprise weighting to oppose it. [KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- More simulations inside the same candidate support produced no improvement.

**Evidence against**

- The current policy is concentrated: target top-1 mass is about 0.68 and entropy about 0.9 nats. Rank-65 moves may rarely matter.
- No measurement shows that a rank-65–128 action becomes best after additional search.
- The measured 128→256 simulation test changed both simulations and considered actions, but did not isolate root width, child width, value quality, or target scaling.
- Gumbel with few simulations is specifically designed not to enumerate a huge action space; blindly widening candidates can dilute the same simulation budget.

**What distinguishes it**

Run paired shadow searches from the same frozen v3 positions and randomness:

- control: top-k 64, considered 64, 128 simulations;
- intervention: top-k 128, considered 64, 128 simulations.

Measure separately:

- action changes at the root;
- completed-Q improvement on the common 64 actions;
- frequency with which ranks 65–128 enter consequential child searches;
- KL between improved targets;
- whether changed actions survive an independent high-budget/full-action Python search.

Width should be promoted from “plausible” to “leading” only if excluded actions or deeper branches regularly rescue value.

### 3. The update-to-data ratio and late-stage optimizer are wrong

**Evidence for**

- One epoch over a six-generation window gives each position roughly six lifetime passes. Two epochs give roughly twelve.
- The only two-epoch continuation degraded by 44 pool Elo and 0.046 ladder weighted score.
- That run combined twelve-pass reuse, constant `1e-3`, Adam, and no weight decay; symmetry and policy-value consistency deteriorated before strength did.
- A learning rate suitable while climbing from a weak donor is not necessarily suitable within eight Elo of the operator’s fixed point.
- KataGo used regularization, stochastic weight averaging, a growing recent-data window, and a final 10× learning-rate drop. Its paper used a fixed per-sample learning rate and a batch of 256—not AlphaGo Zero’s optimizer recipe. [Wu, §2 and Appendix C](https://rlg.mlanctot.info/2020/papers/AAAI20-RLG_paper_36.pdf)
- Lc0’s published histories similarly show repeated late-run learning-rate drops and warn that self-play Elo can move opposite stronger external estimates. [Lc0 training history](https://lczero.org/dev/wiki/project-history/), [Lc0 training discussion](https://lczero.org/blog/2018/10/lc0-training/)

**Evidence against**

- The one-epoch operator does not visibly degrade; it simply makes only about an eight-Elo step.
- Training more conservatively cannot help if the search target contains no better information.

**What distinguishes it**

From v3, use a fixed RL buffer and three weight seeds to compare AdamW:

- `1e-4`,
- `3e-4`,
- `1e-3`,

at 0.25, 0.5, and 1.0 buffer passes. Select by game-held-out policy loss, role-conditional value skill, symmetry, and external strength—not training loss.

### 4. Representation learning is too weak for this game

**Evidence for**

- The trunk has no mechanism for cheap global context, yet Blokus decisions depend on remaining-piece inventory, global accessibility, and spatial competition.
- Inventory is only implicit in all-zero piece planes; placeable corners and phase are absent.
- KataGo’s controlled online ablations reported training-time factors of 1.60× for global pooling, 1.65× for ownership+score targets, 1.30× for opponent-policy prediction, and 1.55× for game-specific features plus search optimizations. These are factors in compute to reach an Elo, not percentages or guaranteed Blokus gains. [KataGo paper, Table 2](https://arxiv.org/abs/1902.10565)
- Lc0 adopted squeeze-excitation, explicitly using global pooling to transmit global state back into every residual block. [Lc0 architecture](https://lczero.org/dev/backend/)
- Larger AlphaZero models have been found more sample-efficient at equal data, contradicting a general “larger only needs more data” assumption. [Neumann & Gros, ICLR 2023](https://openreview.net/forum?id=ZrEbzL9eQ3W)

**Evidence against**

- KataGo’s factors were measured in Go and partly exploit Go-specific target structure.
- The existing supervised capacity experiments cannot resolve architecture effects below their noise floor.
- A trunk change is expensive here because it affects Torch, JAX, and ONNX and prevents straightforward warm-starting.

**What distinguishes it**

Only after restoring a working value/search loop, compare the current trunk with one architectural change at a time. Start with explicit inputs because their semantic correctness is easier to test than a wholesale trunk replacement.

### 5. Training distribution has collapsed around one role, opening family, and lineage

**Evidence for**

- The outcome distribution is severely role-skewed.
- Deterministic candidate arenas are 96.3% White-won among decisive games.
- There is no root Dirichlet exploration on the production path.
- The policy target entropy collapsed in the degrading run.
- KataGo found policy-surprise weighting to be one of its larger post-paper improvements. More recent regret-guided search-control work reports gains of 70–122 Elo in three board games and improved a well-trained 9×9 Go model from 69.3% to 78.2% against KataGo while ordinary continuation barely changed. [Regret-Guided Search Control, ICLR 2026](https://rlg.iis.sinica.edu.tw/papers/rgsc/)

**Evidence against**

- A past-checkpoint pool is mainly a stabilization/diversity mechanism. It does not guarantee discovery of moves absent from every policy in the pool.
- Starting games from non-initial states changes the training distribution and needs careful weighting.

**What distinguishes it**

Measure opening-state coverage, unique state/action counts, policy-support turnover, and exact RNG diversity by generation. Then test 10–20% of games seeded from high-disagreement or high-regret positions, not arbitrary historical openings.

### 6. Network capacity is the ceiling

This remains possible, but it ranks below target and exploration defects. An 8.1M-parameter net may eventually be insufficient, and the existing capacity tests were all teacher-limited supervised tests. However, an xl run would approximately multiply inference cost before there is evidence that the current data contain a useful improvement signal.

### 7. The remaining gap is simply beyond the budget

This is not supported yet. The observed 21% score against level 9 corresponds, under an independent logistic-Elo approximation, to about a 230-Elo gap. With only 100 games, the binomial 95% interval is approximately 14–30%, corresponding to roughly 147–312 Elo—and the true uncertainty is larger because games are not independent. That is substantial but not obviously thousands-of-pounds substantial.

---

## 2. Conclusions in §§1–7 that I think are wrong

### “Measurement instruments have been the dominant failure mode”

**I disagree.** They wasted runs and hid regression, but a better gate does not create a better candidate. The v1 gate froze an approximately +8 Elo update; the paired rerun accepted everything and exposed that the underlying optimizer could degrade. Measurement failure is a major operational defect, not the primary explanation for the low fixed point.

The literature reinforces this distinction. ELF showed that adjacent-model Elo could inflate while Nash averaging detected degradation, and Lc0 warns that self-play Elo can move opposite external-engine estimates. [ELF OpenGo](https://proceedings.mlr.press/v97/tian19a/tian19a.pdf), [Lc0](https://lczero.org/blog/2018/10/lc0-training/)

### “Search width is the leading untested structural candidate”

**I disagree with “leading.”** Width is cheap enough to test and the hard truncation is theoretically unhealthy. But the current evidence does not show that excluded moves matter. The broken/unmeasured value head has a more direct evidential chain to corrupted Gumbel targets.

### “Distillation is a dead end for strength”

**Too broad and not supported.**

What the experiment establishes:

- Current pure supervised training from scratch is weaker than v3.
- Sequentially fine-tuning all of v3 on this corpus produces no detectable external gain.
- Move agreement with Pentobi is not a sufficient proxy for playing strength.

What it does not establish:

- that a 10–20% policy-only teacher mixture during RL is useless;
- that teacher positions are useless as self-play starting states;
- that querying Pentobi on states visited by the student is useless;
- that teacher Q/value targets would be useless;
- that an auxiliary teacher-policy head could not preserve knowledge without overwriting the production policy.

The data scaling curve is still rising, and mixing 20% older corpus data adds 0.058. Those are signs of coverage and diversity limitations, not a completed negative result.

Historically, AlphaGo combined supervised human policy learning with self-play RL successfully, while AlphaGo Zero showed that imitation is not necessary and can impose a ceiling if never superseded. [AlphaGo](https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/), [AlphaGo Zero](https://www.nature.com/articles/nature24270). KataGo’s own current documentation says external data is unnecessary but provides mild gains against some opponents and coverage of states absent from self-play. [KataGo repository](https://github.com/lightvector/KataGo)

My conclusion: **pure distillation is rejected; hybrid use of the corpus remains live.**

### “The auxiliary heads measured as nothing”

**Wrong experimental interpretation.** The experiment was underpowered and tested the wrong outcome.

KataGo’s result is an online RL sample-efficiency result. Auxiliary heads change the trunk, which changes search, which changes the next self-play distribution. Supervised imitation holds the teacher and data distribution fixed, precisely removing that feedback path. Moreover:

- the tested network was only 96×6;
- there were 3,565 games;
- there was effectively one seed;
- the assumed ±0.015 noise floor was later revised to about 0.05;
- score and ownership should be tested jointly to reproduce the KataGo ablation;
- reply loss damaged the value head, indicating loss-weight or gradient-interference problems rather than a clean negative.

My conclusion: **bad experiment, not bad technique.**

### “A very large accumulating buffer is KataGo’s actual practice”

**Incorrect wording.** KataGo used a *growing moving window of recent data*, starting at 250,000 samples and reaching about 22 million; the window grew sublinearly with total generated data. It did not uniformly replay every sample ever generated. [KataGo paper](https://rlg.mlanctot.info/2020/papers/AAAI20-RLG_paper_36.pdf)

That distinction matters. Uniform all-history replay can anchor a plateaued student to low-quality early targets. A growing recent window can preserve diversity while gradually discarding obsolete targets.

### “More simulations does nothing”

It establishes only that 128→256 simulations under the current value function, width, candidate construction, and training target did nothing. MiniZero found that simulation count and scheduling interact with training stage; progressive simulations were initially weaker but later materially stronger in 9×9 Go and Othello. [MiniZero](https://arxiv.org/abs/2310.11305)

### “Capacity is a bad buy”

Reasonable as an immediate spending decision, but not a scientific conclusion. Larger nets can be more sample-efficient, and KataGo/Lc0 both used progressive architecture growth. Capacity should be tested only after the signal pipeline works.

### “A two-week free-box run ought to reach level 7–9”

**Not a responsible forecast.** At top-k 128, two uninterrupted weeks yield about 1.05M games before allowing for evaluation, training interruptions, and failures. That might reach level 6–7 under a corrected operator; level 9 requires a stronger assumption about regained learning slope.

---

## 3. Specific hunt for bugs and silent defects

No bug is proven because I did not inspect code. I would audit these in order:

1. **Value perspective and completed-Q signs**
   - Terminal values at every ply.
   - Sign inversion after each move.
   - Whether completed Q is expressed from root, parent, or player-to-move perspective.
   - Draw mapping and WDL-to-scalar conversion.
   - Min/max or sigma scaling before `softmax(logits + σQ)`.

2. **JAX/Torch numerical parity on real positions**
   - Policy logits before masking.
   - Legal-move probabilities after masking.
   - Value output.
   - Every intermediate block for a small fixed batch.
   - Train/eval normalization behavior and checkpoint loading.

3. **Action encoding and legal-mask ordering**
   - Whether top-k is applied before or after illegal actions are removed.
   - Orientation and square indices under the order-2 symmetry.
   - Whether a legal action can disappear because illegal logits occupy top-k slots.
   - Whether the Python evaluator and JAX trainer assign the same action ID to every placement.

4. **PRNG reuse in lockstep JAX self-play**
   - Unique keys per game, move, search, generation, and device.
   - No broadcasting of the same Gumbel vector across games.
   - Opening and action diversity conditional on identical logits.
   - This is high priority because correlated RNG could create a deterministic-looking training distribution while metrics still appear healthy.

5. **Search-target alignment**
   - The returned completed-Q policy belongs to the exact pre-move state stored beside it.
   - No off-by-one after action application.
   - Legal re-normalization after symmetry augmentation.
   - No conversion from log probabilities to probabilities twice.
   - No accidental detach or stale incumbent logits.

6. **Replay sampling**
   - Split diagnostics by game, not position.
   - Confirm actual rather than configured buffer size.
   - Confirm the number of times each position is sampled over its lifetime.
   - Check whether uniform-by-position sampling overweights long games.
   - Check outcome, role, opening, phase, and checkpoint composition in every minibatch.
   - Confirm old targets retain their original policy/value metadata.

7. **Optimizer continuation**
   - Whether Adam moments are preserved, reset, or mismatched when warming candidates.
   - Whether learning-rate schedules restart each generation.
   - Whether weight decay is actually active in all parameter groups.
   - Gradient norms by head and trunk, including auxiliary-head interference.

8. **Outcome and target generation**
   - Score margin sign and normalization.
   - Final-cell ownership after symmetry and player transformation.
   - Reply target off-by-one and terminal masking.
   - Resignation, truncation, or premature-ending labels, if these mechanisms exist.

9. **Small exact tests**
   - Positions with one legal action.
   - Terminal positions.
   - Short endgames solved by exhaustive minimax.
   - JAX Gumbel search versus an independent reference implementation.
   - Apply the real symmetry and require transformed policy/search output equality to numerical tolerance.

10. **Run provenance**
    - Hash code, full resolved config, checkpoint, optimizer state, and dataset manifest into every generation.
    - Refuse to begin a paid run if committed and resolved configurations differ.
    - The known configuration drift is itself a pipeline defect.

---

## 4. Ranked recommendations, cost, and expected value

Self-play cost below uses the measured top-k 128 throughput of 0.869 games/s: about 32 RTX 3060 Ti hours per 100,000 games. Rental conversion uses approximately £0.74 per 5090 hour from $0.99/hour and a recent Bank of England GBP/USD rate; taxes and storage are excluded. [Bank of England exchange rates](https://www.bankofengland.co.uk/statistics/exchange-rates)

| Rank | Recommendation | Compute estimate | Expected value |
|---|---|---:|---|
| 1 | RL value diagnostics plus target/sign/JAX parity audit | 2–8 GPU-hours, £0 locally | Very high; can invalidate the whole current search target |
| 2 | PRNG, legal-mask, action-map, and completed-Q test suite | 5–15 GPU-hours | Very high; cheap compared with another run |
| 3 | Frozen-position width shadow test: 64/64 versus 128/64 | 15–30 GPU-hours including target generation | High information value; prevents spending a full run on a thin mechanism |
| 4 | Replace per-generation epochs with a fixed update-to-new-data ratio; sweep `1e-4`, `3e-4`, `1e-3` AdamW | 60–100 GPU-hours, about £45–74 rented | High; directly addresses proven regression conditions |
| 5 | Short online branch with value repair, outcome/role balancing, and top-k 128 | 200k games ≈64 self-play hours; budget 80–120 total | Highest plausible strength intervention |
| 6 | RL auxiliary test: control, score+ownership, reply; at least two seeds | 220–320 GPU-hours, about £163–237 rented | Medium-high, but only after value targets are trusted |
| 7 | Hybrid Pentobi use: 10% and 20% policy-only mixture plus opening/high-disagreement seeding | 60–100 GPU-hours | Medium; corpus evidence supports diversity value |
| 8 | Add explicit inventory, placeable-corner, and phase inputs | 350–500 GPU-hours for a fair online comparison, plus substantial engineering | Medium-high upside, expensive implementation |
| 9 | Global pooling/SE trunk and progressive large→xl growth | 500+ GPU-hours plus three-backend work | Potentially high ceiling; wrong first experiment |
| 10 | Reanalyse and checkpoint-pool self-play | Unknown substantial engineering; likely 200+ GPU-hours to test | Useful after search correctness; not a first repair |

For auxiliary heads, pre-tune loss scales offline, but judge them only by online sample efficiency against fixed external opponents. Do not expect KataGo’s 1.65× factor to transfer numerically.

For distillation, I recommend:

- retain 80–90% RL samples;
- use 10–20% teacher samples;
- apply teacher loss principally to policy, not the demonstrably skewed outcome target;
- query Pentobi on student-visited or high-disagreement states where possible;
- keep an auxiliary teacher-policy head or annealed KL regularizer rather than sequentially overwriting v3;
- seed some self-play games from the opening DAG, then finish with normal RL search.

---

## 5. The single cheapest decisive experiment

Run the missing **colour-conditional RL value audit** on v3.

Protocol:

1. Sample a held-out set of complete RL games, stratified by generation but never split within a game.
2. Evaluate:
   - v3;
   - constant predictor;
   - colour-only predictor;
   - colour×game-phase predictor.
3. Report MSE/Brier skill, log loss, calibration, AUC/ranking, and spatial-symmetry error separately for both roles and each phase.
4. Use a game-cluster bootstrap for confidence intervals.
5. Repeat inference through Torch and JAX.
6. On terminal and exhaustively solvable late-game positions, verify exact sign and perspective.

Cost: likely under 1–3 GPU-hours, £0 locally, plus engineering time.

**Pre-registered falsifier of my leading recommendation:** I will withdraw “repair value first” if all of the following hold:

- v3’s value skill is positive versus the colour×phase baseline in both roles, with the 95% game-cluster confidence interval above zero;
- skill and calibration improve toward terminal states;
- Torch and JAX outputs agree to the established parity tolerance;
- symmetry and sign/perspective tests pass;
- no outcome class or phase shows a systematic sign or calibration failure.

If that happens, width/exploration becomes the next leading hypothesis.

---

## 6. Is Pentobi level 9 reachable within the stated budget?

### Direct answer

**Probably yes, but not with the existing operator and not with confidence on a two-week schedule.**

The current level-9 score of 21/100 suggests an approximate 230-Elo deficit, with a very wide effective range. That is close enough that one or two meaningful algorithmic corrections could plausibly bridge it, especially because KataGo-scale efficiency factors are larger than the remaining estimated gap. It is also far enough that another unchanged million games may produce nothing.

### Conditional path

1. **Diagnostics and causal probes:** 50–100 GPU-hours.
2. **Corrected 200k-game branch:** about 64 self-play hours, 80–120 total.
3. **First main run:** 1M games, about 320 RTX 3060 Ti self-play hours.
4. **Continuation if the external slope remains positive:** another 1–3M games, 320–960 hours.

Total planning range: **750–1,500 RTX 3060 Ti-equivalent hours**, including experiments and evaluation overhead. At the measured throughput, the self-play component alone is roughly four to eight continuous weeks. Renting 100–300 hours of the 5090 costs approximately **£74–£222**, but its relevant end-to-end throughput was not supplied, so I cannot responsibly translate that into 3060-equivalent hours.

### Hard budget gates

- After diagnostics, spend nothing if the run cannot be reproduced from a hashed config.
- After 200k corrected games, require at least approximately +50 external Elo or a ladder improvement larger than the known rerun noise.
- After 1M games, require at least +100–150 external Elo and a clear positive slope.
- If those gates fail, stop. The likely reachable target is then **level 6, with level 7 possible**, while level 9 would probably require a new architecture and more than the stated budget.

The existing pipeline should be considered capped around levels 4–5 regardless of additional compute.

---

## 7. Run-parameter analysis

| Parameter | Judgment | Recommendation |
|---|---|---|
| Learning rate | Constant `1e-3` Adam is wrong for late continuation by this project’s own evidence | Sweep `1e-4`, `3e-4`, `1e-3` offline; likely start corrected online continuation near `3e-4`, cosine/step down toward `3e-5`, with AdamW `1e-4` weight decay |
| Schedule | No late-stage decay is unjustified | Decay by external progress and symmetry/value diagnostics; do not reset to `1e-3` each generation |
| Games/generation | 10,000 is not proven wrong, but “generation” is being overloaded | Export every 2,500–5,000 games for freshness; evaluate externally every 25,000–50,000; decouple exports from full-buffer epochs |
| Number of generations | Forty is not a meaningful budget by itself | Specify total games, total optimizer samples, and update-to-data ratio; plan 200k pilot then 1M main run |
| Buffer size | 60k is probably too small for diversity but not obviously too stale | Test 120k–240k games with a sublinearly growing recent window; reserve 25–50% of training draws for the newest 1–2 exports |
| Staleness | “Six generations” is the wrong unit | Track target age in games and Elo distance from the generating checkpoint; cap or downweight targets once policy divergence becomes large |
| Epochs/generation | Two is demonstrably wrong; one still means roughly six lifetime passes | Replace epochs with 2–4 training draws per newly generated position over its lifetime; do not exceed six without evidence |
| Batch size | Cannot be judged: absent from the briefing | Compare effective batches 256, 512, and 1,024, holding total sampled positions and optimizer convention fixed; select for throughput and held-out diagnostics |
| Weight decay | Zero was a mistake in the degrading run | Keep AdamW `1e-4` as the initial default and verify it is active |
| Simulations | Fixed 256 is poor value under current width/value | Use 128 for most work; reconsider progressive or randomized budgets only after value/search repair |
| Gumbel considered | Must be co-designed with top-k | Start the causal test at top-k 128 / considered 64, not 128/32; widening child support is the purpose |
| Gate | Per-generation strength gate is unusable | Remove promotion gating; use catastrophic drift stops and periodic external selection |

The inherited AlphaGo numbers are not a transferable recipe. AlphaGo Zero used SGD, batch 2,048, a 500k-game buffer, regularization, and a promotion gate; AlphaZero used asynchronous training, batch 4,096, 800 simulations and no promotion gate; KataGo used batch 256, a growing recent window and a per-sample learning-rate convention. [AlphaGo Zero](https://www.nature.com/articles/nature24270), [AlphaZero](https://arxiv.org/abs/1712.01815), [KataGo](https://rlg.mlanctot.info/2020/papers/AAAI20-RLG_paper_36.pdf)

The lesson is not to select one of those sets. It is to preserve the ratios they were controlling: fresh data per update, total lifetime replay, regularization, target quality, and optimizer step size.

---

## 8. What I could not determine

I would need the following before approving a long run:

- Current physical and effective batch size.
- Adam betas, epsilon, gradient clipping, mixed precision, and exact weight-decay parameter groups.
- Whether optimizer state is retained between candidates.
- Actual optimizer steps and sampled positions per generation.
- Average positions per game and sampling by game versus position.
- RL value calibration and skill by role, outcome, phase, and checkpoint.
- Exact value perspective and completed-Q normalization used at every tree depth.
- Whether legal masking occurs before top-k.
- Whether JAX PRNG keys are unique across games and searches.
- Exact Pentobi ladder colour assignment, openings, deterministic settings, and evaluation throughput.
- Whether levels 7–9 are expected to be strictly monotonic under the chosen evaluation settings; the observed 16/20/21 scores are not.
- Exact Pentobi corpus generation budget and whether “full move distributions” are raw policy, searched visits, scores, or another quantity.
- How held-out corpus positions were split; position-level splitting would leak near-duplicate positions from the same game.
- Draw, resignation, truncation, score-margin, ownership, and reply-label implementations.
- Whether terminal and small exact-game tests already exist.
- RTX 5090 end-to-end throughput for each proposed search setting.
- Engineering time for threading auxiliary targets through replay and for maintaining parity across Torch, JAX, and ONNX.

The briefing, rather than the code, was used as the sole source for project-specific facts.
