# What actually limits the distilled net (2026-08-02)

One night of controlled runs on the finished 10,000-game v2 corpus, all on the box's
3060 Ti, all free. Every number below is a **Pentobi ladder over levels 1–5, 50 games
per level, 400 simulations** — the same range for every net, so they are comparable with
each other but *not* with any 1–9 figure quoted elsewhere (see
[[reference-ladder-metrics]] on why the range matters).

`weighted` is `Σ(level × wins) / Σ(level × games)`. v3 — the best self-play net — was
measured on this exact scale at **0.539** (78/72/58/48/44% at levels 1–5). Note it takes
level 4 to 48% here, just under the line, where a July ladder at 40 games/level had it at
55%; this is the more carefully measured figure, not a regression.

## The results

| net | L1 | L2 | L3 | L4 | L5 | weighted |
| --- | --- | --- | --- | --- | --- | --- |
| 128×8, lr 1e-4 (the old default) | 62% | 46% | 36% | 20% | 18% | 0.283 |
| 128×8, lr 1e-3 | 62% | 46% | 44% | 34% | 24% | 0.361 |
| 128×8, lr 1e-3, 2,500 games | 58% | 36% | 32% | 18% | 16% | 0.252 |
| 128×8, lr 1e-3, 5,000 games | 68% | 54% | 34% | 26% | 20% | 0.321 |
| **128×8, lr 1e-3, +20% v1 corpus** | **70%** | **60%** | **48%** | **36%** | **30%** | **0.419** |
| 128×8, lr 1e-3, +40% v1 corpus | 64% | 60% | 50% | 36% | 30% | 0.419 |
| 192×12, lr 1e-3 | 66% | 60% | 40% | 26% | 28% | 0.367 |

> **Noise floor, measured after the fact (2026-08-03): ±0.022 weighted.** The `lr 1e-3`
> row above and a later re-run of the *same* configuration scored 0.361 and 0.339. Every
> delta in the next table must be read against that: anything under ~0.022 is unresolved,
> not measured. See [`width-and-capacity-probes.md`](width-and-capacity-probes.md) §4.

## What each factor is worth

| factor | change | weighted delta |
| --- | --- | --- |
| **Learning rate** 1e-4 → 1e-3 | one flag | **+0.078** |
| **Old-corpus mix** 0% → 20% | free, data already existed | **+0.058** |
| **Data volume** 2.5k → 10k games | 4× the corpus | +0.109 |
| **Capacity** 128×8 → 192×12 | 2.3× the parameters | +0.006 |
| Auxiliary heads (ownership, reply) | a week of work | ±0.005 top-1, below noise |

## The four conclusions

**1. The training recipe was the binding constraint, and it was hiding everything else.**
`distill_sl.py` defaults to `lr 1e-4`, documented in its own docstring as *"a fine-tune
rate, not the self-play peak"* — and every distillation run ever done here, v1 and v2
alike, used it for **from-scratch** training. Raising it to 1e-3 is the single largest
effect measured. 3e-3 was slightly worse (0.372 CE vs 0.360), so the optimum is bracketed.

This also explains a result that looked damning: the v2 corpus redesign appeared to buy
nothing (weighted 0.100 vs v1's 0.102 over levels 1–9). Both were measured through a
handicapped recipe.

**Resolved the same night, at 1e-3 on both corpora: v2 does beat v1, by +0.060 weighted**
(v2-only 0.339, v1-only 0.279). That is ~2.7× the ±0.022 floor, so it is real but modest —
and note both single-corpus arms sit *below* the 0.419 that v2 with a 20% v1 mix reaches,
so the two corpora are complementary rather than one dominating. The redesign was worth
doing; it was not worth what the ladder gate needed. (The v1-only ladder exited 3: the
drift circuit-breaker tripping on two consecutive results ≥5% below the best net, which is
expected behaviour when both arms are far under v3, not a failure.)

**2. Data volume helps, but is already decelerating.** Per doubling: +0.069 (2.5k→5k),
then +0.040 (5k→10k). Extrapolating the decay puts 50,000 games at roughly **0.41** —
short of v3's measured 0.539. Renting CPU for a bigger corpus is cheap (~£30) and
positive, but **it does not close the gap on its own**.

**3. Mixing in the "bad" v1 corpus is worth more than quadrupling the good one, and it
saturates at 20%.** +0.058 for free, versus a projected +0.05 for 40,000 more v2 games.
40% is identical to 20% (0.419 both), so this is a one-off boost, not a dose-response.
That pattern — a small injection of differently-distributed positions helping, then
flattening — reads more like regularisation than like a data-volume effect. It is a
reason to prefer *differently generated* data over *more of the same*, but the quick
saturation warns against assuming a second corpus would compound.

**4. Capacity is not worth buying — on imitation.** 2.3× the parameters bought +0.006,
matching the July v1 sizing sweep (a 14× parameter range moved weighted 0.088 → 0.102) and
P8's self-play capacity probe (`xl` tied `large`). A fourth test the following day, `xl` vs
`large` on this corpus, also tied.

Two scoping corrections to the original wording ("stop testing net size"), both of which
matter for the RL phase:

- **+0.006 is far inside the ±0.022 floor.** The right statement is "unresolvable here",
  not "measured to be zero". Same for the auxiliary heads at ±0.005 top-1.
- **Every one of these tests is supervised.** In AlphaZero-family RL, capacity sets the
  *asymptotic* ceiling rather than the current fit, and no `xl` net has ever been run
  through self-play here. Imitation caps out at the teacher, so a bigger net has nothing
  extra to extract and a tie is the expected reading either way. Capacity is a bad buy
  today; it is not a closed question for RL.

## The decisive test: fine-tuning v3 (2026-08-02)

Distillation from scratch produces a net *weaker* than the one we already have, so the
real question was never "how good a launchpad can it build" but "can Pentobi's knowledge
be **added** to v3". `distill_sl.py --arms warm` does exactly that. Measured against v3
under identical ladder settings:

| net | L1 | L2 | L3 | L4 | L5 | weighted | top-1 vs Pentobi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3 as-is | 78% | 72% | 58% | 48% | 44% | **0.539** | — |
| v3 fine-tuned on v2 + 20% v1 | — | — | — | — | — | **0.537** | **0.355** |
| best from-scratch distilled | 70% | 60% | 48% | 36% | 30% | 0.419 | 0.306 |

**No change: 0.537 vs 0.539.** The fine-tuned net is far better at *predicting* Pentobi's
moves — top-1 0.355 against the from-scratch net's 0.306, policy CE 2.379 against 2.960,
the largest imitation gain we have measured — and not one point stronger at winning games.
It also overfits within two passes (CE 2.379 → 2.396 → 2.441 → 2.518), so there is no
headroom to find by training longer.

No damage either: the catastrophic-forgetting risk did not materialise.

## Why imitation keeps failing to become strength

Pentobi's playing strength lives in its **search**, not in its raw move preferences. The
corpus records what Pentobi plays *after* searching, but a net that imitates those choices
still has to search itself to be strong — and v3 already searches. We have been teaching
it the answers to a test it was already passing by a different route.

That single hypothesis also explains the three results that looked anomalous in isolation:

- a 14× parameter increase changing nothing (capacity was never the constraint on a
  quantity that does not convert to strength),
- the deliberately-"bad" v1 corpus matching the carefully redesigned v2 one (target
  quality does not matter much if the target is the wrong thing to learn),
- every measured improvement in held-out accuracy failing to move the ladder.

**Distillation has now been tested from both ends and neither helps.** From scratch it is
worse than v3; as a fine-tune it is identical. The direction should be wound down.

## Where this leaves the strategy

Best distilled net: **0.419**. v3, measured under the same settings: **0.539**. v3
fine-tuned on the corpus: **0.537**.

**Do not buy more Pentobi games.** The data-volume curve was already decelerating toward
~0.41 — below v3 — and we now know that even a *perfect* fit to this corpus lands where we
already are. More games buys a better imitation of something that is not the bottleneck.

**Correction (2026-08-03): the original conclusion here read "the remaining route to level
9 is reinforcement learning from v3". As written that is wrong, and it was written without
reading the RL history.** Self-play continuation from v3 gen-40 has been tried three times
— `search_harder` v1 (0/17 accepted, frozen), v2 (1 accepted in 9), and the paired-gate
rerun (20/20 accepted, ladder 0.344 → 0.298) — for ~$125 and ≤0 external gain, and
[`regression-and-next-steps.md`](regression-and-next-steps.md) §5 explicitly recommends
against a fourth attempt "with any variant of the current operator".

The mechanism is that self-play RL converges to a **fixed point of its improvement
operator**, and v3 gen-40 sits at the fixed point of the operator all three runs used. The
corrected statement is therefore: **the route forward is RL with a *changed* operator, and
the open question is which change raises the fixed point.** That is a property of the
operator — search width, target quality, capacity, curriculum — not of the starting net,
which is also why starting RL from a corpus-distilled net would climb back to the same
ceiling more cheaply rather than exceeding it.

Search width is the leading candidate and is now measured as affordable
([`width-and-capacity-probes.md`](width-and-capacity-probes.md) §1). Playout cap
randomisation remains the efficiency lever, but it cuts cost per game rather than raising
the ceiling, so it is a multiplier on whichever change works — not a substitute for one.

**Still open, but no longer gating any spending decision:**

- ~~Does the v2 corpus beat v1 at 1e-3?~~ **Answered: yes, +0.060 weighted** — see
  conclusion 1. Keep v2; keep v1 as the mix.
- The auxiliary heads were measured through the handicapped recipe, and their deltas were
  below the floor in both instruments. Re-testing at 1e-3 is cheap, but it needs 2–3
  replicates per arm to say anything at all — the reply head's apparent CE win was judged
  against a floor now known to be badly underestimated
  ([`width-and-capacity-probes.md`](width-and-capacity-probes.md) §4). Their value in *RL*
  is separately untested: KataGo's 1.65× for the aux-target group is a self-play result, and
  everything measured here is imitation.
