# Search width and net capacity — two probes for the next run (2026-08-03)

Both run on the box's 3060 Ti, both free. They exist to pick the one variable the next
paid run should change, after [`regression-and-next-steps.md`](regression-and-next-steps.md)
closed both branches of the previous decision tree: `xl`-from-scratch was dropped when P8
tied, and Pentobi distillation was taken and failed its gate
([`distillation-recipe-findings.md`](distillation-recipe-findings.md)).

**Verdicts up front:**

1. **Search width is affordable.** `top_k` 64 → 128 at n=128 costs **1.24× per self-play
   game**, against a pass threshold of 2×. No stall. This discharges R6 of
   [`regression-and-next-steps.md`](regression-and-next-steps.md) §3.3.
2. **Capacity is still not binding, now measured on external data** — but the honest
   reason is that **the experiment cannot resolve an effect this size**. `xl` beat `large`
   by 0.039 nats, and the noise floor is 0.050. The replicate arm was the best of the three.
3. **A methodological finding that outranks both:** run-to-run variation in this pipeline is
   ~0.05 nats held-out CE and ~0.022 weighted ladder. Every single-run A/B in the project's
   history with a delta below that is unresolved, whatever it was previously read as.

---

## 1. Search width — the cost is 1.24×

**Why width and not sims.** `topk_legal` truncates to the prior's top-`k` of ~17,837
actions **at the root and at every child node**
(`games/blokusduo/jax/search.py:151-198`). A policy blind spot below rank `k` can never be
searched, never appear in a training target, and never be corrected — a closed loop. It is
the only untested lever that explains a fixed point immune to more simulations, more
capacity and more data. More *simulations* at fixed width was separately measured as dead
weight at 3× the cost (§3.3 of the regression review).

**Is width visible at the root?** Checked in code before spending on it. The root prior is
truncated to the `top_k` window, then `mctx.gumbel_muzero_policy` receives
`max_num_considered_actions=32`. Gumbel-MuZero selects its considered set by Gumbel-top-k
sampling over the whole window, so at `top_k=128` an action the policy ranks 100th *can*
enter the tree, where at 64 it could not exist. Child nodes widen too, with no
`max_considered` restriction. **The test is well-posed.**

**Method.** Two configs identical but for `jax_selfplay.top_k`
(`run_configurations/blokus_width_topk{64,128}.json`), each measured by
`scripts/benchmarks/cloud_calibration.py --sizes large`. Net `large` (192×12), n=128,
`gumbel_max_considered` 32, jax batch 256.

| arm | games/s | positions/game | s/generation | £/gen @ £0.79/hr |
| --- | --- | --- | --- | --- |
| `top_k=64` (control) | **1.0814** | 57.97 | 440.4 | 0.097 |
| `top_k=128` | **0.8695** | 56.31 | 478.6 | 0.105 |

**Ratio 0.804 — width costs 1.24× per game.** Both arms exited 0, neither approached OOM
(peak 5,461 and 1,635 MiB of 8,192; the 5-second sampler probably missed the second arm's
true peak, so the ordering should not be read). Positions/game is equal within noise
(56.3 vs 58.0, against 56.5 measured on this box in July), so the arms produced
equal-length games rather than one degenerating.

**Four caveats.**

- **This measures cost, not benefit.** Width being affordable is not width raising the
  operator's fixed point. Only a multi-generation run answers that.
- **3060 Ti at jax batch 256, not the 5090 at 1024 we would rent.** Tree footprint scales
  as `batch × sims × top_k`, so production's tree is 4× larger. The arithmetic is
  reassuring — batch 1024 / n=128 / `top_k`=128 is a product of 16.8M, identical to the
  already-validated n=256/`top_k`=64/batch-1024 config that ran at ~2.2 games/s on a 5090
  using 4–6 GB of 32 — and the known cliff (n=512 *with* `top_k`=128) is 4× larger again.
  Confirmation is best folded into gen 0 of the real run rather than bought separately.
- **The harness carries ~10–15% noise.** `train_seconds_per_position` differed 14% between
  arms (0.00126 vs 0.00108) although `top_k` cannot affect training at all. The 1.24× ratio
  is far enough from 2× to survive that; the £/gen column should not be read precisely.
- **Width may be thin at the root in practice.** Target top-1 mass is ~0.68 and entropy
  ~0.9 nats (B6 of [`plateau-investigation.md`](plateau-investigation.md)), so a rank>64
  action rarely wins a Gumbel draw into the considered 32. Width is structurally present;
  whether it is a real channel is exactly what a pilot should measure — see §3.

---

## 2. Capacity on external data — the replicate is the result

**Why re-run a question P8 answered.** P8 fit `large` against `xl` on a frozen buffer of
the net's *own* self-play games. That is partly circular: a capacity-bound net produces
weaker targets and then fits them comfortably, so a tie is the expected reading in both
worlds. The Pentobi corpus is external teacher data and cannot fail that way.

**Method.** Three arms through `scripts/distill_sl.py --arms scratch`, v2 corpus only, lr
1e-3, batch 256, seed 7, holdout 0.05. `large_rep` is the **noise floor**: identical
settings and identical holdout, only the weight initialisation rerolled (`--init-seed 8`,
which leaves `--seed`'s split and subsample untouched). Holdout leakage 0.0000 and
train/holdout rows 279,608 / 15,945 in all three arms, confirming the split really was
shared.

| arm | params | best epoch | held-out policy CE | top-1 | value MSE |
| --- | --- | --- | --- | --- | --- |
| `large` 192×12 | 8,104,223 | 7 | 2.9195 | 0.3066 | 0.4590 |
| `xl` 256×16 | 19,067,231 | 6 | **2.8802** | 0.3107 | 0.4688 |
| `large_rep` 192×12, init-seed 8 | 8,104,223 | 5 | **2.8698** | 0.3109 | 0.4608 |

- **Treatment effect:** `xl` − `large` = **−0.0393 nats** (xl better).
- **Noise floor:** `large` − `large_rep` = **0.0497 nats**.

**The floor is larger than the effect, so there is no evidence capacity binds.** The
replicate is also the best arm outright — an 8.1M-parameter net beating the 19M one on CE,
top-1 *and* value MSE. The pattern is consistent across all three metrics.

**The pre-registered rule was itself wrong.** P8 set "≥0.03 nats → capacity binding", a
threshold chosen against an assumed CE noise of ~±0.015. The measured floor is over 3×
that, so no threshold in that rule is meaningful at one run per arm. The nominal reading
(0.039 clears 0.03) is a false positive, and only the replicate exposes it: without that
arm this would have been written up as "capacity binds, revive the `xl` run" and cost
£100–130 and four days.

**Practical conclusion:** stay at `large`. Capacity has now failed to appear on self-play
data (P8) and on external teacher data, and the effect size is below what this design can
see. Do not buy `xl`; do not build net-growing machinery. This is "not worth purchasing",
**not** "proven absent" — the honest statement is that a 0.03-nat effect is unresolvable
here, and resolving one needs several replicates per arm and a distribution comparison
rather than one run each.

---

## 3. What this settles for the next run

Every axis except width is now fixed by measurement rather than preference:

| knob | value | source |
| --- | --- | --- |
| net | `large` 192×12 | §2, and P8 |
| `top_k` | **128** — the one variable under test | §1 |
| sims / considered | 128 / 32 | more sims at fixed width is dead weight (§3.3) |
| learning rate | 2.5e-4 constant | constant 1e-3 at the fixed point drove the regression (§1.3) |
| optimiser | AdamW, weight decay 1e-4 | the missing AlphaZero guardrail (P1) |
| epochs | 1 | `epochs: 2` drove reuse to ~12 passes and overfit the value head |
| batch | 1024 | the binding quantity is optimizer steps per generation |
| buffer | rolling 60k games | windowed is what AZ/MuZero/KataGo all describe |
| weight flow | `gate_mode: always` + keep-best-by-ladder + drift circuit-breaker | all three arena-gated modes are characterised as broken in this game (§4) |

Baseline for the gate is **0.539 weighted** on the L1–5 mini-ladder at 50 games/level, 400
sims — already measured, and by the same instrument the circuit-breaker uses. Do not
compare it against the 0.344 in the regression review: that is L1–L9 at 100 games/level.

**Start with a pilot, and read the mechanism rather than strength.** At 50 games/level the
ladder's binomial noise is ~±7pp and the per-generation delta at the old fixed point was
~+8 Elo, so three generations cannot resolve strength — a working operator and a dead one
look identical. What three generations *can* resolve:

- **The width indicator: does any target probability mass land on actions the prior ranks
  below 64?** If ~zero, width is a theoretical channel rather than a real one and the
  hypothesis dies for about £2. This is the falsification test the pilot exists for.
- Mechanical health: actual games/s against the predicted 1.24×, VRAM headroom at batch
  1024, and whether training loss *moves* rather than sitting at the 0.76/0.53 flat
  signature of the frozen runs.
- An opportunistic signal: when v3's warm start had genuine headroom its gen-1 candidate
  scored 0.85. Presence of a large early signal would be strong evidence; absence proves
  nothing at gen 3.

**Warm-starting is defensible again, and this is a change of position.** The three failed
continuations (~$125, ≤0 external gain) all ran the *same* operator, which is why v3 gen-40
sat exactly at their fixed point. A wider search is a different operator with a different
fixed point, so a net parked at the old ceiling has headroom if width raises it — and if it
does not climb, the blind-spot hypothesis is refuted cheaply. It would also be the first
continuation run through the fixed instrument and the fixed recipe, changing exactly one
thing.

The corpus has no role in that run: its imitation ceiling (0.419) is below the net we would
start from. It becomes relevant again only if width fails and a trunk change forces a
from-scratch run — then it is a free launchpad to ~0.42 instead of starting at random.

---

## 4. The methodological finding, which outranks both probes

Two independent noise floors were measured today and last night:

| instrument | floor | how |
| --- | --- | --- |
| held-out policy CE | **~0.050 nats** | `large` vs its own replicate, §2 |
| weighted ladder (L1–5, 50 games/level) | **~0.022** | the same config re-run scored 0.361 and 0.339 |

Consequences, applied honestly to results already written down:

- **Survives:** the learning-rate effect (+0.078 weighted, ~3.5× floor), the old-corpus mix
  (+0.058, ~2.6×), data volume 2.5k→10k (+0.109), v2-over-v1 (+0.060, ~2.7×).
- **Was never resolvable:** capacity at +0.006 weighted, and the auxiliary heads at ±0.005
  top-1. Both conclusions happen to be unchanged, but they rest on "below the floor", not
  on "measured to be zero".
- **Should be retracted:** N4/N5's reading that the reply head's −0.021 nat CE gain "clears
  the noise floor". That floor was a single replicate estimated at ±0.015; a comparable
  pipeline measures ~0.05. The configuration differs (96×6, 6 epochs, 3,565 games) so the
  number does not transfer directly, but it corroborates that document's own suspicion that
  its floor was implausibly small. The reply head's apparent win is most likely noise.

**The rule this earns:** a single-run A/B in this pipeline cannot resolve anything below
~0.05 nats CE or ~0.022 weighted. Either run 2–3 replicates per arm and take the floor as
the largest of their deltas, or do not report a delta smaller than that as a result. The
[`supervised-network-improvements.md`](../plans/supervised-network-improvements.md) N4/N5
rows need re-running under that rule before either head is judged.

---

## 5. Provenance

- Width: `temp/benchmarks/width_topk{64,128}.json` on the box; log `/tmp/width_calib.log`;
  VRAM samples `/tmp/width_vram.log`; configs
  `run_configurations/blokus_width_topk{64,128}.json`.
- Capacity: `temp/benchmarks/cap_{large,xl,large_rep}.json` on the box; log
  `/tmp/capacity_corpus.log`. Presets from `config.py:26-29` (`large` 8.10M, `xl` 19.07M),
  parameter counts as reported by the run JSON.
- A first capacity attempt was killed by its own 2-hour-per-arm timeout (`exit=124`) with
  `--v1-mix 0.2` on; building examples from two corpora costs ~70 min per arm and dominates
  runtime. The re-run dropped the mix, which is why these CE numbers are **not** comparable
  to the blended runs in [`distillation-recipe-findings.md`](distillation-recipe-findings.md)
  — only `large`-vs-`xl` within this run is.
- Code read for the Gumbel/`top_k` interaction:
  `games/blokusduo/jax/search.py:151-215`. Incidental confirmation while reading it:
  `root_log_pi = log_pi if config.policy == "gumbel"` bypasses the Dirichlet path entirely,
  so `dirichlet_epsilon` in Gumbel configs is genuinely inert.
