# Distillation net sizing (Phase A) — choosing the net for the Pentobi-L9 corpus

Design investigation for the network that will absorb the Pentobi-L9 distillation corpus
([`../plans/archive/pentobi-distillation.md`](../plans/archive/pentobi-distillation.md) Phase 2) and then carry the
RL-beyond-the-teacher phase. Written 2026-07-24, **before** any distillation training — this doc is
pure analysis: no GPU work, no corpus access. The only thing executed for it was instantiating the
net on CPU to count parameters and microbench a forward pass (§5).

**The brief:** stop reusing the four arbitrary `NET_PRESETS` (`config.py:25-30`) and size the net
from first principles. The presets bundle width and depth in lockstep (64×4 → 128×8 → 192×12 →
256×16), so they can never say *which axis* matters; and `xl` was explicitly sized to mimic AlphaGo
Zero's 256 filters (`config.py:23` says so in its own comment) — a 19×19-Go number transplanted
onto a 14×14 board.

**The dual objective.** The chosen net has two jobs that pull in opposite directions:

1. **Capture Pentobi L9** from the corpus, with headroom to then *surpass* it via RL — argues for
   capacity.
2. **Be fast enough for self-play RL afterwards** — net size compounds brutally: bigger net →
   slower inference → slower self-play → fewer generations per week, over dozens of RL generations
   on a single GPU (home 3060 Ti 8 GB; occasionally a rented 5090). Self-play is inference-bound in
   the low-sim Gumbel regime ([`jax-pipeline-ab.md`](jax-pipeline-ab.md) §4), so this cost is
   ~linear in net FLOPs and is paid on *every move of every game, forever*.

So the target is **the smallest net that captures L9 with some headroom, at self-play throughput we
can afford** — not the biggest net that fits L9.

**Evidence tagging** as in [`cloud-training-recommendation.md`](cloud-training-recommendation.md):
**[measured]** — from an actual run or from the CPU instantiation done for this doc;
**[extrapolated]** — derived from measured numbers by a stated model;
**[open]** — genuinely unknown until Phase B trains on the corpus.

---

## TL;DR

- **Depth is nearly settled by geometry:** 6 residual blocks give every cell a full-board receptive
  field on 14×14; blocks beyond ~6–8 buy composition, not reach (§2.1). **Width is the real unknown**
  and costs quadratically (§2.2). The 17,837-action space is *not* a reason to go big — the conv
  policy head is 0.2% of parameters (§2.3).
- The precedent honestly scaled says our regime is 1–2 size classes below AGZ's 256×20; AlphaGo's SL
  net — the direct analogue of our distillation step — was a **192-filter, 13-layer plain convnet**
  (§3). Our own history: the only external level jump came with medium→large; the capacity probe
  found `xl` fits *self-play* data no better than `large` — but L9 data is stronger, so the capacity
  question genuinely re-opens for distillation (§4).
- **Eight concrete candidates** (§6), width and depth decoupled, 1.06M → 19.07M params, all counted
  by instantiation [measured]: a floor (96×6), a cost-matched shape triangle at ~0.5× large
  (192×6 / 128×14 / 160×10), the incumbent (192×12), a wide-at-large-cost arm (256×8), and the
  `xl` ceiling (256×16).
- **Provisional recommendation (§7): the knee is expected in the ~0.5×-large band — 160×10
  (4.72M params, ~1.5× large's self-play throughput) is the provisional pick, with the incumbent
  `large` 192×12 as the fallback if the corpus proves harder than self-play data.** [open — this is
  exactly what Phase B measures.]
- **Phase B (§8): seed both endpoints (96×6 floor, 256×16 ceiling), then bisect the cost axis
  toward the knee of held-out top-1-accuracy-vs-cost; decision rule = smallest candidate within
  ~1 pp top-1 of the ceiling with acceptable games/s.** Adaptive-sequential, because parallel
  training buys nothing on one GPU.

---

## 1. Why the preset ladder can't answer this

The four presets vary filters and blocks together (F, B) = (64,4), (128,8), (192,12), (256,16), so
every observed preset-to-preset difference confounds width with depth — we have literally no data
point that isolates either axis. And the ladder's top end was anchored on the wrong board:
`config.py:23` documents `xl` as "≈ AlphaGo Zero's 256 filters at 14×14 depth-scaled". AGZ's 256
was chosen (by DeepMind, with TPU pods) for 19×19 Go; nothing about our 196-cell board, ~28-ply
games, or single-GPU budget inherits that choice.

The rest of the config machinery is already shape-agnostic: explicit `num_filters` /
`num_residual_blocks` in a run JSON win over any preset (`config.py:798-813`), and the calibration
tool accepts bare `<F>x<B>` size specs (`alphablokus/calibration.py::parse_net_sizes`). So the
candidates in §6 need **no code changes** — they are config values.

---

## 2. First principles on a 14×14 board

Architecture facts, from [`../../src/alphablokus/games/blokusduo/nn/net.py`](../../src/alphablokus/games/blokusduo/nn/net.py):
input 44×14×14; trunk = one 3×3 stem conv (44→F) + B residual blocks of two 3×3 convs (F→F) each,
all stride 1 pad 1; heads = `ConvPolicyHead` (1×1 conv F→91 orientation planes + a pooled pass
logit) and a value head (1×1 conv → `Linear(196→F)` → `Linear(F→1)` → tanh).

### 2.1 Depth = receptive field first, composition second

The trunk stacks `2B + 1` 3×3 convolutions, so the receptive field of a trunk output cell is a
square of radius `r = 2B + 1` (each 3×3 conv adds 1 to the radius). For *every* cell to see the
*entire* 14×14 board, the radius must cover the worst-case offset of 13 cells (corner to corner):

```
r = 2B + 1 ≥ 13  →  B ≥ 6
```

| Blocks B | RF radius | Coverage |
|---|---|---|
| 4 (`small`) | 9 | corner cells blind to the far ~30% of the board |
| **6** | **13** | **exactly full-board from every cell — the geometric minimum** |
| 8 (`medium`) | 17 | full + 1.3× margin |
| 12 (`large`) | 25 | ~2× the minimum |
| 16 (`xl`) | 33 | ~2.5× the minimum |

Beyond B=6, extra blocks add nothing spatially — they buy *compositional* depth (re-processing
already-global features), which has value but with diminishing returns. For calibration: AGZ's
19×19 board needs B ≥ 9 for full coverage and they ran 20 blocks — ~2.2× their geometric minimum.
Our `large` at 12 blocks is 2.0× ours, i.e. **the incumbent's depth already matches AGZ's
relative overprovision**; `xl`'s 16 exceeds it. Blokus Duo also plausibly needs *less* composition
than Go: no life-and-death recursion, no ladders — the long-range interactions are corner-contact
chains and territory partition, which full-board RF plus a couple of compositional passes should
express. **Working conclusion: B ∈ [6, 12]; depth above 12 is unjustifiable on a 14×14 board.**

### 2.2 Width = per-cell feature capacity, and it costs quadratically

Width F is the number of features computed *per cell*. Cost scales as F²·B (each block is two
F→F 3×3 convs over a fixed 196-cell board — §5's MACs column is ∝ F²B to within 3%), so width is
twice as expensive to buy as depth, per multiplicative step.

What sets the width floor? Two soft arguments:

- **The policy head reads F features per cell through a 1×1 conv into 91 orientation logits.** That
  per-cell linear map has rank ≤ F, so with F < 91 the 91 logits at a cell are linearly constrained
  given the cell's features. `small`'s F=64 was rank-deficient here; F ≥ 96 clears it. (Soft,
  because neighbouring cells' logits couple through the trunk — but a clean reason not to go below
  ~96.)
- **The input is already 44 channels.** F should comfortably exceed the input channel count to avoid
  an information bottleneck at the stem; 96+ does.

What sets the ceiling? The complexity of the L9 policy — genuinely **[open]**. The self-play
evidence (§4) says 192 channels were not saturated *by self-play data* at 12 blocks and 256 added
nothing; whether L9 targets need more than ~128–192 channels per cell is exactly Phase B's
question. **Working conclusion: F ∈ [96, 256], with the interesting region 128–192.**

### 2.3 The action space is NOT a reason to go big

The 17,837-action space *used to* dominate parameter count: the old FC policy head was a single
`Linear(2·196 → 17837)` holding **~95% of the net's parameters**
([`policy-head-architecture.md`](policy-head-architecture.md)). The conv head fixed that: at F=192
it is **17,756 parameters — 0.22% of the net** [measured, §5]. Emitting orientation×row×col logit
planes from a 1×1 conv means head cost is linear in F and independent of action-space size. Any
instinct that "17,837 actions needs a big net" is a fossil of the FC head. Capacity arguments must
stand on trunk/feature grounds alone.

---

## 3. Precedent, scaled honestly

From [`deepmind-run-configs.md`](deepmind-run-configs.md):

| System | Net | Board / actions | Hardware | Relevance |
|---|---|---|---|---|
| AlphaGo (2016) **SL policy net** | 13-layer plain conv, **192 filters** | 19×19 / 361 | 50 GPUs, 29M positions → 57% top-1 | **The direct analogue of our distillation step** — imitating strong play from a fixed corpus |
| AlphaGo Zero (2017) | **20×256** resnet (later 40×256) | 19×19 / 362 | 64 GPU train + TPU inference, 4.9M games | The number `xl` copied |
| AlphaZero (2018) | 20×256 | Go/chess/shogi | 5,000 TPUs self-play | Same net, vastly more data |

Two honest scalings:

- **Board area:** 196 cells vs 361 — 0.54×. Per-cell width capacity doesn't obviously scale with
  area, but total problem size, game length (~28 plies vs ~250), and branching-over-time all do.
  Blokus Duo is also a *scoring* game against a "strong amateur"-class target (Pentobi), not a
  world-champion-level one ([`deepmind-run-configs.md`](deepmind-run-configs.md) §7).
- **Data:** AGZ's 20×256 (~23M params, community estimate) was fed 4.9M games / ~500k-game replay
  windows. Our corpus is **50k games ≈ 1.5M positions (~3.0M after 2× symmetry augmentation)**
  ([`../plans/archive/pentobi-distillation.md`](../plans/archive/pentobi-distillation.md) D5); stage 1 is 13k games
  ≈ 800k augmented positions. Fitting 19M parameters to 1.5M positions is a regularisation problem,
  not a capacity flex — AlphaGo's SL net had ~1 position per ~0.1 param; `xl` on our corpus would
  be ~13 params per position.

The most encouraging precedent is the first row: a **192-filter, 13-layer, non-residual** convnet —
roughly our `large`'s width at *half* its effective conv depth, on a bigger board with a worse
architecture — captured enough of strong human Go play to beat every pre-neural program. On a
smaller board, with residual connections and a modern head, capturing strong-amateur Blokus from
1.5M positions should not need more than that. **Precedent verdict: `xl` (AGZ-sized) is almost
certainly overkill; the AlphaGo-SL-shaped region (128–192 filters, 6–12 blocks) is where the answer
should live.**

---

## 4. Our own evidence

- **`small` (64×4, 0.34M)** was the original production net; it never beat Pentobi L1 (losing ~75%
  vs L1 after 30 gens × 2k games [measured —
  [`cloud-training-recommendation.md`](cloud-training-recommendation.md) §5]) — though that number
  is volume-confounded (`medium` at the same volume also lost ~75% vs L1), §2.1/§2.2 give
  structural reasons `small` was always capped: blind corners (RF radius 9 < 13) and a
  rank-deficient policy-head width (64 < 91).
- **medium → large is the only capacity step that ever moved the external ladder**: L3 → L4 late in
  `blokus_cloud_60` ([`regression-and-next-steps.md`](regression-and-next-steps.md) §3.1). Weak
  single observation, but it is *the* datapoint that capacity mattered at least once.
- **`large` (192×12, 8.10M) is the incumbent and current best** — v3 gen-40 holds a winning record
  through Pentobi L4 (README ladder; `blokus_cloud_v3` post-mortem in
  [`xl-training-scaleup.md`](xl-training-scaleup.md) A1).
- **The capacity probe (post-regression-recovery P8, 2026-07-23) returned a tie**: `large` and `xl`
  trained supervised on the same frozen gen-40 *self-play* buffer with a game-level held-out split
  reached statistically indistinguishable held-out policy CE (< 0.01 nats gap — the pre-registered
  "clear tie" verdict, [`../plans/archive/post-regression-recovery.md`](../plans/archive/post-regression-recovery.md)
  P8). That killed the paid `xl` self-play run and chose the distillation path.

**How far the probe carries — and where it stops.** The probe says: *targets produced by the
current net's own n=256 search contain no structure that 8.1M params can't absorb and 19.1M can.*
It does **not** say `large` can absorb Pentobi L9. L9 targets come from a far stronger, deeper
search over a different playing style; they are plausibly higher-entropy-in-the-right-places and
carry long-horizon judgements self-play never produced. **The capacity question genuinely re-opens
on the L9 corpus** — the probe bounds our *prior* (capacity above `large` is unproven useful, once,
on weaker data), it does not settle Phase B. Both directions remain live: the corpus could need
*more* than `large` (stronger targets) or *much less* (1.5M positions is a small supervised
dataset, and §3's SL precedent fits in the medium band).

---

## 5. Measured sizes and costs [measured]

Method: instantiate `AlphaBlokusDuo(14, 14, 17837, 44, NetConfig(..., policy_head="conv"))` on CPU
and sum `p.numel()`; MACs computed analytically from the layer shapes
(stem `44·F·9·196` + blocks `B·2·F²·9·196` + heads, ≈ `3528·F²B` — the trunk is >98% of compute at
every candidate size); CPU forward microbench = median of 10 reps, batch 64, fp32, 10 torch threads
on the M-series Mac (torch 2.10.0, 2026-07-24). The four preset param counts reproduce the numbers
already published in [`regression-and-next-steps.md`](regression-and-next-steps.md) §6 exactly.

| Net (F×B) | Params | Policy head | Value head | MACs/pos | MACs vs `large` | CPU fwd ms/64 | Self-play games/s (3060 Ti) |
|---|---|---|---|---|---|---|---|
| 64×4 (`small`) | 340,127 | 5,980 | 12,739 | 64.0M | 0.04× | 34.0 | **12.22 [measured]** |
| 96×6 | 1,063,871 | 8,924 | 19,107 | 204M | 0.13× | 53.7 | ~8.3 [extrapolated] |
| 128×6 | 1,860,831 | 11,868 | 25,475 | 359M | 0.23× | 71.8 | ~7.0 [extrapolated] |
| 128×8 (`medium`) | 2,451,679 | 11,868 | 25,475 | 475M | 0.30× | 94.7 | **6.24 [measured]** |
| 128×10 | 3,042,527 | 11,868 | 25,475 | 590M | 0.37× | 117.5 | ~5.5 [extrapolated] |
| 192×6 | 4,118,303 | 17,756 | 38,211 | 799M | 0.51× | 124.5 | ~4.4 [extrapolated] |
| 128×14 | 4,224,223 | 11,868 | 25,475 | 822M | 0.52× | 156.5 | ~4.3 [extrapolated] |
| 160×10 | 4,724,735 | 14,812 | 31,843 | 919M | 0.58× | 156.0 | ~4.0 [extrapolated] |
| 192×8 | 5,446,943 | 17,756 | 38,211 | 1.06G | 0.67× | 158.7 | ~3.6 [extrapolated] |
| 192×12 (`large`) | 8,104,223 | 17,756 | 38,211 | 1.58G | 1.00× | 233.7 | **2.71 [measured]** |
| 224×10 | 9,195,071 | 20,700 | 44,579 | 1.79G | 1.13× | 213.3 | ~2.4 [extrapolated] |
| 256×8 | 9,621,855 | 23,644 | 50,947 | 1.87G | 1.19× | 221.1 | ~2.3 [extrapolated] |
| 256×10 | 11,983,199 | 23,644 | 50,947 | 2.34G | 1.48× | 271.2 | ~2.0 [extrapolated] |
| 256×16 (`xl`) | 19,067,231 | 23,644 | 50,947 | 3.72G | 2.36× | 424.8 | **1.30 [measured]** |

Notes on the cost columns:

- The **games/s** anchors are the four presets measured on the 3060 Ti under production search
  settings (jax Gumbel n=64, K=64, B=1024, bf16 —
  [`cloud-training-recommendation.md`](cloud-training-recommendation.md) §2). Extrapolated rows are
  log-log interpolation of games/s against MACs through those four points. The global fitted
  power-law exponent is −0.54 (throughput falls sub-linearly with FLOPs because fixed tree/env
  machinery shrinks as a share — but the local slope steepens with size, ~−0.9 between `large` and
  `xl`, i.e. big nets pay nearly full FLOP price).
- The **CPU microbench** is a sanity check, not the cost model: at small sizes it is
  overhead-bound (64×4 runs only 12× faster than 256×16 against a 58× MACs gap), so it
  *compresses* ratios. MACs is the honest GPU proxy; the measured games/s ratios sit between the
  two, as expected.
- **VRAM is a non-issue for RL self-play at every candidate** (44×14×14 activations are tiny —
  `config.py:20-21`). The one known 8 GB limit is jax+torch *coexistence* during a training step at
  batch 1024 for `xl` ([`cloud-training-recommendation.md`](cloud-training-recommendation.md) §2);
  Phase B's SL fits are pure torch (no jax resident), and the P8 probe already trained `xl` on the
  box successfully.

---

## 6. The candidates

Eight nets spanning 1.06M → 19.07M params with **width and depth decoupled**. The design logic:
a floor at the geometric depth minimum; the two lockstep references we have history for; a
**cost-matched shape triangle** at ~0.5× large (three shapes, same compute — the experiment the
preset ladder could never run); a wide-at-large-cost arm; and the `xl` ceiling.

| # | Role | F×B | `policy_head` | Params [measured] | Rel. cost (MACs) | Est. games/s, 3060 Ti |
|---|---|---|---|---|---|---|
| C1 | **Floor** — geometric-minimum depth, rank-sufficient width | 96×6 | conv | 1,063,871 | 0.13× | ~8.3 |
| C2 | Reference (`medium`) — run3's net | 128×8 | conv | 2,451,679 | 0.30× | 6.24 [measured] |
| C3 | **Wide-shallow** — width at minimum depth | 192×6 | conv | 4,118,303 | 0.51× | ~4.4 |
| C4 | **Deep-narrow** — depth at medium width (cost-matched to C3 within 3%) | 128×14 | conv | 4,224,223 | 0.52× | ~4.3 |
| C5 | **Balanced mid** — completes the shape triangle at ~0.5× | 160×10 | conv | 4,724,735 | 0.58× | ~4.0 |
| C6 | **Incumbent** (`large`) — current best net, holds L4 | 192×12 | conv | 8,104,223 | 1.00× | 2.71 [measured] |
| C7 | **Wide-at-large-cost** — near-cost-matched shape alternative to C6 | 256×8 | conv | 9,621,855 | 1.19× | ~2.3 |
| C8 | **Ceiling** (`xl`) — "what capturing L9 looks like"; not expected to be picked | 256×16 | conv | 19,067,231 | 2.36× | 1.30 [measured] |

What each comparison isolates:

- **C3 vs C4 vs C5** (identical compute, three aspect ratios): does the corpus reward width,
  depth-beyond-RF, or neither? This is the decoupling experiment, and it runs at the cost level
  where §7 expects the knee.
- **C6 vs C7** (~equal compute): at large-class budget, is the incumbent's 12-block depth doing
  work, or would those FLOPs be better spent on 256 channels at RF-adjacent depth?
- **C1 vs C8**: the Phase-B endpoints — the floor tells us how much of L9 is easy; the ceiling
  anchors the achievable top-1 that the decision rule measures everything against.
- **C2, C6**: continuity with every measured number and trained checkpoint we own.

**How to spell them (no code change):** explicit keys beat presets, so a Phase-B config or probe
arm is just:

```jsonc
"net_config": { "num_filters": 160, "num_residual_blocks": 10, "policy_head": "conv", ... }
```

and the throughput tool takes them directly:
`uv run python -m scripts.benchmarks.cloud_calibration --config <cfg> --sizes 96x6,192x6,128x14,160x10,256x8`.
Deliberately **not** adding eight preset names to `NET_PRESETS` now — the ladder gets revised
*once*, after Phase B crowns a winner, rather than accreting speculative names. (If a preset is
wanted then, the natural move is to add the winner as e.g. `"distill"` and retire the lockstep
sizes to comments.)

---

## 7. Provisional recommendation

**Expect the knee in the ~0.5×-large band; provisional pick: C5 (160×10, 4.72M params) — with the
incumbent C6 (`large`, 192×12) as the fallback if the corpus proves harder than self-play data.**

Reasoning, in order of confidence:

1. **[High confidence] Depth beyond 12 and the `xl` point are out.** Geometry (§2.1), scaled
   precedent (§3), the probe tie on weaker data (§4), and the 2.4× throughput price (§5) all point
   the same way. C8 exists to anchor the ceiling, not to win.
2. **[High confidence] The floor candidates alone are not the answer for the *dual* objective.**
   Even if C1 nearly matches the ceiling's top-1 (possible — 1.5M positions is small), the second
   job is RL headroom *beyond* L9, and the one historical capacity signal (medium→large moving the
   ladder) plus `small`'s structural deficiencies argue for margin above the bare minimum.
3. **[Medium confidence] ~4–5M params fits 1.5M strong-teacher positions with sensible margin.**
   AlphaGo's SL analogue sat at this width class on a bigger board (§3); our corpus is 10–20×
   smaller than theirs per parameter at this size, and weight decay + 2× symmetry augmentation +
   game-level held-out early stopping are already standard in the pipeline. At ~0.5× large the RL
   phase runs ~1.5–1.6× more generations per week than the incumbent — a real, compounding win.
4. **[Low confidence — the open question] Whether L9 targets contain structure that needs the
   large band.** If Phase B shows C6 clearly above the C3–C5 triangle on held-out top-1 (≥ ~1 pp),
   we keep the incumbent size and accept the throughput; its RL viability is proven (v3 ran it for
   40 generations at 10k games/gen). If the triangle ties C6, take the triangle's best shape.

Also carried into Phase B, not decided here: the sizing sweep's arms are **from-scratch** SL fits
(that is what makes them comparable); the separate warm-start-vs-fresh question for the final net
(pentobi-distillation D7 already plans both arms at `large`) only applies to sizes that have a
pre-trained checkpoint — i.e. if the winner ≠ 192×12, it trains from scratch by construction.

---

## 8. Phase B — the adaptive search plan

Phase B = the empirical half: train candidates on the (already-generated) corpus and pick the net.
It rides on pentobi-distillation D6 (corpus dataloader) and D7 (SL trainer); the sizing sweep is
D7 executed per candidate with `net_config` swapped. Everything below runs on the box GPU — an SL
fit over ~1.6–3M augmented positions × a few epochs is **hours per candidate, not days**.

**Metric.** Held-out **top-1 accuracy against the Pentobi-played move** (plus policy CE and value
MSE as secondaries), on a **game-level** held-out split — no position of a held-out game in
training (the split machinery exists: `training/holdout.py::split_games_holdout`, already used by
`scripts/capacity_probe.py`, which is also the template for the per-arm fit loop: train to
asymptote / early-stop on held-out CE, report best).

**The curve being hunted.** Top-1 vs net cost is monotone-saturating — more capacity never fits a
fixed corpus worse (given regularisation + early stopping), it just stops helping. So there is no
threshold to test against, only a **knee** to locate: the cost below which accuracy falls away from
the plateau.

**Protocol — endpoints first, then bisect:**

1. **Seed both endpoints.** Train **C8 (ceiling)** → `acc_ceiling`, the achievable
   "what capturing L9 looks like" number; and **C1 (floor)** → `acc_floor`. Two fits bracket the
   whole curve. (Sanity gate: if `acc_ceiling − acc_floor < ~1 pp`, the corpus is easy — skip to
   the throughput-cheapest acceptable candidate and spend the saved effort on D8's ladder gate.)
2. **Bisect the cost axis toward the knee.** The candidates ordered by cost are
   C1 (0.13×) → C2 (0.30×) → [C3/C4/C5 ≈ 0.5×] → C6 (1.0×) → C7 (1.19×) → C8 (2.36×). Next fit is
   always the cost-midpoint of the current bracketing pair (knee inside vs outside); at the ~0.5×
   band, run the **shape triangle C3/C4/C5** to pick the aspect ratio, since cost is equal there
   and the axis question (§2) is answered by whichever shape wins. Expected total: **4–6 fits of
   the 8**, not all 8.
3. **Decision rule.** Winner = the **smallest-cost candidate within ~1 pp top-1 of `acc_ceiling`
   whose measured self-play throughput is acceptable** — concretely, ≥ ~2 games/s on the 3060 Ti
   under production search settings (the incumbent's 2.71 is proven livable at 10k games/gen ≈
   1 h self-play; below ~2 the box RL loop degrades past what v3 validated). Ties on top-1 break
   toward throughput; ties on both break toward the incumbent (it has trained RL checkpoints and
   validated jax bridging).
4. **Measure games/s per shortlisted candidate — don't trust §5's interpolation.** One command per
   size on the box, existing tooling:
   `uv run python -m scripts.benchmarks.cloud_calibration --config run_configurations/blokus_cloud_calibration.json --sizes <F>x<B>,...`
   (measures real jax-backend self-play games/s and torch training ms/position per size;
   `scripts/benchmarks/benchmark_selfplay_backends.py` as the cross-check). This is deferred to
   Phase B by design — Phase A ranks by MACs, Phase B buys the ranking only where it matters (the
   1–3 nets near the knee).
5. **Value-head check before crowning.** The corpus's value labels (L9-vs-L9 outcomes + margins)
   are much cleaner than self-play outcomes; confirm the winner's held-out value MSE is not the
   axis that actually wanted capacity (record per-arm, as the P8 probe did).
6. **Then D8.** The winner (and the runner-up if close) goes to the mini-ladder gate
   (`scripts/mini_ladder.py`, L3–L6 × 50 games × 400 sims) — held-out top-1 is the *selection*
   metric, the ladder is the *product* metric, and pentobi-distillation's Phase-2 gate (+10 pp at
   any of L5–L7 after SL alone) stays the go/no-go for RL spend.

**Why adaptive-sequential, stated explicitly:** Phase B runs on **one GPU**. Training candidates
"in parallel" time-slices the same device — total wall-clock is identical to sequential, minus
nothing. The efficiency lever on a single GPU is **adaptivity** (each result halves the remaining
search interval, so ~4–6 fits replace 8); parallelism only becomes real with a second GPU (e.g.
box + rented 5090 splitting arms), at which point the same plan parallelises trivially because the
arms are independent.

**What Phase B settles that this doc cannot:** whether L9 targets re-open the capacity gap the P8
probe closed on self-play data (§4); which axis — width or depth — the corpus rewards (§2, via the
triangle); and the true games/s of the knee candidates (§5's interpolation ±).

---

## 9. Provenance

- Param counts / MACs / CPU microbench: instantiation of
  `src/alphablokus/games/blokusduo/nn/net.py::AlphaBlokusDuo` on CPU, torch 2.10.0, 2026-07-24
  (this doc, §5 method; preset counts cross-check against
  [`regression-and-next-steps.md`](regression-and-next-steps.md) §6 — exact match).
- Measured games/s + training ms/position per preset: 3060 Ti calibration run, 2026-07-04
  ([`cloud-training-recommendation.md`](cloud-training-recommendation.md) §2); 5090 anchor: `large`
  19.6 games/s in `blokus_cloud_60` ([`xl-training-scaleup.md`](xl-training-scaleup.md) §2).
- Capacity probe (tie verdict): [`../plans/archive/post-regression-recovery.md`](../plans/archive/post-regression-recovery.md)
  P8; script `scripts/capacity_probe.py`; verdict rule pre-registered in
  [`regression-and-next-steps.md`](regression-and-next-steps.md) §3.4.
- Corpus size / schema / plan: [`../plans/archive/pentobi-distillation.md`](../plans/archive/pentobi-distillation.md)
  (D4 pilot: ~30.3 positions/game [measured]; D5 target: 50k games ≈ 1.5M positions).
- Preset definitions + resolution: `src/alphablokus/config.py:25-30, 798-813`; `<F>x<B>` spec
  support: `src/alphablokus/calibration.py::parse_net_sizes`;
  `scripts/benchmarks/cloud_calibration.py`.
- Policy-head history (FC head ≈ 95% of params → conv head):
  [`policy-head-architecture.md`](policy-head-architecture.md).
- DeepMind net/config numbers: [`deepmind-run-configs.md`](deepmind-run-configs.md) (cited to the
  primary papers there).
