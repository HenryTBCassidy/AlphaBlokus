# Policy head architecture: FC vs convolutional

A research note about an architectural choice in the Blokus Duo neural net that, on inspection, costs us roughly **95% of the network's parameters** for arguably worse training behaviour. Recording the finding here so the decision to act on it (or not) can be made deliberately.

This note is written assuming the reader knows linear algebra (matrices, vectors, dot products) but not necessarily ML-specific terminology. ML jargon is defined the first time it appears.

## TL;DR

- The current Blokus Duo network has ~ 7.3M parameters total. **~95% of them** (~ 7.0M) live in a single fully-connected (FC) layer at the very end of the policy head.
- That FC layer is a faithful copy of the AlphaGo Zero (Silver et al. 2017) policy-head design. For Go it's fine. For Blokus it's wildly disproportionate because our action space (17,837) is ~91× larger than our cell count (196) — Blokus actions index a (piece-orientation, cell) cartesian product, not just a cell like Go.
- The same paper's authors made a different choice for AlphaZero Chess (Silver et al. 2018), using a fully-convolutional policy head with no FC at all. Chess's action space (~4,672) is also structured as (move-type, cell) pairs, like ours.
- A conv-only variant of our policy head would have **~6K params**, not 7M. **~1200× smaller.** This isn't a small optimisation — it potentially rewrites the parameter economy of the whole network.
- "Fewer params = less learning" is the natural intuition but turns out to be misleading: the conv variant brings a **stronger and correct inductive bias** for board games, which generally improves sample efficiency rather than hurting it.

This document explains all of the above slowly. It does **not** prescribe doing the switch — that would be a separate plan, with measurements.

---

## Background: a short glossary

A few ML terms used throughout. If these are already familiar, skip ahead.

- **Tensor**: a multi-dimensional array. "Tensor of shape `[B, 64, 14, 14]`" = a 4D array with axes (batch, channels, rows, columns). Same idea as a numpy array; the name is just an ML convention.
- **Channels**: the second axis of a tensor in an image-like representation. Think of an RGB image as a tensor `[3, H, W]` — channel 0 is red, channel 1 is green, channel 2 is blue. For us each input "channel" is a 14×14 plane describing one aspect of the game state (which cells contain white's piece-1, which contain white's piece-2, etc.).
- **Layer**: a learnable function from one tensor to another. Examples: `Linear`, `Conv2d`, `BatchNorm`. The network is layers composed.
- **Activation function**: a fixed (non-learnable) non-linearity inserted between layers. We use **ReLU** = `max(0, x)`. Without these the whole network would just be a giant linear function and be no more expressive than one matrix multiplication; the non-linearities are what make stacked layers more expressive than a single layer.
- **Batch dimension** / `B`: deep nets process many inputs at once for efficiency. The first axis of every tensor is "which input in this batch." During inference `B=1`; during training `B=batch_size`. From the network's perspective each batch element is independent.
- **Logits**: raw (unnormalised) numbers the network outputs before the final softmax/sigmoid step. Logits can be any real number; softmax turns them into probabilities summing to 1.
- **Receptive field**: for a given output value somewhere in the network, the receptive field is the set of input pixels that can influence that output value. Bigger receptive field = each output "sees" more of the input.
- **Inductive bias**: the implicit assumption a network architecture makes about what kind of function it's trying to learn. Strong inductive bias = "this architecture can only represent a restricted family of functions, but those functions are the kind we expect to need." Weak inductive bias = "this architecture can represent almost any function" — at the cost of needing more data to find the right one. A physicist's analogy: a strong inductive bias is like assuming spherical symmetry to drastically simplify a problem; it works well when the assumption holds, badly when it doesn't.

---

## 1. Anatomy of the current Blokus Duo network

The architecture lives in `games/blokusduo/neuralnets/net.py`. Concretely:

```
INPUT [B, 44, 14, 14]
   │   (44 channels: 21 piece-planes per player + 2 aggregate planes; 14×14 board)
   │
   ▼  STEM
   │     Conv2d(44 → 64, kernel 3×3, padding 1)
   │     BatchNorm + ReLU
   │
[B, 64, 14, 14]
   │
   ▼  TRUNK
   │     4× ResidualBlocks. Each block:
   │       Conv2d(64 → 64, 3×3) → BatchNorm → ReLU
   │       Conv2d(64 → 64, 3×3) → BatchNorm
   │       add the block's input (the "residual" connection)
   │       ReLU
   │
[B, 64, 14, 14]   ← "features": 64 numbers per cell
   │
   ├──────────────────────────────────┬─────────────────────────────┐
   │                                                                │
   ▼  POLICY HEAD                                ▼  VALUE HEAD
   │    Conv2d(64 → 2, kernel 1×1)               │    Conv2d(64 → 1, kernel 1×1)
   │    BatchNorm + ReLU                         │    BatchNorm + ReLU
   │                                             │
[B, 2, 14, 14]                                [B, 1, 14, 14]
   │                                             │
   ▼  Flatten                                    ▼  Flatten
[B, 392]   (= 2·14·14)                       [B, 196]   (= 1·14·14)
   │                                             │
   ▼  Linear(392 → 17,837)   ← THIS              ▼  Linear(196 → 64) → ReLU
[B, 17,837]                                  [B, 64]
   │                                             │
   ▼  log_softmax                                ▼  Linear(64 → 1) → Tanh
 OUTPUT: policy                              [B, 1]
                                                 ▼
                                              OUTPUT: value ∈ [-1, +1]
```

Each "feature" in the `[B, 64, 14, 14]` trunk output is a 64-number summary of one of the 196 board cells, computed by the conv stack from a roughly 9×9 neighbourhood around that cell (each 3×3 conv extends the receptive field by 2, and we have 4 residual blocks × 2 convs each = 8 convs of size 3, plus the stem). So by the time we get to the policy head, "the features at cell (r, c)" already incorporate quite a lot of context from surrounding cells.

### Parameter accounting

How many learnable numbers in each piece?

| Layer / block | Formula | Params | Share of total |
|---|---|---|---|
| Stem conv | `44·64·3·3 + 64` | 25,408 | 0.3% |
| 4 residual blocks | `4·(2·(64·64·3·3 + 0) + small BN)` | ~295K | 4.0% |
| Value head total | (conv + BN + 2 small FCs) | ~12.7K | 0.2% |
| Policy 1×1 conv | `64·2 = 128` | 128 | < 0.01% |
| Policy BatchNorm | 4 params | 4 | < 0.01% |
| **Policy Linear(392 → 17,837)** | `392·17837 + 17837` | **7,009,641** | **95.4%** |
| **Total** | | **~7.35M** | 100% |

The trunk — the "actual reasoning" of the network — has under 5% of the parameters. The final `Linear(392 → 17,837)` is everything else.

---

## 2. What "FC" (fully-connected / linear) means

A fully-connected layer is the simplest learnable function from one vector to another. It's literally a matrix multiplication plus a bias vector:

> **output = W · input + b**

where `input` is a vector of length `M`, `W` is a learnable matrix of shape `(N, M)`, `b` is a learnable bias vector of length `N`, and `output` is a vector of length `N`.

In our case: `input` is the 392-element flattened feature tensor, `W` is shape `(17837, 392)`, `b` is length `17837`. Output is the 17,837 raw action logits.

Three things to internalise:

**a) Every output is computed from every input.** Output `j` = `b[j] + Σᵢ W[j, i] · input[i]`. There's no spatial structure — input position `i=37` and `i=38` are just two indices into a flat vector. The layer has no idea whether they're "next to each other" on the original board or completely unrelated.

**b) Every output has its own dedicated set of weights.** Output `j=100` and output `j=101` each have their own row of `W` — 392 weights each, totally independent. If they happen to correspond to similar real-world things ("piece L at cell (5,7)" vs "piece L at cell (5,8)"), the FC has no built-in awareness of that similarity. Any "these outputs should behave similarly" pattern has to be **learned from data** by having `W[100, :]` and `W[101, :]` end up close to each other through training.

**c) Maximally expressive but maximally generic.** An FC layer can represent any linear function from `M`-dim space to `N`-dim space. This includes useful structured functions, useless random functions, and everything in between. Training has to find the useful one inside this huge space.

The cost is `M × N + N` parameters. For our policy FC: `392 × 17,837 + 17,837 ≈ 7.0 million parameters`.

---

## 3. What "pure convolutional" means

A 2D convolutional layer is also a linear function from one tensor to another, but **constrained** to be translation-equivariant. It applies the **same** weights at every spatial position.

A 1×1 convolution from `C_in` input channels to `C_out` output channels has weights `W` of shape `(C_out, C_in)` — that's it. At every spatial position `(r, c)` independently, it computes:

> **output[k, r, c] = Σᵢ W[k, i] · input[i, r, c]**

The same matrix-multiply applied at every cell. The spatial axes `r, c` don't appear in the weights at all.

For our policy head, a conv-only design would be:

```python
self.policy_head = nn.Sequential(
    nn.Conv2d(64, 92, kernel_size=1, bias=False),  # 92 output channels
    # Now output shape is [B, 92, 14, 14] — 92 logits at every cell.
    nn.Flatten(),                                   # [B, 18,032]
)
```

92 = 91 piece-orientation channels + 1 pass channel. Output: `92 × 14 × 14 = 18,032` raw logits. (Slightly more than our actual 17,837, but the extra outputs are easily masked out or used to encode the pass action — design detail.)

**Param count:** `64 × 92 = 5,888`. Plus 0 if `bias=False`. **~1,200× less** than the FC.

What this constraint buys us:

**a) Spatial weight sharing.** The "rule" for computing the logit of "play piece-orientation `o` at cell `(r, c)`" is the same regardless of where on the board you are. The 64 features at that cell get the same linear treatment. This is a *strong assumption* — it says the function we want to learn doesn't care about *absolute* board position, only about local features.

**b) Spatial locality is encoded for free.** Adjacent cells produce adjacent outputs computed by the same weights — they're automatically "similar functions of similar features."

**c) Less expressive.** The conv layer can ONLY represent translation-equivariant linear functions. Some functions an FC could represent can't be represented by a conv. The question is whether the functions we actually need are in the conv's restricted family.

Linear-algebraically, a conv layer is an FC layer where the matrix `W` is constrained to have a very specific repeating structure (in 1D you'd call it Toeplitz; in 2D it's the natural 2D analogue). The "fewer parameters" come from sharing entries of `W` across many rows and columns — millions of weight slots in the unconstrained FC become repeated copies of the same ~6K underlying weights in the conv.

---

## 4. The two reference architectures

This isn't a question of "we did it wrong." Both designs are documented in DeepMind's reference papers, applied to two different games — and the choice of which to use depends on the **structure of the action space**.

### AlphaGo Zero (2017) — uses FC, like us

From Silver et al. 2017, *Mastering the game of Go without human knowledge*, Methods section:

> "The policy head applies the following modules: (1) A convolution of 2 filters of kernel size 1×1 with stride 1; (2) Batch normalisation; (3) A rectifier non-linearity; (4) **A fully connected linear layer** that outputs a vector of size 19² + 1 = 362 corresponding to logit probabilities for all intersections and the pass move."

Our policy head is exactly this design, scaled to a different board size and action count. So your choice was to follow the canonical reference. That's the responsible default.

For Go this is fine because:
- 19×19 = 361 cells, action space = 362 (one per cell + pass). Ratio ~1.
- Policy FC params = `2·361 × 362 ≈ 261K`. The full AGZ network is ~25M params, so the FC is ~1% — not dominant.

The FC is a sensible choice when the action space is roughly cell-count-sized.

### AlphaZero Chess (2018) — switches to fully convolutional

From Silver et al. 2018, *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* (DeepMind's follow-up):

> "We represent the policy `π(a|s)` by an 8×8×73 stack of planes encoding a probability distribution over 4,672 possible moves. Each of the 8×8 positions identifies the square from which to pick up a piece. The first 56 planes encode possible 'queen moves' for any piece … The final 8 planes encode possible underpromotions … "

There's no FC. The output is directly a tensor of shape `[73, 8, 8]` — 73 channels of 8×8 logits. Each (channel, cell) pair encodes a specific move type at a specific cell.

For Chess:
- 8×8 = 64 cells, action space = 4,672. Ratio = 73.
- Conv policy params ≈ `2 conv stages` × small. Total share of net is tiny.
- An equivalent FC would be ~`hidden_size × 4672` — comparable to our 7M problem.

The choice to drop FC in AZ Chess wasn't arbitrary. It was forced by the action space's structure: actions index `(move-type, cell)` pairs, which is naturally a multi-channel spatial output.

### Where Blokus fits

| | Cells | Actions | Ratio | Action structure | Reference head |
|---|---|---|---|---|---|
| Go (AGZ) | 361 | 362 | ~1 | one-per-cell | FC |
| **Blokus Duo** | **196** | **17,837** | **~91** | **(piece-orientation, cell)** | **arguable** |
| Chess (AZ) | 64 | 4,672 | 73 | (move-type, cell) | conv |

By action-space structure, Blokus is closer to Chess than to Go. Our 91 piece-orientations × 196 cells maps onto "91 channels of 14×14 logits, plus a pass" — same shape AZ Chess's policy head uses with 73 channels of 8×8 logits.

You took the Go reference because at the time, the AGZ paper was probably the canonical reference and the natural starting point for any AlphaZero-style work. The AZ Chess paper's architectural difference is real but quieter — it's in the appendix, easy to miss. Not a mistake; a missed nuance.

---

## 5. Does fewer parameters mean less learning?

This is the natural question. The intuition is "more parameters = more capacity = can learn more." That intuition is sometimes right but in this specific case it's misleading. Three reasons.

### Reason 1: most of the FC's capacity is wasted on what conv encodes for free

The FC has 17,837 output positions, each with 392 dedicated weights. Consider three of those outputs:
- A. "logit for piece L at orientation 3 at cell (5, 7)"
- B. "logit for piece L at orientation 3 at cell (5, 8)"
- C. "logit for piece M at orientation 1 at cell (5, 7)"

A and B are the same piece-orientation at adjacent cells. The "rule" the network needs to learn for them is likely very similar — both should look at features near the cell and produce a logit. But the FC's 392 weights for A and the FC's 392 weights for B are entirely independent. To make A and B behave consistently, training has to **separately discover** the same rule and write it into each output's row of `W`. Out of `2 × 392 = 784` weights for these two outputs, the "true" relationship can be captured by ~392 (the rule) plus a small spatial offset — the rest is redundant.

The conv head simply re-uses the same `64 × 92 = 5,888` weights for every cell. The "same rule applied everywhere" is encoded structurally, not learned.

For A vs C (same cell, different piece-orientation), the situations actually differ — different output channels need different weights — and the conv head correctly has different weights per channel. So the conv head encodes "outputs at different cells use the same weights, outputs at different channels use different weights." This matches the structure of our action space exactly.

### Reason 2: sample efficiency improves with a correct prior

Training a neural net is a search over the space of functions the architecture can represent. **Smaller, structured function classes need less data to search successfully.** This is the "no free lunch" theorem flipped: if you have prior information about the right answer, encoding it as a structural constraint lets you converge faster.

For board games, "the value of a move depends mostly on local board structure" is a very strong, very correct prior. The conv head bakes this in. The FC has to learn it. With limited training data (our self-play is constrained), the FC's broader function class is a liability, not an asset — most of the "extra functions" it can express are useless ones, and training spends time exploring them.

A useful physicist's analogy: imagine fitting an unknown function `f(x)`. If you know `f` is smooth, fitting a polynomial is much more sample-efficient than fitting an arbitrary lookup table. The lookup table has more "capacity" — it can fit any function — but it'll severely overfit on small data. The polynomial's restricted form encodes "smooth-ness" structurally and gets you the right answer faster. Same idea here.

### Reason 3: symmetries become free

Blokus has an 8-fold symmetry group (4 rotations × 2 mirrors). If you rotate the board 90°, the policy should rotate the same way. A conv head respects this for free — same weights everywhere means the rotated input naturally produces the rotated output. The FC, having no spatial weight sharing, has to *learn* to be approximately rotation-equivariant from data. That's wasted capacity, again.

There's a counter-argument worth flagging: **a conv head can only see local features for each output.** "The logit for cell (5, 7)" depends only on the features at (5, 7), which themselves depend on a ~9×9 neighbourhood after the trunk. If there's some genuinely long-range structure that the network needs to capture for a per-cell output, the conv head misses it. But empirically the trunk's receptive field is usually plenty for board games (Stockfish-NNUE and AZ both use similar trunk depths and similar policy-head locality).

**Net effect:** the conv head should train at least as well, probably better, on the same amount of self-play. It's not a "smaller and worse" trade-off; it's a "smaller and structurally-better-suited" trade-off.

---

## 6. What we'd measure if we did the switch

Not a recommendation to switch yet — but if we did, here are the comparisons that would matter:

| Metric | What we'd expect to see |
|---|---|
| Total parameter count | 7.3M → ~340K (22× shrink) |
| Disk size of a checkpoint | ~30 MB → ~1.5 MB |
| Forward-pass time | Similar (both are dominated by the trunk's conv operations) |
| Training-step time | Slightly faster (less to backprop through) |
| Sample efficiency (loss vs games) | Conv-only should converge faster per game |
| Final strength vs Pentobi (long-term) | Should be at least as good, plausibly better with same training budget |
| Symmetry diagnostic | Conv-only should be near-perfect; FC currently has measurable asymmetry |

Some of those are easy to measure (params, disk, step time). Some require a real training run to compare (sample efficiency, final strength). Any switch would want a clean ablation: same self-play data, same MCTS sim count, same number of training steps, only the policy head changing.

Important caveat: any switch invalidates prior trained checkpoints. The `best.pth.tar` we have from previous runs wouldn't load into the new architecture. We'd need to either re-train from scratch or write a one-time weight-transfer script (FC weights → conv-init averaged appropriately — non-trivial).

---

## 7. Why this finding deserves its own document

You asked where this should live. It's a research note because:

1. **It's a discovery, not a commitment.** We haven't decided to do anything. We've identified that a foundational architectural choice may not be optimal for our specific game.
2. **It cuts across multiple future plans.** If we ever decide to scale up training, switch to a bigger net, or measure properly against Pentobi, the policy head architecture matters for all of those.
3. **The reasoning needs to be preserved.** The "AGZ used FC, AZ Chess used conv, Blokus action space is Chess-like" chain of reasoning isn't obvious. Without writing it down, the next person (probably future-us) re-derives it slowly or doesn't at all.

It's specifically **not** a `plans/` document because it doesn't commit us to do anything. If we decide to act on it, we'd open a separate `policy-head-conv-rewrite.md` (or whatever) in `plans/` that handles the actual implementation work, the ablation experiment, the weight-transfer question, and so on.

---

## 8. Out of scope

This document does not:

- **Decide whether to switch.** That's a separate decision with its own trade-offs (re-training time, validation against Pentobi, what to do with existing runs and the immutable 2026-05-22 baseline).
- **Specify how to switch.** That would be a sub-plan with measurements + a step-by-step refactor.
- **Address other architectural questions** (net size, residual block count, training-time regularisation, etc.) — those are downstream concerns once we've decided what the policy head looks like.

---

## Further reading

- Silver et al. 2017, [*Mastering the game of Go without human knowledge*](https://www.nature.com/articles/nature24270). The AGZ paper. Methods section has the FC policy head described in detail.
- Silver et al. 2018, [*A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*](https://arxiv.org/abs/1712.01815). The AZ paper. Appendix describes the 8×8×73 conv policy head for Chess and explains the action encoding.
- *Deep Learning* (Goodfellow, Bengio, Courville), chapter 9 — the canonical textbook treatment of CNNs, weight sharing, and translation equivariance, written at a level a physicist will find approachable.
- Cohen & Welling, [*Group Equivariant Convolutional Networks*](https://arxiv.org/abs/1602.07576). For the deeper mathematical perspective on why weight sharing encodes symmetry (and how to extend it to symmetries beyond translation — relevant if we ever want to encode Blokus's 8-fold symmetry group structurally).
