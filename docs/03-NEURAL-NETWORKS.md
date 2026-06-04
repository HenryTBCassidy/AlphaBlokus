# AlphaBlokus — Neural Network Architectures

## Overview

AlphaBlokus uses two neural network architectures: a simple **4-layer CNN** for Tic-Tac-Toe (Phase 1 validation) and a configurable-depth **ResNet** for Blokus Duo (Phase 2+). Both share the same dual-head output design from the AlphaZero paper — a policy head predicting move probabilities and a value head estimating the game outcome.

**One network, two outputs: "where to move" and "who's winning."**

---

## Architecture Comparison

| Property | Tic-Tac-Toe CNN | Blokus ResNet |
|----------|----------------|---------------|
| Input dimensions | 2 × 3 × 3 (2-channel player split) | 44 × 14 × 14 (per-piece spatial planes) |
| Action space | 10 (9 squares + pass) | 17,837 (14×14×91 + pass) |
| Architecture | 4 Conv layers → FC | Conv → N ResNet blocks → heads |
| Residual connections | No | Yes |
| Configurable depth | No (fixed 4 layers) | Yes (1-8 residual blocks) |
| Configurable width | Yes (num_filters) | Yes (num_filters) |
| Dropout | Yes (configurable rate) | No |
| Policy head | FC trunk → `FC(512→10)` | **Fully-convolutional** (1×1 conv → per-orientation planes + pass head); legacy FC head selectable |
| Parameters (illustrative, 512ch) | ~8.1M | ~19M (4 blocks, conv head) |
| Source file | `tictactoe/neuralnets/net.py` | `blokusduo/neuralnets/net.py` |

> The illustrative parameter counts use 512 channels for comparison with AlphaZero. Production Blokus runs use much smaller nets (e.g. 64 filters × 4 blocks) — see the run configs.

---

## Blokus in Context: AlphaZero's Original Games

How does Blokus Duo stack up against the games AlphaZero was built for? This table compares the neural network architectures from the [AlphaZero paper](https://www.science.org/doi/10.1126/science.aar6404) (Chess and Go) against AlphaBlokus's planned configuration.

| Property | Chess (AlphaZero) | Go (AlphaZero) | Blokus Duo (Ours) |
|----------|-------------------|----------------|-------------------|
| Board size | 8 × 8 | 19 × 19 | 14 × 14 |
| Input encoding | 8 × 8 × 119 planes | 19 × 19 × 17 planes | 14 × 14 × 44 planes |
| Input features | 8 time steps of piece positions, castling rights, en passant, move counts | 8 time steps of stone positions + colour to play | 44 spatial planes: 21 per-piece binary planes per player + 2 aggregate occupancy planes |
| Action space | 4,672 (8×8×73) | 362 (19×19 + pass) | 17,837 (14×14×91 + pass) |
| Residual blocks | 20 | 20 | 1–8 (configurable) |
| Channels per layer | 256 | 256 | 512 |
| Parameters (approx.) | ~45M | ~24M | ~26M (4 blocks) |
| Policy head output | 4,672 logits | 362 logits | 17,837 logits |
| Value head output | Scalar ∈ [-1, 1] | Scalar ∈ [-1, 1] | Scalar ∈ [-1, 1] |
| Training hardware | 5,000 TPUs | 5,000 TPUs | Consumer GPU |
| Training time | 9 hours | 13 days | TBD |
| MCTS sims/move | 800 | 800 | 200+ (configurable) |

### What Stands Out

**Blokus has the largest action space by far.** At 17,837 possible actions, Blokus Duo's action space is ~4× Chess and ~49× Go. This is because every combination of grid position (196) × piece-orientation (91) is a distinct action. A naive fully-connected policy head (`FC(392 → 17,837)`) would be ~7M params — the single most expensive layer in the network. The **fully-convolutional policy head** (the default, see below) avoids this: a 1×1 convolution emits one logit plane per piece-orientation, so the head costs ~47K params instead of ~7M, with a stronger spatial inductive bias to boot.

**Go has a much deeper network for a smaller action space.** AlphaZero uses 20 residual blocks for both Chess and Go, versus our 1–8. The deeper architecture compensates for Go's enormous state space (19×19 board, ~10^170 legal positions) — the network needs more capacity to evaluate positions, even though the number of possible moves per turn is relatively small.

**AlphaZero uses richer input encodings.** Chess encodes the last 8 board positions (119 planes per position), giving the network temporal context about how the game has evolved. Our Blokus encoding uses 44 spatial planes (21 per-piece planes per player + 2 aggregate planes), following the AlphaZero convention of one plane per piece type per player. We don't encode move history — the network sees only the current position — but the per-piece decomposition gives convolutional filters rich structural information about which pieces have been placed and where.

**Compute budget is the real gap.** AlphaZero trained Chess on 5,000 first-generation TPUs for 9 hours (~44,000 TPU-hours). We're targeting a single consumer GPU. This means fewer MCTS simulations per move (200 vs 800), fewer self-play games per generation, and longer wall-clock training times. The configurable ResNet depth (1–8 blocks vs AlphaZero's fixed 20) is a concession to this — we start shallow and scale up as needed.

---

## Tic-Tac-Toe CNN (`AlphaTicTacToe`)

### Purpose

Validate the full AlphaZero pipeline on a solved game before tackling Blokus. The network is intentionally simple — the goal is correctness, not efficiency.

### Architecture Diagram

```
Input: 2 × 3 × 3 (my stones / opponent stones)
    │
    ▼
┌─────────────────────────────┐
│  Conv2d(2→C, 3×3, pad=1)   │  3×3 → 3×3
│  BatchNorm2d → ReLU         │
├─────────────────────────────┤
│  Conv2d(C→C, 3×3, pad=1)   │  3×3 → 3×3
│  BatchNorm2d → ReLU         │
├─────────────────────────────┤
│  Conv2d(C→C, 3×3, pad=1)   │  3×3 → 3×3
│  BatchNorm2d → ReLU         │
├─────────────────────────────┤
│  Conv2d(C→C, 3×3, pad=0)   │  3×3 → 1×1  (no padding!)
│  BatchNorm2d → ReLU         │
└─────────────┬───────────────┘
              │ Flatten: C × 1 × 1 = C
              ▼
       ┌──────┴──────┐
       │   Shared    │
       │  FC Trunk   │
       │             │
       │  FC(C→1024) │
       │  BN → ReLU  │
       │  Dropout(p) │
       │             │
       │  FC(1024→512)│
       │  BN → ReLU  │
       │  Dropout(p) │
       └──────┬──────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌─────────┐     ┌──────────┐
│ Policy  │     │  Value   │
│ Head    │     │  Head    │
│         │     │          │
│ FC(512  │     │ FC(512   │
│   →10)  │     │   →1)    │
│         │     │          │
│ log_    │     │ tanh     │
│ softmax │     │          │
└─────────┘     └──────────┘
     │                │
     ▼                ▼
  π ∈ ℝ¹⁰         v ∈ [-1,1]
```

### Key Design Notes

- **No residual connections** — unnecessary for a 3×3 board with ~4 layers
- **Aggressive spatial reduction** — the 4th conv layer uses no padding, collapsing 3×3 → 1×1. This only works because the input is already small
- **Dropout** — applied in the FC layers (configurable rate). Helps prevent overfitting on the tiny action space
- **Shared FC trunk** — both heads share features through two fully-connected layers before branching

### Forward Pass

```python
# Input: batch × 2 × 3 × 3       # batch_size * 2 * board_rows * board_cols
x = x.view(-1, 2, 3, 3)        # Add channel dim

# 4 conv layers (last one reduces spatial dims)
x = relu(bn1(conv1(x)))         # batch × C × 3 × 3
x = relu(bn2(conv2(x)))         # batch × C × 3 × 3
x = relu(bn3(conv3(x)))         # batch × C × 3 × 3
x = relu(bn4(conv4(x)))         # batch × C × 1 × 1

# Flatten and shared FC
x = x.view(-1, C * 1 * 1)      # batch × C
x = dropout(relu(fc_bn1(fc1(x))))  # batch × 1024
x = dropout(relu(fc_bn2(fc2(x))))  # batch × 512

# Dual heads
pi = log_softmax(fc3(x))        # batch × 10
v  = tanh(fc4(x))               # batch × 1
```

---

## Blokus Duo ResNet (`AlphaBlokusDuo`)

### Purpose

Learn to evaluate Blokus Duo positions and predict good moves from self-play. The ResNet architecture handles the much larger board and action space, with configurable depth to balance training speed against capacity.

### Architecture Diagram

```
Input: 44 × 14 × 14 (44-channel per-piece spatial planes)
    │
    ▼
┌─────────────────────────────────┐
│  Initial Conv Block             │
│  Conv2d(44→C, 3×3, pad=1,      │  14×14 → 14×14
│         bias=False)             │
│  BatchNorm2d(C) → ReLU         │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Residual Block 1               │
│  ┌───────────────────────────┐  │
│  │ Conv2d(C→C, 3×3, pad=1)  │  │
│  │ BatchNorm2d → ReLU       │  │
│  │ Conv2d(C→C, 3×3, pad=1)  │  │
│  │ BatchNorm2d              │  │
│  └───────────┬───────────────┘  │
│         out + residual          │
│              ReLU               │
├─────────────────────────────────┤
│  Residual Block 2               │
│  (same structure)               │
├─────────────────────────────────┤
│  ...                            │
├─────────────────────────────────┤
│  Residual Block N               │
│  (N = config.num_residual_blocks│
│   typically 1-8)                │
└─────────────┬───────────────────┘
              │ features: batch × C × 14 × 14
              │
     ┌────────┴────────┐
     ▼                 ▼
┌────────────────────┐ ┌───────────────┐
│  Policy Head       │ │  Value Head   │
│  (conv, default)   │ │               │
│                    │ │ Conv2d(C→1,   │
│ Conv2d(C→91, 1×1)  │ │   1×1)        │
│  → (B,91,14,14)    │ │ BN(1) → ReLU │
│ reshape + reindex  │ │ Flatten       │
│  → ActionCodec ord │ │ (1×14×14=196) │
│ ┌────────────────┐ │ │               │
│ │ pass head:     │ │ │ FC(196 → C)   │
│ │ AdaptiveAvgPool│ │ │ ReLU          │
│ │ → FC(C → 1)    │ │ │ FC(C → 1)     │
│ └────────────────┘ │ │ Tanh          │
│ concat → log_soft  │ │               │
└─────────┬──────────┘ └───────┬───────┘
          │                    │
          ▼                    ▼
    π ∈ ℝ¹⁷⁸³⁷            v ∈ [-1,1]
```

### Residual Block Detail

Each residual block follows the standard pre-activation pattern from He et al. (2015):

```
input ─────────────────────────────┐
  │                                │ (skip connection)
  ▼                                │
Conv2d(C→C, 3×3, pad=1, bias=F)   │
BatchNorm2d(C)                     │
ReLU                               │
Conv2d(C→C, 3×3, pad=1, bias=F)   │
BatchNorm2d(C)                     │
  │                                │
  └──────────── + ─────────────────┘
               │
             ReLU
               │
             output
```

- **No bias** in conv layers — BatchNorm absorbs the bias term, so including it would be redundant
- **Same spatial dimensions** — padding=1 with kernel=3 preserves 14×14 throughout
- **Identity skip connection** — no projection needed since channel count doesn't change

### Board Representation (44 × 14 × 14)

The neural network input is a 44-channel spatial tensor built by
`BlokusDuoBoard.as_multi_channel(current_player)`:

```
Channels  0-20:  Current player's 21 pieces (one 14×14 binary plane each)
                 Channel i = 1 where piece (i+1) sits, 0 elsewhere.
                 All-zero if that piece hasn't been played yet.
Channels 21-41:  Opponent's 21 pieces (same layout)
Channel  42:     Aggregate current player occupancy (union of 0-20)
Channel  43:     Aggregate opponent occupancy (union of 21-41)
```

Internally, the board stores a single `_piece_placement_board` (int8, 14×14)
where +piece_id = White, -piece_id = Black, 0 = empty. The `as_multi_channel`
method derives the 44-channel tensor from this signed array using a
perspective-relative encoding: `ppb * current_player > 0` selects "my" pieces,
`ppb * current_player < 0` selects "opponent" pieces.

Piece inventory is implicit: an all-zero plane means "not yet played."
No separate inventory vector or column-padding is needed.

**Canonical form:** `board.canonical(player)` negates the placement board and
swaps all per-player state (remaining pieces, placement caches). After
canonicalisation, `as_multi_channel(1)` always shows the current player's
pieces in channels 0-20.

**Design rationale:** Four encoding options were evaluated: column-padded single tensor (1×14×18), late fusion (2×14×14 + inventory vector), per-piece spatial planes (44×14×14), and FiLM conditioning. The per-piece approach was chosen because it follows the AlphaZero convention (one plane per piece type per player), gives conv filters access to piece identity and spatial position simultaneously, and implicitly encodes piece inventory without a separate input. The sparsity of per-piece planes (each piece covers 1-5 of 196 squares) is comparable to AlphaZero Chess (where per-piece planes are 1.6-3.1% dense) and is well-supported by the "Representation Matters" paper (2024) which found richer representations improve performance even when sparse. See `docs/plans/archive/board-encoding-options.md` for the full analysis of all four options.

### Action Space Encoding

The policy head outputs a probability distribution over 17,837 possible actions:

```
Action index = (row × 14 × 91) + (col × 91) + piece_orientation_id

Where:
  row ∈ [0, 13]           → 14 rows
  col ∈ [0, 13]           → 14 columns
  piece_orientation_id ∈ [0, 90]  → 91 unique piece-orientations
  + 1 pass action (index 17,836)

Total: 14 × 14 × 91 + 1 = 17,837
```

**91 piece-orientations** come from symmetry reduction:
- 21 pieces × 8 orientations (4 rotations × 2 flips) = 168 raw orientations
- After removing duplicates: 91 unique orientations (basis orientations)
- Examples: the 1-square piece has only 1 orientation; asymmetric pentominoes have all 8

The `PieceManager` uses an `OrientationCodec` to map between `(piece_id, orientation)` tuples and contiguous integer IDs for fast conversion in both directions.

### Output Heads

**Policy Head (`policy_head: "conv"`, the default — F4).**

A fully-convolutional head, which exploits the structure of the action space: every action is a `(grid cell, piece-orientation)` pair, so the natural output is one logit *per orientation per cell*.

1. 1×1 convolution `C → 91`, producing one logit plane per piece-orientation: `(B, 91, 14, 14)`.
2. Reshape (channel-major) and reindex through a precomputed permutation buffer so the flat vector lands in `ActionCodec` order. (The permutation reconciles the conv's `(orientation, row, col)` layout with the codec's `(y, x, orientation)` board-coordinate layout; pinned by a one-hot probe test against the real `ActionCodec.encode`.)
3. Pass action: a separate small head — `AdaptiveAvgPool2d(1) → Flatten → Linear(C → 1)` — emits a single pass logit rather than wasting a whole 14×14 plane on it.
4. Concatenate the `14×14×91` move logits with the pass logit → `(B, 17,837)`, then `log_softmax`.

This head costs ~47K params (at C=512) versus the ~7M of the fully-connected alternative — a ~150× reduction in that layer and a ~19× smaller checkpoint, while training cleanly and giving a stronger board-game inductive bias.

**Legacy fully-connected head (`policy_head: "fc"`).** The original head, kept selectable: `Conv2d(C→2, 1×1) → BN → ReLU → Flatten(392) → Linear(392 → 17,837)`. The two heads have **incompatible** `state_dict`s, so a checkpoint trained with one head can't be loaded into the other. Use `"fc"` only to load an old FC-head checkpoint.

Both heads output log-probabilities; during prediction `torch.exp(pi)` converts back to probabilities for MCTS.

**Value Head:**
1. 1×1 convolution reducing C channels → 1 channel
2. BatchNorm + ReLU
3. Flatten: 1 × 14 × 14 = 196
4. FC: 196 → C (compress to channel width)
5. ReLU
6. FC: C → 1
7. `tanh` → value in [-1, +1]

The value represents the expected outcome: +1 = current player wins, -1 = current player loses, 0 = even position.

---

## Loss Functions

Both architectures use the same loss functions from the AlphaZero paper:

### Policy Loss

```
L_π = KL(π_target ‖ π_predicted) = Σ π_target · (log π_target − log π_predicted),  batch-mean
```

Implemented as `F.kl_div(outputs, targets, reduction='batchmean')`, where `outputs` are the network's log-probabilities and `targets` are the MCTS visit distribution (normalised to sum to 1). KL divergence and cross-entropy differ only by the target's own entropy — a constant w.r.t. the network's parameters — so they produce **identical gradients**. KL is used because it reads as a true divergence (zero when the prediction matches the target) and is the natural pairing with a `log_softmax` output.

### Value Loss

```
L_v = 1/N × Σ (v_target - v_predicted)²
```

Mean squared error between the target value (actual game outcome: +1, -1, or 1e-4 for draws) and the network's value prediction.

### Total Loss

```
L = L_π + L_v
```

No weighting between the two losses. No explicit regularisation term (weight decay or L2 penalty) — this is noted as a potential improvement.

---

## Training Process

### Data Flow

```
Self-play episodes (MCTS + current network)
        │
        ▼
Training examples: [(board, π_target, v_target), ...]
        │
        ▼
Random batch sampling (batch_size examples per step)
        │
        ▼
Forward pass → L_π + L_v
        │
        ▼
Backward pass → Adam optimiser update
        │
        ▼
Repeat for num_epochs
```

### Training Loop Detail

1. **Optimiser:** Adam with configurable learning rate. Optional LR schedule via `lr_scheduler` config — `null` (constant, default) or `"cosine"` (`CosineAnnealingLR` with `T_max = num_generations × epochs`).
2. **Batching:** `DataLoader` with `shuffle=True` over the cumulative training pool, `batch_size` per step
3. **Epochs:** Configurable number of passes over the data per generation
4. **CUDA support:** GPU transfer when `cuda=True` in config; optional fp16 autocast on the forward pass when `fp16_inference=True` (inference only — training stays fp32)
5. **Logging:** Per-batch pi_loss, v_loss, total_loss to Parquet (+ W&B if configured); per-epoch held-out diagnostics (policy entropy, top-1/5 accuracy, value calibration) when an eval set is supplied
6. **Persistence:** All metrics written as hive-partitioned Parquet per generation (no pickle) — see [07-DATA-STORAGE.md](07-DATA-STORAGE.md)

### Prediction

```python
# 1. Multi-channel tensor from the (canonical) board, add batch dim
tensor = torch.tensor(board.as_multi_channel(1), dtype=torch.float32)
if self.net_config.cuda:
    tensor = tensor.contiguous().cuda()
tensor = tensor.unsqueeze(0)  # (C, H, W) → (1, C, H, W)

# 2. eval mode (disables dropout, uses running BN stats); no grad;
#    optional fp16 autocast on CUDA when fp16_inference is set
self.nnet.eval()
with torch.no_grad(), self._inference_autocast():
    pi, v = self.nnet(tensor)

# 3. exp() back to probabilities; .float() casts fp16 → fp32 if autocast ran
return torch.exp(pi).float().cpu().numpy()[0], v.float().cpu().numpy()[0]
```

`predict_batch` is the same forward pass with a real batch dimension — it stacks N boards into one `(N, C, H, W)` call. MCTS uses it to evaluate a whole batch of leaves in a single GPU round-trip (see [02-ALGORITHMS.md](02-ALGORITHMS.md)).

### Checkpointing

- **Save:** stores `state_dict` (weights) **and** `optimizer_state_dict`, plus `scheduler_state_dict` when a scheduler is configured — so Adam's momentum buffers and the LR schedule survive a resume
- **Load:** `torch.load(filepath, map_location=...)` with automatic CPU fallback when CUDA unavailable; restores optimiser/scheduler state if present in the file
- **Directory structure:** `{run_directory}/Nets/{filename}`
- **Naming convention:** `best.pth.tar` (current best), `temp.pth.tar` (candidate's pre-arena snapshot), `accepted_{gen}` / `rejected_{gen}` (per-generation outcomes), `elo_baseline.pth.tar` (frozen gen-0 anchor)

---

## Parameter Counts

### Tic-Tac-Toe CNN (C=512, dropout=0.3)

| Layer | Parameters |
|-------|-----------|
| conv1 (2→512, 3×3) | 9,216 |
| conv2 (512→512, 3×3) | 2,359,296 |
| conv3 (512→512, 3×3) | 2,359,296 |
| conv4 (512→512, 3×3) | 2,359,296 |
| bn1-bn4 | 4 × 1,024 = 4,096 |
| fc1 (512→1024) | 525,312 |
| fc2 (1024→512) | 524,800 |
| fc3 (512→10) - policy | 5,130 |
| fc4 (512→1) - value | 513 |
| fc_bn1, fc_bn2 | 3,072 |
| **Total** | **~8.1M** |

### Blokus ResNet (C=512, N=4 residual blocks, conv policy head)

| Layer | Parameters |
|-------|-----------|
| Initial conv (44→512, 3×3) + BN | 202,752 + 1,024 |
| Per residual block (2 convs + 2 BNs, no bias) | 2 × 512 × 512 × 9 + 2 × 1,024 = 4,720,640 |
| 4 residual blocks | 4 × 4,720,640 = 18,882,560 |
| Policy head — move conv (512→91, 1×1) + pass `Linear(512→1)` | 46,683 + 513 ≈ 47K |
| Value conv (512→1, 1×1) + BN | 512 + 2 |
| Value FC1 (196→512) + FC2 (512→1) | 100,864 + 513 |
| **Total** | **~19.2M** |

With the conv policy head, the network is **trunk-dominated**: the residual blocks (~18.9M) are essentially the entire parameter budget. The policy head — the single largest layer (~7M) under the legacy FC design — is now a rounding error (~47K). Swapping in the legacy FC head would push the total back to ~26M.

---

## Configuration

Both architectures are configured through `NetConfig`:

```python
@dataclass(frozen=True)
class NetConfig:
    learning_rate: float            # Adam LR
    dropout: float                  # Dropout rate (TTT only)
    epochs: int                     # Training epochs per generation
    batch_size: int                 # Batch size
    cuda: bool                      # GPU acceleration
    num_filters: int                # Conv channel width
    num_residual_blocks: int        # ResNet depth (Blokus only)
    lr_scheduler: str | None = None # None (constant) or "cosine"
    fp16_inference: bool = False    # fp16 autocast on the forward pass (CUDA only)
    policy_head: Literal["fc", "conv"] = "conv"  # Blokus policy head (TTT ignores it)
```

Concrete values live in the run configs (`run_configurations/*.json`) — they change often and are deliberately not duplicated here.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ResNet over plain CNN for Blokus | ResNet | Residual connections prevent degradation in deeper networks. Blokus needs more capacity than TTT. Matches AlphaZero paper architecture |
| Multi-channel input | 2 ch (TTT), 44 ch (Blokus) | One plane per piece per player, following AlphaZero convention. Richer information for conv filters |
| 14×14 input with per-piece planes | Spatial encoding | Piece inventory is implicit in the per-piece planes (all-zero = not played). No column padding needed |
| Fully-convolutional policy head (default) | Conv head | Action space is `(cell, orientation)` pairs, so a 1×1 conv to per-orientation planes is the natural read-out. ~150× fewer params than the FC head in that layer, ~19× smaller checkpoints, stronger spatial bias. Legacy FC head kept selectable via `policy_head="fc"` |
| KL-divergence policy loss | `F.kl_div(..., 'batchmean')` | Identical gradient to cross-entropy (differ only by the target's constant entropy); reads as a true divergence and pairs naturally with `log_softmax` |
| Adam optimiser, optional cosine schedule | `lr_scheduler` config | Constant LR by default; `"cosine"` (`CosineAnnealingLR`) available for longer runs |
| No weight decay / L2 | Simplicity | Following the minimal AlphaZero recipe first. Regularisation may improve generalisation — listed as open question |
| Checkpoint includes optimiser + scheduler state | Resumability | Adam momentum and the LR schedule survive a resume rather than resetting |

---

## Open Questions / Future Work

- **Weight decay:** L2 regularisation in the loss function to prevent overfitting (currently none)
- **fp16 *training*:** fp16 *inference* is implemented; full mixed-precision training (autocast + `GradScaler` on the backward pass) could roughly double training throughput but isn't done
- **LR warm-up:** the cosine schedule has no warm-up phase; a short warm-up could help stability in early generations
- **Batch normalisation momentum:** default PyTorch momentum (0.1) may not be optimal for small batch counts
- **Gradient clipping:** could prevent training instability in early generations
