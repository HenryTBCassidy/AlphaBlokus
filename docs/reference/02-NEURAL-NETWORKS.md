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
| Parameters (typical) | ~8.1M (512ch) | ~26M (512ch, 4 blocks) |
| Source file | `tictactoe/neuralnets/net.py` | `blokusduo/neuralnets/net.py` |

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

**Blokus has the largest action space by far.** At 17,837 possible actions, Blokus Duo's action space is ~4× Chess and ~49× Go. This is because every combination of grid position (196) × piece-orientation (91) is a distinct action. The policy head's final FC layer (392 → 17,837 ≈ 7M params) is a direct consequence — it's the single most expensive layer in the network.

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
┌───────────────┐ ┌───────────────┐
│  Policy Head  │ │  Value Head   │
│               │ │               │
│ Conv2d(C→2,   │ │ Conv2d(C→1,   │
│   1×1)        │ │   1×1)        │
│ BN(2) → ReLU │ │ BN(1) → ReLU │
│ Flatten       │ │ Flatten       │
│ (2×14×14=392) │ │ (1×14×14=196) │
│               │ │               │
│ FC(392 →      │ │ FC(196 → C)   │
│   17837)      │ │ ReLU          │
│               │ │ FC(C → 1)     │
│ log_softmax   │ │ Tanh          │
└───────┬───────┘ └───────┬───────┘
        │                 │
        ▼                 ▼
  π ∈ ℝ¹⁷⁸³⁷         v ∈ [-1,1]
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

The `PieceManager` maintains a `BidirectionalDict` mapping between `(piece_id, orientation)` tuples and integer IDs for fast conversion in both directions.

### Output Heads

**Policy Head:**
1. 1×1 convolution reducing C channels → 2 channels (spatial features extraction)
2. BatchNorm + ReLU
3. Flatten: 2 × 14 × 14 = 392
4. Linear projection: 392 → 17,837
5. `log_softmax` for numerical stability during training

The output is log-probabilities. During prediction, `torch.exp(pi)` converts back to probabilities for MCTS.

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
L_π = -1/N × Σ π_target · log(π_predicted)
```

This is the cross-entropy between the MCTS-derived target policy and the network's predicted policy. The target policy comes from MCTS visit counts (normalised to sum to 1). The network outputs log-probabilities directly, so the loss is just the negative dot product averaged over the batch.

**Implementation note:** The current code computes `-torch.sum(targets * outputs) / targets.size()[0]` which is equivalent to cross-entropy when `outputs` are log-probabilities. There's a TODO to switch to `torch.nn.CrossEntropyLoss` for clarity.

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

1. **Optimiser:** Adam with configurable learning rate (no schedule currently — noted as future work)
2. **Batching:** Random sampling with replacement from the training pool
3. **Epochs:** Configurable number of passes over the data per generation
4. **CUDA support:** Automatic GPU transfer when `cuda=True` in config
5. **Logging:** Per-batch pi_loss, v_loss, total_loss, and running averages
6. **Persistence:** Training metrics saved as pickle per generation, consolidated to Parquet at end of run

### Prediction

```python
# 1. Get multi-channel tensor from board
tensor = board.as_multi_channel(1)
tensor = torch.FloatTensor(tensor.astype(np.float64))

# 2. Add batch dimension
tensor = tensor.unsqueeze(0)  # (C, H, W) → (1, C, H, W)

# 3. Set network to eval mode (disables dropout, uses running BN stats)
self.nnet.eval()

# 4. No gradient computation needed for inference
with torch.no_grad():
    pi, v = self.nnet(tensor)

# 5. Convert log-probabilities back to probabilities
return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
```

### Checkpointing

- **Save:** `torch.save({'state_dict': nnet.state_dict()}, filepath)` — saves only model weights (not optimiser state)
- **Load:** `torch.load(filepath, map_location=...)` with automatic CPU fallback when CUDA unavailable
- **Directory structure:** `{run_directory}/Nets/{filename}`
- **Naming convention:** `best.pth.tar` for the current best network, `temp.pth.tar` for the candidate during arena evaluation

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

### Blokus ResNet (C=512, N=4 residual blocks)

| Layer | Parameters |
|-------|-----------|
| Initial conv (44→512, 3×3) + BN | 202,752 + 1,024 |
| Per residual block (2 convs + 2 BNs, no bias) | 2 × 512 × 512 × 9 + 2 × 1,024 = 4,720,640 |
| 4 residual blocks | 4 × 4,720,640 = 18,882,560 |
| Policy conv (512→2, 1×1) + BN | 1,024 + 4 |
| Policy FC (392→17,837) | 6,991,904 + 17,837 (bias) ≈ 7.0M |
| Value conv (512→1, 1×1) + BN | 512 + 2 |
| Value FC1 (196→512) + FC2 (512→1) | 100,352 + 513 |
| **Total** | **~26M** |

The policy head's final FC layer (392 → 17,837 ≈ 7M params) is the single largest layer due to Blokus's massive action space.

---

## Configuration

Both architectures are configured through `NetConfig`:

```python
@dataclass(frozen=True)
class NetConfig:
    learning_rate: float     # Adam LR (e.g., 0.001)
    dropout: float           # Dropout rate (TTT only, e.g., 0.3)
    epochs: int              # Training epochs per generation (e.g., 10)
    batch_size: int          # Batch size (e.g., 64)
    cuda: bool               # GPU acceleration
    num_filters: int         # Conv channel width (e.g., 512)
    num_residual_blocks: int # ResNet depth (Blokus only, e.g., 4)
```

### Typical Configurations

| Parameter | Tic-Tac-Toe (test) | Blokus (full) |
|-----------|-------------------|---------------|
| learning_rate | 0.001 | 0.001 |
| dropout | 0.3 | 0.3 |
| epochs | 10 | 10 |
| batch_size | 64 | 64 |
| cuda | false | true |
| num_filters | 512 | 512 |
| num_residual_blocks | 1 | 4 |

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ResNet over plain CNN for Blokus | ResNet | Residual connections prevent degradation in deeper networks. Blokus needs more capacity than TTT. Matches AlphaZero paper architecture |
| Multi-channel input | 2 ch (TTT), 44 ch (Blokus) | One plane per piece per player, following AlphaZero convention. Richer information for conv filters |
| 14×14 input with per-piece planes | Spatial encoding | Piece inventory is implicit in the per-piece planes (all-zero = not played). No column padding needed |
| log_softmax + manual cross-entropy | Performance | Numerically stable when combined with the target × log(pred) loss formula. Could switch to torch.nn.CrossEntropyLoss for clarity |
| Adam optimiser (no schedule) | Simplicity | Works well out of the box. Learning rate scheduling is listed as future work |
| No weight decay / L2 | Simplicity | Following the minimal AlphaZero recipe first. Regularisation may improve generalisation — listed as open question |
| No optimiser state in checkpoint | Simplicity | Means training restarts from fresh optimiser state each generation. Acceptable because each generation trains from scratch on cumulative data |

---

## Open Questions / Future Work

- **Learning rate scheduling:** Warm-up then cosine/step decay could improve training stability, especially for longer runs
- **Weight decay:** L2 regularisation in the loss function to prevent overfitting
- **Mixed precision training (FP16):** Could roughly double training throughput on modern GPUs
- **Batch normalisation momentum:** Default PyTorch momentum (0.1) may not be optimal for small batch counts
- **Gradient clipping:** Could prevent training instability in early generations
- **Cross-entropy loss refactor:** Switch from manual `targets * outputs` to `torch.nn.CrossEntropyLoss` for clarity
- **Optimiser state persistence:** Saving Adam's momentum buffers across generations for smoother training
