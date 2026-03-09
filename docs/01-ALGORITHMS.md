# AlphaBlokus — Algorithms

## Overview

AlphaBlokus implements three tightly coupled algorithms that form the AlphaZero training loop:

1. **Monte Carlo Tree Search (MCTS)** — Explores the game tree guided by a neural network
2. **Self-Play** — Generates training data by having the network play against itself
3. **Arena Evaluation** — Determines whether a newly trained network is stronger than the previous best

These are orchestrated by the **Coach**, which runs the full loop: self-play → training → arena → accept/reject.

```
┌─────────────────────────────────────────────────────────────┐
│                     Coach.learn()                           │
│                                                             │
│   ┌──────────┐     ┌──────────┐     ┌──────────────────┐   │
│   │Self-Play │────>│ Training │────>│Arena Evaluation   │   │
│   │(MCTS +   │     │(gradient │     │(new net vs old    │   │
│   │ network) │     │ descent) │     │ net, both w/MCTS) │   │
│   └──────────┘     └──────────┘     └────────┬─────────┘   │
│        ▲                                      │             │
│        │           ┌──────────┐               │             │
│        │           │  Accept  │◄──── win% ≥ threshold?     │
│        │           │  new net │               │             │
│        │           └────┬─────┘               │             │
│        │                │                     │             │
│        │           ┌────┴─────┐          ┌────┴─────┐      │
│        └───────────│ Save as  │          │  Reject  │      │
│                    │  "best"  │          │  revert  │      │
│                    └──────────┘          └──────────┘      │
│                                                             │
│   Repeat for N generations                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Monte Carlo Tree Search (MCTS)

### Core Idea

MCTS builds a search tree by running many simulations from the current game state. Each simulation walks down the tree, expanding a new node when it reaches an unexplored position, and evaluating it with the neural network. The visit counts at the root node then determine the move to play.

### Implementation (`core/mcts.py`)

The MCTS tree is stored as **6 flat dictionaries** keyed by string representations of game states:

| Dictionary | Key | Value | Purpose |
|------------|-----|-------|---------|
| `Qsa` | `(state, action)` | float | Q-value (average value of taking action `a` in state `s`) |
| `Nsa` | `(state, action)` | int | Visit count for `(state, action)` pair |
| `Ns` | `state` | int | Total visit count for state `s` |
| `Ps` | `state` | array | Neural network policy prediction for state `s` |
| `Es` | `state` | float | Game-ended status cache (0 if not ended) |
| `Vs` | `state` | array | Valid moves mask for state `s` |

### PUCT Formula

At each node, MCTS selects the action with the highest Upper Confidence Bound:

```
best_action = argmax_a [ Q(s,a) + U(s,a) ]

where:
  U(s,a) = cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

  Q(s,a) = average value from simulations through (s,a)
  P(s,a) = neural network's prior probability for action a
  N(s)   = total visit count for state s
  N(s,a) = visit count for (state, action) pair
  cpuct  = exploration constant (default: 1.0)
```

**Intuition:**
- **Q(s,a)** exploits known good moves (high average value)
- **U(s,a)** explores under-visited moves, weighted by the network's prior
- As N(s,a) grows, U(s,a) shrinks — the algorithm naturally shifts from exploration to exploitation
- `cpuct` controls the exploration/exploitation trade-off

### Single Simulation Walkthrough

```
def search(state):
    1. CHECK TERMINAL
       If game_ended(state) != 0:
           return -game_ended(state)  # negate because of alternating players

    2. EXPAND LEAF NODE
       If state not in Ps (never visited):
           - Run neural network: policy, value = nnet.predict(state)
           - Mask invalid moves from policy
           - Renormalise policy (or set uniform if all invalid)
           - Store: Ps[s] = policy, Vs[s] = valid_moves, Ns[s] = 0
           - Return -value  # negate for alternating perspective

    3. SELECT BEST ACTION
       For each valid action a:
           if (s,a) visited before:
               u = Q(s,a) + cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
           else:
               u = cpuct * P(s,a) * sqrt(N(s) + EPS)  # encourage first visit
       best_action = argmax(u)

    4. RECURSE
       next_state, next_player = game.get_next_state(state, player, best_action)
       canonical_state = game.get_canonical_form(next_state, next_player)
       value = search(canonical_state)

    5. BACKUP
       Q(s,a) = (N(s,a) * Q(s,a) + value) / (N(s,a) + 1)  # running average
       N(s,a) += 1
       N(s) += 1
       return -value  # negate for alternating perspective
```

### Move Selection from Root

After running `num_mcts_sims` simulations, the root node visit counts determine the move:

```python
def get_action_probabilities(state, temperature):
    for _ in range(num_mcts_sims):
        search(state)

    counts = [Nsa[(s, a)] for a in range(action_size)]

    if temperature == 0:
        # Exploitation: pick the most-visited move
        best = argmax(counts)
        probs = [0] * action_size
        probs[best] = 1
    else:
        # Exploration: sample proportional to visit counts^(1/temp)
        counts = [c ** (1.0 / temperature) for c in counts]
        total = sum(counts)
        probs = [c / total for c in counts]

    return probs
```

### Temperature Schedule

- **First `temp_threshold` moves:** temperature = 1 (explore, build diverse training data)
- **After threshold:** temperature → 0 (exploit, play strongest move)

This produces varied opening play while maintaining strong endgame play.

### Known Performance Issues for Blokus

**1. Action space iteration (CRITICAL):**
The current implementation iterates over ALL actions to find the best PUCT score:

```python
# Current: O(17,837) per node visit
for a in range(game.get_action_size()):
    if valid_moves[a]:
        # compute PUCT score
```

For Blokus (17,837 actions), this is prohibitively slow. Typical positions have 50-300 valid moves, meaning >98% of iterations are wasted checking invalid actions.

**Fix:** Maintain a list of valid action indices and iterate only those:
```python
# Target: O(num_valid_moves) per node visit
valid_indices = np.nonzero(valid_moves)[0]
for a in valid_indices:
    # compute PUCT score
```

**2. Tree reconstruction:**
A fresh MCTS tree is built for each self-play game. Between moves within a game, the subtree rooted at the chosen action could be reused.

**3. String-based state keys:**
States are converted to strings for dictionary keys via `board.tostring()`. For 14x14 boards, this creates 784-byte keys on every lookup. A hash-based approach would be faster.

---

## MCTS Caching & Optimisation Strategies

The current MCTS implementation has several opportunities for performance improvement that become critical at Blokus scale.

### Valid Move Caching

The Blokus game logic already maintains a **placement point cache** — after each piece insertion, it updates the set of valid corner positions where future pieces can be placed. This avoids recomputing all valid moves from scratch.

The caching strategy (designed but not yet fully implemented) works in layers:

```
Layer 1: Placement Points
   - Track all valid corner positions for each player
   - Updated incrementally after each piece placement
   - Already partially implemented in Player class

Layer 2: Per-Point Move Generation
   - For each placement point, enumerate all piece-orientation combos that can
     be legally placed touching that corner
   - Cache result: invalidate only when nearby squares change
   - populate_moves_for_new_placement() — the critical unimplemented function

Layer 3: Aggregate Valid Moves
   - Union of all per-point valid moves = total valid moves for the position
   - valid_moves() aggregates from all placement points
```

### Initial Move Pre-computation

First moves for both players are **pre-computed and cached** at game initialisation. Since the first move must cover a specific starting square (White: (9,9), Black: (4,4)), the set of valid first moves is fixed and can be computed once.

### Transposition Table

Different move sequences can reach the same board position in Blokus (transpositions). A transposition table would recognise these and reuse MCTS evaluations:

```
Approach:
  - Hash board state (Zobrist hashing: XOR of random values per piece-position)
  - Before expanding a node, check if this state exists in the table
  - If found: reuse P(s), V(s), Q(s,a), N(s,a)
  - Trade-off: memory usage vs computation savings
```

**Estimated impact:** Moderate. Blokus has fewer transpositions than Chess (piece placement is permanent), but they do occur when different piece-ordering sequences produce the same board.

### Subtree Reuse Between Moves

After player A makes a move, the MCTS tree from A's perspective has a subtree rooted at the chosen action that remains valid for the next search. Currently, this entire tree is discarded.

```
Before move:
        root
       / | \
      a   b   c    ← player A's moves
     /|\  ...
    x  y  z         ← player B's responses (still valid!)

After choosing move 'a':
    Current:  rebuild entire tree from scratch
    Better:   promote subtree under 'a' as new root
```

**Estimated impact:** Significant. Saves ~30-50% of MCTS simulations per move in the mid-game.

### Batched Neural Network Inference

The single biggest bottleneck in MCTS is neural network evaluation — one forward pass per leaf node expansion. Batching multiple leaf evaluations into a single GPU forward pass dramatically improves throughput.

**Approaches:**

1. **Root parallelisation:** Run N independent MCTS trees in parallel, batch their leaf evaluations
2. **Leaf parallelisation:** Within a single tree, collect multiple leaves before evaluating, then backpropagate all at once
3. **Virtual loss:** During parallel search, add a "virtual loss" to nodes being evaluated to encourage exploration of different paths

**Estimated impact:** 4-16x speedup depending on batch size and GPU utilisation.

---

## Self-Play (`core/coach.py`)

### How Self-Play Works

Each generation, the Coach runs `num_eps` complete games of self-play:

```python
def execute_episode(game, nnet):
    """Play one complete game, return training examples."""
    board = game.initialise_board()
    player = 1  # white starts
    episode_step = 0
    train_examples = []

    while True:
        episode_step += 1
        canonical = game.get_canonical_form(board, player)

        # Temperature: explore early, exploit late
        temp = 1 if episode_step < temp_threshold else 0

        # MCTS produces move probabilities
        pi = mcts.get_action_probabilities(canonical, temp)

        # Augment with symmetries for training data
        symmetries = game.get_symmetries(canonical, pi)
        for sym_board, sym_pi in symmetries:
            train_examples.append([sym_board, player, sym_pi, None])

        # Sample move from probabilities
        action = np.random.choice(len(pi), p=pi)
        board, player = game.get_next_state(board, player, action)

        # Check game end
        result = game.get_game_ended(board, player)
        if result != 0:
            # Assign values: +1 for winner's positions, -1 for loser's
            return [(x[0], x[2], result * ((-1) ** (x[1] != player)))
                    for x in train_examples]
```

### Training Data Format

Each training example is a tuple of:
- **Board state** (canonical form — always from current player's perspective)
- **Policy vector** (MCTS visit count distribution — the "correct" move probabilities)
- **Value** (+1 if current player eventually won, -1 if lost)

### Symmetry Augmentation

For games with symmetries (rotations, reflections), each position generates multiple training examples. This is free data augmentation:

- **Tic-Tac-Toe:** 8 symmetries (4 rotations × 2 reflections)
- **Blokus Duo:** Not yet implemented. The 14x14 board has no rotational symmetry (starting squares break it), but 180-degree rotation with colour swap is valid. Needs careful analysis.

### Training Window

Not all historical self-play data is used for training. A sliding window controls how many generations of data to include:

```python
def window_size(generation, max_lookback):
    """Linearly grow the training window from 5 to max_lookback."""
    return min(5 + generation, max_lookback)
```

**Rationale:** Early generations produce low-quality data (random play). As the network improves, older data becomes less representative. The growing window balances:
- **Recency:** Prioritise recent, higher-quality games
- **Diversity:** Include enough data to prevent overfitting to recent patterns

---

## Arena Evaluation (`core/arena.py`)

### Purpose

After training, the new network must prove it's stronger than the previous best. The Arena plays them head-to-head.

### Protocol

```python
def play_games(num_games):
    """
    Play num_games between new net and old net.
    Each pair of games alternates who plays first.
    Both players use MCTS with temperature=0 (pure exploitation).
    """
    new_wins, old_wins, draws = 0, 0, 0

    for i in range(num_games):
        if i % 2 == 0:
            result = play_one_game(player1=new_net, player2=old_net)
        else:
            result = play_one_game(player1=old_net, player2=new_net)

        # Tally results...

    return new_wins, old_wins, draws
```

### Acceptance Criterion

```
accept = new_wins / (new_wins + old_wins) >= update_threshold
```

- **Default threshold:** 0.55 (new net must win 55%+ of decisive games)
- If accepted: save new net as "best", continue training from it
- If rejected: revert to old net, discard new weights

**Why threshold > 0.50?** A 50% threshold would accept networks that are merely equal. 55% provides statistical confidence that the new network is genuinely stronger, reducing noise in the training process.

### Alternating Start Positions

Games are played in pairs with swapped colours. This controls for first-move advantage:
- Game 1: new_net plays White, old_net plays Black
- Game 2: old_net plays White, new_net plays Black

If both players are equal, each should win one game regardless of first-move advantage.

### Arena MCTS Settings

Arena games use **temperature = 0** throughout (no exploration). This ensures the evaluation reflects pure playing strength, not randomness from temperature-based sampling.

---

## The Complete Training Loop

Putting it all together, one generation of training:

```
Generation N:

1. SELF-PLAY                          [~70% of time]
   ├── Create fresh MCTS tree
   ├── Play num_eps games (e.g. 100)
   │   ├── Each game: MCTS search → sample move → repeat until terminal
   │   └── Augment positions with symmetries
   ├── Store (board, policy, value) tuples
   └── Save to self_play_history_{N}.pickle

2. TRAINING                           [~15% of time]
   ├── Save current net as "temp" checkpoint
   ├── Combine examples from last window_size(N) generations
   ├── Shuffle all examples
   ├── Train for num_epochs
   │   ├── Each epoch: iterate batches
   │   │   ├── Forward pass: board → (log_policy, value)
   │   │   ├── Policy loss: -sum(target_pi * log_policy) / batch_size
   │   │   ├── Value loss: sum((target_v - pred_v)^2) / batch_size
   │   │   ├── Total loss: policy_loss + value_loss
   │   │   └── Backward pass + Adam step
   │   └── Log loss metrics per batch
   └── Save training metrics to pickle

3. ARENA EVALUATION                   [~15% of time]
   ├── Load "temp" checkpoint as old net
   ├── Play num_arena_matches (e.g. 100) with alternating starts
   ├── Both players use MCTS with temp=0
   ├── Count wins/losses/draws
   ├── If new_wins / (new_wins + old_wins) >= 0.55:
   │   ├── ACCEPT: save new net as "best"
   │   └── Save as accepted_{N}.pth.tar
   └── Else:
       ├── REJECT: revert to "temp" (old net)
       └── Save as rejected_{N}.pth.tar

4. REPORTING
   ├── Write timing data to parquet
   ├── Save arena results to parquet
   └── After all generations: generate HTML report
```

### Configuration Comparison

| Parameter | Test Run | Full Run |
|-----------|----------|----------|
| num_generations | 2 | 50 |
| num_eps (games per gen) | 10 | 100 |
| num_mcts_sims | 2 | 200 |
| num_arena_matches | 2 | 100 |
| batch_size | 10 | 32 |
| epochs | 1 | 5 |
| cpuct | 1 | 1 |
| learning_rate | 0.001 | 0.001 |
| update_threshold | 0.55 | 0.55 |

---

## AlphaZero Paper Reference

The original AlphaZero paper (Silver et al., 2018) provides key implementation details that guide this project:

- **MCTS simulations:** 800 per move in Chess, scaled by game complexity
- **cpuct:** Uses a formula that increases with the number of visits, not a fixed constant. We use a fixed constant (1.0) as a simplification
- **Dirichlet noise:** Added to root node priors for exploration: `P(s,a) = (1-ε)p_a + ε·η_a` where `η ~ Dir(α)`. Not yet implemented in AlphaBlokus
- **Temperature:** τ=1 for first 30 moves, τ→0 afterwards. Our `temp_threshold` serves the same purpose
- **Training:** 700,000 training steps with batch size 4096 for Chess. We use much smaller scale
- **Evaluation:** New network accepted if it wins 55%+ of 400 games. We use the same threshold with fewer games

**Paper:** Silver, D. et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science* 362.6419 (2018): 1140-1144.
