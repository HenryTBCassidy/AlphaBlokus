# AlphaBlokus — Algorithms

## Overview

AlphaBlokus implements the algorithms that form the AlphaZero training loop:

1. **Monte Carlo Tree Search (MCTS)** — explores the game tree guided by a neural network
2. **Self-Play** — generates training data by having the network play against itself
3. **Arena Evaluation** — decides whether a newly trained network is stronger than the previous best
4. **Strength evaluation** — measures the new network against fixed baselines (frozen gen-0, and a perfect-play minimax oracle for Tic-Tac-Toe)

These are orchestrated by the **Coach** (`core/coach.py`), which runs one generation at a time.

```
Coach.learn() — one generation:

  Self-Play ─────> Training ─────> Arena ─────> Strength eval
  (MCTS + net)     (SGD)           (new vs old)  (vs gen-0 / minimax)
                                        │
                                  score ≥ threshold?
                                  (wins + ½·draws) / games
                                    /          \
                                  Yes           No
                                   │             │
                                Accept         Reject
                              (save new)     (keep old)
                                   \           /
                                    ┗━━━━━━━━━┛
                                         │
                                   Next generation
```

---

## Monte Carlo Tree Search (MCTS)

### Core idea

MCTS builds a search tree by running many simulations from the current position. Each simulation descends the tree picking promising actions, stops when it reaches an unexpanded or terminal position, evaluates that leaf with the neural network, and propagates the value back up. After a fixed number of simulations, the **visit counts** at the root define the move-probability distribution.

### Tree storage (`core/mcts.py`)

The tree is stored implicitly as a set of dictionaries keyed by **board state** (`state_key` — for Blokus this is the 196-byte signed `int8` placement board; for TTT the flat board bytes). Because the keys are positions, not move sequences, transpositions are handled for free.

| Dictionary | Key | Value | Purpose |
|------------|-----|-------|---------|
| `q_values` | `(state, action)` | float | Mean value of taking `a` in `s` |
| `visit_counts` | `(state, action)` | int | Visits to the `(s, a)` edge |
| `state_visits` | `state` | int | Total visits to `s` |
| `policy_priors` | `state` | array | Network policy for `s`, masked to legal moves and renormalised |
| `game_ended_cache` | `state` | float | Cached game-ended result (0 = ongoing) |
| `valid_moves_cache` | `state` | array | Cached legal-move mask for `s` |
| `virtual_visits` | `(state, action)` | int | In-flight virtual loss during a batch (empty between moves) |

### PUCT formula

At each expanded node the search picks the action maximising the PUCT score:

```
best_action = argmax_a [ Q(s,a) + U(s,a) ]

  U(s,a) = cpuct · P(s,a) · √N(s) / (1 + N(s,a))

  Q(s,a) = mean value of simulations through (s,a)   (0 for an unvisited edge)
  P(s,a) = network's prior probability for action a
  N(s)   = total visits to state s
  N(s,a) = visits to the (s, a) edge
  cpuct  = exploration constant (set per run in config)
```

`Q(s,a)` exploits moves with high observed value; `U(s,a)` explores under-visited moves weighted by the network's prior, and shrinks as `N(s,a)` grows — so the search shifts from exploration to exploitation on its own. Only **legal** actions are scored: the loop iterates `np.where(valids)[0]`, never the full 17,837-wide space.

### Dirichlet root noise

To guarantee the search explores moves the (possibly wrong, early-training) policy rates poorly, AlphaZero mixes Dirichlet noise into the **root** priors during **self-play only**:

```
P(s,a) = (1 − ε)·p_a + ε·η_a ,   η ~ Dir(α)   over the legal moves at the root
```

- Applied once per move, before the simulations, so every descent that move sees the noised priors.
- Gated by `MCTSConfig.dirichlet_epsilon`. At `ε = 0` (the default) the noise codepath is a no-op and the search is bit-identical to noise-free behaviour — which is why **arena and Elo evaluation never request it** (they pass `add_root_noise=False`), keeping evaluation deterministic.
- It draws from the episode's seeded RNG, so noised self-play stays reproducible.
- AlphaZero used `ε = 0.25`; `α ≈ 0.03` suits Blokus's few-hundred early legal moves (Go-like).

### How a simulation batch runs

Simulations are run in batches of `K = mcts_batch_size` (the F-series batched-inference optimisation; see [What makes MCTS fast](#what-makes-mcts-fast)). One batch of size `K`:

1. **Descend `K` times.** Each descent walks from the root picking the PUCT-best action until it reaches a terminal or unexpanded leaf. As it traverses an edge it deposits a **virtual loss** on that edge (counts as `VIRTUAL_LOSS = 3` extra visits each worth value −1), which depresses that edge's PUCT score so the next descent in the same batch is steered down a *different* path. This keeps the `K` collected leaves diverse.
2. **Evaluate the distinct leaves once.** The unexpanded leaves are de-duplicated and sent through a single batched `predict_batch` call — one GPU forward pass for the whole batch instead of `K` separate calls. Terminal leaves need no network call (their value is the cached game result).
3. **Expand** each freshly evaluated leaf: mask its priors to legal moves, renormalise, initialise its node stats.
4. **Back-propagate** every descent's leaf value up its path (running-mean `Q` update, sign flipped each ply), then **remove** the virtual loss it deposited, so `virtual_visits` returns to empty.

At `K = 1` (the default) a single descent deposits virtual loss that affects no other descent and is removed immediately — so the arithmetic is exactly the classic one-leaf-per-NN-call recursive search. `K > 1` trades a slight search approximation for far better GPU utilisation.

### Move selection from the root

After `num_mcts_sims` simulations, the root visit counts define the policy:

```python
counts = [visit_counts.get((root, a), 0) for a in range(action_size)]

if temperature == 0:           # exploitation: most-visited move (ties broken at random)
    probs = one_hot(argmax(counts))
else:                          # exploration: sample ∝ counts^(1/temperature)
    counts = [c ** (1 / temperature) for c in counts]
    probs  = counts / sum(counts)
```

The raw (pre-temperature) visit distribution's entropy is also recorded per move as a search-confidence diagnostic.

### Temperature schedule

- **First `temp_threshold` moves:** temperature = 1 — sample proportional to visit counts, producing varied openings and diverse training data.
- **After the threshold:** temperature → 0 — play the most-visited (strongest) move.

---

## What makes MCTS fast

The bottleneck in AlphaZero-style search is twofold: legal-move generation (CPU, Python) and neural-net evaluation (GPU). AlphaBlokus attacks both, plus avoids redundant work across moves.

### 1. Precomputed legal-move tables (Pentobi-style)

The single biggest CPU win. Instead of re-deriving piece geometry on every `valid_move_masking` call, all of the static geometry — which `(piece, orientation, anchor)` placements exist, which cells each occupies, which it forbids, which it opens as new attach points — is enumerated **once at startup** into lookup tables. At runtime the only board-dependent input is a 196-byte `forbidden` array; move generation becomes "look up a short candidate list per attach point, then verify each candidate with a handful of byte reads."

This design is taken directly from **[Pentobi](https://github.com/enz/pentobi)** (Markus Enzenberger) — its precomputed `(anchor, adj_status, piece) → candidate move-id list` table is the heart of its speed, and notably uses **no bitboard**, just byte arrays plus precomputation. We reimplemented the same algorithm in Python (our own code, not a port). It is ~9× faster per call than the readable reference generator and produces bit-identical results. Full detail in [Move Generation](#move-generation-blokus-duo) below.

### 2. Batched neural-net inference with virtual loss

A batch-of-one forward pass barely uses the GPU. Collecting `K` leaves per outer step (diversified via virtual loss, see [How a simulation batch runs](#how-a-simulation-batch-runs)) and evaluating them in a single `predict_batch` call collapses `K` GPU round-trips into one. Identical leaves within a batch are de-duplicated so the network sees each unique position once. Combined with optional fp16 inference on CUDA, this is the main GPU-side speedup.

### 3. Implicit subtree reuse within a game

A fresh MCTS tree is created per game, but the **same tree is kept across every move of that game** — `play_self_play_episode` (and `NetworkPlayer` in evaluation) reuse one `MCTS` instance for the whole game. Because the tree is keyed by board state, all statistics for any position already explored remain available and are reused the moment a later search revisits that position. This gives the benefit usually described as "promote the chosen child to the new root" — without any explicit re-rooting step — and additionally reuses work across in-game transpositions. The tree is only discarded between games (via the `startGame` hook on `NetworkPlayer`).

### 4. Per-state caches inside the search

Within a search, `get_game_ended` results, legal-move masks, and masked priors are each computed once per state and cached (`game_ended_cache`, `valid_moves_cache`, `policy_priors`). Repeated descents through the same internal node pay nothing for them.

---

## Move Generation (Blokus Duo)

Legal-move generation is the per-game CPU bottleneck for Blokus. The codebase has **two implementations** that produce identical legal-move sets — a readable reference and an optimised fast path that is validated against it.

### The reference generator

The straightforward "generate and test" approach (`BlokusDuoGame._generate_valid_moves`). It centres on **placement points** (a.k.a. attach points): board cells diagonally adjacent to a friendly piece and not side-adjacent to one. By Blokus's corner-touch rule these are the only cells where a new piece can make contact, so they anchor the search.

```
_valid_moves(board, player):
    if first move:
        return initial_actions   # precomputed at game init; the fixed set covering the start square

    for each placement_point:
        for each remaining piece:
            for each orientation:
                for each filled cell of the piece:
                    slide the piece so that cell lands on the placement point
                    if it fits on-board AND every cell is empty AND no friendly side-adjacency:
                        yield Action
    deduplicate (a move can be reachable from several placement points)
```

A short-circuit variant `_has_valid_moves` returns `True` on the first move found — used by `get_game_ended` to detect a stuck player without enumerating everything.

The board maintains two caches that this path relies on, both updated incrementally after each placement: the **placement-point set** and a **side-danger zone** (a boolean 14×14 array marking cells orthogonally adjacent to a friendly piece, so the no-edge-touching rule is one lookup instead of four neighbour checks). Piece-orientation arrays are precomputed once by `PieceManager`.

### The optimised generator (Pentobi-style precomputed tables)

`games/blokusduo/movegen_tables.py` + `movegen_runtime.py`, enabled by the `use_optimised_movegen` config flag. This is the [Pentobi-derived](https://github.com/enz/pentobi) design credited above. Built once per process (~200 ms):

- **Geometry table** — every legal `(piece, orientation, anchor)` placement that fits on the board, numbered `move_id` 0..13,728 (**13,729** total, verified against Pentobi's hard-coded constant by `scripts/count_onboard_placements.py`). Each stores its occupied cells, the cells it forbids, the cells it opens as attach points, and its `action_id` in the 17,837-wide action space.
- **Lookup table** — indexed by `(anchor, adj_status, piece) → short list of candidate move-ids`, where `adj_status` is a 6-bit summary of which of 6 specific neighbour cells around the anchor are currently forbidden. This pre-filters out placements that can't fit given the local blocked cells.

At runtime the only per-board input is `forbidden = occupied OR side-danger`. For each attach point: read its 6 neighbours into a 6-bit `adj_status`, look up the candidate list per remaining piece, and confirm each candidate with an unrolled OR over its (≤5) cells' forbidden bytes (a `NULL_CELL` sentinel slot, always 0, lets the loop run a fixed 5 iterations without bounds checks). A `seen` set dedups moves reachable from multiple attach points.

**Equivalence is proven, not assumed:** a position-level test compares the two generators' legal-move *sets* across thousands of stratified positions, and a stronger determinism test confirms self-play at a fixed seed produces bit-identical training trajectories with the fast path on or off.

> Deferred further speedups (Cython, true bitboard) are tracked in [`docs/plans/move-gen-further-optimisation.md`](plans/move-gen-further-optimisation.md). They are not needed to train.

---

## Self-Play (`core/self_play.py`)

Each generation the Coach runs `num_eps` complete self-play games. One game (`play_self_play_episode`):

```python
def play_self_play_episode(game, mcts, temp_threshold):
    board, current_player, move_count = game.initialise_board(), 1, 0
    train_examples = []

    while True:
        move_count += 1
        canonical = game.get_canonical_form(board, current_player)
        temp = 1 if move_count < temp_threshold else 0

        # MCTS-improved policy; root Dirichlet noise requested (no-op unless ε > 0)
        pi = mcts.get_action_prob(canonical, temp=temp, add_root_noise=True)

        # Symmetry augmentation: store every symmetric (board, policy) pair
        for sym_board, sym_pi in game.get_symmetries(canonical, pi):
            train_examples.append((sym_board, current_player, sym_pi, None))

        action = np.random.choice(len(pi), p=pi)
        board, current_player = game.get_next_state(board, current_player, action)

        result = game.get_game_ended(board, current_player)
        if result != 0:
            # Fill in values: +result for positions whose player matches the
            # last mover, −result for the others. Policy cast to float32.
            return [(b.as_multi_channel(1),
                     np.asarray(p, dtype=np.float32),
                     result * ((-1) ** (player != current_player)))
                    for (b, player, p, _) in train_examples]
```

This single function is the source of truth for the episode loop — both the serial Coach path and the parallel worker pool call it, which is what makes their output identical at the same seed.

### Training data format

Each example is `(board_state, policy_vector, value)`:

- **Board state** — canonical multi-channel encoding (current player's planes first).
- **Policy vector** — the MCTS visit-count distribution (the improved target the network is trained toward).
- **Value** — `+1` if the player to move at that position eventually won, `−1` if lost, ~0 for a draw.

### Symmetry augmentation

Each visited position is stored once per symmetry of the game — free data augmentation:

- **Tic-Tac-Toe:** 8 symmetries (the full D₄ group: 4 rotations × 2 reflections).
- **Blokus Duo:** 2 (identity + main-diagonal reflection). The fixed starting squares at array indices (4,4) and (9,9) lie on the main diagonal, so only the transpose preserves them; every other rigid motion of the square either moves a start square off the diagonal or swaps the two (which the first-mover rule forbids). The policy is reindexed under transposition via a precomputed action permutation. (See `games/blokusduo/game.py` `get_symmetries` / `transpose_policy`.)

### Training window

Not all history is used. A sliding window controls how many recent generations of self-play feed each training step:

```python
def window_size(generation, max_lookback):
    if generation <= 5:
        return 5
    # grows linearly from 5 toward max_lookback, then plateaus there
    if 2 * max_lookback > generation:
        return round(generation * (max_lookback - 5) / (2 * max_lookback) + 5)
    return max_lookback
```

Early generations are random and low-quality; as the network improves, old data becomes less representative. The growing-then-plateauing window balances **recency** (recent, stronger games) against **diversity** (enough data to avoid overfitting recent patterns).

---

## Arena Evaluation (`core/arena.py`)

### Purpose

After training, the new network must prove it is stronger than the previous best. The Arena plays them head-to-head, both using MCTS at temperature 0 (pure exploitation, no temperature randomness).

### Alternating start positions

Games are played in two halves with the colours swapped, so each network plays White in half the games. This controls for first-move advantage: if the two are equal they should split the games regardless of who starts.

### Acceptance criterion (score-based)

The decision lives in one place — `core/acceptance.py` — so the training loop and the report can never disagree:

```
score  = (new_wins + 0.5 · draws) / (new_wins + prev_wins + draws)
accept = score ≥ update_threshold      (default 0.55; False if no games played)
```

Draws count as half a point each, in both numerator and denominator. This is the chess-style criterion rather than the draws-excluded `new_wins / (new_wins + prev_wins)` form — the latter silently rejects every generation in forced-draw regimes (TTT under near-perfect play, late-stage Blokus where both nets have converged), which score-based handles cleanly. A threshold above 0.5 demands the new network be genuinely (not just marginally) stronger before it's accepted.

- **Accepted:** saved as `best.pth.tar` and `accepted_{gen}.pth.tar`; training continues from it.
- **Rejected:** weights reverted to the pre-training checkpoint; the candidate saved as `rejected_{gen}.pth.tar` for inspection.

---

## Strength evaluation vs fixed baselines

Independent of accept/reject, every generation the new network is measured against fixed references (`Coach._evaluate_strength_vs_baselines`). Unlike the arena (which compares against a *moving* target), these give an absolute "is it actually getting stronger?" signal:

- **Elo vs frozen gen-0** (when `elo_games_per_gen > 0`): the random-init network is frozen at run start as a fixed anchor. Score rate → Elo difference via the standard `400 · log10(score / (1 − score))`, added to a display anchor rating.
- **Minimax oracle (TTT only):** games against a perfect-play opponent. Since TTT is a forced draw under optimal play, draw-rate → 1.0 with loss-rate → 0 is the "the model has learned optimal play" signal.
- **Symmetry diagnostic:** on a fixed set of reference positions, the KL divergence between the network's policy on a position and on its symmetric image — zero for a perfectly equivariant network. Catches learned directional biases that augmentation should be averaging out.

All of these run noise-free and reset the MCTS tree between games.

---

## The complete training loop

One generation, end to end:

```
Generation N:

1. SELF-PLAY                         (dominant share of wall-clock)
   ├── Fresh MCTS tree per game (reused across that game's moves)
   ├── Play num_eps games (serial, or across worker processes if num_parallel_workers > 1)
   │   └── Each move: MCTS search → sample by temperature → augment with symmetries
   └── Save the generation's examples to SelfPlayHistory/self_play_{N}.parquet

2. TRAINING
   ├── Snapshot current net → temp.pth.tar (the previous-best for the arena)
   ├── Combine + shuffle examples from the last window_size(N) generations
   ├── Train num_epochs:
   │   ├── Forward: board → (log-policy, value)
   │   ├── Policy loss: KL(target_pi ‖ predicted) ;  Value loss: MSE(target_v, predicted)
   │   ├── Backward + Adam step
   │   └── Per-epoch held-out diagnostics (entropy, top-1/5 accuracy, value calibration)
   └── Log loss + throughput metrics

3. ARENA                             (new vs previous-best, both MCTS @ temp 0)
   ├── num_arena_matches games, colours swapped halfway, replays recorded
   ├── accept iff (wins + ½·draws) / games ≥ update_threshold
   └── Accept → save best.pth.tar ;  Reject → revert to temp.pth.tar

4. STRENGTH EVAL                     (logged whether or not accepted)
   ├── Elo vs frozen gen-0 baseline
   ├── Minimax oracle (TTT only)
   └── Symmetry diagnostic

5. REPORTING
   └── Flush metrics to Parquet (+ mirror to W&B if configured)

After all generations: render the interactive HTML report.
```

---

## AlphaZero paper reference

The original AlphaZero paper (Silver et al., 2018) guides the design. Where we diverge, it's deliberate scaling-down for a single-GPU project:

- **MCTS simulations:** 800 per move in Chess, scaled to game complexity. We use far fewer (config-driven).
- **cpuct:** the paper uses a formula that grows with visit count; we use a fixed per-run constant.
- **Dirichlet noise:** added to root priors for exploration, `P(s,a) = (1−ε)·p_a + ε·η_a`, `η ~ Dir(α)`. **Implemented** (self-play only; see [Dirichlet root noise](#dirichlet-root-noise)).
- **Temperature:** τ = 1 for the first ~30 moves, τ → 0 afterwards — our `temp_threshold` does the same.
- **Training:** 700k steps at batch size 4096 for Chess. We train at much smaller scale.
- **Evaluation:** new network accepted at 55% over 400 games. We use the same threshold (score-based) with fewer games.

**Paper:** Silver, D. et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science* 362.6419 (2018): 1140–1144.
