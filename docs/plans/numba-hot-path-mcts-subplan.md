# Subplan — MCTS Per-Node Array Storage (N3 / N4)

Detailed execution subplan for N3+N4 of [numba-hot-path.md](numba-hot-path.md): replace
the MCTS `(state, action)` dicts with per-node aligned arrays so the ~32% select-time
gather (`np.fromiter` over 19.7M `dict.get` calls/game) disappears. **This is not a Numba
change** — pure numpy/Python restructuring. It is the determinism-sensitive piece, so this
subplan enumerates every read/write site and the exact before→after, and is written so a
fresh worker can execute it without re-deriving the bit-identical constraints.

**The cardinal rule:** the returned visit-count distribution must be **bit-identical** at a
fixed seed, at both K=1 and K=16. The PUCT float expressions and the backprop running-mean
update **keep their exact operation order** — we only change *where the inputs come from*
(node arrays vs dict gather), never the arithmetic.

---

## Current storage (what we're replacing)

In `MCTS.__init__` (`core/mcts.py`):

| Attribute | Type | Keyed by | Role |
|---|---|---|---|
| `q_values` | `dict` | `(state, action)` | running-mean Q per edge |
| `visit_counts` | `dict` | `(state, action)` | N per edge |
| `virtual_visits` | `dict` | `(state, action)` | in-flight virtual loss per edge |
| `state_visits` | `dict` | `state` | total visits through a node (== Σ N) |
| `policy_priors` | `dict` | `state` | sparse priors, **aligned to `acts`** |
| `valid_moves_cache` | `dict` | `state` | `acts` = ascending legal action ids |
| `game_ended_cache` | `dict` | `state` | terminal value cache |

`game_ended_cache` **stays as-is** (per-state, cheap, not part of the gather). The other six
collapse into one per-state record.

## Target storage

```python
@dataclass
class _Node:
    acts:    NDArray  # int32[m]   ascending legal action ids   (was valid_moves_cache[s])
    P:       NDArray  # float32[m] priors aligned to acts        (was policy_priors[s])
    N:       NDArray  # int32[m]   visit counts aligned          (was visit_counts gather)
    Q:       NDArray  # float64[m] q-values aligned              (was q_values gather)
    virtual: NDArray  # int32[m]   virtual loss aligned          (was virtual_visits gather)
    n_total: int      # total visits through this node          (was state_visits[s])
```

`self.nodes: dict[StateKey, _Node]` replaces the six. The edge "key" becomes the **index
`i` into `acts`** (ascending, so `i` ↔ action is stable). `path` carries `(_Node, i)` instead
of `(state, action)` — this also removes a dict lookup per backprop step.

**Dtype must match today exactly** (this is a bit-identical trap):
- `P` is `float32` (NN output, masked+renormalised) — `_select_action` promotes it to
  float64 with `.astype(np.float64)`, *keep that*. After Dirichlet noise `P` becomes float64
  (the noise is float64) — that already happens today; preserve it.
- `Q` is `float64` (Python-float running mean stored as float64 today via the dict).
- `N`/`virtual` integer. Under NEP50, `N[i] + 1` (int32 + Python int) stays int32, and
  `int32 * float64 / int32` promotes to float64 — numerically identical to today's
  `int * float / int`. Verified by the golden test, but know it's load-bearing.

---

## Read/write site inventory

**Internal (`core/mcts.py`) — all migrate together in N3.1:**

| Method | Today | After |
|---|---|---|
| `_expand_leaf` | sets `policy_priors[s]`, `valid_moves_cache[s]`, `state_visits[s]=0` | builds `_Node(acts, P, N=0, Q=0, virtual=0, n_total=0)`; `nodes[s]=node` |
| `_select_action(s)` | gathers n/q/virtual via `fromiter(dict.get)` | takes `node`; reads `node.N/Q/P/virtual/n_total`; returns **index `i`** |
| `_descend_to_leaf` | `path.append((s,a))`; `virtual_visits[(s,a)] += VL` | `path.append((node,i))`; `node.virtual[i] += VL`; `a = int(node.acts[i])` |
| `_backprop` | running-mean on `q_values[(s,a)]`, `visit_counts`, `state_visits[s]` | same arithmetic on `node.Q[i]`, `node.N[i]`, `node.n_total` |
| `_remove_virtual_loss` | `virtual_visits[(s,a)] -= VL`, pop if ≤0 | `node.virtual[i] -= VL` (array slot returns to 0; no pop) |
| `_apply_root_dirichlet_noise` | reads/writes `valid_moves_cache[s]`, `policy_priors[s]` | `node.acts`, `node.P = (1-ε)·node.P + ε·noise` |
| `get_action_prob` | `counts=[visit_counts.get((s,a),0) for a in range(A)]` | scatter `node.N` into a zeros(A) vector at `node.acts` |
| `get_episode_stats` | `tree_size=len(state_visits)` | `len(self.nodes)` |
| `_estimate_tree_memory_bytes` | iterates the dicts | sums `_Node` array `nbytes` |

**External — migrate in N4 (kept working in N3 via read-only compat properties):**

| Site | Today | Plan |
|---|---|---|
| `core/players.py:91` | `[mcts.visit_counts.get((s,a),0) for a in range(A)]` | route through new `mcts.root_visit_counts(board)` (N4.1) |
| `tests/test_core/test_batched_inference.py:256–322` | **writes** `valid_moves_cache/policy_priors/visit_counts/q_values/virtual_visits/state_visits` to hand-build a tree, then recomputes PUCT | rewrite to build a `_Node` directly (N3.2 — must move *with* the storage change; a read-only shim can't satisfy a writer) |
| `tests/test_core/test_mcts.py:83–84` | reads `len(state_visits)`, `len(policy_priors)` | read-only shim covers it; tidy to `len(mcts.nodes)` in N4.2 |
| `tests/test_core/test_profiling.py:88` | `len(state_visits)` | shim; tidy N4.2 |
| `tests/test_core/test_dirichlet_noise.py:50,68,75,96` | reads `dict(visit_counts)`, `policy_priors[s]` | shim covers it (read-only) |
| `scripts/benchmark.py:86,151`, `scripts/profile_mcts_memory.py:155,156` | `len(state_visits)`, `len(q_values)` | new `mcts.num_states()` / `num_edges()` (N4.1) |

---

## Determinism-critical details

1. **`_select_action` fast/slow virtual-loss branch.** Today the branch is
   `if self.virtual_visits:` (any virtual loss *anywhere*). Switch to `if node.virtual.any():`
   (virtual on *this* node). This is **bit-identical** because when a node's `virtual` is all
   zero the slow-path `np.where(v==0, …)` arms reduce to exactly the fast-path expressions
   (`eff_n = n + 0 = n`, so `1+eff_n == 1+n` to the bit). The per-node condition only changes
   *which* path runs for nodes without virtual loss, not the numbers. Verified by the K=16
   golden test.
2. **`argmax` tie-break.** Keep `int(np.argmax(scores))` and return `node.acts[i]`. `acts` is
   ascending, so first-max tie-break is unchanged.
3. **Backprop order.** Iterate `reversed(path)`, flip `value` sign each ply, same
   running-mean formula. `N[i]==0` is the first-visit sentinel (≡ "`(s,a)` absent from the
   dict" today, since `_expand_leaf` zeroes `N`).
4. **`get_action_prob` counts.** Scatter produces the *same integer per action* as the
   `dict.get` list; the temperature/entropy code downstream is untouched, so the returned
   probs are identical. (Bonus: drops the per-call `range(17837)` dict probe.)
5. **Dirichlet dtype drift.** `node.P` goes float32→float64 after noise (the noise is
   float64) — this already happens today via `policy_priors[s]`; do not "fix" it.

---

## Step sequence (each = one commit; run the gate after each)

Gate after **every** step: `uv run --extra dev python -m pytest tests/test_core/ tests/test_blokusduo/ -q` (CPU; no GPU needed). The load-bearing tests are `test_batched_inference.py` (K=1/K=16 visit counts, `atol=1e-6` — for us expect **exact**), `test_parallel_self_play.py` (worker determinism), `test_dirichlet_noise.py`.

| Step | Commit | Gate must be green |
|---|---|---|
| **N3.1** | Introduce `_Node`/`self.nodes`; migrate all internal methods (table above); add **read-only** compat properties (`visit_counts`, `q_values`, `virtual_visits`, `state_visits`, `policy_priors`, `valid_moves_cache`) reconstructed from `nodes`. | everything **except** the white-box writer in `test_batched_inference.py` |
| **N3.2** | Rewrite the `test_batched_inference.py` white-box helper to build a `_Node` via a small test-only constructor and recompute PUCT from node arrays. Assertions (selection + visit counts identical) unchanged. | full `tests/test_core/` + `tests/test_blokusduo/` |
| **N4.1** | Add `MCTS.root_visit_counts(board) -> NDArray` + `num_states()`/`num_edges()`. Route `get_action_prob` *and* `players.py` through `root_visit_counts` (dedups the identical comprehension). Point the two `scripts/` at the new counters. | full suite |
| **N4.2** | Delete the read-only compat properties; tidy the remaining test reads (`len(state_visits)`→`len(mcts.nodes)`, etc.) to the node API. | full suite |

**Compat property sketch (N3.1, removed in N4.2)** — keeps read-only external consumers green:

```python
@property
def visit_counts(self) -> dict:                      # read-only reconstruction
    return {(s, int(node.acts[i])): int(node.N[i])
            for s, node in self.nodes.items()
            for i in range(node.acts.shape[0]) if node.N[i] > 0}   # N>0 ≡ dict.get(...,0)
# state_visits → {s: node.n_total}; policy_priors → {s: node.P}; valid_moves_cache → {s: node.acts}; etc.
```

---

## Effort & risk

- **N3.1** ~1–1.5 days (the core change; mechanical but every method touched at once).
- **N3.2** ~0.5 day (test rewrite — the fiddly bit; the assertion stays, only the setup moves).
- **N4.1** ~0.5 day. **N4.2** ~0.5 day.
- Total **~2.5–3 days**. Risk **medium**: determinism-sensitive, but the golden tests make a
  regression loud and bisectable (one commit per step). Keep PUCT/backprop arithmetic verbatim.

## Expected payoff

The ~32% select slice collapses to a few % (the gather is gone; the vectorised PUCT math
stays). Combined with N2's move-gen kernel this is where the plan's ~2.5–3.5× single-worker
comes from. Confirm in N6 (re-profile on the box).
