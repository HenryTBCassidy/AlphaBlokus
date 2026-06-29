# Numba / Cython Self-Play Hot-Path Speedup

Implementation plan for compiling/refactoring the CPU-bound self-play hot path.
Self-play is **~80% CPU-side Python** even at the run-2 config (128f×8b, 800-cap
branching sims); the GPU sits ~idle (inference ~19%). Measured on the box
(gpu-linux, 2026-06-23) via `scripts/profile_self_play.py` at
`run_configurations/blokus_run2_bignet.json`:

| Slice | own-time | Dominated by |
|---|---|---|
| **Move generation** | ~40% | `movegen_runtime._fill_legal_actions_for_anchor` (4.5 s of a 13.7 s game) |
| **UCB select** | ~32% | `np.fromiter` + 3 gen-exprs over **19.7M `dict.get((state,action))`** calls in `mcts._select_action` |
| Board/transition/encode | ~10% | `with_piece`, `as_multi_channel`, `_compute_side_danger`, `_update_placement_points` |
| NN inference | ~19% (timing) | conv2d / batch_norm / `.cpu()` |

**Expected payoff: ~2.5–3.5× single-worker** (≈ same in games/s, since we're
CPU-bound). This is the low-risk interim win from the self-play speed assessment —
**not** the 10×, which needs the native/vectorised re-architecture (separate plan).
The useful side effect: after this lands, inference becomes ~40–50% of the now-smaller
wall, i.e. the bottleneck visibly moves to the GPU — the signal that the *next* plan
should be the batched-inference re-architecture, not more CPU micro-opt (re-measured in N6).

**Honest framing on "is Numba a free win here?"**
- **Move-gen (N2): yes** — a near-textbook `@njit` target over integer/boolean numpy
  tables, bit-identical, guarded by an existing 5,000-position equivalence suite.
- **Select (N3/N4): not a Numba problem** — the cost is `dict[(state_bytes, action)]`
  lookups, which Numba can't index. The fix is changing MCTS node storage to per-node
  aligned arrays; the gather then *disappears* and the existing vectorised PUCT math
  runs as-is. No JIT. This is the bigger, determinism-sensitive piece, and it touches
  more surface than "decorate a function" because `core/players.py` and a white-box
  test read/write those dicts directly (migrated in N4).

**Cython is the fallback, not the default.** If Numba typing stalls N2 (the frozenset
inventory / dedup set), a `.pyx` is the escape hatch — but the loop is integer/boolean,
which Numba handles cleanly. Recommend Numba.

---

## Checklist

| # | Item | Effort | Priority | Done | Files |
|---|------|--------|----------|------|-------|
| N1 | Add `numba` dep; prove `@njit(cache=True)` hits across forkserver workers + torch coexists; warm-compile at worker init | 0.5 day | High | ✅ | `pyproject.toml`, `games/blokusduo/movegen_runtime.py` |
| N2 | Numba move-gen kernel (`_fill_mask` + `has_any_move` kernel) behind the F2 flag; bit-identical on the dev-5000 cache | 2–4 days | High | ✅ | `games/blokusduo/movegen_runtime.py`, `tests/test_blokusduo/test_movegen_equivalence.py` |
| N3 | Add per-node array record in MCTS; migrate `_select_action`/descent/expand/backprop to it (gather killed); keep dict attrs as read-only compat | 2–3 days | High | ✅ | `core/mcts.py`, `tests/test_core/test_batched_inference.py` |
| N4 | Flip authority to node arrays; retire `(s,a)` dicts; add `root_visit_counts()` accessor; migrate `players.py` + rewrite the white-box test | 2–3 days | High | ✅ (N4.1 + N4.2) | `core/mcts.py`, `core/players.py`, `tests/test_core/*`, `scripts/*` |
| N5 | Incremental side-danger in `with_piece` (carry opponent's zone, OR the placed piece's halo) — the ~10% slice | 1–2 days | Medium | ✅ | `games/blokusduo/board.py` |
| N6 | Re-profile on the box (per-sim ladder) before/after; write up the delta | 0.5 day | High | ✅ | `docs/research/numba-hot-path-results.md` |

High-priority path (N1→N4 + N6): **~1.5–2 weeks** solo. N5 optional, promote only if N6 shows board-transition is then the top remaining slice.

> **Progress (2026-06-23, worktree `numba-hot-path`):**
> - **N1 ✅** — `numba==0.65.1` added. Coexistence + cache probe on the Mac venv (py311,
>   torch 2.10, numpy 2.4.3): `import torch` + `import numba` clean; `@njit(cache=True)`
>   writes `.nbi`/`.nbc` to `__pycache__` (cache persists → forkserver workers load, not
>   recompile). Warm-up wired into `get_default_generator` (the worker-init path via
>   `_maybe_enable_f2`). **Box (gpu-linux) re-confirmation still owed — fold into N6.**
> - **N2 ✅** — `_fill_mask_kernel` + `_has_any_move_kernel` added, gated by
>   `ALPHABLOKUS_NUMBA_MOVEGEN` (default on; `=0` is the pure-Python A/B baseline).
>   Bit-identical: `test_movegen_equivalence` (5,000 positions) + full `test_blokusduo/`
>   (160 passed) + `test_mcts`/`test_parallel_self_play`/`test_batched_inference`
>   (32 passed) all green; ruff clean. **Speed: `valid_move_masking` 244 µs → 10.3 µs
>   (~24×)** on a mid-game position (Mac CPU). `get_game_ended` flat on move-rich
>   positions (early-exit hides the kernel; its win is the near-terminal "no-move" scans).
>   Committed (`0497201`, `6f4e642`).
> - **N3 ✅** — MCTS storage migrated from the `(state, action)` dicts to per-state
>   `_Node` arrays (`acts`/`priors`/`n`/`q`/`virtual`/`n_total`); `_select_action` reads
>   them directly (the ~32% gather is gone). Read-only compat properties retained for
>   tests. Bit-identical at a fixed seed: full `tests/test_core/` + `tests/test_blokusduo/`
>   (254 passed), incl. the rewritten white-box `test_select_action_matches_scalar_loop`
>   (2,000 randomised trials, both virtual-loss paths), the K=1/K=16 search-equivalence,
>   and `test_parallel_self_play` worker determinism. ruff clean. **Box benchmark done
>   (2026-06-29, same-session ladder): 3.78 s/game = 2.12× vs baseline (N3 step +1.37× over
>   N2); inference share now 43.6% — the bottleneck has crossed over to the GPU.**
> - **N4.1 ✅** — `root_visit_counts()` / `num_states()` / `num_edges()` accessors added;
>   `get_action_prob` and `core/players.py` routed through `root_visit_counts` (no more
>   dict-surface dependency in production). **N4.2 deferred (deliberate):** the read-only
>   `visit_counts`/`q_values`/… properties are retained — they're off the hot path, no
>   production consumer depends on them, and the determinism tests use them for whole-tree
>   snapshots, so deleting them is churn with no perf benefit.
> - **N3 + N4.1 committed** (`5342f15`, tag `bench/n3-n4.1`) after review.

### Safety nets (the bit-identical contract)

Every step stays bit-identical at a fixed seed; these are the existing guards:

- **Move-gen (N2):** `tests/test_blokusduo/test_movegen_equivalence.py` —
  `test_movegen_equivalence_dev_cache` + `test_has_any_move_equivalence_dev_cache`
  check the F2 path against the reference array generator on **5,000 stratified
  positions** (`tests/fixtures/blokus_duo_positions/dev_5000.npz`). The Numba kernel
  sits behind the same flag, so these cover it for free — output is integer/boolean,
  so the tolerance is **exact**.
- **MCTS (N3/N4):** `tests/test_core/test_batched_inference.py` pins the K=1 / K=16
  visit-count contract (`atol=1e-6`) and white-box-reconstructs the PUCT score;
  `test_core/test_mcts.py`, `test_dirichlet_noise.py`, `test_profiling.py` pin
  invariants; `test_core/test_parallel_self_play.py` pins end-to-end worker
  determinism. Run the full `tests/test_core/` after **each** N3/N4 sub-change so a
  regression is bisectable. PUCT float **operation order** must be preserved (the
  current `_select_action` is bit-identical to the original scalar loop by
  construction — keep the numpy arithmetic verbatim; only change where its inputs
  come from).

---

## N1. Numba dependency + cache / coexistence probe

**Current state:** no JIT toolchain; deps are numpy/torch/etc. in `pyproject.toml`.

**Fix:** add `numba` (pulls `llvmlite`). Verify two real failure modes *before* writing
any kernel:

1. **Multi-worker compile cost.** Self-play spawns 16 fresh workers **per generation**
   (`forkserver` on Linux). JIT compiles on first call *per process* (~1–3 s). Without
   mitigation that's `16 × num_generations` recompiles. `@njit(cache=True)` persists
   compiled code to disk and reloads it across processes/runs. **Verify the cache
   actually hits across a forkserver worker** (it should — keyed by function +
   signature; confirm the cache dir is writable and shared). Warm it once in the worker
   initialiser (`_worker_init_self_play` → call `get_default_generator()` which triggers
   one warm call) so the first real move doesn't pay compile latency.
2. **torch + numba coexistence.** Confirm `import numba` next to `import torch` on the
   box's Python 3.11 venv doesn't clash (llvmlite LLVM vs torch libs). One-line import
   test on gpu-linux.

**Deliverable:** `numba` in `pyproject.toml`, a warm-compile call at worker init, and a
short note in N6's results doc confirming (1) and (2) hold on the box. No behaviour
change yet.

---

## N2. Numba move-gen kernel

**Current state:** `F2MoveGenerator.valid_move_mask` is the production path
(`use_optimised_movegen` flag). Its inner loop `_fill_legal_actions_for_anchor` is
**~33% of one game alone**. Per anchor it iterates remaining pieces, slices a candidate
list from `_lookup_move_ids`, and per candidate does a 5-cell `forbidden[...]` OR, a
Python `set` membership/`add` dedup, and an `out[...] = True` write.

Everything it reads is **already numpy**: `begin[anchor,adj,piece]` (uint32
[196,64,21]), `size` (uint16 [196,64,21]), `move_ids` (flat uint16),
`cells[move_id,:5]` (uint8, NULL_CELL=196 padded), `action_id[move_id]` (uint32),
`adj_status_cells[anchor,:6]` (int), `forbidden` (uint8, size `NUM_CELLS+1` with the
pad slot always 0). The only non-Numba pieces, and their fixes:

- `seen` Python `set` → `np.zeros(num_moves, np.bool_)` scratch array (reset per call, or
  track touched ids to clear cheaply),
- `remaining_pieces` `frozenset` → sorted `int32` array,
- the loop over `board.placement_points(player)` (a dict) → pass the anchor cells as an
  `int32` array (`row*N + col` for each key).

**Fix:** lift forbidden→anchor→fill into one `@njit(cache=True)` kernel:

```python
@njit(cache=True)
def _fill_mask(forbidden, anchors, adj_status_cells,
               lookup_begin, lookup_size, lookup_move_ids,
               move_cells, move_action_id, remaining, out, seen):
    for ai in range(anchors.shape[0]):
        anchor = anchors[ai]
        if forbidden[anchor]:
            continue
        adj = (forbidden[adj_status_cells[anchor, 0]]
               | (forbidden[adj_status_cells[anchor, 1]] << 1)
               | (forbidden[adj_status_cells[anchor, 2]] << 2)
               | (forbidden[adj_status_cells[anchor, 3]] << 3)
               | (forbidden[adj_status_cells[anchor, 4]] << 4)
               | (forbidden[adj_status_cells[anchor, 5]] << 5))
        for pi in range(remaining.shape[0]):
            piece_idx = remaining[pi] - 1
            begin = lookup_begin[anchor, adj, piece_idx]
            size  = lookup_size[anchor, adj, piece_idx]
            for off in range(size):
                mid = lookup_move_ids[begin + off]
                if seen[mid]:
                    continue
                c = move_cells[mid]
                if not (forbidden[c[0]] | forbidden[c[1]] | forbidden[c[2]]
                        | forbidden[c[3]] | forbidden[c[4]]):
                    seen[mid] = True
                    out[move_action_id[mid]] = True
```

`valid_move_mask` keeps the first-move fallback (`_fill_first_move_mask`), builds the
`forbidden` / `anchors` / `remaining` arrays, then calls `_fill_mask`. `has_any_move`
(the `get_game_ended` existence check) gets a sibling early-exit `@njit` kernel
returning bool.

**Tasks:** (a) array-marshal the inputs; (b) the two kernels; (c) wire `F2MoveGenerator`
to call them; (d) keep the pure-Python F2 loop reachable for A/B (a `use_numba_movegen`
sub-flag or env toggle) so N6 can measure the kernel against it.

**Why it's the clean win:** integer/boolean only → **exact** output → the 5,000-position
equivalence suite validates it with zero tolerance. Expect ~5–20× on this kernel; at
~40% of the wall it's the single biggest lever here.

**Risk: low.** Watch the NULL_CELL pad slot (`forbidden` is already `NUM_CELLS+1`, slot
196 must stay 0) and the uint dtypes (`begin+off` must not overflow — `begin` is uint32,
fine). Gate behind the F2 flag.

---

## N3. Per-node array record in MCTS — kill the select gather (core change)

> **Detailed execution subplan:** [numba-hot-path-mcts-subplan.md](numba-hot-path-mcts-subplan.md)
> — full read/write-site inventory, per-method before→after, the bit-identical traps, and
> the N3.1/N3.2/N4.1/N4.2 commit sequence. Read it before touching `core/mcts.py`.

**Current state:** tree stats live in global dicts keyed by `(state_key, action)`:
`q_values`, `visit_counts`, `virtual_visits`, plus per-state `state_visits`,
`policy_priors`, `valid_moves_cache`, `game_ended_cache`. `_select_action` rebuilds —
**per descent, per node** — three arrays via
`np.fromiter(self.visit_counts.get((s,a),0) for a in acts)` (+ q, + virtual). That's the
**19.7M `dict.get` / ~32% of the game**. The PUCT arithmetic is already vectorised and
cheap; the cost is purely the dict-of-`(bytes,int)` gather (why P2 bought ~0%).

**Fix:** store stats **per node, aligned to that node's legal-move array** (`acts`,
already in `valid_moves_cache[s]`). One record per expanded state:

```python
class _Node:                  # one per expanded state_key
    acts:    NDArray  # int32[m]   ascending legal action ids
    P:       NDArray  # float64[m] priors, aligned to acts (== policy_priors today)
    N:       NDArray  # int32[m]   visit counts, aligned
    Q:       NDArray  # float64[m] q-values, aligned
    virtual: NDArray  # int32[m]   in-flight virtual loss, aligned
    n_total: int      # == state_visits[s] (sum of N; kept explicit for sqrt order)
```

- `_expand_leaf` builds the node (it already computes `acts` + masked `P`; just also
  allocate `N=zeros`, `Q=zeros`, `virtual=zeros`, `n_total=0`).
- `_select_action(node)` becomes pure array math on `node.N/Q/P/virtual/n_total` — **the
  gather is gone**; keep the exact numpy PUCT expressions and the `argmax` tie-break.
  Returns the chosen **index `i`** into `acts` (and `acts[i]`).
- `_descend_to_leaf` carries `(s, i)` in `path` (index, not action); deposits virtual
  loss via `node.virtual[i] += VIRTUAL_LOSS`.
- `_backprop` updates `node.Q[i]`/`node.N[i]`/`node.n_total` by index (same running-mean
  order as today).
- `_remove_virtual_loss` decrements `node.virtual[i]`.

**Compat (so the suite stays green this step):** keep the public dict attributes as
**read-only properties** reconstructed from nodes (`visit_counts` →
`{(s, node.acts[i]): node.N[i] ...}`), and keep `state_visits`/`policy_priors`/
`valid_moves_cache` working as before (the latter two map straight onto node fields).
This lets `players.py` and all existing tests keep passing on N3; N4 removes the shims.
`get_action_prob` builds its dense `counts` by scattering the root node's `N` into an
`action_size` vector at `acts` positions (cheaper than today's `range(action_size)` loop).

**Risk: medium** — core, determinism-sensitive, but mechanical. Do it in small commits;
run `tests/test_core/` after each. Keep PUCT float ops verbatim.

---

## N4. Retire the `(s,a)` dicts; migrate external consumers

**Current state (after N3):** node arrays are authoritative *internally*, but the dict
attributes still exist as compat shims, and two external consumers depend on them:

- `core/players.py:91` — builds the raw visit distribution with
  `[self._mcts.visit_counts.get((s,a),0) for a in range(n_actions)]` (production: arena/
  Elo replay records).
- `tests/test_core/test_batched_inference.py:256–322` — white-box: **writes** into
  `valid_moves_cache`/`policy_priors`/`visit_counts`/`q_values`/`virtual_visits`/
  `state_visits` to hand-build a tree, then recomputes PUCT to assert selection.
- `scripts/benchmark.py`, `scripts/profile_mcts_memory.py` — read `len(state_visits)` /
  `len(q_values)` for tree-size stats.

**Fix:**
- Add `MCTS.root_visit_counts(board) -> NDArray[float]` (dense `action_size` vector from
  the root node). Route both `get_action_prob` and `players.py` through it (dedups the
  identical list-comprehension that exists in both today).
- Rewrite the `test_batched_inference.py` white-box helper to construct a `_Node`
  directly via a small test-only builder, and recompute PUCT from node arrays. The
  **assertion** (selection + visit counts identical) is unchanged — only the setup
  surface moves to the node API.
- Update the two `scripts/` tree-size reads to `mcts.num_states()` / `mcts.num_edges()`
  helpers (or keep a `state_visits`-length property).
- Remove the N3 compat shims once all readers are migrated.

**Risk: medium** — this is the surface the scoping pass under-counted. Mechanical once
the accessor exists; the white-box test rewrite is the fiddly bit. Keep it its own
commit(s).

---

## N5. Incremental board transition (optional, ~10%)

**Current state:** `BlokusDuoBoard.with_piece` copies the 196-byte board, builds two
fresh placement-point dicts, and **recomputes both players' side-danger from scratch**
every move (`_compute_side_danger`, `_update_placement_points` ~0.38 s combined; ~45 µs
per `with_piece`, called on every descent edge).

**Fix options (choose by the N6 re-profile):**
- Incremental side-danger: a placed piece only changes danger in its 1-cell halo — update
  that neighbourhood instead of a full-board numpy recompute.
- Store placement points as arrays, not dicts (also feeds N2's anchor marshalling — the
  anchor int-array N2 needs would come straight from here).
- A small `@njit` halo update if it stays hot.

**Why optional:** ~10% → Amdahl ceiling ~1.11×. Only worth it if N2+N3/N4 land and board
transition is then the top remaining CPU slice. Deferred unless N6 promotes it.

---

## N6. Re-profile and write up the delta

Re-run on the box at the run-2 config, before/after:
- `scripts/profile_self_play.py --mode cprofile` (function attribution),
- a 16-worker `run_self_play_episodes_parallel` games/s measurement (the production wall).

Record the new phase split — expect inference to climb to ~40–50% of the now-smaller
wall, i.e. the bottleneck moving to the GPU side (the signal to start the
native/batched-inference plan rather than more CPU work). Write to
`docs/research/numba-hot-path-results.md`; update the self-play speed assessment. Include
the N1 cache/coexistence confirmation.

---

## What this plan deliberately does NOT do

- **It is not the 10×.** Honest ceiling ~2.5–3.5× single-worker. The 10× needs the native
  (C++/Pentobi-port) batched-inference engine or vectorised-GPU MCTS — a separate, larger
  plan this de-risks by showing where the wall moves (N6).
- **No new parallelism model** (free-threaded Python / process-model changes are out of
  scope — the assessment deprioritised them).
- **No net/MCTS hyperparameter changes.** Pure speed; trajectories stay bit-identical at a
  fixed seed.
