# MCTS Memory Reduction — lift the per-worker tree footprint

Sub-plan of [lean-self-play-workers](../lean-self-play-workers.md). The lean-workers
work established that the **per-worker MCTS tree (~2 GB) is what caps how many
self-play workers we can run** — and it's the one cost no OS trick can share
(the tree is private and constantly mutated, so copy-on-write / shared memory
can't touch it). The *only* lever is making the tree itself smaller in code,
which is **fully OS-agnostic** — it pays off identically on WSL, native Windows,
and the upcoming native-Linux partition, so none of this work is substrate-specific.

**Goal:** the strongest model needs the most self-play data per unit time, which
means the most concurrent workers, which means the smallest per-worker tree.
Halving tree memory roughly doubles the workers that fit in 32 GB.

**Approach: measure before cutting.** We do *not* guess at optimisations. First
build a profiler that attributes tree memory to its actual consumers and records
a hard baseline (M1–M2); only then design the changes against that evidence (M3),
appending the concrete change rows once the targets are known. This mirrors the
[profiling-investigation](profiling-investigation.md) →
[report](../../research/profiling-report.md) pattern that worked for the cycle profile.

The MCTS is dict-based ([`core/mcts.py`](../../../core/mcts.py)); the suspect
consumers, in rough order of expected size:
- `policy_priors: dict[StateKey, PolicyVector]` — a policy vector per state (dense = 17,837 floats ≈ 71 KB/state)
- `valid_moves_cache: dict[StateKey, ValidMoves]` — a legal-move representation per state
- `q_values` / `visit_counts` / `virtual_visits: dict[StateAction, …]` — keyed by `(196-byte key, action)`; the keys dominate
- `state_visits`, `game_ended_cache: dict[StateKey, …]` — per-state scalars + their 196-byte keys

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| M1 | **Build an MCTS memory profiler** — attribute peak/end-of-game tree memory across each dict (entries, bytes/entry, total), report bytes-per-state and total tree MB for one self-play game | 0.5 day | High | ✅ |
| M2 | **Profile + publish the baseline** — run M1 on a representative self-play game at production settings; write a findings report ranking the consumers + record the baseline tree-MB number to beat | 1 hr + run | High | ✅ |
| M3 | **Design the reductions** *(gated on M2)* — turn the ranked findings into concrete, ordered change rows (M4+); each with an expected MB saving and a re-measure against the M2 baseline | 0.5 day | High | ✅ |
| M4 | **Compact `valid_moves_cache` → valid-action index array** — store `np.where(valids)[0]` (int32) instead of a dense `float64[17837]` mask; update the two read sites | 1 hr | High | ✅ |
| M5 | **Sparsify `policy_priors` → priors aligned to valid actions** — store `masked[valid_actions]` (keep `float64`) instead of the dense vector; iterate the aligned arrays in `_select_action` | 1.5 hr | High | ✅ |
| M6 | **(Optional, lossy) dtype trim** `float64`→`float32` on the sparse priors — halves what's left; *not* bit-identical to the float64 past, so gated behind re-validation | 1 hr | Low | ❌ not needed (see below) |

Baseline to beat (from [M2](../../research/mcts-memory-report.md)): **1,874 MB peak / 279.8 KB/state**.

**Result (M4+M5, same seed/config):** **1,874 MB → 19.3 MB peak (~97×), 279.8 → 2.9 KB/state.**
Bit-identical: the game is unchanged (6,540 states · 6,599 state-actions · 25 moves both before and after), all 271 tests pass. The two arrays dropped from dense `float64[17,837]` (139 KB each) to sparse `[~414]` (`policy_priors` 3.2 KB, `valid_moves_cache` 1.6 KB per state). Memory is no longer the worker-count constraint — 20 workers ≈ 380 MB of trees vs ~37 GB before.

**M6 decision: skip it.** At 19.3 MB/worker the tree is already negligible; `float32` would shave ~4 MB (the `float64` priors half) off an already-tiny number while *breaking* bit-identical reproducibility — a bad trade. Revisit only if a far larger net or sim count ever makes the sparse priors a real cost again.

---

## M1. Build an MCTS memory profiler

Extend the existing `memory` mode in [`scripts/profile_self_play.py`](../../../scripts/profile_self_play.py)
(which already plays one game under `tracemalloc` and reports peak heap + tree
size) to **attribute** memory rather than just total it. After playing one
self-play game at production settings (`blokus_scaled_15.json`, 300 sims), report
for **each** MCTS dict:

- entry count (`len`),
- total bytes (deep size of keys + values — use `tracemalloc` top-allocators-by-line for the authoritative split, cross-checked with a deep sizer for per-dict totals),
- bytes per entry, and average bytes per *state*.

Plus the headline figures: **peak tree memory during the game** and **at game end**, total states, and a one-line breakdown (`policy_priors X MB · valid_moves_cache Y MB · StateAction keys Z MB · …`).

Two memory views matter and should both be reported:
- **Peak during the game** — what actually sets the per-worker RAM ceiling (this is the number that gates worker count).
- **Growth shape** — does the tree grow unbounded across the game, or plateau? (Tells us whether tree-reuse/pruning is even relevant.)

Keep it `tracemalloc`-based and single-process (no pool) so the attribution is
clean and reproducible. Output stays markdown/numbers; any chart goes to the
gitignored `temp/` (per the no-build-artifacts-in-git rule).

## M2. Profile + publish the baseline

Run the M1 profiler on the home PC (or Mac — tree memory is OS-agnostic, so the
Mac is fine for this and avoids the remote-run friction) at production MCTS
settings. Produce `docs/research/mcts-memory-report.md` with:

- the **baseline**: peak tree MB/worker and bytes/state (the number every later change is measured against — treat it as immutable like the cycle baseline),
- the **ranked consumers** (which dict owns what fraction of the peak),
- the **growth curve** (tree MB vs move number),
- and a short "biggest levers" section translating the ranking into candidate attacks (input to M3).

This is both the diagnosis *and* the benchmark: M4+ changes re-run this exact
profile and report the delta against the baseline.

## M3. Design the reductions

M2 was decisive: **99.6% of the 1,874 MB tree is two dict values** — `policy_priors`
and `valid_moves_cache`, each a dense `float64[17,837]` array stored *per state*
(139 KB each). Everything else (q/visits/caches/keys) is ~8 MB combined, so the
key-interning and dtype-of-scalars ideas from the original brainstorm aren't worth
it — the entire win is in those two arrays.

**The insight that makes it cheap and safe:** the MCTS only ever touches these
arrays *at the valid moves*. `valid_moves_cache[s]` is read only as
`np.where(valids)[0]` (the valid action ids; `mcts.py:362`) and `.sum()` (a count;
`mcts.py:205`). `policy_priors[s]` is read only as `priors[a]` for those same
valid `a` inside the UCB loop. So we never need the dense 17,837-vector — only the
values at the (tens–hundreds of) legal moves. And because `np.where` returns
indices in **ascending order — the exact order the loop already iterates** —
storing the legal-move slice in that order and keeping `float64` is
**bit-identical**, not approximate. That's M4 + M5 (lossless). M6 (`float32`) is a
further halving but changes the floats, so it's optional and gated.

Expected: the two arrays drop from `17,837 × 8 B` to `n_valid × 8 B` per state.
At the measured average legal-move count this should cut the tree by **~10×+**
(re-measure confirms), turning per-worker memory from the cap into a non-issue.
After M4+M5 land, re-benchmark worker count on the target substrate to confirm the
freed RAM actually buys more concurrent workers.

## M4. Compact `valid_moves_cache` → valid-action index array

Today `_expand_leaf` stores `self.valid_moves_cache[s] = valids` — a dense
`float64[17,837]` mask. Replace it with the legal-action **index array**:

```python
acts = np.where(valids)[0].astype(np.int32)   # the valid action ids, ascending
self.valid_moves_cache[s] = acts              # (rename to valid_actions for clarity)
```

Update the two read sites:
- `mcts.py:205` `int(self.valid_moves_cache[s].sum())` → `len(self.valid_moves_cache[s])` (the count is unchanged).
- `mcts.py:362` `for a in np.where(valids)[0]:` → `for a in self.valid_moves_cache[s]:` (same ids, same order — `np.where` already gave ascending order, so iteration is identical).

Grep for any other references first. Removes the `valid_moves_cache` half
(~864 MB at baseline). Re-run the profiler; expect the dict to drop from 139 KB to
`4 × n_valid` bytes per state. Bit-identical.

## M5. Sparsify `policy_priors` → priors aligned to valid actions

With M4's index array in place, store priors **only for the valid moves, in the
same order**, keeping `float64` so the arithmetic is unchanged:

```python
acts = self.valid_moves_cache[s]              # from M4 (ascending valid ids)
self.policy_priors[s] = masked[acts]          # float64, length n_valid, aligned with acts
```

Rework `_select_action` to walk the aligned arrays instead of indexing a dense
vector by action id:

```python
acts   = self.valid_moves_cache[s]
priors = self.policy_priors[s]                # priors[i] is the prior for acts[i]
for i in range(len(acts)):
    a = int(acts[i])
    prior_a = priors[i]                        # was priors[a]
    ...                                        # rest of the UCB expression unchanged
```

Membership checks (`if s not in self.policy_priors`, `mcts.py:295/335`) are
unaffected. Removes the `policy_priors` half (~864 MB). Re-run the profiler;
combined with M4 the tree should be a small fraction of the 1,874 MB baseline.
Bit-identical (same float64 values, same order).

## M6. (Optional, lossy) dtype trim float64 → float32

The sparse priors are still `float64`. Casting to `float32` halves what remains,
but it changes the UCB floats at the ~1e-7 level — so it is **not** bit-identical
to the float64 trajectories (it stays deterministic and parallel-consistent, just
different from before). Low priority: the sparse change already does the heavy
lifting, and this trades reproducibility-vs-history for a final ~2× on an
already-small number. Only do it (with a re-validation pass) if M4+M5 leave memory
still tight at the worker count we want.

---

## Notes

- **OS-agnostic by construction** — every change here is in `core/mcts.py` data structures; nothing depends on the start method, OS, or fork/spawn. Do it now, it transfers to the native-Linux partition unchanged.
- **Stays on `perf/lean-self-play-workers`** — this is the same effort (make the self-play workers light enough to run many at once), just attacking the dominant cost rather than the process-spawn mechanism.
- **Correctness gate** — any change must keep self-play trajectories bit-identical at a fixed seed (the parallel-determinism + movegen-determinism tests are the guard); these are memory-layout changes, not algorithm changes.
