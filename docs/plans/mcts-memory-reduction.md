# MCTS Memory Reduction — lift the per-worker tree footprint

Sub-plan of [lean-self-play-workers](lean-self-play-workers.md). The lean-workers
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
[profiling-investigation](archive/profiling-investigation.md) →
[report](../research/profiling-report.md) pattern that worked for the cycle profile.

The MCTS is dict-based ([`core/mcts.py`](../../core/mcts.py)); the suspect
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
| M3 | **Design the reductions** *(gated on M2)* — turn the ranked findings into concrete, ordered change rows (M4+); each with an expected MB saving and a re-measure against the M2 baseline | 0.5 day | High | |

*(M4+ — the actual code changes — are intentionally left unnumbered until M2's
data tells us which levers are worth it and in what order; see M3.)*

---

## M1. Build an MCTS memory profiler

Extend the existing `memory` mode in [`scripts/profile_self_play.py`](../../scripts/profile_self_play.py)
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

## M3. Design the reductions *(gated on M2)*

With M2's ranking in hand, turn the biggest levers into concrete, ordered change
rows (appended here as M4, M5, …), each with an expected saving and a re-measure
step. **Do not pre-commit to these before M2** — they're hypotheses to confirm
against the data:

- **Sparse `policy_priors`** — store only the legal-move entries, not a dense 17,837-vector (MCTS only ever reads priors for valid moves). Likely the single biggest win if priors are dense.
- **Compact `valid_moves_cache`** — store packed bits or the `np.where` index array instead of a full bool[17,837] mask.
- **Shrink the key overhead** — the `StateAction` dicts key on `(196-byte board key, action)`; intern state keys to a small int id (or move to id-indexed arrays) so the 196 bytes aren't repeated across every dict and every action.
- **Tree reuse / pruning** — on committing a move, keep the chosen subtree and drop the rest, bounding peak size (only worth it if M2 shows the tree grows across the game rather than resetting per move).
- **dtype trims** — `float32`→`float16` for priors/Q where precision allows; `int32`→`int16` for visit counts.

M3 ranks these by M2's measured impact, sequences them to minimise churn, and
each resulting row re-runs M2's profile to confirm the saving is real before the
next change. Once M4+ land, re-benchmark worker count on the target substrate to
confirm the freed memory actually buys more concurrent workers.

---

## Notes

- **OS-agnostic by construction** — every change here is in `core/mcts.py` data structures; nothing depends on the start method, OS, or fork/spawn. Do it now, it transfers to the native-Linux partition unchanged.
- **Stays on `perf/lean-self-play-workers`** — this is the same effort (make the self-play workers light enough to run many at once), just attacking the dominant cost rather than the process-spawn mechanism.
- **Correctness gate** — any change must keep self-play trajectories bit-identical at a fixed seed (the parallel-determinism + movegen-determinism tests are the guard); these are memory-layout changes, not algorithm changes.
