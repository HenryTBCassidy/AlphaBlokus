# Fix the training OOM properly — memory-fit as a first-class, pre-flight concern

**What this covers.** The v2 warm-start run was OOM-killed (exit 137 / SIGKILL) at the **first training
step of generation 6** — the generation where the 60k-game replay buffer first reaches full capacity
(3.88M positions). This is the *third* time a memory/crash issue has surfaced at scale, so this plan is
deliberately **structural**: it fixes the immediate mechanism *and* closes the systemic gap that keeps
letting these through, rather than patching one instance.

## Why this keeps recurring (read before fixing)
The three incidents had *different* root causes (dense encoding → lazy encoding fix; a fork+JAX
pin-memory deadlock → forkserver fix; now a forkserver×buffer memory blow-up). But the **pattern** has
one systemic cause worth naming:

1. **The memory peak is delayed to the buffer-fill generation.** The buffer fills over
   `replay_buffer_games / num_eps` generations (60k/10k = 6). Peak RAM only occurs *then*. So every
   cheap/short validation (gens 1–2) passes and the OOM only appears deep into an expensive paid run.
   **This is the number-one reason it slips through.**
2. **We validate "does it run?" not "does it fit in RAM at full buffer?"** CI uses tiny configs; short
   runs never reach the buffer-fill point; there is no cheap *full-scale memory* test.
3. **Fixes interact, untested at the production config.** #40 (`forkserver`) and #41 (60k buffer) were
   each validated in isolation; their memory interaction was never exercised together.
4. **The one guard we had has a blind spot.** `check_ram_budget` (`diagnostics.py:85`) estimates
   *buffer bytes + flat overhead* vs `psutil.virtual_memory().total`, but does **not** model the
   DataLoader **worker multiplication** or the container cgroup limit — so it gave false confidence.

**The mechanism (hypothesis, to confirm in M1):** #40 switched the training DataLoader to `forkserver`
(`base_wrapper.py:43` `_get_multiprocessing_context`), which recreates workers from a clean process and
**pickles the dataset to each worker** instead of sharing it copy-on-write as plain `fork` did. The
`_LazyPolicyDataset` (`base_wrapper.py:99`) is built from the `ReplayBuffer`'s in-RAM
`deque[GameExamples]` (`replay_buffer.py:38`, ~18 GB at 60k games), so N workers ⇒ ~N copies. That's why
`blokus_cloud_60` (40k buffer + old `fork` = shared) survived but v2 (60k + `forkserver` = copied) OOM'd.

**Prerequisite / relation:** builds on the O8 work in `archive/oom-hardening.md` (that fixed the
*dense-encoding* OOM; this fixes the *worker-multiplication* OOM its guard doesn't model).

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| M1 | Profile + **confirm the mechanism** — reproduce the OOM cheaply, measure peak RSS vs worker count | 2 h | High | ✅ |
| M2 | Fix the mechanism so workers don't each copy the buffer (shared-memory / on-disk-backed dataset) | 3 h | High | ✅ |
| M3 | Fix `check_ram_budget` to model worker multiplication + cgroup limit; abort pre-flight with guidance | 2 h | High | ✅ |
| M4 | Cheap full-buffer **memory probe** script — know the peak RAM before renting a GPU | 1.5 h | High | |
| M5 | Memory-cost model doc + add the probe/guard to the CLOUD-TRAINING pre-flight checklist | 1 h | Medium | |
| M6 | Validate: 60k buffer + 8 workers fits (or aborts cleanly); reproduce-then-fixed; CI green | 1 h | High | |

### M1 findings (mechanism confirmed)

Measured peak **process-tree** RSS (main + all worker children) for the current
in-RAM `_LazyPolicyDataset` + `forkserver` DataLoader on a synthetic Blokus
buffer (compact 14×14 int8 boards, sparse 150-nnz policies), on a 24 GB Mac:

| buffer | dataset pickle | w=0 | w=2 | w=4 | w=8 |
|--------|---------------|-----|-----|-----|-----|
| 65k positions (1000 games) | 0.091 GB | 1.01 GB | 2.81 GB | 4.08 GB | 6.37 GB |
| 195k positions (3000 games) | 0.274 GB | 1.81 GB | 5.10 GB | 7.53 GB | — |

- **The dataset is pickled in full to every worker**, and its size is *exactly
  linear* in buffer size (0.091 → 0.274 GB as positions go 65k → 195k, ~1.4
  KB/position in this synthetic; production policies have more nonzeros so it is
  larger). Confirms the `forkserver` per-worker-copy hypothesis — plain `fork`
  shared it copy-on-write, which is why the 40k-buffer `fork` run survived.
- Peak RSS rises monotonically with worker count; the per-worker cost is a fixed
  torch-import + prefetched-dense-batch base (~0.4–0.6 GB, buffer-independent)
  **plus** the pickled buffer copy (buffer-proportional).
- **Extrapolation to production (60k games ≈ 3.9M positions):** the pickled copy
  alone is ≈5.5 GB/worker → ≈44 GB across 8 workers, on top of the ~18 GB
  resident buffer → OOM at the buffer-fill generation. Matches the exit-137.
- **Fix direction:** the buffer-fill OOM is driven by the *buffer-proportional*
  pickled copy, so the fix must remove the buffer from what workers receive
  (memmap-backed dataset — M2). The fixed prefetch base is bounded and not the
  cause.

---

## M1. Profile and confirm the mechanism (do NOT fix blind)
Reproduce cheaply (locally / a big-RAM CPU box — no GPU needed for the memory path): build a full-size
`ReplayBuffer` (60k games of synthetic/real positions) + the training `DataLoader` at
`dataloader_workers ∈ {0, 2, 4, 8}` and measure **peak RSS** for each. Confirm whether peak scales with
worker count (the forkserver per-worker-copy hypothesis) or is flat (some other cause). Report the
measured GB-per-worker. This decides M2's fix and calibrates M3's estimate. (We over-asserted a
mechanism once already — measure first.)

## M2. Stop workers each copying the buffer
Given M1's finding, the likely fix is to make the dataset **shared, not per-worker-copied**, while
keeping the `forkserver` JAX-deadlock fix intact. Options, best first:
- **On-disk-backed dataset:** the `SelfPlayStore` already persists each generation's games to parquet
  (`replay_buffer.py:37`). Have `_LazyPolicyDataset` read positions **lazily from the on-disk parquet /
  a memmap** rather than from the in-RAM deque, so workers share the OS page cache instead of each
  holding ~18 GB. Structural + scales to any buffer size.
- **Shared-memory arrays:** put the buffer's compact-board / policy arrays in `torch`/`multiprocessing`
  shared memory so workers reference rather than copy.
- **Fallback / stopgap:** reduce default `dataloader_workers` and/or `pin_memory` for large buffers (cuts
  the multiplier but doesn't fix the ceiling — acceptable only until a real fix lands).
Keep `num_workers=0` and Mac/CPU behaviour unchanged.

## M3. Make the pre-flight guard actually catch this
`check_ram_budget` (`diagnostics.py:85`) must model the **real** peak, not just buffer bytes:
`peak ≈ buffer_bytes × (1 + workers × copy_fraction) + dataloader_transient + framework`, using the M1
per-worker number. And compare against **actually-available** memory — read the cgroup limit
(`/sys/fs/cgroup/.../memory.max`) when present, not just `psutil.virtual_memory().total` (a container
can be capped far below host RAM). On failure, **abort before training** with the estimate, the limit,
and the specific knobs to lower (`replay_buffer_games`, `dataloader_workers`, `pin_memory`). Test it
fires for the 60k+8-workers case on a small budget and passes for safe configs.

## M4. Cheap full-buffer memory probe
`scripts/benchmarks/memory_probe.py`: given a run config, build the full-size buffer + dataloader at the
config's worker count and print peak RSS + the guard's estimate-vs-actual. Runnable on the box or a
cheap big-RAM pod — **so the memory cost is known before renting a GPU.** This is the missing "does it
fit at full scale?" test that short validation runs cannot provide (the peak is at the buffer-fill gen).

## M5. Memory-cost model + protocol
Document `peak_RAM ≈ f(replay_buffer_games, dataloader_workers, pin_memory, positions/game, net)` in
`docs/research/` so scale knobs can be reasoned about without crashing, and add to
`docs/guides/CLOUD-TRAINING.md`'s pre-flight checklist: **run `memory_probe.py` (or confirm
`check_ram_budget` passes) at the full buffer before launching a paid run.**

## M6. Validate
- With M2, 60k buffer + 8 workers fits within a target RAM (show the peak-RSS drop vs M1's baseline).
- The M3 guard aborts cleanly when over-budget and passes when safe.
- Full CI green (ruff, format, mypy, base + jax tests).

---

## Notes for the executing agent
- **Measure before fixing (M1).** Style contract as usual (types, ruff, mypy strict, loguru, real
  objects in tests). Keep `--resume`, `num_workers=0`, and Mac/CPU paths unchanged.
- The through-line of this plan is **making the full-buffer memory peak visible and gated *before* we
  spend money** — that's what breaks the recurrence, more than any single mechanism fix.
- One commit per row; tick Done as each lands; archive on completion.
