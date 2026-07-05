# Training-phase memory-cost model

How much RAM a training run needs, as a function of the scale knobs — so buffer
size and DataLoader parallelism can be reasoned about *before* a paid run rather
than discovered by an exit-137 hours in. Companion to
[`docs/plans/fix-training-oom.md`](../plans/fix-training-oom.md) (the fix that
made this a first-class, pre-flight concern) and the guard/probe that implement
it (`training/diagnostics.py`, `scripts/benchmarks/memory_probe.py`).

## Why the peak is invisible until it kills you

The rolling replay buffer fills over `replay_buffer_games / num_eps` generations
(e.g. 60k / 10k = 6). **Peak RAM only occurs at that buffer-fill generation** —
so cheap validation runs (gens 1–2) and CI (tiny configs) never reach it, and the
OOM only appears deep into an expensive run. The whole point of the model + probe
below is to make that peak computable and measurable up front.

## The model

```
peak_RAM ≈ resident_buffer + dataloader_workers·per_worker + framework

resident_buffer = (replay_buffer_games + num_eps) · bytes_per_game
per_worker      = worker_base + prefetch_factor·batch_size·dense_position_bytes
                              + copy_fraction·resident_buffer
framework       ≈ 6 GB   (torch runtime + net + optimizer + CUDA context)
```

Constants (Blokus, from `training/diagnostics.py`, deliberately generous so the
guard errs toward *over*-estimating — the safe direction):

| term | value | what it is |
|------|-------|-----------|
| `bytes_per_game` | 256 KB | one buffered self-play game (compact boards + sparse policies + object overhead, symmetry-augmented) |
| `dense_position_bytes` | 106 KB | one position in flight: `(44,14,14)` planes + densified 17,837 policy, float32 |
| `worker_base` | 0.7 GB | a forkserver/spawn worker's torch re-import + heap |
| `copy_fraction` | **0.0** (was ~1.0) | fraction of the buffer each worker *copies* — see below |

## The `copy_fraction` term — the bug this closed

`forkserver`/`spawn` DataLoader workers receive a **pickled copy** of the
dataset. The old in-RAM `_LazyPolicyDataset` referenced every position in the
buffer, so each worker pickled a full copy — `copy_fraction ≈ 1.0`:

```
peak ≈ resident_buffer·(1 + dataloader_workers) + …
```

At 60k games (~18 GB) × 8 workers that is ~160 GB — the OOM. Plain `fork` shared
the buffer copy-on-write (`copy_fraction ≈ 0`), which is why the earlier 40k-buffer
`fork` run survived and the `forkserver` 60k run did not.

**M1 measurement** (synthetic Blokus buffer, in-RAM dataset, `forkserver`): the
pickled dataset is *exactly linear* in buffer size — 0.091 GB at 65k positions,
0.274 GB at 195k (~1.4 KB/position; production policies have more nonzeros, so
larger). Extrapolated to 60k games ≈ 3.9M positions: ~5.5 GB/worker copied →
~44 GB across 8 workers on top of the resident buffer.

**The fix (M2):** the memmap-backed dataset (`training/memmap_dataset.py`) spills
the buffer to flat memmap files once per generation and hands workers only the
paths, so they share the OS page cache — the pickled dataset is a few hundred
bytes regardless of buffer size, i.e. `copy_fraction ≈ 0`. The in-process path
(`dataloader_workers = 0`, the Mac/CPU default) is untouched and stays in RAM.

## Using it before a paid run

- **Guard (automatic):** `check_ram_budget` computes the model above and compares
  it against `0.8 ×` available RAM — the tighter of physical RAM and the **cgroup
  limit** (a container is often capped far below the host; `psutil` reports the
  host). It aborts *before* training with the estimate, the limit, and the knobs
  to lower. Runs on every `Coach` start.
- **Probe (manual, recommended for a paid run):**
  `uv run python -m scripts.benchmarks.memory_probe --config <cfg>` builds the
  **full** synthetic buffer and drives the real DataLoader at the config's worker
  count, printing measured peak process-tree RSS next to the guard estimate and
  available RAM. Run it at full scale on the target box; on a small dev box use
  `--games` to reduce (the per-worker multiplier is visible at any buffer size).

## Knobs, by memory impact

1. `replay_buffer_games` (+ `num_eps`) — the resident buffer, the dominant term.
2. `net_config.perf.dataloader_workers` — `per_worker` × this; the term M1 showed
   the old guard ignored. Post-fix it no longer multiplies the buffer, but the
   prefetch + worker-base cost still scales with worker count.
3. `net_config.perf.prefetch_factor` / `batch_size` — the in-flight dense batches.
4. `net_config.perf.pin_memory` — page-locked host copies of those batches.
