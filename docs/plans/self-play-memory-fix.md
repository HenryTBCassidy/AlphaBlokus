# Self-Play Memory Fix — Sparse Storage, Per-Batch Densify, Train-from-Disk

**Status: revised after the gen-3 OOM (2026-06-04).** The sparse-policy fix shipped but was **necessary, not sufficient** — the scaled run still OOM'd. Root cause is now precisely understood (below). Companion to [`../research/self-play-speed-investigation.md`](../research/self-play-speed-investigation.md) and [`full-cycle-optimisation.md`](full-cycle-optimisation.md). To be executed on `further-full-cycle-optimisations`.

## The problem (measured 2026-06-04, run `blokus_scaled_15_20260603_222031`)

The 15-gen scaled run **crashed at generation 3** (~1.0 h in). W&B system metrics at death: `system.memory_percent` → **100%**, `proc.memory.availableMB` → **24.6 MB**, `proc.memory.rssMB` → **27,256 MB** (the 28 GB WSL cap). The log stops mid-training with no traceback — the signature of the **OOM-killer (SIGKILL)**. 2 nets were accepted before it died (Elo 400 → 620), so training itself works. GPU was idle (~2.8 GB). **It was an in-WSL out-of-memory crash at the training step — not a host issue, not WSL instability.**

A full memory analysis of the code (RAM vs disk) found **four** stacked costs:

| # | Cost | Where | Size | Status |
|---|------|-------|------|--------|
| 1 | **Train-time densification** — `train()` densifies the *entire* buffer into dense tensors in one `np.array([...])` | RAM (transient) | ~15–20 GB by gen 3 (~140k × (board 34.5 KB + dense `pi` 71 KB)) | **❌ THE OOM** |
| 2 | **In-RAM replay buffer** (`train_examples_history`) duplicates what's on disk | RAM | ~5 GB | ❌ redundant — already on disk as parquet (`SelfPlayHistory/`, 887 MB) |
| 3 | **Dense board** (`44×14×14` float32) — now the dominant *per-example* cost | RAM + disk | 34.5 KB / example | ❌ should be `uint8` / bit-packed |
| 4 | Per-worker net + CUDA context | RAM ×8 | ~2.5 GB × 8 ≈ 20 GB | ⚠ separate concern → speed doc |

**Key correction to the original plan:** M2 intended to "densify only at train-batch assembly", but the implementation densifies **all examples at once** (`pis_np = [as_dense(p) for p in raw_pis]; torch.tensor(np.array(pis_np))`). That rebuilds the full dense cost the sparse storage just saved — at gen 3 the buffer is large enough that this transient spike is the 27 GB that OOM'd. The sparse-policy work was real (it got the run from **gen 1 → gen 3**), but the densify-everything step undoes it at train time.

**Disk is a non-issue.** WSL has **942 GB free** (1 TB ext4 vhdx, C: has 642 GB free), and the self-play data is *already* written there each generation. Streaming training from disk is fully viable — we're holding a RAM copy we don't need.

**Target behaviour:** RAM stays **flat** across a run regardless of how many games accumulate — bounded by the worker footprint, not the buffer. With M2–M4 below, peak dense memory at train time is one mini-batch (~18 MB), not the whole buffer.

**Invariant:** every change must be **numerically identical** to today — same examples, same training, same games at a fixed seed. Sparse / uint8 / on-disk are storage encodings of the same data.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| M1 | **Sparse `pi` storage** — store `(indices, values)` not the dense 17,837-vector; densify on read | — | — | ✅ shipped (partial — necessary, not sufficient) |
| M2 | **Per-batch densification** — replace the densify-everything-at-once in `train()` with a `Dataset`/`collate_fn` that densifies each mini-batch (peak ~18 MB, not ~15 GB). **The direct OOM fix.** | 1.5 hr | High | |
| M3 | **Compact board storage** — store board planes as `uint8` (binary) or bit-packed (≤1 KB); densify to float32 per mini-batch alongside M2. Cuts the now-dominant per-example cost 4–32× | 1.5 hr | High | |
| M4 | **Train-from-disk** — stream the lookback window from the `SelfPlayHistory/` parquet at train time (pool closed, ~20 GB freed); drop `train_examples_history` as the in-RAM training source | 3–4 hr | High | |
| M5 | **Correctness gate** — numerically identical to the dense in-RAM path at a fixed seed (loss/grads); lossless board + `pi` round-trips; parallel-self-play determinism preserved; short full run reproduces | 1.5 hr | High | |
| M6 | **Memory benchmark** — RAM flat through a **full multi-gen run past gen 3** at the full 70k buffer + 8 workers; restore `max_queue_length=70000`; re-profile and confirm no climb | 1 hr | High | |

---

## M2. Per-batch densification *(the OOM fix)*

Today `base_wrapper.train()` does, in effect:
```python
boards_np, raw_pis, vs_np = zip(*examples)          # whole buffer
pis_np = [as_dense(p, action_size) for p in raw_pis]  # ← densifies EVERY example
policies = torch.tensor(np.array(pis_np))             # ← ~10 GB dense array at gen 3
```
Replace with a `torch.utils.data.Dataset` that **holds the sparse examples** and densifies in `__getitem__` (or a `collate_fn` that densifies a batch), so only `batch_size` (256) dense policy/board vectors ever exist at once. No change to the training maths — same tensors reach the optimiser, just materialised lazily. This is the single change that removes the gen-3 OOM.

## M3. Compact board storage

The board encoding is `44×14×14` and almost entirely binary (piece-placement planes) — yet it's stored as float32 (34.5 KB/example), now the dominant per-example cost after M1. Store as `uint8` (immediate 4×) or bit-pack the binary planes (up to ~32×), and densify/cast to float32 in the same per-batch step as M2. Keep the encoding semantics identical — only the storage dtype changes.

## M4. Train-from-disk

The lookback window is **already persisted** to `SelfPlayHistory/generation=N/…parquet` each generation, yet `train_examples_history` keeps a full RAM copy as the training source. Switch training to a disk-backed `Dataset` that reads the lookback window's parquet(s) (at train time the self-play pool is closed, so its ~20 GB is freed and disk I/O isn't competing). Drop the in-RAM history as the training source. Combined with M2+M3, training RAM is bounded by the mini-batch, independent of buffer size or worker count. Storage stays parquet — sparse `pi` columns + `uint8` board are written directly (the in-RAM↔disk densify/dense detour in `coach.save_self_play_history` goes away).

## M5. Correctness gate

The safety net — the data pipeline is exactly where a silent bug corrupts a whole overnight run. (a) lossless `densify(sparsify(pi)) == pi` and board uint8/bit-pack round-trips over 1,000+ real examples; (b) one generation's training loss/grads at a fixed seed identical to the dense in-RAM path; (c) the disk-backed `Dataset` yields the same batches (order + content) as the in-RAM path under a fixed seed; (d) parallel-self-play determinism unchanged. Full suite green before this is trusted in a run.

## M6. Memory benchmark *(the proof)*

Restore `max_queue_length=70000`, 8 workers, full config, and run **past generation 3** (where it died). Confirm `proc.memory.rssMB` stays flat across generations (≈ worker footprint, no climb), `system.memory_percent` never approaches 100, and self-play s/game doesn't drift up. This is the evidence the problem is gone — measured against the gen-3 OOM, not asserted.

---

## Notes / scope

- **Symmetry was sidestepped, not reworked.** The shipped fix sparsifies *after* `get_symmetries` runs on the dense `pi`, so the order-2 transpose still operates on dense vectors and no action-index remap was needed. Leave it that way unless profiling says otherwise.
- **Worker CUDA contexts (cost #4, ~20 GB) are out of scope here** — that's the worker-count / threading / inference-server axis tracked in the speed-investigation doc. M2–M4 make training RAM independent of the buffer; the worker baseline is the remaining floor.
- **Don't rush it into an unattended run.** Land M5 green first.
