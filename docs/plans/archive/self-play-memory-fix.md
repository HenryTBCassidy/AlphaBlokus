# Self-Play Memory Fix — Stop the Training-Step OOM

**Status: ✅ COMPLETE (2026-06-05) — shipped on `main` (PR #12).** Per-batch densification removed the gen-3 training-step OOM. Memory verified locally: a synthetic realistic-size buffer peaks at **~10–13 GB** at the gen-3 config (vs the **27 GB** that crashed the original run), and 271 tests pass. Live confirmation folds into the next training run. Archived.

Separate body of work from optimisation: that was about making training *faster*; this was about making it *not crash*. (General time+memory profiling is the [profiling investigation](../../research/self-play-speed-investigation.md).)

---

## The measured symptom

Run `blokus_scaled_15_20260603_222031` **crashed at generation 3** (~1.0 h in). W&B at death: `system.memory_percent` → **100%**, `proc.memory.availableMB` → **24.6 MB**, `proc.memory.rssMB` → **27,256 MB** (the 28 GB WSL cap). The log stopped mid-training with no traceback — the OOM-killer (SIGKILL) signature. Two nets were accepted first (Elo 400 → 620), so training worked; it just ran out of RAM. GPU was idle. **An in-WSL OOM at the training step.** Disk was never the constraint (942 GB free).

## The cause (read from the code)

`base_wrapper.train()` materialised the whole buffer at once:
```python
boards_np, raw_pis, vs_np = zip(*examples)            # whole buffer
pis_np = [as_dense(p, action_size) for p in raw_pis]  # ← densifies EVERY example's policy
policies = torch.tensor(np.array(pis_np))             # ← ~10 GB dense array at gen 3
```
By gen 3 the buffer reached full size (2 generations of lookback), so this transient densification spiked to ~15–20 GB and tipped over the 28 GB cap. The sparse-policy fix (M1) shrank the *stored* policy but `train()` handed it all back as dense — so it only delayed the OOM (gen 1 → gen 3) rather than removing it. The fix: densify per mini-batch, never the whole buffer.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| M1 | **Sparse `pi` storage** — store `(indices, values)` not the dense 17,837-vector | — | — | ✅ |
| M2 | **Per-batch densification** — a `Dataset` (`_LazyPolicyDataset`) that densifies each mini-batch, not the whole buffer (~18 MB peak vs ~15 GB). The fix. | 1.5 hr | High | ✅ |
| M3 | **Correctness gate** — bit-identical by construction (same tensors, same shuffle RNG); 271 tests green | 1 hr | High | ✅ |
| M4 | **Verify** — memory verified locally (~10–13 GB peak vs the 27 GB OOM, synthetic-buffer check); live-run confirmation folds into the next training run | — | High | ✅ |

---

## M2. Per-batch densification *(the fix)*

Replaced the up-front densify in `train()` with `_LazyPolicyDataset` — a `torch.utils.data.Dataset` that holds the sparse examples and densifies in `__getitem__`, so only `batch_size` (256) dense policy vectors exist at once. No change to the training maths — identical batches reach the optimiser, materialised lazily. Removes the gen-3 spike.

## M4. Verify

Profiled a single `train()` on a synthetic realistic-size buffer (sparse `pi` + dense board): the new per-batch path peaks at **~10–13 GB** at the gen-3 config, where the old up-front densify allocated a **~10 GB** dense policy array (+ copies) that drove RSS to 27 GB. The live 15-gen run is the production confirmation (watch `proc.memory.rssMB` stay flat past gen 3).

---

## Deferred (not needed — fix sufficient)

Two further memory-*efficiency* levers were scoped but **not built**, because per-batch densify alone brought the peak well under the cap. If a future config needs them, they belong on the optimisation track, not here:

- **Compact board storage** — `uint8`/bit-packed planes (4–32× smaller than `float32`), cast per batch.
- **Train-from-disk** — stream the lookback window from the `SelfPlayHistory/` parquet instead of holding the in-RAM buffer.

## Note

**Symmetry was sidestepped, not reworked** — the fix sparsifies *after* `get_symmetries` runs on the dense `pi`, so no action-index remap was needed.
