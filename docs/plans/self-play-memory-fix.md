# Self-Play Memory Fix — Stop the Training-Step OOM

**Status: focused bug fix.** The scaled run crashes with an out-of-memory kill at generation 3. We already know the cause from the code — the training step densifies the *entire* replay buffer into one array — so this is a direct fix, not an investigation. **No profiling gate:** the cause is code-confirmed, and the proof is the run surviving past gen 3. (General time+memory profiling is a separate, upcoming investigation — see [`../research/self-play-speed-investigation.md`](../research/self-play-speed-investigation.md) — not part of this fix.)

Separate body of work from optimisation: that's about making training *faster*; this is about making it *not crash*. The sparse-policy storage (M1) already shipped on `main`.

---

## The measured symptom

Run `blokus_scaled_15_20260603_222031` **crashed at generation 3** (~1.0 h in). W&B at death: `system.memory_percent` → **100%**, `proc.memory.availableMB` → **24.6 MB**, `proc.memory.rssMB` → **27,256 MB** (the 28 GB WSL cap). The log stops mid-training with no traceback — the OOM-killer (SIGKILL) signature. Two nets were accepted first (Elo 400 → 620), so training works; it just runs out of RAM. GPU was idle. **An in-WSL OOM at the training step.**

## The cause (read from the code)

`base_wrapper.train()` materialises the whole buffer at once:
```python
boards_np, raw_pis, vs_np = zip(*examples)            # whole buffer
pis_np = [as_dense(p, action_size) for p in raw_pis]  # ← densifies EVERY example's policy
policies = torch.tensor(np.array(pis_np))             # ← ~10 GB dense array at gen 3
```
By gen 3 the buffer reaches full size (2 generations of lookback), so this transient densification spikes to ~15–20 GB and tips over the 28 GB cap. The sparse-policy fix (M1) shrank the *stored* policy but `train()` hands it all back as dense — so it only delayed the OOM (gen 1 → gen 3) rather than removing it.

The other RAM users are not the trigger: the ~5 GB in-RAM buffer is modest, and the 8× worker CUDA contexts (~20 GB) are freed before the training step runs. So the fix is squarely the densify-everything spike.

**Invariant:** the fix is **numerically identical** to today at a fixed seed — same tensors reach the optimiser, just materialised lazily.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| M1 | **Sparse `pi` storage** — store `(indices, values)` not the dense 17,837-vector | — | — | ✅ shipped |
| M2 | **Per-batch densification** — a `Dataset`/`collate_fn` that densifies each mini-batch, not the whole buffer (~18 MB peak vs ~15 GB). The fix. | 1.5 hr | High | ✅ |
| M3 | **Correctness gate** — training numerically identical to the dense path at a fixed seed; lossless round-trips; parallel-self-play determinism preserved; full test suite green | 1 hr | High | ✅ |
| M4 | **Verify on the run** — launch the 15-gen run; confirm `proc.memory.rssMB` stays flat and it survives **past gen 3**. The overnight run doubles as the proof (and gets results). | — | High | |

---

## M2. Per-batch densification *(the fix)*

Replace the up-front densify in `train()` with a `torch.utils.data.Dataset` that holds the sparse examples and densifies in `__getitem__` (or a `collate_fn`), so only `batch_size` (256) dense policy vectors exist at once. No change to the training maths — identical batches reach the optimiser, materialised lazily. This removes the gen-3 spike.

## M3. Correctness gate

The data pipeline is where a silent bug corrupts a whole overnight run. (a) lossless `densify(sparsify(pi)) == pi` over real examples; (b) one generation's training loss/grads at a fixed seed identical to the pre-change path; (c) parallel-self-play determinism unchanged; (d) full suite green. Do this before launching.

## M4. Verify on the run

Restore the full config (`max_queue_length=70000`, 8 workers) and launch the 15-gen run. Watch `proc.memory.rssMB` on W&B: it should stay flat across generations and sail past gen 3. That's the proof — measured against the original crash, not asserted.

---

## Deferred (only if M4 still shows pressure)

If per-batch densify alone *doesn't* keep RAM flat at the full config, two further memory levers are available — but they are **not expected to be needed**, and they're really memory-*efficiency* optimisations, so they'd move to the optimisation track rather than block the run:

- **Compact board storage** — store the `44×14×14` board planes as `uint8`/bit-packed (4–32× smaller) instead of `float32`, cast per batch.
- **Train-from-disk** — stream the lookback window from the `SelfPlayHistory/` parquet (already on disk) instead of holding the in-RAM buffer as the training source.

## Note

**Symmetry was sidestepped, not reworked** — the shipped fix sparsifies *after* `get_symmetries` runs on the dense `pi`, so no action-index remap was needed. Leave it.
