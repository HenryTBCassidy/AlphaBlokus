# Lazy board encoding in the training Dataset

## Context

`blokus_run2_bignet` (128f×8b, 1000 games/gen, `max_generations_lookback: 5`, 800 sims)
was **OOM-killed by the kernel at gen-15 training** (2026-06-22 02:21, ~30 GB RSS on a
31 GB box — SIGKILL, no traceback, log stops at "Starting Training For Generation #15").
The partial run was excellent (Elo climbing to +604 vs baseline by gen 14, still rising,
decisive accepts) so the config is right; the only blocker is memory.

**Root cause.** `_LazyPolicyDataset.__init__` (`games/base_wrapper.py`) stacks *every*
board into one dense float32 tensor up front:

```python
self._boards = torch.from_numpy(np.asarray(boards, dtype=np.float32))
```

The source board arrays already live in the replay-buffer deques, so this is a second
full copy. A board is `(44, 14, 14)` float32 = 34.5 KB; the retained window is
`lookback (5) × ~64k examples/gen ≈ 320k examples ≈ 11 GB`. Buffer (~11 GB) + the
up-front stacked copy (~11 GB) ≈ 22 GB, plus the bigger net, CUDA, and torn-down-worker
residue → over 31 GB. Policies in this same class were *already* made lazy (kept sparse,
densified per item in `__getitem__`); **boards are the one remaining up-front
densification.** `_prepare_training_data` only extends references + shuffles (no copy),
and `_ensure_eval_set`'s all-boards stack early-returns after gen 1, so the Dataset is
the single fix point.

## The change

Make boards lazy too: keep references to the buffer's board arrays (no copy) and
materialise each to a float32 tensor *per item* in `__getitem__`, exactly as policies
are. The `DataLoader` already collates one batch (256) at a time, so only a batch of
boards is ever dense at once. Peak drops from ~2× window to ~1× (~11 GB) + overhead
(~16 GB) — comfortable on 31 GB through the full 60-gen run (lookback is fixed at 5, so
the window stops growing). **No storage/parquet format change**, so run2 resumes cleanly
from its gen-14 checkpoint + self-play history.

## Checklist

| # | Item | Done |
|---|------|------|
| L1 | `_LazyPolicyDataset` holds board *references*; materialises board float32 tensor per item in `__getitem__` (mirror the policy treatment) | ✅ |
| L2 | Validate: existing training tests pass (4/4); synthetic window confirms boards referenced not copied (`np.shares_memory` True) + identical `__getitem__` output | ✅ |
| L3 | Live confirmation on the box: a big-config run clears the generation that used to OOM (the training-step window densification) | ⏳ next run |

## Validation

- `uv run pytest tests/test_core -k "train or dataset or coach"` and the base-wrapper tests
  — behaviour must be identical (same tensors out, just built lazily).
- Memory check: construct `_LazyPolicyDataset` over a synthetic buffer sized like the run2
  window and confirm constructing it no longer allocates a second full board copy.

## Outcome

The fix landed in commit `1ac3630`: `_LazyPolicyDataset` now references the buffer's board
arrays and materialises one batch at a time, removing the ~11 GB up-front copy that OOM-killed
run2 at gen-15 training. Verified in tests (boards referenced not copied; identical
`__getitem__` output; full suite green). **L3 is the only open item** — it's just live
confirmation, which will come for free on the next big-config run (paused for now); if that run
clears the previously-fatal generation, the fix is proven end-to-end. Archived as code-complete.

## Out of scope (deferred)

Shrinking the *buffer itself* (e.g. storing the 196-byte placement board and re-encoding,
or boards as `uint8`) would cut the remaining ~11 GB further and is the path to >8.5k
games/gen — but it changes the stored representation (parquet schema + self-play), would
break clean resume of run2, and isn't needed to unblock the current run. Revisit only if a
future run pushes lookback or games/gen well past run2's.
