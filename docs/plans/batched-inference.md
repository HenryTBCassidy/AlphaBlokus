# Batched-inference MCTS (F3 of full-cycle-optimisation)

Sub-plan for the third F-step in [`full-cycle-optimisation.md`](full-cycle-optimisation.md). Replaces the current "one MCTS sim → one batch-of-1 GPU call" pattern with "K parallel sims → one batched GPU call per K sims", using **virtual loss** to keep the K parallel paths diversified.

Expected wall-clock impact: **~1.5-2× on top of F1+F2-8w**, mainly by recovering the GPU-contention loss F2 surfaced. Post-F2 measurement showed inference at **77.5%** of per-process time at 8 workers — that's the headroom F3 is going after.

Companion docs:
- [`docs/research/pentobi/move-generation.md`](../research/pentobi/move-generation.md) — Pentobi's actual MCTS doesn't batch (their `Search::play_in_tree` runs one sim at a time), so F3 isn't a Pentobi-port. It's the AlphaZero-paper approach.
- AlphaGo Zero (Silver et al. 2017) Methods §3 — describes virtual loss exactly. Worth reading before P3.
- [`docs/plans/full-cycle-optimisation.md`](full-cycle-optimisation.md) — master plan + F3 row this implements.

---

## Design decision: within-worker leaf parallelism with virtual loss

Two F3 designs are possible. This plan locks in **Option A**.

### Option A (chosen): within-worker batching

- Each worker still runs its own independent MCTS.
- Per "outer step", the MCTS collects `K` leaf positions via `K` parallel descents from the root, using virtual loss to bias each descent toward an unvisited path.
- One batched `nnet.predict_batch(K boards)` call replaces K separate `predict(board)` calls.
- All K results are backpropped, virtual losses removed.

### Option B (rejected for now): cross-worker inference server

- A separate process collects leaf-eval requests from all 8 workers.
- Batches across processes via shared memory or a queue.
- Better asymptotic GPU utilisation because it coalesces requests across processes too.
- Significantly more complex: IPC, process lifecycle, error handling, debuggability.

### Why Option A

1. **Simpler — no IPC.** Single-process MCTS with a batch-of-K predict call. Trivially testable in unit tests; no second process to spawn, monitor, or shut down.
2. **Composes naturally with F1.** Each worker independently batches its own MCTS. The 8-worker setup post-F3 is just 8 copies of single-worker-batched MCTS, each making batched calls. No coordination needed.
3. **Catches most of the gain.** GPU contention at 8 workers is bad because each worker is issuing batch-of-1 launches with high overhead-to-compute ratio. With Option A, each worker issues batch-of-K (K=8 or 16) launches — fewer launches, better compute-per-launch. The benefit per worker is roughly K-fold-ish on inference, and inference is 77.5% of per-process time post-F2.
4. **Option B can build on top.** If Option A doesn't fully saturate the GPU (likely it won't — 8 workers × K=16 isn't a full GPU's worth of batch), Option B becomes a follow-up that wraps Option A's `predict_batch` with a server. The MCTS-side code wouldn't change.

### When to revisit (Option B)

If after F3 lands the inference share is *still* > 50% of per-process time, or the GPU still shows contention waits in profiling, Option B becomes worth a new sub-plan. Don't try to retrofit it into Option A mid-flight.

---

## Why this is the right next step

Three reasons F3 is the right thing to do now:

1. **Inference is now the binding constraint.** Post-F2 at 8 workers, 77.5% of per-process time is `nnet.predict()` calls. Move-gen is down to 8.7%. The biggest remaining slice is inference, and F3 attacks it directly.
2. **F2 made F3 high-leverage.** Pre-F2 the inference share was ~37% — F3 would have been a modest win. Post-F2 it's 77% and the GPU is contention-limited at 8 workers. F3's batched calls cut both launch overhead AND queue contention.
3. **The infrastructure is ready.** F1 brought workers; F2 surfaced the bottleneck cleanly. We have a working `predict()` path and an existing MCTS with detailed profiling instrumentation we can re-use to verify the speedup phase by phase.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| P0 | Establish baseline. Use the F2-8-worker measurement from 2026-05-28 (`temp/profile_baseline_f2_w8_2026-05-28/`). Per-process inference share **77.5%** is the slice we're attacking. No need to re-run | 5 min | High | ✓ Confirmed: self-play 6.49s/game, total 4.3 min (7.58× vs serial); inference 77.5% / move-gen 8.7% of per-process time |
| P1 | Add `predict_batch(boards: Sequence[IBoard]) -> tuple[list[NDArray], list[float]]` to `INeuralNetWrapper`. Implement in `BaseNNetWrapper` by stacking boards into a single tensor and calling the network once. Verify it produces identical results to N separate `predict()` calls | 1-1.5 hr | High | ✓ Added to interface + `BaseNNetWrapper`; matches serial `predict()` to 1e-6 on TTT |
| P2 | Write tests pinning **K=1 batched ≡ existing serial** equivalence. Run on TTT (fast) and a few representative Blokus positions. Establishes the safety net before the real refactor. Also pins the determinism contract: same seed + same batch size = same MCTS visit counts | 1 hr | High | ✓ `tests/test_core/test_batched_inference.py` — 5 tests: TTT + Blokus equivalence, single-board, order-independence, MCTS visit-count determinism |
| P3 | Implement virtual loss in `core/mcts.py`. Add per-(state, action) virtual visit counter. Modify UCB formula to include virtual loss. Add functions to apply/remove virtual loss along a path | 2-3 hr | High | ✓ `VIRTUAL_LOSS=3`, `virtual_visits` dict, `_select_action` (vl-aware, exact pre-F3 arithmetic when vl=0), `_remove_virtual_loss`; bookkeeping verified balanced |
| P4 | Refactor `search()` from recursive single-leaf to iterative leaf-collection. Need to track each descent's path (state-action sequence) so we can backprop + remove virtual loss after the batched eval | 3-4 hr | High | ✓ `_descend_to_leaf`/`_expand_leaf`/`_backprop` + `_Descent` dataclass; `search()` kept as size-1 batch; per-descent path tracked |
| P5 | Wire batched inference. In `get_action_prob`, replace the inner `for _ in range(num_sims)` loop with `for _ in range(num_sims // K): collect K leaves → predict_batch → backprop K`. Add `mcts_batch_size: int = 1` to `MCTSConfig`. ``K=1`` keeps the existing path bit-identical | 1-2 hr | High | ✓ `_simulate_batch` with leaf-dedup; `mcts_batch_size` added (default 1); K=1 bit-identical (golden test), remainder batch keeps total sims = num_mcts_sims for all K |
| P6 | Validate. Run the K=1 equivalence test + the K=N self-consistency test on the dev_5000 fixtures. Run the F2 determinism test with F3 enabled to confirm no regression | 1 hr | High | ✓ K=1 golden bit-identical; K>1 deterministic (K=4/8/16); K=N self-consistent on Blokus dev positions; full suite **248 passed** (F2 determinism suite green with F3 present) |
| P7 | Measurement. Re-run `scripts/run_benchmark.sh --config run_configurations/profile_baseline_f4.json --num-workers 8 --output-dir temp/profile_baseline_f2_f4_w8_<date>` with `mcts_batch_size=8` and `mcts_batch_size=16`. Compare per-process component split against the F2 8w baseline. Update master plan progress tracker | 1 hr | High | |
| P8 | Microbenchmark (optional but cheap). Time `predict_batch(K)` vs K serial `predict()` calls on the GPU directly. Confirms the GPU-side speedup independent of MCTS overhead | 30 min | Medium | |
| P9 | Archive plan. `git mv` to `docs/plans/archive/`. Append a Scope additions section if anything landed beyond this checklist. Update master plan F3 row to **Done** ✓ | 5 min | Low | |

Total: ~12-15 focused hours.

---

## P1. `predict_batch` on the wrapper

Today `INeuralNetWrapper.predict(board)` returns `(policy, value)` for one position. F3 needs a batched analogue:

```python
def predict_batch(
    self, boards: Sequence[IBoard],
) -> tuple[list[NDArray], list[float]]:
    """Run the network on N boards in a single forward pass.

    Equivalent to ``[self.predict(b) for b in boards]`` but executes
    the forward pass once with batch dimension ``len(boards)``.
    Crucial for F3 — the speedup comes entirely from collapsing N
    batch-of-1 GPU calls into 1 batch-of-N call.

    Returns:
        ``(policies, values)`` — lists of length ``len(boards)``,
        each policy a length-``action_size`` numpy array, each value
        a Python float in ``[-1, 1]``.
    """
```

Implementation in `BaseNNetWrapper`:

```python
def predict_batch(self, boards):
    # Convert each board to its multi-channel array, stack, push to device
    arrs = [b.as_multi_channel(1) for b in boards]
    tensor = torch.tensor(np.stack(arrs), dtype=torch.float32)
    if self.net_config.cuda:
        tensor = tensor.contiguous().cuda()
    self.nnet.eval()
    with torch.no_grad():
        log_pi, v = self.nnet(tensor)
    policies = torch.exp(log_pi).cpu().numpy()
    values = v.view(-1).cpu().numpy()
    return list(policies), [float(x) for x in values]
```

Almost trivially this should match `predict()` modulo float-precision noise. P2's test pins that exact equivalence.

The TTT and Blokus wrappers shouldn't need changes — the base class handles the batching uniformly.

---

## P3. Virtual loss

Virtual loss biases the K parallel descents toward different paths by **temporarily pretending visited actions had unfavourable outcomes**. Two ways to implement:

### Counter-based (simpler, what we'll do)

Add `self.virtual_visits: dict[StateAction, int]` parallel to `self.visit_counts`. When descending, increment `virtual_visits[(s, a)]` by some count (typically 1-3). When computing UCB:

```
effective_visits = visit_counts[(s,a)] + virtual_visits[(s,a)] * VLOSS_WEIGHT
effective_q = (q_values[(s,a)] * visit_counts[(s,a)] - virtual_visits[(s,a)] * VLOSS_WEIGHT) / max(effective_visits, 1)
u = effective_q + cpuct * prior * sqrt(state_visits) / (1 + effective_visits)
```

After backprop completes for a path, decrement `virtual_visits` along that path.

`VLOSS_WEIGHT` is a small constant — AlphaGo Zero used 3. Effectively says "treat each in-flight visit as if it had returned a -3 reward" — discouraging that path.

### Why this works

If sim_1 picks child A, virtual_visits[(s, A)] becomes 1. When sim_2 evaluates the same node, action A's UCB is reduced relative to other actions, so sim_2 picks a different child. Over K descents, the K paths spread across the tree.

The trade-off: virtual loss biases the search slightly. With K large relative to per-node visit count, the bias matters. With K small (8-16) and total sims (300+), the bias is small. The AlphaZero paper used K up to 8 with no observed quality loss.

---

## P4. Iterative leaf collection

The current `search(canonical_board)` is recursive — descends to leaf, evaluates with NN, returns value, backprops on the way back up. For F3 we need to **separate descent from evaluation**: K descents collect K leaves first, then ONE batched eval, then K backprops.

The descent needs to return:
- The leaf's `canonical_board` (the position to evaluate)
- The path taken: list of `(state, action)` pairs from root to leaf, so we can backprop + remove virtual loss

```python
def _descend_to_leaf(self, canonical_board: IBoard) -> tuple[IBoard, list[tuple[StateKey, int]]]:
    """Descend from root, picking UCB-best action at each node (with
    virtual loss applied), until reaching a leaf or terminal state.

    Returns the leaf board + the path taken (state-action pairs).
    Applies virtual loss along the path AS we descend, so subsequent
    descents diversify.
    """
```

Then in `get_action_prob`:

```python
K = self.config.mcts_batch_size
for batch_step in range(self.config.num_mcts_sims // K):
    leaves = []
    paths = []
    for _ in range(K):
        leaf, path = self._descend_to_leaf(root_board)
        leaves.append(leaf)
        paths.append(path)
    # ONE batched GPU call
    priors_list, values_list = self.nnet.predict_batch(leaves)
    # Backprop + remove virtual loss
    for leaf, path, priors, value in zip(leaves, paths, priors_list, values_list):
        self._expand_leaf(leaf, priors)
        self._backprop(path, value)
        self._remove_virtual_loss(path)
```

Edge cases to handle in the iterative descent:
- Reaching a terminal state during descent: no NN eval needed, just backprop the terminal value
- Two descents reaching the same not-yet-expanded leaf: the second one's expansion would be redundant. Easiest fix: deduplicate the leaves before predict_batch, share the result back to each duplicate's backprop. Or accept the redundancy as a small inefficiency for now (only matters for early sims when the tree is small)

---

## Out of scope for this plan

- **Cross-worker inference server (Option B above).** Considered, deferred. If Option A leaves GPU under-saturated, opens a new sub-plan then.
- **`predict()` deprecation.** The new `predict_batch` doesn't replace the existing `predict` — keep both. Eventually we might want `predict` to just call `predict_batch([board])`, but not in this plan.
- **Dirichlet root noise.** AlphaGo Zero adds noise to the root policy priors for exploration. We don't do this currently; F3 doesn't change it. Could be a separate optimisation later.
- **Tree-parallel MCTS** (multiple threads sharing one tree). Different technique, different trade-offs, not on our roadmap.
- **Mixed-precision (fp16) inference.** Tangential to F3. Could compound with batched inference for further GPU speedup later.

---

## Notes for whoever picks this up

- **K=1 must stay bit-identical to today's MCTS.** This is the safety check that proves the refactor didn't break anything. If the K=1 test fails, the refactor has a bug — fix before moving on.
- **Virtual loss bookkeeping is error-prone.** Every increment needs a matching decrement. Forget one and you'll see q-values drifting strangely across moves. The path-tracking design makes this explicit (each descent records its path; the decrement loop iterates the path) — don't try to optimise it away.
- **Two descents hitting the same leaf** is a real scenario in early-game when the tree is small. Don't crash; deduplicate or accept the redundancy. Test on the empty-board case explicitly.
- **`mcts_batch_size` is a config knob, not a constant.** Different runs may want different K (smaller for short MCTS runs, larger for long ones). Default `K=1` so the new code is opt-in.
- **GPU memory matters.** Batched inference uses K× the activation memory of a single call. With our small net (64f×4b) and K=16, this is sub-100MB — fine on the 3060 Ti. Worth checking before scaling K above 32.
- **Don't try to share the MCTS tree across workers.** F3 is per-worker batching. Cross-worker sharing is Option B and a different sub-plan.
- **Profiling instrumentation should still work.** The existing `_total_inference_time_s` counter measures time-in-`predict()`. With batched inference, it'll measure time-in-`predict_batch()` which is fewer calls but each one bigger. The component split should still be meaningful — verify after P5.
