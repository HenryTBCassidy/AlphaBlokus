# P2 (B1) — Vectorise `_select_action`

Subplan for **P2** of [self-play-throughput.md](self-play-throughput.md): replace the
per-legal-move Python `for`-loop in `MCTS._select_action` (`core/mcts.py`) with a single
vectorised numpy UCB computation, then `argmax`. `_select_action` runs on every descent at
every internal node and scales with Blokus's large branching factor, so it is the prime
CPU hot-spot in self-play (B1 in
[self-play-speed-investigation.md](../../research/self-play-speed-investigation.md)). The win
compounds across *every* worker config (CPU and GPU), is OS-agnostic, and needs no new
toolchain.

**Hard constraint: the change must stay bit-identical to the loop.** UCB selection drives
the whole search trajectory, so any float drift cascades into different games. The guards:

- `tests/test_core/test_batched_inference.py::test_k1_matches_pre_f3_golden` — pins exact
  visit counts + policy frozen from the pre-F3 recursive search (the K=1 golden).
- `test_batched_search_deterministic_ttt` / `test_batched_search_self_consistent_blokus` —
  K>1 determinism + well-formedness.
- `tests/test_core/test_parallel_self_play.py` — full-game 1-worker-vs-4-worker trajectory
  equality.
- A new direct equivalence test (B1.2) comparing the vectorised function against a
  reference scalar implementation across randomised stats, including in-flight virtual loss.

### Bit-identicality hazards (and how each is handled)

1. **`prior` dtype promotion.** `policy_priors[s]` is **float32** (the NN policy comes from
   `torch.exp(log_pi).float().numpy()`). In the scalar loop `cpuct * prior_a` promotes the
   float32 scalar to float64 (python-float × numpy-scalar → float64). Under numpy 2.x
   (NEP-50 weak promotion) `cpuct * priors_float32_array` *stays* float32. → Cast `priors`
   to float64 once at the top of the vectorised path so the arithmetic matches the scalar
   promotion bit-for-bit.
2. **`sqrt(state_visits)` vs `sqrt(state_visits + EPS)`.** Visited edges use
   `sqrt(state_visits)`; first-visit edges use `sqrt(state_visits + EPS)`. Both are
   loop-invariant — compute each once as a scalar (`math.sqrt`, correctly-rounded, identical
   to `np.sqrt`).
3. **Operation order.** Keep the exact left-to-right order `cpuct * prior * sqrt / denom`
   (numpy elementwise evaluates in the same order, so each element is bit-identical).
4. **The `virtual == 0` visited case must use plain `q`, not `(n*q - 0)/n`.** The two are
   not bit-equal in float. `np.where(v == 0, q, (n*q - v)/eff_n)` preserves the split.
5. **Tie-breaking.** The loop keeps the first action with a strictly-greater score, and
   `acts` is ascending, so ties resolve to the lowest action id. `np.argmax` returns the
   first occurrence of the max → identical, *provided* tied scores are bit-equal (they are,
   since each score is computed identically to the scalar version).

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| B1.1 | Vectorise `_select_action` — gather aligned `n`/`q`/`virtual` arrays, compute all four UCB branches with numpy, `argmax`; preserve the empty-virtual fast path | 2 hr | High | ✅ |
| B1.2 | Add a direct scalar-vs-vectorised equivalence test (randomised stats + virtual loss) and confirm the full determinism suite is green | 1 hr | High | ✅ |

> **Result — correctness (2026-06-15):** Bit-identical confirmed — the K=1 golden, the
> 1-vs-4-worker trajectory tests, and the new randomised scalar-vs-vectorised
> equivalence test (2000 trials × {with, without} virtual loss) all pass; full
> non-slow suite green (273 passed). Microbenchmark on this function: **2.0× at
> 150 legal moves, 2.3× at 400** (Blokus-scale); ~0.55× at TTT's 9 moves
> (numpy overhead, negligible for self-play).
>
> **Result — end-to-end throughput (2026-06-15, Linux re-baseline): ~0% — the
> function was not the bottleneck.** Same-substrate before/after on the box
> (pre-P2 `79f03f5` vs +P2 `6cfa88e`, 150-episode self-play each):
>
> | Config | Linux pre-P2 (g/s) | Linux +P2 (g/s) | Δ |
> |--------|--------------------|------------------|----|
> | gpu8 | 1.154 | 1.194 | +3.4% |
> | cpu16 | 0.723 | 0.714 | −1.2% (noise) |
> | server16 | 1.189 | 1.195 | +0.5% |
>
> The 2× function speedup vanished at the self-play level: `_select_action` is
> only ~6% of self-play wall-clock (not the suspected 22–35%), so Amdahl caps the
> end-to-end gain near zero. The per-edge dict gather (kept here) is the residual
> cost inside the function; only the arithmetic got cheaper, and the arithmetic
> wasn't the bottleneck. **Decision: keep P2** — it's bit-identical and not slower,
> so there's no reason to revert, but it is *not* a throughput lever. The real
> ceiling/bottleneck on Linux is re-characterised in **P3**
> ([p3-linux-rebaseline.md](p3-linux-rebaseline.md)). *(The big numbers vs the
> Windows baseline — 1.58×/3.38× — were the P1 OS migration, not P2.)*

---

## B1.1. Vectorise `_select_action`

**Current state** (`core/mcts.py:352`): a `for i in range(len(acts))` loop that, per legal
move, does 2–4 dict lookups, recomputes `math.sqrt(state_visits)` *every iteration*, branches
on visited/virtual, and tracks the running best with `if u > cur_best`.

**Fix.** Gather the per-edge statistics aligned to `acts` (the ascending legal-id array) and
compute the score vector in one numpy pass:

- `priors` ← `policy_priors[s].astype(np.float64, copy=False)` (hazard 1).
- `n`, `q` ← `np.fromiter` over `acts.tolist()` reading `visit_counts.get`/`q_values.get`
  (default 0 / 0.0 — `n == 0` ⇔ unvisited, since a visited edge always has `n ≥ 1`).
- `sqrt_sv = math.sqrt(state_visits)`, `sqrt_sv_eps = math.sqrt(state_visits + EPS)`.

The four scalar branches collapse to (with `eff_n = n + v`):

```
visited:   base = where(v == 0, q, (n*q - v)/eff_n)         # hazard 4
           u    = base + cpuct*prior*sqrt_sv/(1 + eff_n)    # 1+eff_n == 1+n when v==0
unvisited: u    = where(v == 0, cpuct*prior*sqrt_sv_eps,
                        -1.0 + cpuct*prior*sqrt_sv/(1 + v))
score = where(n > 0, u_visited, u_unvisited)
return int(acts[argmax(score)])
```

**Fast path.** When `self.virtual_visits` is empty — *always* the case at K=1, and the common
case at K>1 — skip gathering `v` entirely and use the two-branch (`v == 0`) form. This keeps
the golden-pinned K=1 path lean and avoids a dict lookup per edge. Only gather `v` (via
`virtual_visits.get`) when the dict is non-empty.

Why this is faster: hoists the per-iteration `math.sqrt` out (computed twice total, not
`len(acts)` times), moves all multiply/divide/compare into vectorised C, and drops the
Python-level branch + running-best bookkeeping. The remaining per-edge dict lookups stay,
but they are the cheap part; the arithmetic was the waste.

## B1.2. Equivalence test + suite green

The K=1 golden covers only TTT and `v == 0`. Add a focused unit test that pins the function
itself: populate an `MCTS` state's caches (`valid_moves_cache`, `policy_priors`,
`state_visits`, `q_values`, `visit_counts`, `virtual_visits`) with randomised values across
many seeds — including a fraction of edges with non-zero virtual loss and a mix of
visited/unvisited — and assert `_select_action` returns exactly the action chosen by a
reference copy of the original scalar loop. This guards both the fast (`v == 0`) and slow
(virtual-loss) paths directly, independent of any end-to-end golden.

Then confirm the existing guards are green:

```bash
uv run pytest tests/test_core/test_batched_inference.py tests/test_core/test_parallel_self_play.py
uv run pytest -m "not slow"
```
