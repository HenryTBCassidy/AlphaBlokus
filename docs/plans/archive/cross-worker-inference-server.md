# Cross-Worker Inference Server (Option B)

> **Outcome (2026-06-03): built, correctness-validated (bit-identical), but measured 0.99× — no speedup. Not adopted; kept behind the default-off `inference_server` flag.** The premise expired when F4 shrank the net 21.6×: GPU inference is no longer the bottleneck, so the ~30% self-play GPU util is CPU-bound idle, not batch-starved. See [Result](#result-2026-06-03-implemented--validated-but-no-speedup--not-adopted). Real remaining levers are CPU-side — being investigated separately.

Build a dedicated inference-server process that batches MCTS leaf evaluations **across all self-play/arena/Elo workers** into single large GPU calls, instead of each worker issuing its own small batched calls (F3 / "Option A"). This is the deferred follow-up to F3, promoted because measurement shows the GPU is badly under-saturated: during self-play the RTX 3060 Ti runs at **~28–30% utilisation** (and ~35% in Elo), even though inference is **77.5%** of per-worker time. The win is recovering the gap between F3's *isolated* GPU-batching potential (**11.5× at K=16, ~20× at K≥32** — P8 microbenchmark) and the **1.86×** we actually realised end-to-end, which is lost entirely to 8-worker GPU contention.

**Prerequisite:** F3 (batched inference, "Option A") is landed — this builds directly on the per-worker `predict_batch(K)` call site and does not change MCTS leaf selection. **Companion docs:** [`archive/batched-inference.md`](batched-inference.md) (Option A + the Option-B design notes this expands), [`full-cycle-optimisation.md`](full-cycle-optimisation.md) (master tracker — add the result there), [`../08-TRAINING-ESTIMATES.md`](../../08-TRAINING-ESTIMATES.md).

**Expected speedup:** **~1.5–2.5× on self-play wall-clock (central estimate ~2×)**, bounded below by the per-worker CPU tree-traversal that stays serial between batches (Amdahl floor ≈ 1/(1−0.775) ≈ 4.4× absolute ceiling on the inference slice; realistically ~2×). **Bonus:** one copy of the net on the GPU instead of eight (memory saving). The C1/C9 measurements confirm or revise this.

**Safety invariant (carried from F3):** the server is opt-in behind a config flag; with it **off**, the run is **bit-identical** to today's F3 path. Inference is per-row independent in eval mode, so a board's evaluation does not depend on its batch-mates — results stay reproducible regardless of how requests are coalesced.

---

## Result (2026-06-03): implemented + validated, but **no speedup — not adopted**

Built, correctness-proven, and benchmarked on the RTX 3060 Ti. Outcome:

| Path | Self-play s/game | GPU util | Notes |
|------|------------------|----------|-------|
| server-OFF (F3) | **0.76** | mean 31% | per-worker batching |
| server-ON (F5), 5ms wait | 0.94 (**0.81×, slower**) | mean 25% | default wait too high |
| server-ON (F5), 1ms wait | **0.76 (0.99×, a wash)** | mean 20% | tuned best case |

(8 workers, 150 sims, K=16, 24 games, conv/F4 net.) **Correctness is perfect** — server-on is bit-identical to server-off (max policy/value diff = 0.0, even in fp16).

**Why no gain — the F3-era premise no longer holds.** The server was motivated by F3's finding that inference was **77.5%** of per-worker time on the **old 7.3M-param FC net**, with the GPU contended. But **F4 shrank the net 21.6× (→ 340K params)**, making a forward pass cheap. So the ~30% self-play GPU utilisation is **not** "GPU starved for a bigger batch" — it's **CPU-bound idle** (per-worker move-gen + MCTS tree traversal in Python). Batching cheap inference harder doesn't fill that idle, and the per-leaf IPC round-trip *adds* latency — which is why GPU util went *down*, not up. Net effect: a wash at best.

**Decision: do not adopt.** The implementation is correct and safe (default-off flag, bit-identical), so it stays in the tree **parked** — it would only pay off if the net grows large enough that GPU inference becomes the bottleneck again (the pre-F4 regime). Real remaining speed levers given a CPU-bound profile: the adaptive-sim endgame taper (`IDEAS.md` I1, cheap), further CPU-side move-gen/MCTS work (Amdahl-limited), or cloud GPUs for scale. C8 (arena/Elo routing) is deferred — nothing to extend.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| C1 | Baseline + batch-size target: record the F3-8w before-numbers (self-play s/game, GPU ~30%, inference 77.5%); refresh the `predict_batch` saturation sweep on the **conv** net (K up to 64/128) to pick the server's max batch `N` | 1 hr | High | Partial — baseline measured (self-play GPU ~30%); conv-net `N` sweep pending PC/GPU |
| C2 | Inference-server core: transport-agnostic batching loop (size-OR-timeout flush, batch capped at `max_batch`) with `predict_fn` + request-source injected; unit-tested with synthetic requests, no MCTS/IPC/CUDA | 4–6 hr | High | ✅ `core/inference_server.py` + `tests/test_core/test_inference_server.py` (4 tests: routing/per-row-equivalence, size-cap, timeout flush, policy rules) |
| C3 | Low-latency IPC transport: shared-memory tensors for request (board planes) and response (policy+value); round-trip latency microbenchmark vs an in-process call | 4–6 hr | High | ✅ `core/inference_channel.py` (per-worker slot, shared-mem arrays, Queue+Event signalling) + multiprocess round-trip test |
| C4 | Worker-side client: swap the F3 `predict_batch(K)` call site to "submit to server, block for result", behind a `inference_server: bool` config flag (off = in-process F3 path) | 3–4 hr | High | ✅ `predict_encoded` refactor in `base_wrapper` (265 tests still green) + `InferenceClientNet` + client test |
| C5 | Orchestrator lifecycle: spawn + supervise the server alongside the worker pool; clean startup/shutdown; handle a worker or the server dying without hanging the run | 3–4 hr | High | ✅ self-play path in `parallel_self_play.py` (flag-gated; off = byte-identical) + game-only worker builder; smoke ran end-to-end on GPU |
| C6 | Determinism + correctness tests: golden bit-identical eval (server-on vs server-off) on fixed boards; full-run reproducibility at a fixed seed; equivalence over 1,000+ random positions | 3 hr | High | ✅ `scripts/validate_inference_server.py` on GPU: **bit-identical** (max policy/value diff = 0.0, fp16) + smoke |
| C7 | Flush-rule config + sweep: expose `server_max_batch` (N) and `server_max_wait_ms` (T); short sweep to set sane defaults (no fragile single-point tuning — the OR-rule self-corrects) | 2 hr | Medium | ✅ config fields added; `max_wait_ms` 5→1 was the only sensitive knob (see C9) |
| C8 | Route arena + Elo through the server too (two-net games via `run_two_player_games_parallel`); confirm the eval phases get the same batching benefit | 3 hr | Medium | `Deferred` — C9 showed no speedup to extend (see Result) |
| C9 | End-to-end measurement: re-run `scripts/run_benchmark.sh` at 8 workers, server on vs off; record self-play s/game, GPU util, total wall-clock and the speedup in the master tracker | 1 hr | High | ✅ `scripts/benchmark_inference_server.py`: **0.99× (no speedup)** — see Result |
| C10 | Hardening + docs: server-crash recovery, optional auto-adaptive batch window (stretch), update `REMOTE-TRAINING.md` + master plan, then archive this plan | 2–3 hr | Medium | Docs: this Result section + master tracker updated. Not adopted → kept behind default-off flag, parked |

---

## C1. Baseline + batch-size target

**Current state.** Measured 2026-06-03: during self-play the 3060 Ti runs **~28–30%** GPU util (~2 GB / 8 GB), ~35% in Elo. F3 8-worker baseline (2026-06-01): self-play 3.73s/game, inference 77.5% of per-worker time, ~14× vs serial. P8 microbenchmark: `predict_batch` GPU-side gain 9.23× (K=8), 11.53× (K=16), saturating ~20× at K≥32 — on the **old FC net**.

**Fix.** Lock the before-numbers in the master tracker. Re-run `scripts/benchmark_predict_batch.py` on the **conv (F4) net** (340K params — much smaller, so the saturation point likely shifts) for K up to 64/128, to choose the server's max batch `N` (the point past which GPU throughput stops improving). This `N` is the only "tuning" number that materially matters; the time window `T` just backstops it.

**Effort:** 1 hr (mostly the sweep; numbers already mostly in hand).

---

## C2. Inference-server core

**Fix.** A standalone process (`core/inference_server.py`) that:
- loads the net once onto the GPU (eval mode, fp16);
- accepts eval requests (board planes + a request id);
- accumulates them and **flushes when either the batch reaches `N` positions OR `T` ms elapse, whichever first** — the self-correcting rule, no guessing;
- runs one `predict_batch` forward pass and returns each result tagged with its request id.

Unit-test in isolation with synthetic requests (correct routing, flush triggers, throughput vs serial). No MCTS or IPC yet — pure server logic.

**Effort:** 4–6 hr.

---

## C3. Low-latency IPC transport

**Current state / risk.** This is *the* crux. In the MCTS hot loop a worker submits and **blocks** for its result thousands of times per game. Naive IPC (pickle a tensor down a `Queue`) can cost more than the local GPU call it replaces — making the whole thing slower.

**Fix.** Use **shared-memory tensors** (`torch.multiprocessing` shared tensors / a shared-memory ring buffer): workers write board planes and read results from memory the server also maps, with only small control messages on a queue. Microbenchmark the round-trip latency in isolation and confirm it's well under the GPU-batch time it unlocks.

**Effort:** 4–6 hr.

---

## C4. Worker-side client

**Current state.** Each worker, post-F3, calls `predict_batch(K)` locally (its own GPU call).

**Fix.** Behind a new `inference_server: bool` config flag, replace that single call site with "write the K leaves to the shared buffer, signal the server, block on the result." MCTS leaf *selection* (virtual loss, K collection) is unchanged — only *where* the evaluation runs. Flag **off** → in-process F3 path, bit-identical.

**Effort:** 3–4 hr. **Depends on:** C2, C3.

---

## C5. Orchestrator lifecycle

**Fix.** In `core/parallel_self_play.py`, spawn the inference server alongside the worker pool, pass workers the shared-memory handles, and manage the lifecycle: clean startup ordering (server ready before workers submit), graceful shutdown, and a watchdog so a dead worker or a dead server fails the run loudly instead of hanging it.

**Effort:** 3–4 hr. **Depends on:** C4.

---

## C6. Determinism + correctness tests

**Fix.** Inference is per-row independent in eval mode, so server-on must produce **bit-identical** evaluations to server-off. Tests: (a) golden eval equality on fixed boards regardless of batch-mates; (b) a full short run reproducible at a fixed seed, server-on vs off; (c) equivalence over 1,000+ random positions (mirrors the F2 equivalence-test discipline). Make the full suite pass before wiring into real runs.

**Effort:** 3 hr. **Depends on:** C4.

---

## C7. Flush-rule config + sweep

**Fix.** Expose `server_max_batch` (N, from C1) and `server_max_wait_ms` (T). Short sweep over a couple of (N, T) pairs to set defaults — deliberately light, because the size-OR-timeout rule self-corrects (fast arrival hits N, slow arrival hits T). Note an *optional* future refinement (auto-adapting N/T to observed arrival rate) but don't build it now.

**Effort:** 2 hr. **Depends on:** C5.

---

## C8. Route arena + Elo through the server

**Current state.** Arena and Elo are two-net games via `run_two_player_games_parallel` (also parallel, also inference-bound — measured ~3.4s/game arena).

**Fix.** Have the eval phases use the server too. Two nets are in play (new vs prev/baseline), so either run two server instances or tag requests by net id. Confirm the eval phases get the same batching benefit.

**Effort:** 3 hr. **Depends on:** C5.

---

## C9. End-to-end measurement

**Fix.** Re-run `scripts/run_benchmark.sh --num-workers 8` server-on vs server-off, same config and seed. Record self-play s/game, GPU utilisation (expect it to climb from ~30% toward saturation), arena/Elo s/game, and total wall-clock. Log the achieved speedup in the `full-cycle-optimisation.md` progress tracker. **This is the number that decides whether the complexity earned its keep.**

**Effort:** 1 hr. **Depends on:** C7, C8.

---

## C10. Hardening + docs

**Fix.** Server-crash recovery (restart or fail-fast cleanly); document the new flag + architecture in `REMOTE-TRAINING.md` and the master plan; capture the result and any scope additions; then `git mv` this plan to `archive/`. Optional stretch: the auto-adaptive batch window from C7.

**Effort:** 2–3 hr. **Depends on:** C9.

---

## Risks & open questions

- **IPC latency is the make-or-break (C3).** If the shared-memory round-trip isn't fast enough, the win evaporates. Microbenchmark it *before* full integration — if it can't beat the local call's overhead, stop and reconsider.
- **Single point of failure.** A dead server stalls all workers (C5 watchdog mitigates).
- **Debuggability.** Multi-process is harder to debug than F1–F4's independent workers — hence the strong opt-out flag and golden equivalence tests.
- **Amdahl floor.** Per-worker CPU tree-traversal stays serial between batches; that caps the realistic gain at ~2× even if the GPU fully saturates. If C1/C9 show the floor is lower than hoped, the honest call may be "good enough, ship cloud GPUs instead."
- **Bigger picture.** For *scale* (games needed to beat Pentobi), more/faster GPUs (the cloud 4090 ladder) remain the dominant lever; this server maximises the single-PC iteration loop, it doesn't replace that.

---

## Complementary cheap win (not in this plan)

**Adaptive sim budget** (`docs/IDEAS.md` I1): taper `num_mcts_sims` down in the low-branching endgame (300 sims for <30 legal moves is gross overkill). Orthogonal to this server, near-zero infrastructure, reclaims real self-play wall-clock. Worth a tiny separate plan; bundle the measurement if pursued alongside.
