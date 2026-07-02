# JAX Self-Play Pipeline — Throughput and A/B Validation

Results for plan steps G7/G8 of `docs/plans/jax-selfplay-pipeline.md` (executed 2026-07-02 on
the box: RTX 3060 Ti 8GB, i5-13600KF, jax 0.10.2 + CUDA 12, mctx 0.0.71). Companion to the
search-agreement validation recorded in the plan's G4 note
(`temp/benchmarks/validate_jax_search.json`).

---

## 1. Where the time actually goes (component profile)

At the run3 net (128f×8b), S=400, B=128, bf16 — one batched move's search decomposed:

| Component | Wall time |
|---|---|
| 400 × net forward (B=128) | 1.55 s |
| 400 × full recurrent body (step + game-end + legal masks + forward + top-k) | 2.13 s |
| **Full mctx search (S=400, K=128)** | **32.7 s** |

**94% of search wall-time is mctx's internal per-simulation tree machinery**, not the
environment (solved in the spike) and not the net. The overhead scales with `top_k` — the
per-(node,action) tree arrays it reads/writes each sim — and is nearly flat in batch size:

| Config (bf16, S=400) | s/move-batch | sims/s | moves/s |
|---|---|---|---|
| B=128, K=128 | 32.7 | 1,566 | 3.9 |
| B=128, K=64 | 6.0 | 8,502 | 21.3 |
| B=128, K=32 | 4.4 | 11,644 | 29.1 |
| B=512, K=64 | 28.2 | 7,266 | 18.2 |
| B=1024, K=64 | 58.6 | 6,991 | 17.5 |
| B=1024, K=32 | 38.1 | 10,758 | 26.9 |
| B=2048, K=32 | 78.0 | 10,503 | 26.3 |
| **B=1024, K=64, S=64** (Gumbel-shaped) | **3.1** | 20,912 | **326.7** |

The spike's pre-registered "2× margin for mctx overhead" was wrong by ~7× at K=128. The G4
agreement data rescues the situation: K=64 already tracks the exact search better than
production's own K=16 virtual-loss batching (top-1 0.735 vs 0.715), so K=64 is both the
quality *and* the speed choice.

## 2. Backend throughput (same checkpoint, same flat S=400, 128f×8b)

| Backend | games/s | sims/s | VRAM |
|---|---|---|---|
| python, 16 workers, K=16, fp16 (production) | 0.91 | 11,916 | 0.4 GB |
| jax PUCT S=400 (best K/B) | ~0.9–1.0 | ~10.5–11.6k | ~2–6.5 GB |
| jax Gumbel-shaped S=64 (B=1024, K=64) | **~11.4 (est. 327 moves/s ÷ 28.6)** | 20,912 | ~3 GB |

Read: **PUCT-mode jax ≈ parity** with the whole 16-worker python pipeline — while using zero
CPU workers (the box's 16 cores are freed for anything else, e.g. a concurrent python arena) —
and **low-sim Gumbel search is ~12× production games/s** on the same card. The plan's original
2.5–3× PUCT expectation died with the mctx-overhead discovery; the G10 lever carries the
speedup instead, exactly as the spike's multiplicative-levers argument predicted.

## 3. A/B training validation (three arms, 10 generations each)

Setup: `run_configurations/ab_{python,jax,gumbel}_10.json` — 64f×4b, 1000 games/gen, buffer
5000, temp_threshold 12, Dirichlet 0.25/0.03, arena gate 50 games @0.55, Elo 50 games/gen vs
frozen gen-0. Arms differ ONLY in generation backend/search: python-PUCT S=300 (16 workers,
K=16) vs jax-PUCT S=300 (B=256, K=64, bf16) vs jax-Gumbel n=64 (max_considered 16). Evaluation
(arena/Elo/Pentobi) runs the identical python machinery in all arms.

<!-- RESULTS FILLED AFTER THE CHAIN COMPLETES -->

### 3.1 Wall-clock

| Arm | total | per-gen self-play | notes |
|---|---|---|---|
| python-PUCT | TBD | TBD | |
| jax-PUCT | TBD | TBD | |
| jax-Gumbel n=64 | TBD | TBD | |

### 3.2 Strength

| Comparison | Result |
|---|---|
| Internal Elo trajectory (vs frozen gen-0) | TBD |
| Head-to-head: jax-PUCT final vs python final (100 games, S=400, noise-free) | TBD |
| Head-to-head: jax-Gumbel final vs python final (100 games) | TBD |
| Pentobi L1 (20 games, 800 sims) per final net | TBD |

### 3.3 Verdict vs the G8 acceptance gate

TBD — gate: jax-trained net not weaker beyond noise AND ≥2× wall-clock. Expected shape from
§2: the equivalence claim rests on the jax-PUCT arm; the wall-clock claim rests on the Gumbel
arm.

## 4. Recommendations

- Production jax runs: `selfplay_backend: "jax"`, `search_policy: "gumbel"` with n≈64 sims,
  `top_k: 64`, `batch_size: 256+`, bf16. (TBD pending §3.)
- PUCT-mode wall-clock on GPU is bounded by mctx's per-sim tree traffic; if a
  no-behaviour-change speedup is ever needed, the fix is a custom fixed-shape tree (out of
  scope here — logged as future work in the plan).
- The 5070 Ti moves the net-forward ceiling (~2.5×), which matters most in the low-sim regime
  where forwards are back to being the bottleneck (S=64: 3.1s/move-batch ≈ 50% forward time).
