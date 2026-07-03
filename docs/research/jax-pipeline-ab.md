# JAX Self-Play Pipeline — Throughput and A/B Validation

Results for plan steps G7/G8 of `docs/plans/archive/jax-selfplay-pipeline.md` (executed 2026-07-02 on
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

One operational failure worth recording: the first chain attempt crashed both jax arms with
CUDA OOM — XLA preallocates 75% of VRAM by default, starving the torch training step and the
CUDA eval workers that share the card. Fixed at the jax import gateways
(`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `core/jaxplay/__init__.py` +
`games/blokusduo/jaxenv/__init__.py`); the rerun held at 1.2–1.7 GB VRAM throughout.

### 3.1 Wall-clock (10 generations × 1000 games)

| Arm | self-play total | self-play games/s | whole run |
|---|---|---|---|
| python-PUCT S=300 (16 workers, K=16) | 3,476 s | 2.88 | 1h 11m |
| jax-PUCT S=300 (B=256, K=64, bf16) | 25,197 s | 0.40 | 7h 13m |
| jax-Gumbel n=64 (B=256, K=64, bf16) | **740 s** | **13.5** | **20 min** |

At this deliberately small net (64f×4b) the python pipeline is at its best (CPU search is
cheap relative to the tiny net) and mctx's per-sim overhead is at its relatively worst — so
jax-PUCT is 7× *slower* here, worse than the 128f×8b parity of §2. The asymmetry flips with
net size: at 128f×8b python drops to 0.91 games/s while jax-Gumbel holds ~11 (§2), i.e.
**~12× at production scale, 4.7× (self-play) / 3.5× (end-to-end) even at this
python-friendly config.**

### 3.2 Strength

| Comparison | Result |
|---|---|
| Head-to-head: **jax-PUCT** final vs python final (100 games, S=400, noise-free, alternating colours, randomised openings) | **58–37–5 → 60.5%** for jax (95% CI 50.7–69.5%) |
| Head-to-head: **jax-Gumbel** final vs python final (100 games) | **50–43–7 → 53.5%** for gumbel (95% CI 43.8–63.0%) |
| Pentobi L1 (20 games, 800 sims): python / jax-PUCT / jax-Gumbel finals | 0/20 · **1/20** · 0/20 — near-uninformative at this run size (run3 needed 30+ gens at 2× net to reach 25%); the only game won by any arm went to jax-PUCT, consistent with its head-to-head edge |
| Internal Elo trajectories | python 492→802, jax-PUCT 470→630, gumbel 470→591 — **not cross-comparable** (each run's Elo is anchored to its own random gen-0 net; the jax arm *trails* python on internal Elo yet *beats* it head-to-head). Method note: never compare Elo curves across runs; head-to-head is the arbiter. |

The jax-PUCT net being significantly *stronger* (CI excludes parity) is plausibly real signal,
not just luck: the G4 agreement data showed the top-K search tracks the exact PUCT search more
faithfully than python's K=16 virtual-loss batching — better search → better targets. Gumbel
at 16× fewer sims trains a net statistically indistinguishable from the python baseline.

### 3.3 Verdict vs the G8 acceptance gate

Gate: *jax-trained net not weaker beyond noise AND ≥2× wall-clock win.*

- **Strength: PASS, both arms** (jax-PUCT 60.5% — stronger; gumbel 53.5% — parity).
- **Wall-clock: PASS via the Gumbel arm** (3.5× end-to-end at the python-friendliest config;
  ~12× at production net size), **FAIL for PUCT mode** (parity at 128f×8b, 7× slower at
  64f×4b — mctx per-sim tree traffic, see §1).

**Decision: flip Blokus production configs to `selfplay_backend: "jax"` +
`search_policy: "gumbel"`.** PUCT mode stays fully supported as the fidelity/cross-check mode
(and trains slightly stronger nets if wall-clock is no object); the python backend stays the
default for the `RunConfig` dataclass, TicTacToe, and all evaluation.

## 4. Recommendations

- Production jax runs: `selfplay_backend: "jax"`, `search_policy: "gumbel"` with n≈64 sims,
  `top_k: 64`, `batch_size: 256+`, bf16 — see `run_configurations/blokus_jax_gumbel_30.json`.
- Revisit `gumbel_max_considered` (16) and n once a longer run exists; the Gumbel paper's
  n=16–32 settings would roughly double throughput again if strength holds.
- PUCT-mode wall-clock on GPU is bounded by mctx's per-sim tree traffic; if a
  no-behaviour-change speedup is ever needed, the fix is a custom fixed-shape tree (out of
  scope here — logged as future work in the plan).
- The 5070 Ti moves the net-forward ceiling (~2.5×), which matters most in the low-sim regime
  where forwards are back to being the bottleneck (S=64: 3.1s/move-batch ≈ 50% forward time).
