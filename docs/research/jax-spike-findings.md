# JAX Spike Findings — Parity, Throughput, Go/No-Go

Results of the de-risk spike in `docs/plans/jax-spike.md` (J1–J7, executed 2026-07-02).
The question: can Blokus Duo rules run as fixed-shape JAX array ops, bit-identical to the
Python engine, and fast enough batched on the RTX 3060 Ti to project ≥10× self-play
throughput for a full JAX/mctx rewrite?

**One-line answer: parity is exact everywhere; the environment becomes ~50× faster than the
entire 16-worker production fleet and stops mattering; the pipeline becomes purely
inference-bound at ~3× current production sims/s on the 3060 Ti — which misses the
pre-registered 10× bar on current hardware, but converts self-play into a problem that GPU
compute and Gumbel-style low-sim search solve multiplicatively.**

---

## Environment

| | Mac (dev) | Box (measurement) |
|---|---|---|
| Hardware | MacBook, CPU only | RTX 3060 Ti 8GB, i5-13600KF 14C/20T |
| jax | 0.10.2 (CPU wheel, `--extra jax`) | 0.10.2 + CUDA 12 (`--extra jax-cuda`) |
| Code | branch `feat/jax-spike` | same, commit `8507d74` |
| GPU state | — | idle (run3 finished 02:47); nothing else on the card |

Artifacts: `temp/benchmarks/jax_spike_gpu.{html,json}` (box), `temp/benchmarks/jax_spike_1d32851_cpu.{html,json}` (Mac), raw log `temp/jax_spike_bench.log` on the box.

## Parity (criterion 1 — hard requirement): PASS, exact

- **Legal masks**: bit-exact equality with the F2/numba generator on **5,000/5,000**
  dev-cache positions, and with the slow reference generator on a 500-position stratified
  subsample (three-way oracle). `tests/test_blokusduo/test_jaxenv_parity.py`.
- **Step / game-end / scoring**: every cached action sequence replayed ply-by-ply — signed
  board, inventories, last-piece records, player-to-move all bit-identical; `get_game_ended`
  matches at every final position (both perspectives) and per-ply on a 1-in-25 stride.
  Coverage is real: 242 sequences contain passes, 251 end terminal, 16 end drawn.
  `tests/test_blokusduo/test_jaxenv_step_parity.py`.
- The branches random play can't reach (+15 all-placed, +5 monomino-last) are pinned by
  synthetic-state tests. `tests/test_blokusduo/test_jaxenv_scoring.py`.
- The full suite passes **on the CUDA backend too** (15/15 on the box) — the kernels do all
  rule arithmetic in int32 matmuls, so there is no float-tolerance parity risk.
- One real trap found and pinned: the engine's start squares are **array indices**
  (4,4)→cell 60 / (9,9)→cell 135, not board coordinates (the plan's design sketch had them
  converted through `CoordinateIndexDecoder`, which is wrong — caught by the parity suite
  in its first run, exactly as intended).

**Criterion 3 (no dynamic-shape escape hatches): PASS.** Everything is `jit`/`vmap`-clean —
no host callbacks, no per-state Python. The one Python-side cost is table construction
(~200 ms, once per process, reusing `build_move_tables`).

## Throughput (criterion 2)

### Box GPU (RTX 3060 Ti), batch sweep

| Measurement | B=64 | B=256 | B=1024 | B=4096 | B=8192 |
|---|---|---|---|---|---|
| `legal_mask` (masks/s) | 118k | 204k | 397k | **718k** | 430k¹ |
| Random rollout (mask+sample+step, steps/s) | 326k | 427k | 549k | **570k** | 558k |
| Net pseudo-self-play fp32 (steps/s) | 13.0k | 17.2k | 18.6k | 19.0k | OOM² |
| Net pseudo-self-play **bf16** (steps/s) | 28.2k | 30.7k | **31.6k** | OOM² | OOM² |
| Net forward only fp32 (forwards/s) | 12.6k | 16.8k | 19.0k | 19.4k | OOM² |
| Net forward only bf16 (forwards/s) | 25.6k | 30.3k | **31.2k** | OOM² | OOM² |

¹ B=8192 mask throughput dips: the three int32 `(B, 17837)` intermediates are ~585 MB each at that batch.
² 8 GB card; the OOM'd configs are logged as SKIPPED in the run log — not silently dropped. bf16 B≥4096 needs ~16 GB-class VRAM.

Python baselines on the same box, same session: F2/numba mask **16,375/s**, random rollout
**9,691 steps/s** (both single-process; production runs 16 such workers).
Production sims/s baseline (post-numba N6, 16 workers, run2/run3 config): **10,499 sims/s**
/ 0.947 games/s (`docs/research/numba-hot-path-results.md`).

During the GPU rollout phases the card sat at 100% utilisation drawing **~203 W** — dense
tensor work. During production self-play it reports 98% "utilisation" at **~134 W** — many
tiny kernels in flight. Same metric, very different truths; the power draw is the honest one.

### What the numbers mean

- **The environment is solved.** One GPU runs mask+sample+step at 570k steps/s — ~59× a
  single numba worker, ~3.7× an ideal 16-worker fleet, and **54× the production pipeline's
  total sims/s**. Blokus legality as three int8 matmuls is not the bottleneck of anything
  anymore.
- **The pipeline becomes pure inference.** The fused net rollout (31.6k steps/s bf16) is
  within ~1% of the bare forward throughput (31.2k forwards/s): mask, sampling and step are
  free next to the 128f×8b net. The run3-shaped net costs ~0.95 GFLOP/forward; at 31.2k
  forwards/s that's ~30 effective TFLOPS — the 3060 Ti is the ceiling, definitively.
- **Mac CPU JAX is a non-event** (26–40k masks/s ≈ one numba process), and XLA-CPU runs
  convs inside `fori_loop` pathologically slowly (~40× the bare forward) — the net rollout
  is a GPU-only measurement. Development/parity work on the Mac is fine; performance work is not.

## Verdict against the pre-registered criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Exact parity on all 5,000 positions + replay | **PASS** |
| 2 | Projected ≥10× production sims/s with 2× mctx margin (~210k raw) | **FAIL on this GPU at this net** — 31.6k raw = 3.0× production; 1.5× with the margin |
| 3 | `jit`/`vmap`-clean, no escape hatches | **PASS** |

So, per the letter of the plan: **no-go on the 3060 Ti at the run3 net size**. But the
measurement decomposition matters more than the headline, because criterion 2 didn't fail
the way a no-go was expected to fail:

- The risk the spike existed to retire — "can the rules run on GPU at all, exactly, and
  fast?" — is fully retired. The env is 50×+ overbuilt relative to any net we'd run.
- What remains is a **hardware arithmetic fact**: after numba, production already extracts
  ~1/3 of the 3060 Ti's total bf16 inference capability for this net (10.5k of ~31k
  forwards/s). No software rewrite can conjure more than ~3× on this card at this net size —
  the JAX pipeline simply collects that 3× and removes the CPU from the equation entirely.
- Post-rewrite, throughput scales with exactly two levers, both multiplicative and both
  already on the table:
  - **Sims per move.** Run3 spends ~ 392 sims/move (887,130/2,265, N6 data). mctx's Gumbel
    AlphaZero achieves equal-or-better policy improvement at 16–64 sims/move in published
    results (Danihelka et al.; the pgx training runs use it). At n=64 that's ~ 6× fewer
    sims → **~ 18 games/s projected (≈19× production)** on the 3060 Ti alone.
  - **Tensor compute.** RTX 5070 Ti ≈ 2.5–2.7× tensor throughput and 16 GB (unlocks the
    OOM'd bf16 large-batch configs) → **~45 games/s projected (≈47×)** combined with Gumbel.
    Cloud A100/H100 or a second GPU scale the same way — self-play actors are embarrassingly
    parallel.

**Recommendation:** treat this as a **conditional go**. The rewrite is only worth doing as
the package *JAX/mctx pipeline + Gumbel low-sim search + the 5070 Ti* (any two of the three
levers clear the 10× bar; all three land ~ 30–50×). A rewrite that keeps 392-sims PUCT search
on the 3060 Ti buys 3× and is not worth weeks of work. If the 5070 Ti is not purchased and
Gumbel is off the table, the honest fallback is the plan's stated no-go path: incremental
inference work on the current pipeline (bf16/torch.compile, larger K) for ≲1.5×.

## Caveats and open items

- The 2× margin for mctx tree overhead is still an estimate — nothing here ran a real tree
  search. The pseudo-self-play loop (mask → forward → sample → step) is the optimistic
  per-sim cost; Gumbel's small-n search also has per-sim gather/scatter costs.
- Games/s figures from the rollout benchmark use a fixed 72-ply horizon (~2.5× the ~28.5-ply
  average), so its games/s understate steady-state throughput; steps/s is the honest metric,
  and a production actor would recycle finished games (standard continuation trick).
- The dummy net omits the fixed action-permutation between conv-head planes and ActionCodec
  order and uses random weights — throughput-identical, strength-meaningless.
- bf16 B≥4096 and fp32 B≥8192 need more than 8 GB; on a 16 GB card the forward curve may
  still have a little headroom beyond 31k.
- Training-side throughput (the other half of a full rewrite) was out of scope, per the plan.

## If go: full-rewrite scope sketch

1. **Env module** — productionise the spike env (now promoted to `games/blokusduo/jaxenv/`) (they are the
   env; the parity suite comes with them).
2. **mctx integration** — Gumbel AlphaZero policy over batched games; recurrent_fn = step +
   encode + forward; validate policy-improvement on TicTacToe-in-JAX or directly on Duo vs
   the Python MCTS at equal sims.
3. **Net port** — Flax/NNX ResNet matching `games/blokusduo/neuralnets/net.py` including the
   action permutation; weight-conversion both ways so Pentobi benchmarking (which stays on
   the Python/GTP harness) and existing checkpoints keep working.
4. **Actor/learner loop** — on-device replay buffer, continuation-style actors, training
   step in JAX; W&B + parquet reporting glue.
5. **Eval bridge** — checkpoint → PyTorch (or direct jax inference behind
   `INeuralNetWrapper`) for the Arena/Elo/Pentobi machinery.

Estimated as a 3–5 week project solo; step 2 is the risk concentration and should be spiked
first (a week, same pattern as this one).
