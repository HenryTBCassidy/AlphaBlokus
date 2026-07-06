# Training the `xl` net — cheapest + fastest path, and whether to train it at all

Scoping for the decision that follows `blokus_cloud_v3`: if the large net (192f×12b) genuinely
plateaus around Pentobi Level 4, what is the most cost-effective and fast way to train the `xl`
preset (256f×16b, ~19M params)? Written 2026-07-06, **before** v3's end-of-run pool Elo exists —
everything here is *ready to execute if v3 confirms the plateau*, not a commitment. No code was
changed and no GPUs were rented for this doc.

**Evidence tagging**, as in [`cloud-training-recommendation.md`](cloud-training-recommendation.md):
**[measured]** — from an actual run (`blokus_cloud_60` on a RunPod 5090, or the 3060 Ti calibration);
**[extrapolated]** — derived from measured numbers by a stated model;
**[verify]** — confirm at rental/decision time (prices drift; v3's diagnostics aren't in yet).

---

## TL;DR

1. **`xl` is probably not the next lever — check v3's diagnostics first.** The measured weak link
   at `large` is **value-head data starvation**, not capacity ([`blokus-cloud-60-analysis.md`](blokus-cloud-60-analysis.md)
   §2–3): policy loss was still falling briskly at run end, entropy was healthy, and the ladder was
   still climbing. Doubling games/gen at `large` costs **+$0.28/gen** and doubles the independent
   value labels; switching to `xl` at the same games/gen costs **+$0.43/gen** and adds zero new
   data. Spend on data first unless v3 shows the pre-registered capacity signal (§1).
2. **If `xl` is confirmed: a single rented RTX 5090 wins.** ~50 min/gen [extrapolated from
   measured], so a 100-generation × 10k-games run is **~3.5 days and ~$82 (~£65)** — a config
   change, zero engineering. H100s are ~3× the price for ≤2× the throughput on this workload; skip.
3. **Multi-GPU is a wall-clock lever, not a cost lever.** Self-play is ~71% of an `xl` generation,
   so Amdahl caps synchronous sharding at 3.4×: 2×5090 gives 1.55× for +29% $/run; 4×5090 gives
   2.1× for +88%. A full async actor-learner removes the serial barrier but costs 2–3 weeks of
   engineering plus training-dynamics risk, to compress a run that already fits in half a week.
   **Don't build it for this decision.** The cheap variant (sharded synchronous self-play, ~2–4
   eng-days) is worth having as a capability — registered as IDEAS I6, to be picked up when runs
   get long enough to hurt (≳1 week single-card).

---

## 1. Is `xl` even the right lever? The premise check

The trigger for this doc is v3 "plateauing around Level 4". But *plateau at the gate* and *out of
capacity* are different diagnoses, and `blokus_cloud_60` already showed how easily they're
confused — its late-run stall was an LR/gate-hysteresis artifact, not capacity
([`blokus-cloud-60-analysis.md`](blokus-cloud-60-analysis.md) §3, refined by
[`../plans/archive/lr-scheduler-options.md`](../plans/archive/lr-scheduler-options.md)). v3 runs constant-LR
precisely to remove that confound, so its end-of-run curves are the clean read.

**The pre-registered capacity signal** (from the 60-run analysis, rec #6, unchanged here): move to
`xl` only when a healthy-LR run shows **policy loss flattening** *and* **the Pentobi ladder stalls
across two consecutive benchmarks**. At the end of `blokus_cloud_60` neither held: policy loss fell
0.53 → 0.40 over its final 13 generations [measured] and the ladder stepped L3 → L4 in the same
window [measured].

**What the data says is actually starving: the value head.** 10k games/gen = 10k independent
outcome labels shared across ~565k training positions (~56.5 samples/game after symmetry
[measured]); value loss was the one curve that degraded late in the 60-run [measured]. This is
AlphaGo's decorrelation problem ([`deepmind-run-configs.md`](deepmind-run-configs.md) §AlphaGo),
and its fix is **more games**, not more filters.

**The cost math, per generation on a 5090 at $0.99/h** (v3-shaped recipe: Gumbel n=128, K=64,
B=1024, 60k-game buffer, 100 arena games; full derivation in §2):

| Option | s/gen | $/gen | What it buys |
|---|---|---|---|
| `large`, 10k games/gen (v3 baseline) | ~1,410 | $0.39 | — |
| **`large`, 20k games/gen** | ~2,430 | **$0.67 (+$0.28)** | **2× independent value labels**, 2× data throughput |
| `xl`, 10k games/gen | ~2,990 | $0.82 (+$0.43) | 2.4× params; zero new data |

Same marginal spend, opposite targets. Given the measured diagnosis, **more-games-at-large is the
better first purchase of the two** — it attacks the identified weakness, it needs no calibration of
an unmeasured training step (§2), and it composes with a later `xl` run (the games remain valid
training data... for a `large` continuation; an `xl` restart re-generates anyway).

**Honest bottom line:** if v3's pool Elo flattens but its policy loss is still falling and value
loss is elevated → run `large` bigger (15–20k games/gen, ~$32–40 per further 60 gens), not `xl`.
Only if both capacity signals fire is `xl` the move. The rest of this doc prices that move.

---

## 2. Measured anchors and the `xl` cost model

**Phase split at `large` on the rented 5090** (`blokus_cloud_60`: 10k games/gen, Gumbel n=64,
K=64, B=1024, 40k buffer, 40 arena games) [measured]:

| Phase | s/gen | Share |
|---|---|---|
| Self-play (jax Gumbel) | ~510 (19.6 games/s) | **69%** |
| Training (torch, ~2.5M-sample buffer pass) | ~187 (14.1k samples/s) | 25% |
| Arena + report | ~44 | 6% |
| **Total** | **~750 (12.5 min)** | |

**Scaling factors to `xl`:**

- **Self-play ×2.08** [measured] — the 3060 Ti calibration ran both presets under identical search
  settings: `large` 2.71 games/s vs `xl` 1.30 ([`cloud-training-recommendation.md`](cloud-training-recommendation.md) §2).
  Self-play cost is dominated by net forwards in the low-sim Gumbel regime
  ([`jax-pipeline-ab.md`](jax-pipeline-ab.md) §4), so the ratio transfers across cards.
- **Sims ×2** [extrapolated] — v3's recipe doubles Gumbel sims vs the measured run (n=64 → 128);
  mctx per-sim cost is ~linear in n at fixed K ([`jax-pipeline-ab.md`](jax-pipeline-ab.md) §1).
- **Training ×2.35 (upper bound)** [extrapolated] — FLOPs ratio 8×/3.4×. Measured preset-to-preset
  training ratios scale *sub*-FLOPs (medium→large was 1.64× against a 3.4× FLOPs ratio), so the
  real number is likely ~1.6–2.4×; the xl training step was never measured (it OOM'd on the 8 GB
  3060 Ti — a jax/torch VRAM *coexistence* limit, a non-issue at 32 GB, and `xla_mem_fraction` is
  already a config knob). **[verify: run `cloud_calibration.py` on the rented card before
  committing]**.
- **Arena ~×1.9 at 100 games** [extrapolated] — python-backend inference scales with net FLOPs;
  arena is 6% of wall-clock, so even a bad estimate here barely moves totals.

**`xl` per-generation model on one 5090** (10k games/gen, n=128, 60k buffer, 100 arena games):

| Phase | s/gen | Share |
|---|---|---|
| Self-play (~4.7 games/s) | ~2,120 | **71%** |
| Training (~3.4M-sample pass) | ~660 | 22% |
| Arena + report | ~210 | 7% |
| **Total** | **~2,990 (~50 min)** | |

That 71/22/7 split is what prices every architecture below.

---

## 3. Architecture A — one big card, serial (status quo)

### Card choice (RunPod on-demand, checked 2026-07-06 **[verify at rental time]**)

| Card | $/h | Est. throughput vs 5090 | $/completed run (relative) | Verdict |
|---|---|---|---|---|
| **RTX 5090 (32 GB)** | **$0.99** | 1× [measured — v3 runs on one] | **1×** | **Pick.** Validated end-to-end: container, jax+torch coexistence, S3 resume all proven on this exact card. |
| RTX 4090 (24 GB) | $0.69 | ~0.55–0.65× [extrapolated: bf16 + bandwidth ratios] | ~1.1× | Fine fallback if 5090s are scarce; slower wall-clock for similar $. |
| L40S (48 GB) | $0.99 | ~0.6× | ~1.6× | No — 4090-class throughput at 5090 price. |
| H100 PCIe (80 GB) | $2.89 | ≤2× [extrapolated] | ~1.5× | No. Big-batch training silicon; our workload is thousands of small-batch 14×14 conv forwards. Pays ~3× the rate for at most ~2× the speed, on an unvalidated stack. |
| H100 SXM / H200 / B200 | $3.29–5.89 | — | worse | No. |

The 60-run's throughput already tells us the 5090 outperforms the old extrapolation (19.6 games/s
measured vs 7.5 projected in [`cloud-training-recommendation.md`](cloud-training-recommendation.md)
§3) — which is why the `xl` run below prices far under that doc's "£140–180" guess.

### Run cost (`xl`, 10k games/gen, ~50 min/gen, 5090 @ $0.99/h)

| Run length | Wall-clock | Cost | Note |
|---|---|---|---|
| 60 gens | ~50 h (~2 days) | ~$50 (~£39) | Minimum informative run at this net size ("judge bigger nets late" — [`deepmind-run-configs.md`](deepmind-run-configs.md) §5) |
| **100 gens** | **~83 h (~3.5 days)** | **~$82 (~£65)** | **Recommended commitment** |
| 150 gens | ~125 h (~5 days) | ~$123 (~£97) | Extend via `--resume` only while pool Elo / ladder still climb |

**Effort: ~zero.** The `xl` preset exists in `NetConfig`; this is a JSON config + the standard
runbook. One mandatory pre-step: `cloud_calibration.py` on the rented card (~$1) to replace the two
[extrapolated] factors above with measurements — exactly the stage-0 gate the cloud plan built.

Two cheap de-risking options worth carrying into the run config, neither blocking:

- **Bootstrap `xl` from the existing self-play archive.** Before going live, train the fresh `xl`
  net for a few epochs on the v3 buffer (the parquet archive is on the object store) — a supervised
  warm start that skips the random-play phase for roughly one generation's worth of GPU time. Cheap
  insurance, not load-bearing. [Idea, unvalidated — decide at run time.]
- **Keep games/gen at 10k initially but budget to raise it** — `xl` will eventually hit the same
  value-starvation wall; at `xl` +10k games/gen costs ~$0.58/gen.

---

## 4. Architecture B — multi-GPU actor-learner

### What the code supports today (verified on `main`, 2026-07-06)

- **No multi-GPU anything.** No `pmap`/`shard_map`/`jax.sharding` in `src/`; torch resolves a bare
  `torch.device("cuda")` with no index; the jax backend puts params on the default device via plain
  `jnp.asarray` (`games/blokusduo/jax/checkpoint.py`). Nothing reads *or* fights
  `CUDA_VISIBLE_DEVICES`, so per-process env pinning would work cleanly.
- **The self-play compute is cleanly shardable.** `run_wave(params, key, carry)` is a jitted pure
  function over a leading batch dim; slots are independent (`games/blokusduo/jax/actors.py`). The
  host-side `TraceHarvester` is per-slot and constructed per generation — one per producer process
  is trivial.
- **The seams for any split already exist:**
  - *Game consumption*: all backends stream completed games through a `sink` callback into
    `ReplayBuffer.add_game` (`training/coach.py`) — a game-granular streaming interface.
  - *Weight publication*: weights already travel actor-ward via the filesystem
    (`parallel_worker_init.pth.tar`, re-converted by the jax backend each generation).
  - *Race-free polling primitive*: `progress.json` is written atomically (temp + `os.replace`)
    *after* all generation artifacts — a remote actor could key off it safely.
- **What would fight an async split:** checkpoints are written **non-atomically** (direct
  `torch.save` to the final path); `SelfPlayStore` is one clobber-prone parquet per generation with
  no multi-producer naming; the replay buffer is an unlocked in-process deque; resume assumes
  generation-atomic completion; and the arena gate conflates "training head" and "self-play
  weights" in one `self.nnet` object.
- **The unadopted inference server is not the reuse vehicle.** It lost on measured throughput
  (0.72× at the bignet config — [`inference-server-bignet.md`](inference-server-bignet.md), "do not
  reopen"), and its shared-memory transport is single-node. Its transport-agnostic core could host
  a network `RequestSource`, but centralised inference is the wrong shape now that self-play is
  jax-native. The reusable assets are the worker pool's *patterns* (seed derivation, sink
  streaming, checkpoint-as-weight-sync), not its machinery.

### B-lite: sharded synchronous self-play (single node, N GPUs)

Keep the serial generation loop; split only the self-play phase. N child processes, each pinned to
one GPU via `CUDA_VISIBLE_DEVICES`, each running the existing jax backend on `num_eps/N` games with
`xla_mem_fraction` raised (dedicated card), streaming games back to the coach's `sink` over an mp
queue — the same contract the CPU worker pool already implements. The coach still owns the buffer,
the single parquet write, training, and the gate. `spawn` start method (the fork-vs-JAX hazard is
already documented in [`../plans/archive/harden-long-runs.md`](../plans/archive/harden-long-runs.md)).

- **Effort: ~2–4 eng-days** — producer wrapper + fan-in, per-producer RNG derivation (the
  `derive_episode_seed` pattern exists), config knob, validation run. No storage or coach-loop
  changes. A `shard_map` alternative exists (the compute is structurally ready) but puts sharding
  inside the compiled path and mctx for no capability gain; the N-process form is simpler and
  failure-isolated.
- **Speedup — Amdahl-bounded by the 71% self-play share.** `xl` per-gen at 10k games, N×5090 pods
  priced linearly (RunPod multi-GPU pods bill ~N× the single-card rate **[verify]**):

| Setup | $/h | s/gen | Speedup | $/gen | $/100-gen run | Wall-clock |
|---|---|---|---|---|---|---|
| 1×5090 (baseline) | $0.99 | 2,990 | 1× | $0.82 | **$82** | 83 h |
| 2×5090 | $1.98 | ~1,930 | 1.55× | $1.06 | $106 (+29%) | 54 h |
| 4×5090 | $3.96 | ~1,400 | 2.14× | $1.54 | $154 (+88%) | 39 h |
| 8×5090 | $7.92 | ~1,140 | 2.63× | $2.51 | $251 (+205%) | 32 h |
| ∞ GPUs | — | ~870 | **3.4× max** | — | — | — |

Every rung buys wall-clock and *loses* $/run, because train+arena stay serial on one card while
N−1 cards idle through them. That's the whole story of synchronous sharding.

### B-full: async actor-learner (the classic AlphaZero split)

Actors generate continuously against the latest published weights; the learner trains
continuously; no serial barrier, so all GPUs stay busy and $/game returns to roughly the
single-card rate. At 4×5090 (3 actors + 1 learner), the cycle becomes learner-bound at
~870 s/gen-equivalent → a 100-gen-equivalent run in **~24 h for ~$95** — 3.4× faster than
single-card at ~1.15× the cost [extrapolated].

**Staleness is not the risk** — our synchronous loop *already* runs every self-play game on
weights exactly one generation old, and both AGZ (25k games per weight refresh) and AZ (continuous
publication, no gate) tolerated equal or greater staleness at vastly larger scale
([`deepmind-run-configs.md`](deepmind-run-configs.md)). Publishing per-generation from the learner
reproduces today's staleness exactly.

**The engineering is the risk.** The cut touches every weak point listed above: atomic checkpoint
writes, a locked/multi-producer buffer path, multi-file `SelfPlayStore` naming + discovery, a
resume-semantics rewrite, and a real design decision on the gate (actors track gated-best à la AGZ,
or latest à la AZ — the latter changes training dynamics we've validated only in gated form; IDEAS
I4 parked exactly this). Estimate: **2–3 weeks** plus an A/B validation run to confirm strength
parity — call it ~$150–250 of validation compute on top.

**Verdict: not for this decision.** B-full's payoff is compressing multi-week runs or amortising a
standing multi-GPU fleet; our binding run is 3.5 days on one card. It becomes worth revisiting if
the roadmap reaches sustained 4+ GPU usage (e.g. an L9 push at 500+ generations or 50k+ games/gen).

---

## 5. A vs B, side by side (`xl`, 100 gens × 10k games)

| | A: 1×5090 serial | B-lite: 2×5090 | B-lite: 4×5090 | B-full: 4×5090 async |
|---|---|---|---|---|
| Wall-clock | 83 h | 54 h | 39 h | ~24 h |
| $ / completed run | **$82** | $106 | $154 | ~$95 (+ ~$150–250 one-off validation) |
| Engineering | **none** | 2–4 days | 2–4 days | 2–3 weeks |
| Risk | none (validated stack) | low (new fan-in path) | low | medium-high (gating semantics, buffer/checkpoint atomicity, resume rewrite) |
| Reusable capability | — | yes — composes with every future run | yes | yes, the end-state |

A gets ~100% of the outcome for ~53% of B-lite-4×'s money and 0% of anyone's engineering time. The
only thing it costs is ~2 extra days of calendar — during which no human attention is required
(the S3-sync + `--resume` + watchdog machinery from the cloud plan is exactly for this).

---

## 6. Recommendation

**Phased, each phase gated, ready to fire when v3's pool Elo lands:**

| Phase | Trigger | Action | Cost | Effort |
|---|---|---|---|---|
| 0 | v3 ends | Read the diagnostics against §1's capacity signal: policy-loss slope, value-loss trajectory, pool-Elo curve, ladder trend. | $0 | hours |
| 1 | Plateau real, **capacity signal absent** (expected case: value loss elevated, policy loss still falling) | Stay at `large`: 15–20k games/gen continuation via `--resume`, ~60 gens. | ~$32–40 | config only |
| 2 | Capacity signal **fires** (now, or after phase 1 plateaus with healthy value loss) | **`xl` on one rented 5090**: calibrate (~$1), then 100 gens × 10k games, extend by `--resume` while curves climb. Optional supervised warm start from the v3 game archive. | ~$50–125 | config only |
| 3 | Runs start exceeding ~1 week single-card (e.g. `xl` at 20k+ games/gen, or an L9-push run length) | Build **B-lite** (sharded sync self-play — IDEAS I6) and rent 2–4×5090 pods for wall-clock. | +30–90% $/run for 1.5–2.1× speed | 2–4 days |
| 4 | Sustained multi-GPU usage becomes the norm | Revisit **B-full** async actor-learner (IDEAS I4) with the phase-3 machinery as its foundation. | — | 2–3 weeks |

**Budget envelope: phases 0–2 land at ~$85–165 (~£67–130) total**, single card, no code changes.
Distributed self-play is deliberately deferred, not rejected — it's independently valuable, its
cheap form is small, and phase 3 names the moment it starts paying.

**Do-not-do list:** don't rent an H100 (worse $/result on this workload); don't build B-full to
speed up a 3.5-day run; don't touch the live v3 pod — every action above starts *after* its
end-of-run pool Elo is read.
