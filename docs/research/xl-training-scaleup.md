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

> **⚠ Superseded in part by the [v3 post-mortem addendum](#addendum-2026-07-07--blokus_cloud_v3-post-mortem)
> (2026-07-07).** v3's data arrived and *inverted* this doc's diagnosis: the value head is healthy
> and the **policy** is what's stuck, with ~zero external transfer of the run's internal gains.
> §1's Phase-1 "more games at large" call is **retracted**, the `xl` trigger fires on its letter
> but not its spirit, and the recommendation is revised (addendum A6). §§2–5 (cost model,
> architecture A vs B) stand, with measured corrections in addendum A5.

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

> **Superseded by the addendum's A6** (2026-07-07) — phases 1–2 below assumed the 60-run's
> value-starvation diagnosis, which v3 falsified. Kept for the record.

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

---

# Addendum (2026-07-07) — blokus_cloud_v3 post-mortem

`blokus_cloud_v3` finished: 40 generations, constant LR 1e-3 (verified flat in the
`LearningRate` parquets — the 60-run's LR confound is genuinely gone), weights-only warm start
from `blokus_cloud_60`'s gen-57 net, 24.3 h on the RunPod 5090 (~$24), 21/40 candidates accepted.
Everything below is parsed programmatically from `temp/runs/blokus/blokus_cloud_v3/`
(run log, `Tournament/`, metric parquets) and `temp/benchmarks/v3_final_ladder_L1-9.html`;
nothing is eyeballed. This addendum revisits each claim of the main doc against that data.

## A1. What v3 measured [all measured]

**Pentobi ladder — flat at Level 4.** Final gen-40 net (100 games/level, 400 sims) vs the
starting gen-57 net and the mid-run check:

| Net | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | Headline |
|---|---|---|---|---|---|---|---|---|---|---|
| gen-57 donor (40 g/level) | 80 | 75 | 60 | 55 | 45 | 20 | — | — | — | L4 |
| v3 gen-14 (40 g/level) | 80 | 75 | 60 | 55 | 45 | 20 | — | — | — | L4 |
| v3 gen-40 final (100 g/level) | 77 | 71 | 61 | 53 | 47 | 40 | 16 | 20 | 21 | L4 |

L5 sat at 45–47% in every ladder we have (needs >50%; the final CI is [38%, 57%]). Two reads
worth separating: (i) L1–L5 did not move at all across 40 generations; (ii) L6 improved 20% → 40%
— the one ladder cell that moved, though CIs overlap ([10,35] vs [31,50]). And the L6→L7 cliff
(40% → 16%) marks where Pentobi's deeper search regime starts — levels 7–9 are a different
opponent class, near-flat at 16–21%.

**Training losses (end-of-generation values parsed from the tqdm bars).** Value loss:
0.525 → minimum 0.363–0.385 (gens 32–37), ending 0.419 — *better* than the 60-run's 0.46
finish, on the same 60k buffer the main doc's §1 prescribed. Policy loss: a round trip —
0.493 (gen 1) → 0.669 peak (gen 13) → 0.476–0.488 (gens 32–37) → 0.554 (gen 40). Flat-to-
oscillating; never below its warm-start level. **The 60-run's pattern is inverted: value healthy,
policy stuck.**

**Eval-set entropy rose all run**: 1.21 → 1.69 nats — the opposite of the 60-run's sharpening
(4.10 → 1.59). The improvement operator is *diffusing* the policy, not sharpening it. Top-1
agreement with the (this-run, gen-1) eval set stayed 89–100% throughout — the net is not
drifting far from its warm-start search targets.

**Pool BayesElo (anchor gen0 = the gen-57 donor, = 0):** rose noisily to +240 (gen 40), peaking
+286 (gen 32), with real non-monotonic excursions (gen 1 = −71, gens 24–25 ≈ +81 after a 5-gen
rejection streak).

**Arena and tournament games are heavily quantized.** 14 of the 19 rejections scored *exactly*
50–50–0 (P ≈ 8% each under independent games — 14/14 is impossible by chance); four acceptances
were 100–0–0 sweeps. The pool tournament shows the same structure at 30 games/pairing: many
exact 15–15–0 and 30–0–0 results, and **gen32 vs gen40 was 30 draws out of 30**. Eval-time play
is deterministic-per-(seed, colour) (`temp=0`; `opening_temp`/`opening_moves` exist in
`evaluation/players.py` but default to 0 **[verify how arena pairings are seeded before trusting
close gate scores]**), so between near-equal nets the colour/seed decides and mirrored pairs
split — the gate and the tournament lose most of their resolution exactly where we need it.

## A2. Question A — the value-starvation premise is falsified; Phase 1 is retracted

The main doc's §1 recommended "more games at `large`" *specifically to fix value-head data
starvation*. v3 — which already ran the 60-analysis's buffer fix (60k games) — shows the value
head comfortably healthy (0.36–0.42) while the policy is what stopped improving. Whatever
capped v3, it was not a shortage of independent value labels. **Phase 1 ("more games at large,
~$32–40") is retracted as a strength lever.** More games would buy more of a curriculum that is
demonstrably no longer teaching anything Pentobi-relevant (A4). What replaces it is in A6.

## A3. Question C first (it gates B) — the gen-32 "peak" is not evidence of regression

The pooled ratings say gen 32 = +286 > gen 40 = +240. The raw pairings say the two nets are
**directly indistinguishable — 30/30 draws head-to-head**; the −46 gap is entirely graph-inferred
through quantized third-party results (gen 32 beat gen 36 30–0 where gen 40 beat it 26–4, etc.),
and with mirrored deterministic games a 30-game pairing carries ~15 independent openings of
information. Δ46 Elo is inside that noise floor. The loss curves do wobble over the final
generations (policy 0.476 → 0.554, value 0.363 → 0.419 across gens 36–40, through two rejection
streaks whose reverts staled the buffer onto one incumbent's games) — mild late instability is
plausible but unproven.

**Consequences:** no checkpoint-selection change (gen 32 and gen 40 are equivalent within
measurement; `best.pth.tar` is fine). The run-length lesson is real though: nothing measurable
was gained after ~gen 32 — the last 8 generations were ~5 h/$5 spent inside the noise. The §6
stop rule ("stop when curves flatten across two benchmark windows") is reaffirmed, with the
addition that pool-Elo differences under ~±50 at 30-games/pairing should be read as ties, and
the tournament/gate should get opening diversification before we lean on either again (A6-P0).

## A4. Question B — the xl trigger fires on its letter and fails on its spirit

The pre-registered signal (§1) was: *policy loss flattening* AND *ladder stalled across two
runs*. Mechanically both hold: policy loss is flat-to-rising at a healthy constant LR, and both
the 60-run and v3 ended at L4. Read literally, this doc's own rule says "go xl".

I don't think that's the honest read. The trigger was designed to detect **capacity saturation**
— a net that can no longer fit what self-play is teaching. v3 shows no underfitting signature
anywhere: value loss at its best-ever (0.36), top-1 eval agreement 89–100%, and the policy-loss
levels aren't even comparable to the 60-run's (different Gumbel targets: n=128/considered-32 vs
n=64/16; and in a non-stationary self-play regime rising CE can mean *harder targets*, not a
worse net). What v3 actually exhibits is the signature of a **self-play curriculum ceiling**:

- **Internal gains stopped transferring.** The pooled +240 is the generous number; the *direct*
  head-to-head of gen 40 vs the gen-57 anchor is 15–8–7 ≈ 62% ≈ **+83 Elo** (wide CI at 30
  games). Somewhere between +83 and +240 of internal, in-lineage improvement bought **0 pp** on
  L1–L5. Even the conservative +83 should have moved L5 by ~+11 pp if it transferred — the
  point estimate moved +2 pp. The prompt-level framing "+240 bought zero" overstates the
  numerator, but the decoupling itself is real and is the key fact.
- **That decoupling is the classic in-lineage exploitation pathology**, not a capacity symptom.
  A capacity-bound net fails to improve *internally* too; v3's lineage kept finding wins against
  its own ancestors (26–4 over gen 17, sweeps over mid-run nets) while gaining nothing against
  an out-of-distribution opponent. Bigger nets are, if anything, better at memorising lineage
  quirks — `xl` inherits this failure mode at 2× the price.
- **The same-capacity net was climbing recently.** This exact 192×12 architecture went L3 → L4
  late in the 60-run and its donor kept accepting candidates at 55–76%. The *recipe* stopped
  extracting improvement, at the same parameter count where improvement was recently cheap.
- **The operator is diffusing, not sharpening.** Entropy rising 1.21 → 1.69 nats across 40
  generations of "improvement" means the n=128-over-top-32 Gumbel targets are, on net, flatter
  than the policy they're meant to improve. That is an improvement-operator statement, not a
  capacity statement (and Blokus's ~400-branch opening vs DeepMind's 800–1,600 sims says the
  same thing from first principles — §"deepmind-run-configs" and the 60-analysis both flagged
  operator thinness as the growth-rate bottleneck, rec #2, which v2/v3 only half-took: sims
  64→128 but still a thin considered set, and the plateau arrived anyway).

**Verdict: capacity is *not confirmed*. `xl` is demoted** from "Phase 2, fires on this signal"
to "after the recipe levers, and only alongside a genuine underfitting signature" (A6-P4). One
honest hedge: L6 moving 20% → 40% while L1–L5 sat still is a small counter-signal that *some*
general improvement occurred; it keeps "real gain hidden by ladder noise" alive as a minority
read. The next run should tighten it (A6-P0 makes the gate/tournament informative; rerunning
the donor's ladder at 100 games/level would pin the baseline CI).

## A5. Question D — cost model corrections; architecture verdict unchanged

The §2 model predicted ~23.5 min/gen for v3's recipe; v3 measured **37.2 min/gen** (steady
state: self-play 1,353 s / training 760 s / arena 65 s = 61% / 34% / 3%). Two attributable
misses, both instructive [measured]:

1. **Training 2.7× the model** (760 s vs ~280 s): v3 ran `dataloader_workers: 0` — the
   workaround for the JAX-fork DataLoader crash — which starves the training GPU. The fix
   (spawn-context workers, [`../plans/archive/harden-long-runs.md`](../plans/archive/harden-long-runs.md))
   is worth ~21% of total wall-clock at `large` and more at `xl`. Until it lands, every cost
   table in §§2–5 should be scaled by ~1.33–1.6×.
2. **Self-play 1.33× the model** (1,353 s vs ~1,020 s): the model scaled mctx cost with sims
   only (n 64→128 ⇒ 2×); v3 also doubled `gumbel_max_considered` 16→32, and root-candidate
   width costs real time too. Corrected rule of thumb: self-play cost ≈ linear in n × mild
   in considered-set width (measured composite: 2.65× for 2× sims + 2× considered).

Re-priced `xl` on one 5090 (10k games/gen): ~79 min/gen with `workers=0`, ~60 min/gen once the
DataLoader fix lands → **100 gens ≈ $99–130 (~4–5.5 days)** [extrapolated], vs the original $82.
The architecture conclusions are unaffected in structure: still single-5090 (the H100 $/result
argument only strengthens as training share grows), still no async build. One number moves:
at v3's measured 61% self-play share the synchronous-sharding Amdahl cap drops to ~2.6×
(back to ~3.3× once training is fixed) — B-lite buys slightly less than §4 claimed until
harden-long-runs lands. IDEAS I6 unchanged.

## A6. Revised recommendation (supersedes §6 phases 1–2)

The lever is the **improvement operator and the curriculum**, not net size and not game count.
Phased, cheapest and most-diagnostic first; every phase stays on the validated 1×5090 + `large`
stack:

| Phase | What | Why / gate | Cost | Effort |
|---|---|---|---|---|
| P0 | **Make the measurements informative + reclaim wall-clock.** (a) Opening diversification for arena + pool tournament (`opening_temp`/`opening_moves` already exist in `evaluation/players.py` — likely config-only **[verify]**); (b) land the spawn-context DataLoader fix (harden-long-runs); (c) rerun the gen-57 donor ladder at 100 games/level to pin the baseline. | 14 exact-50/50 gates and a 30/30-draw pairing mean we currently can't see <~50-Elo effects — every later phase is judged through this instrument. | ~$3–5 | ~1 day |
| P1 | **Thicken the improvement operator at `large`**: Gumbel n 128→512, `gumbel_max_considered` 32→64 (toward DeepMind's 800-sim regime), ~40-gen continuation from gen-40 via `--resume`. Optionally A/B one arm of jax-PUCT self-play (it trained the *stronger* net head-to-head in the backend A/B, 60.5%). | The one lever the 60-analysis ranked as the growth bottleneck (rec #2) that v3 only half-took — and the entropy-diffusion finding points straight at target quality. Gate: L5 > 50% or pool-Elo slope clearly positive through P0's instrument. | ~$35–45 (~2.9× self-play cost/gen) | config only |
| P2 | **Break the lineage: opponent + opening diversity in self-play.** Fraction of self-play games vs a pool of past checkpoints (AlphaGo's RL pool — the known cure for in-lineage exploitation), plus a self-play opening-temperature schedule. | Directly attacks the +internal/0-external decoupling. Gate: transfer reappears (ladder moves with pool Elo). | ~$30–50/run | ~2–4 days |
| P3 | **External teacher: Pentobi seeding/distillation** (SL on Pentobi games or Pentobi-anchored targets — [`cloud-training-recommendation.md`](cloud-training-recommendation.md) §7.4 already named this the biggest lever if tabula-rasa stalls). | If P1+P2 still don't move L5–L6, the curriculum needs information self-play cannot generate. | compute trivial; data-gen + code real | ~1–2 weeks |
| P4 | **`xl`** — only alongside a genuine underfitting signature (value or policy loss stuck *high* while the P0-instrumented gate shows healthy acceptance) after P1–P3. Pricing per A5: ~$99–130 per 100 gens. | Capacity spend is wasted while the operator/curriculum is the binding constraint — v3 is the demonstration. | ~$99–130 | config only |

**Bottom line, stated plainly:** v3 falsified this doc's premise in the most useful way — it
cleared the two suspects the 60-run left standing (LR schedule, value starvation) and exposed
the one it couldn't see: an improvement operator too thin, and a curriculum too narrow, to push
a Level-4 net further from its own games alone. `xl` would have been ~$100 spent training a
bigger net on the same dead-end curriculum. Architecture A (one 5090) remains the right vehicle
for whichever run comes next; nothing in v3 changes the multi-GPU verdict.
