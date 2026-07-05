# Cloud training recommendation — one GPU, ~£100

What card to rent, what net to train, and how to spend ~£100 for the strongest Blokus Duo net we
can get — with a staged plan up the Pentobi ladder. Written 2026-07-04 for
[`docs/plans/archive/cloud-scale-training.md`](../plans/archive/cloud-scale-training.md) (C13).

**Evidence tagging.** Every number is one of:
**[measured]** — from an actual run on the 3060 Ti (mostly [jax-pipeline-ab.md](jax-pipeline-ab.md));
**[extrapolated]** — derived from measured numbers by a stated model;
**[calibrate]** — must be confirmed by `scripts/benchmarks/cloud_calibration.py` on the rented card
*before* committing the budget. Nothing here was obtained by training large nets to compare —
that's what the calibration tool + staged plan replace.

---

## TL;DR

- **Rent an RTX 5090** (fallbacks: RTX 4090, L40S) on a marketplace neocloud (RunPod/Vast/Lambda),
  ~£0.55–0.85/h **[verify at rental time]**. VRAM is irrelevant to this workload — buy bf16
  throughput + memory bandwidth per £.
- **Train the `large` preset (192f×12b, ~8M params)** with `run_configurations/blokus_cloud.json`:
  jax+Gumbel self-play, 10,000 games/gen, 40k-game buffer, all C3 perf knobs on.
- £100 buys **far more** than any run to date (~10× run3's games at 3.4× its net, with room to
  triple the run length): at `large` the binding constraint is **learning dynamics at scale, not
  compute** — though the budget does genuinely bind one step up at `xl` (§3). Hence the staged
  plan below — spend ~£27 proving the recipe, then extend the same run with `--resume` while the
  Elo/ladder curves still climb.
- Expected outcome, stated honestly: the current 64f×4b net loses ~75% vs Pentobi **level 1**
  [measured]. This plan should convincingly beat L1–L2 and plausibly reach L3–L5
  **[extrapolated — wide error bars]**. Beating **L9** within £100 is unlikely and is *not*
  promised; §7 estimates what it might take.

## 1. Card class

The workload is thousands of small bf16 conv forwards (B≤1024 over 44×14×14) from the jax search,
plus a torch training pass. It is compute/bandwidth-bound, **never VRAM-bound** (the whole xl net +
activations + XLA arena fits in <8 GB [measured on the 3060 Ti]).

| Card | Indicative rate **[verify]** | Why / why not |
|---|---|---|
| **RTX 5090** | ~£0.55–0.85/h | ~4× the 3060 Ti's bf16 tensor throughput, 1.8 TB/s bandwidth. Best £/throughput for small-batch conv. **Recommended.** |
| RTX 4090 | ~£0.30–0.55/h | ~2–2.5× the 3060 Ti. Cheaper/h; slightly worse £/result on long runs, fine fallback. |
| L40S | ~£0.65–0.95/h | Datacentre reliability, similar-to-4090 throughput at a higher rate. Pick when 5090s are scarce. |
| A100/H100 | ~£1.3–2.5/h | Optimised for big-batch training; wasted on 14×14 boards. Skip. |

Nothing in the code assumes a card: the container (C8) uses pip-bundled CUDA, the XLA VRAM cap is a
config knob (C1), and batch/net sizes are config. `--gpus all` on any CUDA-12 host works.

## 2. Where the time goes — what we know

The calibration tool itself was run end-to-end on the 3060 Ti, inside the training container
(2026-07-04, `blokus_cloud_calibration.json` search settings: Gumbel n=64, K=64, B=1024, bf16;
2,048-game bursts after a jit warmup; training measured over 8,192 synthetic buffer positions with
the C3 perf knobs on, torch in eager — inductor was unavailable in that first image):

| Preset | Self-play games/s **[measured]** | Training ms/position **[measured]** |
|---|---|---|
| small 64f×4b | 12.22 | 0.775* |
| medium 128f×8b | 6.24 | 0.444 |
| large 192f×12b | 2.71 | 0.728 |
| xl 256f×16b | 1.30 | — (OOM, see below) |

Caveats: an image rebuild ran concurrently during the medium/large bursts (CPU contention →
treat those as mild *underestimates*); small's training number carries first-size warmup, medium's
0.444 is the cleanest small-net figure. **One measured position per game ≈ 56.5 training
examples** (~28 moves × 2 symmetry augmentation) — the buffer cost model below uses this. The xl
*training* measurement OOM'd on the 8 GB card (jax's 0.6 VRAM share + a batch-1024 xl training
step) — a real coexistence limit on small cards, a non-issue at 24 GB+; the tool now skips a
failed size instead of dying **[calibrate xl on the rented card]**.

## 3. Net size — the cost model

Per-position net cost scales ∝ blocks × filters² (3×3 convs on a fixed 14×14 board):

| Preset | Size | Params | Relative net FLOPs | Measured games/s ratio |
|---|---|---|---|---|
| small (today's prod) | 64f×4b | ~1M | 0.125× | 1.96× medium |
| medium (run3) | 128f×8b | ~3.7M | 1× | 1× |
| **large (recommended)** | **192f×12b** | **~8M** | 3.4× | 0.43× |
| xl (stretch) | 256f×16b | ~19M | 8× | 0.21× |

Projected to a 5090 (~3× net forward, ~2× tree machinery vs the 3060 Ti **[extrapolated]**), at
10k games/gen with a 40k-game buffer (= 40k × 56.5 ≈ **2.26M positions** trained per generation
at epochs 1):

| Preset | 5090 games/s (est.) | self-play/gen | train/gen | ~min/gen* | £/gen @ £0.70/h |
|---|---|---|---|---|---|
| medium | ~16 | ~10 min | ~6 min | **~20** | £0.23 |
| large | ~7.5 | ~22 min | ~9 min | **~39** | £0.45 |
| xl | ~3.6 | ~46 min | ~17 min | **~79** | £0.92 |

\* incl. ~25% overhead for arena/Elo/report — the python-backend eval phases get *slower* as nets
grow and are the least-well-modelled part **[calibrate]**; if calibration shows eval eating >30%
of a generation, cut `num_arena_matches`/`elo_games_per_gen` before cutting games/gen.

So at ~£0.70/h: `large` ≈ **£0.45/generation** → the 60-gen headline run is **~£27**, and ~200
generations fit £90. `xl` ≈ £0.92/gen → 60 gens ≈ £55 and 100+ gens overruns the budget — **the
budget genuinely binds at xl**, which is exactly why it's the stretch, not the plan.

**Why `large`, not `xl`, as the committed run:** capacity only pays when enough games/generations
feed it (deepmind-run-configs.md §5: bigger nets learn slower per gen but raise the ceiling — judge
late). `large` is 3.4× run3's proven net at a per-gen price that lets us run 3–5× run3's
generations *and* 5× its games/gen inside a third of the budget. `xl` doubles the per-gen price
again, can't fit a long run in £100, and sits furthest from validated territory (checkpoint
bridging, eval wall-clock, the small-card VRAM coexistence limit above).

## 4. The recommended run (`run_configurations/blokus_cloud.json`)

`large` preset, 60 generations × 10,000 games/gen (600k games — 10× run3's total), buffer 40,000
games (staleness 4 gens, reuse 4), Gumbel n=64/K=64/B=1024 bf16 self-play, arena gate 40 games
@0.55, Elo 20 games/gen, perf knobs: bf16 autocast, TF32, cudnn.benchmark, channels_last, compile,
8 DataLoader workers, pinned memory, metric sync every 25 batches, batch 1024. Object-store block
per the runbook so the run survives preemption.

Learning rate stays 0.001+cosine **[validated at run3 scale]**; if calibration's first generations
show training loss oscillating at batch 1024, drop to 512 rather than tuning LR mid-run.

## 5. Staged plan — train → benchmark → extend

| Stage | What | Cost (est.) | Gate to continue |
|---|---|---|---|
| 0 | Rent card → `docker run` calibration config + `cloud_calibration.py --rate-gbp-per-hour <r>` | ~£1 | Table roughly matches §3; pick final preset (`large` unless it says otherwise) |
| 1 | `blokus_cloud.json`, 60 gens | ~£27 [extrapolated from measured 3060 Ti throughput] | Internal Elo still climbing; arena accept rate healthy |
| 2 | Pentobi ladder: `pentobi_benchmark --levels 1-5 --games 40` on `best.pth.tar` (+ a mid-run checkpoint) | ~£1–2 | Beats L1 at >50% → continue; else diagnose before spending more |
| 3 | Extend: bump `num_generations`, `--resume` (same buffer, same Elo baseline). Repeat ladder every ~40 gens; climb `--levels` as levels fall | ~£50–65 (another 100–140 gens) | Stop when Elo/ladder plateaus across two consecutive benchmarks |
| 4 (optional) | If still climbing at plateau-free ~£70 spent: restart at `xl` seeded by... no — continue `large`; an `xl` restart is a *new* budget decision (§7) | — | — |

Everything in stages 2–3 is push-button: the ladder JSONs land in `PentobiLadder/` and render in
the report; `--resume` + the object store make extension and preemption safe (C7/C11).

## 6. Expected outcome — the reality check

Recorded plainly, per the plan's brief:

- Baseline: 64f×4b after 30 gens × 2k games loses ~75% vs **L1** [measured]. run3-class
  (medium, 30×2k) reached ~25% vs L1 [measured].
- This plan is ~10× run3's games with 3.4× the net and a better search (the jax-PUCT arm already
  beat python head-to-head 60.5% [measured]). Beating **L1–L2 is the expected floor**; L3–L5 is a
  reasonable hope **[extrapolated — no measurement exists between "25% vs L1" and "beats L9",
  treat as a guess to be measured, not a promise]**.
- **L9 within £100: unlikely.** Pentobi L9 plays at strong-amateur level with deep search; nothing
  in our data locates it on our Elo curve. Do not promise it.

## 7. Beyond £100

If stage 3 plateaus below the target level, the next levers, in rough £-efficiency order
**[extrapolated]**:

1. **More of the same** (extend `large`): ~£27 per further 60 gens. Cheap while curves climb.
2. **`xl` restart** (fresh run, 150+ gens × 10k games): ~£140–180 — worthwhile if `large` plateaus
   with healthy training diagnostics (capacity-limited, not data-limited).
3. **Stronger eval search at benchmark time** (more sims vs Pentobi) is free strength at play time
   — always max this before buying more training.
4. Seeding from Pentobi games (AlphaGo-style SL bootstrap, 01-BACKGROUND/deepmind-run-configs §1)
   — a code investment, not a compute one; the biggest known lever if tabula-rasa stalls.

A realistic envelope for "beats mid-ladder (L5)" is **£100–300 total**; for "beats L9" the honest
answer is **unknown** — somewhere between £300 and £1500+, and possibly gated on seeding/search
work rather than raw compute. Each ladder benchmark tightens these numbers; that's the point of
building the measurement into the loop.
