# Cloud training recommendation — one GPU, ~£100

What card to rent, what net to train, and how to spend ~£100 for the strongest Blokus Duo net we
can get — with a staged plan up the Pentobi ladder. Written 2026-07-04 for
[`docs/plans/cloud-scale-training.md`](../plans/cloud-scale-training.md) (C13).

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
- £100 buys **far more** than any run to date: the Gumbel backend made generation cheap, so the
  binding constraint is **learning dynamics at scale, not compute**. Hence the staged plan below —
  spend ~£15 proving the recipe, then extend the same run with `--resume` while the Elo/ladder
  curves still climb, stepping up to `xl` only if calibration and the curves justify it.
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

From the 3060 Ti **[measured]** (128f×8b = `medium`, Gumbel n=64, K=64, bf16):

- Self-play: **~11.4 games/s** at B=1024 (327 moves/s ÷ 28.6 moves/game); 13.5 games/s at 64f×4b.
- At n=64, wall time is ≈50% net forward / ≈50% mctx tree machinery — so *net size scales only
  half the cost* until the net dominates.
- The torch training loop was CPU-feed-bound (single-threaded dense-encoding of 17,837-length
  policies); C3's DataLoader workers + bf16 + channels_last exist precisely so a fast card isn't
  starved. **[calibrate: training ms/position with perf knobs on]**

## 3. Net size — the cost model

Per-position net cost scales ∝ blocks × filters² (3×3 convs on a fixed 14×14 board):

| Preset | Size | Params | Relative net FLOPs |
|---|---|---|---|
| small (today's prod) | 64f×4b | ~1M | 0.125× |
| medium (run3) | 128f×8b | ~3.7M | 1× |
| **large (recommended)** | **192f×12b** | **~8M** | 3.4× |
| xl (stretch) | 256f×16b | ~19M | 8× |

Self-play games/s model **[extrapolated]**: `t_game = t_tree + t_net`, with the measured medium
split (50/50 on the 3060 Ti) and a 5090 speedup of ~3× on net forward, ~2× on tree machinery:

| Preset | 3060 Ti games/s | 5090 games/s (est.) | 10k games/gen | train/gen (1.16M-position buffer) | ~min/gen* |
|---|---|---|---|---|---|
| medium | 11.4 [measured] | ~27 | ~6 min | ~1 min | **~9** |
| large | ~5.2 [extrapolated] | ~14 | ~12 min | ~3 min | **~19** |
| xl | ~2.5 [extrapolated] | ~7 | ~23 min | ~7 min | **~38** |

\* incl. ~25% overhead for arena/Elo/report — the python-backend eval phases get *slower* as nets
grow and are the least-well-modelled part **[calibrate]**; if calibration shows eval eating >30%
of a generation, cut `num_arena_matches`/`elo_games_per_gen` before cutting games/gen.

At ~£0.70/h, `large` costs **~£0.22/generation**; the 60-gen headline run is **~£14**. Even `xl`
fits ~150 generations in £100. The budget is not the wall — knowing *when to stop pushing a stale
recipe* is. That's what the stages are for.

**Why `large`, not `xl`, as the committed run:** capacity only pays when enough games/generations
feed it (deepmind-run-configs.md §5: bigger nets learn slower per gen but raise the ceiling — judge
late). `large` is 3.4× run3's proven net at a per-gen price that lets us run 3–5× run3's
generations *and* 5× its games/gen inside a third of the budget. `xl` triples the per-gen price and
sits furthest from validated territory (checkpoint bridging, eval wall-clock). Step up to it with
evidence, not hope.

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
| 1 | `blokus_cloud.json`, 60 gens | ~£15 [extrapolated] | Internal Elo still climbing; arena accept rate healthy |
| 2 | Pentobi ladder: `pentobi_benchmark --levels 1-5 --games 40` on `best.pth.tar` (+ a mid-run checkpoint) | ~£1–2 | Beats L1 at >50% → continue; else diagnose before spending more |
| 3 | Extend: bump `num_generations`, `--resume` (same buffer, same Elo baseline). Repeat ladder every ~40 gens; climb `--levels` as levels fall | ~£30–60 | Stop when Elo/ladder plateaus across two consecutive benchmarks |
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

1. **More of the same** (extend `large`): ~£15 per further 60 gens. Cheap while curves climb.
2. **`xl` restart** (fresh run, 150+ gens × 10k games): ~£60–100 — worthwhile if `large` plateaus
   with healthy training diagnostics (capacity-limited, not data-limited).
3. **Stronger eval search at benchmark time** (more sims vs Pentobi) is free strength at play time
   — always max this before buying more training.
4. Seeding from Pentobi games (AlphaGo-style SL bootstrap, 01-BACKGROUND/deepmind-run-configs §1)
   — a code investment, not a compute one; the biggest known lever if tabula-rasa stalls.

A realistic envelope for "beats mid-ladder (L5)" is **£100–300 total**; for "beats L9" the honest
answer is **unknown** — somewhere between £300 and £1500+, and possibly gated on seeding/search
work rather than raw compute. Each ladder benchmark tightens these numbers; that's the point of
building the measurement into the loop.
