# P3 — Linux re-baseline (re-profile + worker-split sweep)

Subplan for **P3** of [self-play-throughput.md](self-play-throughput.md). The
premise-fixing phase: the roadmap's original evidence was measured on Windows and
is now known misleading (P1's OS migration was the real win; P2's vectorisation
was ~0% end-to-end). Before building anything in P4 we characterise the **current
Linux reality** — where self-play wall-clock and memory actually go on the small
net + `forkserver`, and how games/s scales with worker config — then let the data
pick the P4 target (more GPU workers? hybrid split? all-GPU? central server?).

**No new product code in P3** — it is measurement only. True mixed-device (GPU+CPU
*simultaneously*) workers need the per-worker `num_gpu_workers` feature, which is
**P4**. P3 brackets that decision with the three single-mode curves below: if
all-GPU-N keeps climbing, hybrid may be unnecessary; if all-GPU caps early (VRAM)
while CPU/server add throughput, hybrid is worth building. *(Optional probe: a
crude hybrid estimate by co-running one GPU and one CPU self-play process and
summing games/s — noisy, only if the curves are ambiguous.)*

Companion: profiling tool rationale in
[self-play-speed-investigation.md](../../research/self-play-speed-investigation.md)
(Part A). Box access + run mechanics in
[REMOTE-TRAINING.md](../../guides/REMOTE-TRAINING.md).

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| R1 | **Throughput sweep** — games/s vs worker config on Linux: all-GPU-N, CPU-only-N, server-N | ~1 hr (runs) | High | ✅ |
| R2 | **Time profile** — py-spy (`--subprocesses`) on a multi-worker run + Scalene/line_profiler on a 1-worker run; rank the real Linux hot slices | ~2 hr | High | |
| R3 | **Memory profile** — tracemalloc on the in-episode tree + replay buffer; confirm the ~19 MB/worker tree holds and size the buffer | ~1 hr | Medium | |
| R4 | **Write-up + P4 recommendation** — bottleneck breakdown + games/s-vs-config curve → name the P4 target; update the roadmap evidence | ~1 hr | High | |

---

## R1. Throughput sweep

**Goal:** a games/s-vs-config curve on Linux/`forkserver`, same harness as the P2
re-baseline (`run_self_play_episodes_parallel`, timed exactly as `core.coach`).

**Matrix** (150 eps each, production net 64f×4b, K=16, 300 sims — i.e. the
`bench_workers_*` settings, varying only workers/device):

| Mode | `worker_cuda` | `inference_server` | N (`num_parallel_workers`) |
|------|---------------|--------------------|----------------------------|
| all-GPU | true | false | 8, 10, 12, 14, 16 *(stop when VRAM OOMs or g/s rolls over)* |
| CPU-only | false | false | 12, 16, 20 |
| server | false | true | 16, 20, 24 |

**Watch VRAM** for all-GPU: each worker holds its own CUDA context (the Windows
cap was 8 ≈ 2.5 GB/worker). `nvidia-smi --query-gpu=memory.used --format=csv` between
runs; cap N where it approaches 8 GB.

**Mechanics** (mirrors the P2 run; box gets code via `git`):
1. Box on the branch tip (`ssh gpu-anywhere 'cd ~/AlphaBlokus && git fetch && git checkout <branch> && git reset --hard origin/<branch>'`).
2. Drop the timing driver on the box (untracked; same one P2 used):
   ```python
   # bench_throughput.py — times run_self_play_episodes_parallel for a config, prints games/s
   import sys, time
   from core.config import load_args
   from core.game_factory import instantiate_game_and_network
   from core.parallel_self_play import run_self_play_episodes_parallel
   cfg = load_args(sys.argv[1]); game, nnet = instantiate_game_and_network(cfg)
   nnet.save_checkpoint(filename="parallel_worker_init.pth.tar")
   t0 = time.perf_counter()
   run_self_play_episodes_parallel(config=cfg, generation=1,
       checkpoint_path="parallel_worker_init.pth.tar", num_workers=cfg.num_parallel_workers)
   dt = time.perf_counter() - t0; n = cfg.num_eps
   print("RESULT {} N={} cuda={} server={} g/s={:.4f} s/game={:.3f}".format(
       cfg.run_name, cfg.num_parallel_workers, cfg.worker_cuda,
       getattr(cfg, "inference_server", False), n/dt, dt/n), flush=True)
   ```
3. Generate the matrix configs on the box from the `bench_workers_*` templates
   (override `num_parallel_workers`/`worker_cuda`/`inference_server` via a small
   loop), run each under `tmux` with `PATH=$HOME/.local/bin:$PATH` (uv), tee to a log.
4. Collect the `RESULT` lines → games/s-vs-config table + curve.

**Output:** the curve, the all-GPU VRAM cap on Linux, and which single mode wins.

### R1 results (2026-06-16, Linux, 150-ep self-play, net 64f×4b, K=16, 300 sims)

| N | all-GPU (g/s) | CPU-only (g/s) | server (g/s) |
|----|---------------|----------------|--------------|
| 8 | 1.180 | — | — |
| 10 | 1.333 | — | — |
| 12 | 1.392 | 0.660 | — |
| 14 | 1.564 | — | — |
| 16 | 1.594 | 0.728 | 1.196 |
| 18 | 1.613 | — | — |
| 20 | 1.626 | 0.763 | 1.343 |
| 24 | — | — | 1.448 |

Peak VRAM (all-GPU): **3.97 GB @ N=18, 4.38 GB @ N=20** (~220 MiB/worker CUDA context).

**Conclusions:**
1. **All-GPU wins decisively** at every comparable point. GPU N=16 (1.59) beats the
   best CPU-only (0.76 @ N=20) by ~2.1× and the best server (1.45 @ N=24).
2. **VRAM is a non-issue** — the old "capped at 8 by VRAM" was a Windows artifact.
   The net is 1.5 MB; 20 CUDA contexts = 4.4 GB, half the 8 GB card. The real cap is
   the **20 hardware threads** (gains flatten at N≈16: N16→N20 is only +2%).
3. **Hybrid (per-worker device) is unnecessary.** The bottleneck is CPU cores shared
   by *all* workers regardless of inference device, so adding slower CPU-inference
   workers alongside GPU workers can't help — there's no idle resource to fill.
4. **P4 recommendation: drop the artificial 8-worker cap; default to ~16 GPU workers**
   (1.59 g/s, leaves ~4 threads for the main process + OS; ~18 if the box is otherwise
   idle). No new `num_gpu_workers` feature needed — this is a config-default change.

### Follow-up runs (batching + verification)

Full write-up, with the *why*, in
**[../research/linux-throughput-rebaseline.md](../../research/linux-throughput-rebaseline.md)**.
Headlines:

- **Within-worker batching (`mcts_batch_size` K) is still ~5×** (all-GPU N=16):
  K=1/4/8/16/32 → 0.30/1.00/1.34/1.59/1.65 g/s. Keep K=16 (K=32 only +4%, adds
  search distortion). It helps by slashing the *number* of CPU↔GPU round-trips
  across 16 contending processes — not by saturating GPU compute. **Keep it.**
- **Cross-worker inference server still not worth it** (server < all-GPU) — it swaps a
  GPU round-trip for an IPC round-trip + sync. Distinct from within-worker batching;
  the server never won (matches F5's 0.99×).
- **Reproducibility:** gpu8 = 1.194/1.180/1.204 across sessions (~2%); curve is real.
- **Thread-pin (`set_num_threads(1)`) is minor for GPU workers** (gpu8 1.20→1.13
  disabled) — *not* the Windows→Linux confound. Code-change confounds are ruled out;
  the Windows gap is OS/driver (WDDM) and/or warmup-inflated measurement — not cited
  as fact.

## R2. Time profile

On a **single-worker** Linux self-play run (clean attribution, no IPC noise) and a
**multi-worker** run (real contention):

- **py-spy** (handles multiprocessing, ~zero overhead) on the multi-worker run:
  `uvx py-spy record --subprocesses -o temp/pyspy.svg -- uv run python main.py --config <one-gen-selfplay-config>` → flame graph across workers.
- **Scalene** on a 1-worker run: splits Python vs native (numpy/torch) vs system
  time per line — tells us if a hot line is interpreter overhead (Cython/Numba
  could help) or already-native. `uv run --with scalene python -m scalene ...`.
- **line_profiler** on the functions py-spy/Scalene flag (`_descend_to_leaf`,
  `state_key`, `get_next_state`, `_select_action`, move-gen) for exact line timings.

**Output:** a ranked time breakdown (e.g. "move-gen X%, state_key Y%, get_next_state
Z%, inference W%, …") with each slice's Amdahl ceiling — the thing P2 lacked.

## R3. Memory profile

- **tracemalloc** on the in-episode MCTS tree (the per-episode dicts grow as sims
  accumulate) — confirm the sparse-MCTS ~19 MB/worker still holds at 300 sims /
  long games, so the all-GPU-N and CPU-N worker counts in R1 are RAM-safe.
- Size the **replay buffer** held between phases (board+policy+value per move ×
  episodes × lookback) — the RAM that duplicates what's already on disk.

**Output:** a per-component bytes table; confirm headroom for the R1 worker counts.

## R4. Write-up + P4 recommendation

Combine R1's curve and R2/R3's breakdowns into the **current Linux profile** and a
single recommendation for **P4's target config**: more GPU workers, a hybrid split
(needs the `num_gpu_workers` feature → P4 builds it), all-GPU-N, or the central
server. Update the roadmap's re-baseline table with the full curve, and record the
real bottleneck so P4/P5 are grounded in data, not suspicion.
