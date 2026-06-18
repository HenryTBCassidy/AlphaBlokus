# Linux Performance — Findings & Recommendations (2026-06-18)

The headline + the decision. Full detail in the two reports:
**[Time profile](linux-time-profile.md)** · **[Memory profile](linux-memory-profile.md)**.
Plan: [linux-performance-profiling.md](../plans/linux-performance-profiling.md).

Profiled the current Linux config and two larger-net candidates (64f×4b / 128f×8b /
256f×10b), single-worker (cProfile/Scalene) and multi-worker (py-spy, 16 workers), plus
the training step ramped to scale. The question was: **is the system performant enough to
do serious runs at ~10,000 games/gen (and maybe a bigger net), or do we optimise first?**

---

## The two things that matter

### 1. 🔴 Memory blocks 10k games/gen — and it's a small, clear fix
The **training step OOMs at ~450–460k examples ≈ ~8,500 games/gen** on the 32 GB box.
**10k games/gen (~530k examples) needs ~37 GB → OOM by ~5 GB.** Self-play itself is fine
(~23 GB at 16 workers); the training step is the sole wall.

Root cause (verified in code): boards are stored as **dense 44×14×14 float32 planes**
(~34.5 KB each) and the training `Dataset` copies them all into one big float32 tensor
up-front. The policy is *already* sparse — boards are the leftover. **Fix: store boards
compactly (196-byte int8) and encode to planes lazily per mini-batch**, mirroring the
sparse-policy pattern already in the code. Cuts buffer board RAM ~175× and removes the
up-front tensor → 10k+ games/gen fits easily. **Effort ~½ day, low-risk, testable.**

### 2. 🟢 Speed is *not* worth optimising before the runs
At 16 workers we're at **~1.6 games/s**, core/GPU-contention-bound. Single-worker, the
only slice with a real Amdahl ceiling is **move-gen (~47%, ceiling ~1.9×)** — but that's a
multi-week bitboard rewrite whose benefit *shrinks* in the parallel regime. Everything
else (`_select_action` gather ~14%, inference ~10%, board ~7%) is a sub-1.2× lever.
**No speed optimisation clears the effort/return bar right now.**

---

## Recommendations (ranked)

| # | Action | Why | Effort | Verdict |
|---|--------|-----|--------|---------|
| 1 | **Lazy board encoding in the training `Dataset`** | The *only* thing blocking 10k games/gen; small + low-risk | ~½ day | **Do it — if you want ≥10k games/gen** |
| 2 | Speed rewrites (bitboard move-gen, aligned-array select) | High effort, diluted benefit at 16 workers, we're already at 1.6 g/s | weeks | **Skip for now** |
| 3 | Bigger net | 128f×8b is cheap (~1.2× self-play, same train mem); 256f×10b is **11.6× slower to train** + 1.55× self-play | — | **128f×8b OK; avoid 256f×10b unless quality demands it** |
| 4 | **Pentobi benchmarking + serious run** | The actual goal; system is ready once (1) lands (or if ~7.5k/gen is acceptable, ready now) | — | **The real next step** |

## The decision

- **If you want 10k+ games/gen:** do **one** small thing — the lazy-board-encoding memory
  fix (~½ day) — then go straight to Pentobi benchmarking + a serious run. Don't touch the
  speed code.
- **If ~7,500 games/gen is acceptable:** the system is **ready now** — no optimisation
  needed. Go straight to Pentobi + serious run.
- **Either way:** the answer to "is speed optimisation worth it before serious runs?" is
  **no.** The move-gen rewrite is the only real lever and it doesn't pay off at the
  production worker count. Spend the effort on the memory fix (only if chasing 10k/gen)
  and on getting a real Pentobi-benchmarked run going.

## Net-size note
If increasing capacity: **128f×8b** is the sweet spot (cheap on both axes, same memory
ceiling). **256f×10b** triples params again but makes each training step ~7.4 min (11.6×)
and self-play 1.55× slower — only worth it if model quality clearly demands it, and it
makes the all-GPU/K=16 batching even more important (inference share 10% → 25%).

## Artifacts
- Reports: [linux-time-profile.md](linux-time-profile.md), [linux-memory-profile.md](linux-memory-profile.md)
- Charts: `assets/time_breakdown_64f4b.png`, `assets/train_memory_ramp.png`, `assets/cost_vs_netsize.png`
- Flamegraphs (16-worker, open in browser): `assets/pyspy_{64f4b,128f8b,256f10b}.svg`
- Raw log: `temp/prof/allprof.log` (gitignored)
