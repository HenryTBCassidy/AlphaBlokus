# Linux Time Profile — Self-Play (2026-06-18)

Where self-play spends its time on the Linux box, re-measured at the current config
and two larger-net candidates. Part of [linux-performance-profiling.md](../plans/linux-performance-profiling.md)
(Step 2). Companion: [linux-memory-profile.md](linux-memory-profile.md). Headline +
recommendations: [linux-performance-findings.md](linux-performance-findings.md).

**Method.** RTX 3060 Ti, i5-13600KF (14C/20T), Linux, commit `6cfa88e`. 300 sims,
K=16, F2 move-gen on, conv policy head. Two views:
- **Single-worker** (`profile_self_play.py cprofile`/`timing`, 1–3 games) — clean
  per-function attribution, no IPC/GPU contention.
- **Multi-worker** (py-spy `--subprocesses`, 16 workers) — the real production wall.

---

## 1. Single-worker: where the CPU time goes (64f×4b, the production net)

_(Time-split pie — chart generated locally, not committed; the breakdown is in the own-time table below.)_

cProfile, 1 game (7.21 s total), **own time** (the optimisation targets):

| Function | own s | category |
|----------|-------|----------|
| `movegen_runtime._fill_legal_actions_for_anchor` | 1.82 | **move-gen** |
| `game._placements_at_point` | 0.67 | move-gen |
| `game._all_cells_valid` | 0.52 | move-gen |
| `mcts._select_action` gather genexprs (×3) | 0.83 | **UCB select** |
| `mcts._expand_leaf` (own) | 0.22 | tree |
| `mcts._select_action` (own) | 0.21 | UCB select |
| `board.as_multi_channel` | 0.18 | encoding |
| `_at_least_one_corner` / `_generate_valid_moves` / `_valid_placement` / `_compute_adj_status` | ~0.35 | move-gen |
| board danger/placement-point updates | ~0.23 | board |

**Phase view** (cumulative, non-overlapping): `_descend_to_leaf` 3.77 s · `_expand_leaf`
2.56 s · `predict_batch` 0.73 s. Category split: **move-gen ~47%**, UCB select ~14%,
backprop/tree/transition ~22%, **inference only ~10%**, board/encoding ~7%.

**Takeaway:** single-worker self-play is **CPU-bound on move generation** (~47%), not
inference. The second slice is `_select_action` (~14%) — and note it's the **dict-gather
generators**, not the arithmetic, that cost (which is why P2's vectorisation of the
arithmetic was ~0%: it didn't touch the gather).

## 2. How the breakdown shifts with net size

_(Cost-vs-net-size chart — generated locally, not committed; numbers in the table below.)_

| net | params | single-worker s/game | inference share |
|-----|--------|----------------------|-----------------|
| 64f×4b | 340K | 4.91 | ~10% |
| 128f×8b | ~2.7M | 5.89 | ~16% |
| 256f×10b | ~13M | 7.61 | ~25% |

Bigger net → inference climbs from 10% → 25% of self-play; move-gen stays ~constant in
absolute terms (~2.7 s). So a bigger net **shifts cost toward GPU inference**, which makes
the all-GPU / K=16 batching conclusion ([linux-throughput-rebaseline.md](linux-throughput-rebaseline.md))
matter *more*, not less. Self-play time over the whole net range only grows 1.55×
(single-worker) — but training cost explodes (see memory report / right panel).

## 3. Multi-worker (16 workers) — the production wall

games/s under py-spy (sampling adds overhead — real values are the ~1.6 from the
[throughput re-baseline](linux-throughput-rebaseline.md); these are for *relative* shape):

| net | games/s (under py-spy) |
|-----|------------------------|
| 64f×4b | 1.10 |
| 128f×8b | 0.94 |
| 256f×10b | 0.45 |

Flamegraphs (16-worker, per net size) are generated locally, not committed — regenerate
with `uvx py-spy record --subprocesses` over a multi-worker self-play run.

At 16 workers we're between CPU-core saturation and GPU-round-trip contention: single
worker is 4.91 s/game → 16 perfectly-parallel workers would give 3.3 games/s, but we see
~1.6 — roughly half lost to GPU serialization + core oversubscription (16 workers + main
on 20 threads, 8 of them slower E-cores). The 256f×10b drop (0.45) shows inference
contention biting hard at the big net.

## 4. Amdahl ceilings (single-worker, what perfecting each slice could buy)

| Slice | share | ceiling (1/(1−p)) |
|-------|-------|-------------------|
| Move generation | ~47% | **1.9×** |
| UCB select (gather) | ~14% | 1.16× |
| Inference (64f net) | ~10% | 1.11× |
| Board/encoding | ~7% | 1.08× |

Only **move-gen** has a ceiling worth a structural project (bitboards, ~1.9× *if*
perfected) — and that ceiling shrinks in the multi-worker regime where GPU contention
caps gains. Everything else is a sub-1.2× lever. This is the data that says "don't chase
small slices" (the P2 lesson).
