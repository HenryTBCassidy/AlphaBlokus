# Linux Performance Profiling — Time & Memory Before Scale-Up

We switched OS (Windows → native Linux), which already changed the throughput
picture materially (see [linux-throughput-rebaseline.md](../research/linux-throughput-rebaseline.md)
— the all-GPU-16 / K=16 production config). Before committing to **serious**
training runs we want the system as performant as it can be — so this plan
**re-profiles time and memory on Linux at the current config, and extends the
profile to the scenarios we're about to scale into**: ~**10,000 games/generation**
(10× today) and a **larger neural net**. The deliverables are two readable reports
— one for time, one for memory — with charts/tables, plus a ranked shortlist of
optimisations to do *before* the big runs.

**Why now:** the last profiling report ([profiling-report.md](../research/profiling-report.md))
was measured on **Windows, the old config, at 1K games/gen** — it is stale on all
three axes. And the scale-up changes the balance: more games/gen stresses **memory**
(replay buffer + the training step that OOM-crashed before), and a bigger net shifts
cost back toward **GPU inference**, which could even overturn the all-GPU/K=16
conclusion (those were measured on the tiny 340K-param net).

**Prereqs / assets:** Linux box (`gpu-anywhere`); prod config
`run_configurations/blokus_linux16_15.json`; existing harness
[`scripts/profile_self_play.py`](../../scripts/profile_self_play.py) (`timing` /
`cprofile` / `memory` modes), [`scripts/profile_mcts_memory.py`](../../scripts/profile_mcts_memory.py),
report generator [`scripts/profile_report.py`](../../scripts/profile_report.py);
MCTS already carries detailed per-phase timers (`MCTSEpisodeStats`). Tool rationale
+ Amdahl method: [self-play-speed-investigation.md](../research/self-play-speed-investigation.md).

> **Open parameter (confirm before Step 1):** the exact **larger-net candidates** to
> profile. Current = 64f×4b (340K params). Proposed candidates: **128f×8b** and
> **256f×10b**. Henry to confirm/adjust the target net size(s).

---

## Checklist

| # | Item | Effort | Priority | Output | Done |
|---|------|--------|----------|--------|------|
| 1 | **Harness + scenario matrix** — confirm tools on Linux; pin the configs to profile (net sizes × worker setup) and the single- vs multi-worker split | ~2 hr | High | run scripts + matrix table | ✅ |
| 2 | **Time report** — fine-grained (cProfile/Scalene/line_profiler, single worker) + real-contention (py-spy `--subprocesses`, 16 workers), across net sizes; charts | ~0.5 day | High | [linux-time-profile.md](../research/linux-time-profile.md) | ✅ |
| 3 | **Memory report** — in-episode tree + replay-buffer growth + training-step peak, at 1K **and 10K** games/gen, across net sizes; charts + OOM go/no-go | ~0.5 day | High | [linux-memory-profile.md](../research/linux-memory-profile.md) | ✅ |
| 4 | **Scale & net-size projection** — wall-clock/gen + peak-RAM for {1K, 10K, higher} × {current, bigger} net; time-bound vs memory-bound verdict | ~2 hr | High | projection tables (in both reports) | ✅ |
| 5 | **Bottleneck shortlist + optimisation recommendations** — rank levers by Amdahl ceiling × effort; what to fix before the serious runs | ~2 hr | High | [linux-performance-findings.md](../research/linux-performance-findings.md) | ✅ |

> **Outcome (2026-06-18):** profiling complete; reports + recommendations delivered (see
> Output column). Headline: **speed optimisation is not worth it before serious runs**; the
> one gating issue is **training-step memory — 10k games/gen OOMs**, fixable with lazy board
> encoding in the training `Dataset` (~½ day). Full verdict + decision tree in
> [linux-performance-findings.md](../research/linux-performance-findings.md).

---

## Step 1. Harness + scenario matrix

**Goal:** lock down *what* we profile and confirm *how*, so Steps 2–4 are
turn-the-crank. No new findings here — just a defensible measurement plan.

**Confirm the tools run on Linux** (some are new to this box):
- `profile_self_play.py` modes `timing` / `cprofile` / `memory` — already ours.
- **py-spy** (`uvx py-spy record --subprocesses`) — the only tool that sees across
  the 16 worker processes; needed because the GPU round-trip/contention cost only
  appears at multi-worker scale (the K-sweep proved this).
- **Scalene** (`uv run --with scalene`) — splits Python vs native (numpy/torch) vs
  system time per line; tells us whether a hot line is interpreter overhead
  (Cython/Numba could help) or already-native (won't).
- **tracemalloc** — already wired in `profile_self_play.py memory` + `profile_mcts_memory.py`.

**The scenario matrix** (the two axes that matter for the scale-up):

| Axis | Values |
|------|--------|
| Net size | current **64f×4b** (340K) · **128f×8b** · **256f×10b** *(confirm in intro)* |
| Worker setup | **1 worker** (clean per-function attribution) · **16 workers** (real contention, prod config) |

Self-play hyperparameters held at prod (300 sims, K=16). For each profiling run,
record the exact config + commit so the reports are reproducible.

**Single- vs multi-worker rationale:** single-worker gives clean cProfile/Scalene
attribution (no IPC/contention noise); multi-worker (py-spy) gives the *real* wall
the production run will see. We need both — they answer different questions.

## Step 2. Time report

**Goal:** answer "where does self-play wall-clock go *now*, on Linux, per net size"
— the thing P2 got wrong by guessing.

1. **Fine-grained, single worker:** `profile_self_play.py cprofile` (top functions
   by cumulative + total) and Scalene (Python-vs-native split) on 1 game, for each
   net size. Plus the built-in MCTS phase timers (inference vs move-gen vs
   game-ended vs tree-work) via `MCTSEpisodeStats`.
2. **Real contention, 16 workers:** py-spy `--subprocesses` flame graph on a short
   prod-config run, per net size — captures GPU round-trip/contention that
   single-worker can't show.
3. **Amdahl ceilings:** for each slice, `1/(1-p)` — the max the whole self-play
   could gain from perfecting that slice. (Stops us chasing 6%-slices like P2 did.)

**Deliverable — `docs/research/linux-time-profile.md`:** GitHub-renderable tables +
an interactive HTML (`profile_report.py` → `temp/…html`, gitignored), matching the
style of the old [profiling-report.md](../research/profiling-report.md). Charts:
**pie** for the per-phase time split, **bar** for top-N functions, one set per net
size. Headline line per scenario: seconds/game and the dominant slice.

## Step 3. Memory report

**Goal:** answer "does 10K games/gen fit in 32 GB, and where does the memory go" —
*before* a 26-hour run discovers it doesn't. Memory is the scale-up's real risk
(the gen-3 OOM was at the training step).

Profile the three memory targets from [self-play-speed-investigation.md](../research/self-play-speed-investigation.md):
1. **In-episode MCTS tree** — `profile_mcts_memory.py` / `profile_self_play.py memory`
   (tracemalloc): per-state bytes × tree growth at 300 sims, per net size. Confirm
   the sparse-tree ~19 MB/worker still holds → 16 workers RAM-safe.
2. **Replay buffer** — bytes per (board, policy, value) example × moves/game ×
   games/gen × lookback, measured at **1K** and extrapolated + spot-checked at
   **10K** games/gen. This is the RAM held *between* phases.
3. **Training-step peak** — a single `train()` on a **synthetic buffer sized for 10K
   games/gen** (the OOM site). Per net size (bigger net = bigger activations +
   params). Report peak RSS + a **go/no-go vs 32 GB**.

**Deliverable — `docs/research/linux-memory-profile.md`:** per-component **bytes
table** + **line charts** of tree-growth-vs-sims and buffer-RAM-vs-games/gen, per
net size, and the explicit 10K-games/gen OOM verdict.

## Step 4. Scale & net-size projection

**Goal:** turn the measurements into the numbers Henry needs to choose the run
config. Combine Step 2 (time) + Step 3 (memory) into a projection:

| Scenario | games/gen | net | self-play h/gen | peak RAM | bound by |
|----------|-----------|-----|-----------------|----------|----------|
| current prod | 1,000 | 64f×4b | ~0.17 h | … | … |
| target | 10,000 | 64f×4b | ~1.75 h | … | time / mem? |
| target + big net | 10,000 | 128f×8b | … | … | … |
| stretch | >10,000 | … | … | … | … |

For each: is it **time-bound** (optimise the hot slice / add hardware) or
**memory-bound** (cap the buffer / stream to disk / shrink lookback)? This is the
decision input for the scale-up.

## Step 5. Bottleneck shortlist + optimisation recommendations

**Goal:** a ranked, honest list of what to actually fix before the serious runs —
each lever sized by **Amdahl ceiling × effort**, so we don't repeat the P2 mistake
(2× on a 6% slice = ~0%).

- Rank the time slices and memory components surfaced in Steps 2–3.
- For each candidate (e.g. bitboard move-gen, `get_next_state` buffer reuse,
  buffer-to-disk streaming, sims/move taper): expected gain, effort, risk.
- Recommend the **do-before-scale-up** subset; everything else is logged for later.
- Spin the chosen optimisations into a **follow-on implementation plan** (this plan
  stays measurement + recommendation; it does not itself implement).

---

## Notes

- **Determinism / honesty:** single-run noise is ~2% (established in the
  re-baseline); where a number drives a go/no-go (esp. the OOM verdict), repeat it.
- **No silent caps:** if a profiling run is shortened (fewer games/sims to fit
  wall-clock), say so in the report and note the extrapolation.
- **This plan deliberately does *not* implement optimisations** — it measures and
  recommends. Implementation is a separate, scoped follow-on so we choose levers
  from data, not suspicion.
