# Stream B — results log (2026-08-04)

Raw measurements only. No interpretation. CIs are Wilson/binomial on win counts unless noted.
Box: `gpu-linux` (RTX 3060 Ti, 8 GB, 20 vCPU, 31 GB RAM), checkout `/home/henry/AlphaBlokus`, single GPU — all jobs serialised.

Reference numbers (given, not re-derived): v3 gen-40 weighted 0.3444, per-level
77/71/61/53/47/40/16/20/21 (100 games/level, 400 sims) — source confirmed on-box at
`temp/runs/blokus/blokus_cloud_v3/PentobiLadder/ladder_accepted_40.pth.tar_20260803T144300Z.json`.
No wall-clock is stored in that file (matches A8's finding).

---

## Timing instrumentation (prerequisite for scheduling B1)

Command: `uv run python scripts/pentobi_benchmark.py --config run_configurations/blokus_cloud_v3.json --net accepted_32.pth.tar --level <L> --games 8 --sims 400 --workers 8`

| Level | Games | Wall clock (8 games, `--workers 8`, net accepted_32) | real/user/sys |
|---|---|---|---|
| 1 | 8 | 69.24s | real 1m9.244s / user 4m27.415s / sys 0m7.212s |
| 9 | 8 | 566.09s | real 9m26.090s / user 31m58.997s / sys 0m9.521s |

Ratio L9/L1 ≈ 8.2x. At this game count the harness only actually used 4 of the 8
requested workers at L9 (8 games / 2-games-per-chunk = 4 chunks; the critical path was
one worker's two sequential ~4-5 min games). Whether a 100-game run keeps this same
per-chunk cost or gets better parallel utilisation across all 8 workers was **not**
directly measured — L2-L8 were not sampled at all. So the following is a bracket, not a
schedule:

- Naive linear interpolation of per-8-games cost across levels 1→9 and scaling ×12.5 to
  100 games/level gives **very roughly ~10 hours** for one net's L1-9 sweep at 100
  games/level, dominated by L6-9.
- Chunk-critical-path reasoning (2 games/chunk, ≤8-way parallel) gives a lower bound of
  **~35-60 minutes just for L9 alone at 100 games** (scaling the observed 566s/8-games
  chunk-bound cost), consistent with the ~10 hour whole-ladder figure once L1-L5's
  smaller cost is added.
- Two nets (gen-32, gen-36) × full L1-9 sweep ≈ **~20 hours**, before the L6-9 top-up.
- L6-9 top-up (+200 games/level, both nets) is dominated by the same slow levels: rough
  estimate **~25-30 hours** more.
- **Total rough B1 estimate: on the order of 1.5-2.5 days of continuous, serialised GPU
  time.** This is a bracket from 2 measured points (L1, L9) and a linear-in-level
  assumption for L2-L8 that was not verified — could be an over- or under-estimate if
  Pentobi's think-time doesn't scale linearly with level.

Given this, B1 was launched as a background queue on the box rather than run
synchronously (see status below); exact per-level timings will be recorded in
`temp/benchmarks/streamb1/*.log` as it progresses.

---

## B1 — Ladder gen-32 / gen-36

Status: **launched, in progress, not complete.** Started 2026-08-04 17:59:30 UTC on
gpu-linux as a detached background process (PID group led by `bash temp/run_b1_queue.sh`,
`setsid`+`nohup`+`disown`, survives the SSH session ending). Queue (serial, one net/level
combination at a time):

1. `accepted_32.pth.tar` L1-9, 100 games/level, seed 0 — **running now**
2. `accepted_32.pth.tar` L6-9, 200 games/level, seed 1000 (top-up)
3. `accepted_36.pth.tar` L1-9, 100 games/level, seed 0
4. `accepted_36.pth.tar` L6-9, 200 games/level, seed 1000 (top-up)

Correction to the timing estimate above: the real 100-game run uses **all 8 workers**
(chunks of 6-7 games/worker observed), not the 4 effectively used in the small 8-game
timing sample — so the ~10h/net and ~46h-total brackets above are pessimistic upper
bounds; real throughput should be higher. Exact per-level durations will be in
`temp/benchmarks/streamb1/*.log` (timestamped START/END lines) and
`temp/benchmarks/streamb1_driver.log` on the box once each step finishes.

**Not yet available:** win/loss counts, weighted scores, CIs for gen-32/gen-36. Decision
rule (gen-32 vs gen-40 by >0.022 weighted) cannot be evaluated until this finishes.

**Relaunch at 16 workers (per coordinator instruction):** original queue killed at ~5 min
in (nothing lost, per-level restructured), relaunched per-level (not `--sweep`) so
per-level START/END timestamps give real timing. GPU memory checked immediately after
warm-up: 5032-5036 MiB of 8192 MiB — under the 7 GB threshold, no need to drop to 12
workers. 16 `multiprocessing-fork` worker processes confirmed live at ~98% CPU each; load
average settled at 10-12 on 20 cores (not saturated).

**Real per-level timing, net accepted_32, level 1, 100 games, 400 sims, 16 workers:**
17:06:08Z → 17:11:40Z = **332s (5m32s)**.

**Correction:** the "no 2x speedup" framing above was wrong and is retracted. The ~300s
figure it was compared against was not an instrumented measurement — it was an
extrapolation from partial tqdm progress-bar sampling during a level transition, not a
timed 8-worker run. The only instrumented 8-worker number that exists is the original
8-game test (69.24s/8 games → ~860s/100 games extrapolated), against which 332s at 16
workers is roughly 2.6x faster. There is no clean 8-vs-16 comparison and, per the
coordinator, we are not running one — the scaling question is dropped entirely. What
matters from here is the absolute per-level wall clock for scheduling, not a workers
scaling factor.

**Level 2 (clean):** 17:11:40Z → 17:16:58Z = **318s**.

L1 (332s) and L2 (318s) are essentially equal, which contradicts the linear-in-level
model used earlier to bracket the total queue time (that model was built from the L1/L9
8-game ratio of 8.2x and assumed even per-level steps). **Not extrapolating the total
queue time until L4-L6 give clean numbers** — per coordinator, correctly holding off
rather than re-deriving from an already-contradicted model.

**Confound disclosure:** while B4's probe script was being written and CPU-smoke-tested
on this same box (see B4 section), those test runs briefly competed for CPU with the
running B1 queue: overlap with the last ~78s of level 2's window, and with most of level
3's window (load average briefly hit 20.9 on 20 cores during that overlap). Level 1 and
level 2's numbers above are unaffected (smoke testing started after level 1 finished and
only briefly overlapped level 2's tail); **level 3's timing should be treated as
possibly-contaminated** and not used to firm up the per-level curve. **No further
box-side work of any kind (no smoke tests, no probes, nothing) will run while B1 is in
progress** — monitoring only from here, per coordinator instruction.

**Per-level wall clocks, running total (updated as levels land; win rates + CIs added
once each level's log is parsed):**

| Net | Level | Games | Wall clock | Note |
|---|---|---|---|---|
| accepted_32 | 1 | 100 | 332s | clean |
| accepted_32 | 2 | 100 | 318s | clean |
| accepted_32 | 3 | 100 | 365s | confounded, see above — timing not trustworthy |
| accepted_32 | 4 | 100 | pending | |

## B2 — Eval-time search scaling

Status: **blocked on human approval for the re-scoped version (coordinator is
re-scoping, likely L7-L9 only at 30-50 games/level). Will not start until that approval
lands — not merely GPU-blocked.**
Launch script staged at (local) `run_b2_queue.sh` — needs the "best checkpoint" name as
its one argument, which is a B1 output, not decided yet. Also unresolved: which net
counts as "best" — `blokus_cloud_v3/Nets/best.pth.tar` (by internal Elo/arena, dated Jul 7)
may not agree with whatever B1 finds strongest by ladder; README's "current best" text
predates this investigation's L1-9 ladder work and describes a different net
characterisation (192f×12b, ~70 gens, level 6 frontier) than the config actually in
`blokus_cloud_v3` (preset "large"). Needs resolving against B1's result, not assumed.

## B3 — Learning-rate sweep, offline

Status: **impossible as specified (confirmed) and blocked on human approval for the
fix.** Coordinator confirmed: only `Nets/` survived for both `blokus_cloud_v3` and
`blokus_paired_gate_rerun`, so B3 as written cannot run. The likely fix is a fresh
self-play data-generation job with the best checkpoint (which would also supply a fresh
eval set another stream needs) — that is new scope outside this session and requires
human approval. **Will not start it.**

**Data-availability finding (original):** no `SelfPlayHistory` parquet buffer
exists for `blokus_cloud_v3` or `blokus_paired_gate_rerun` on either the box
(`/home/henry/AlphaBlokus`) or the local main checkout — both run directories contain
only `Nets/`, ladder/arena/report artifacts, not the raw self-play buffer. Runs that do
have local `SelfPlayHistory`: `blokus_run3_overnight`, `blokus_run1_taper`,
`blokus_linux_15`, `blokus_jax_gumbel_30`, and others (see box listing). Needs a decision
on which buffer stands in for "existing run data" before B3 can start.

`scripts/capacity_probe.py` already implements frozen-buffer training + game-level
holdout split + held-out CE (via `alphablokus.training.holdout`) and looks like the
right base for this — it supports `--arms large-warm --warm-start <ckpt> --lr <lr> --seed
<seed>`, so the 3 LR x 3 seed matrix is 9 invocations with fixed `--history-dir`,
`--file-indices`, `--warm-start`.

## B4 — Width shadow test

Status: **script written and CPU-smoke-tested. Will NOT be run in this stream.**
Coordinator moved B4 to after the pilot (B6): the critical path to the only
money-spending decision is B1 → replay-buffer generation → B3 → B6, B4 isn't on it, and
at an estimated 15-30 box hours it would delay that decision point for no benefit. B4
only informs whether to include width in a *later* run. `scripts/width_shadow_probe.py`
is left in place, committed, unrun (see commit note at the end of this section).

New file: `scripts/width_shadow_probe.py` (new script only — nothing under `src/`
touched). Does not reuse `validate_jax_search.py` as-is (confirmed it can't: no
`gumbel_max_considered` CLI exposure, and computes none of (b)/(c)/(d)); instead
duplicates the small amount of `search.py`'s gumbel-branch logic needed to retain the
mctx `search_tree` (`make_search` intentionally discards it), so completed-Q can be read
via `mctx.qtransform_completed_by_mix_value` — the same qtransform production actually
uses for the gumbel path (search.py's gumbel branch passes no override).

**Key design finding, verified empirically on this box (not assumed):** jax's
counter-based PRNG does NOT give matching Gumbel noise on the shared 64 actions between a
`top_k=64` and `top_k=128` call from the same key once more than one position is batched
together (checked: batch-row 0 matches, batch-row 1 does not, for `jax.random.gumbel`
under shapes `(2,64)` vs `(2,128)`). The script therefore runs **one position at a time**
(not batched) so the "identical randomness" requirement is actually true rather than
assumed, and it asserts in-loop that intervention's top-64-by-prior exactly equals
control's top-64 (should always hold — both are exact `top_k` by the same prior logits).

**Bug found and fixed during CPU testing:** the (d) referee call originally used
`get_action_prob(canonical, temp=1e-3)`, which overflowed (`OverflowError: 34 ** 1000`) —
`src/alphablokus/search/mcts.py::get_action_prob` handles `temp=0` as a *special case*
(exact argmax over visit counts), it is not a smooth limit as temp→0. Fixed to call with
`temp=0` directly; confirmed working in isolation and inside the full probe.

**CPU smoke tests run** (tiny settings, `JAX_PLATFORMS=cpu`, `net accepted_32.pth.tar`,
192f×12b): 2 positions/sims=16 (no decision changes, ran clean); 6 positions/sims=8 (no
changes, ran clean); 10 positions/sims=8/considered=6/top-k 12→24 (**1/10 decisions
changed, exercised the (d) referee path end-to-end successfully** — deeper 24-sim referee
agreed with neither control nor intervention on that one changed position, `n=1` so not
meaningful on its own, just confirms the code path runs). All three runs: 0
`topk_prefix_mismatch_warnings` (sanity check passing as expected).

**Not run, and not scheduled in this stream** (see status line above) — the real run at
spec settings (top_k 64/128, considered 64, 128 sims, 100+ frozen positions,
`--deeper-sims` at a real "much deeper" value) is deferred to after B6.

**Findings-pile note (general evaluation-harness footgun, not specific to B4):**
`src/alphablokus/search/mcts.py::get_action_prob` treats `temp=0` as a *special exact-
argmax case*, not the smooth limit of temp→0 — a small-but-nonzero temperature such as
`1e-3` computes `visit_count ** (1/temp)` = `visit_count ** 1000`, which overflows
(`OverflowError`) for any visit count > 1. This is a real trap for anyone writing an
evaluation harness that wants "deterministic best move" and reaches for a tiny
temperature instead of exactly `0`; it happened to bite this probe script, not
production code, but is worth carrying forward as a general note.

## B5 — bf16 vs fp32 self-play A/B

Status: queued, staged, not yet run. Per coordinator: runs immediately after B1 finishes,
before B4 (it's short). Launch script staged locally (not yet copied to the box) at
`run_b5_queue.sh`: two `scripts/validate_jax_search.py` invocations (`--dtype bfloat16`,
`--dtype float32`), same checkpoint/positions/top-k sweep, script's internal rng key is a
hardcoded `jax.random.PRNGKey(0)` so the two dtype runs are directly comparable without
extra seeding work. Gives top-1 agreement and visit-distribution overlap (Σ min(p,q))
directly — this is a ready-made harness for the "top-k selection agreement" half of B5;
training-target-distribution comparison is the same overlap metric read as training
targets.

## B6

Not run. Per instructions, requires explicit human approval — out of scope for this session.
