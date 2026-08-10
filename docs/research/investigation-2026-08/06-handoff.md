# Handoff — state as of 2026-08-04 evening

Written so a new session started **in the AlphaBlokus directory** resumes with nothing lost.
Read `03-synthesis.md` first (the action plan), then this.

## Read-order for a new session

1. `03-synthesis.md` — the four-stream action plan. **Includes two corrections marked
   `[V, corrected 2026-08-04]` — read those, they overturn earlier claims.**
2. `00-briefing-erratum.md` — corrections to the original briefing, which contains errors.
3. `04-codex-bug-sweep.md` + `05-fable-bug-sweep.md` — two independent bug audits.
4. `findings-codex.jsonl` / `findings-fable.jsonl` — same schema, diffable.
5. `stream-b-results.md` — box measurements as they land.
6. `probes-fable/` — executed probe scripts from the audit.

## Stream status

| Stream | State | Where |
|---|---|---|
| **A** — instrument repairs | **Done** | PR #69, branch `stream-a-plan-2026-08-04`. 1011 tests pass (baseline 911) |
| **B** — box measurements | **B1 running on the box** | `gpu-linux:/home/henry/AlphaBlokus/temp/benchmarks/streamb1/` |
| **C** — bug sweep | **Done** | PR #70, branch `audit/fable-bug-sweep-stream-c`. 7 new exact tests |
| **D** — value fixes | **Not started, now unblocked** (was gated on A landing) | — |

Also pushed: `feat/width-shadow-probe` (B4 probe script, written, never run, no PR).

## B1's result — the "wrong checkpoint" worry is closed

Weighted over levels 1–8, 100 games/level, 400 sims:

| net | weighted | vs gen-40 |
|---|---|---|
| gen-32 | 0.375 | −0.003 (tie) |
| gen-40 (reference) | 0.378 | — |
| gen-36 | 0.479 over L1–6 vs gen-40's 0.519 | −0.040 (worse) |

**gen-40 was not the wrong warm-start checkpoint.** gen-32 ties it, gen-36 is worse. The
internal tournament's 46-Elo ranking of gen-32 over gen-40 does not reproduce on the ladder
— the fifth time internal Elo has disagreed with it. Not final: gen-32's L9 and the
200-game L6–9 top-ups were still outstanding.

## Open decisions (all were awaiting Henry)

1. **~5,000-game self-play data-generation job** (frozen weights, no gradient updates,
   ~80 min, £0). **This is the critical-path blocker.** Both historical replay buffers were
   deleted, so there is no dataset for B3's LR sweep, no fresh eval set for A6 (there is no
   "A12" — Stream A is A1–A10; see the numbering note in `03-synthesis.md`), and no
   way to settle the value-head vintage confound. One job fixes all three. Without it
   Stream B ends after B1 and B6 cannot be configured.
2. **B2 scope** — L7–9 at 50 games/level across 400/1,600/6,400 sims (~19 h), or 6,400 at
   L9 only (~10 h). Not on the critical path.
3. **B5 kill condition** — the original ">99.5% bf16-vs-fp32 top-64 agreement" was invented
   with no basis; 97.6% was measured on a small random net (uninformative worst case).
   Proposed replacement, pre-registered before seeing trained-net data: **≥99% top-64
   overlap AND target-distribution KL below the noise floor, on a trained net.**
4. **PR #68 follow-up** — when it merges, its two width configs still carry
   `dirichlet_epsilon`/`temp_threshold` and #69's new validator will reject them. Intended
   fail-loud behaviour, but needs the one-line strip on that branch or a follow-up.
5. **Pricing-engine cleanup** (unrelated repo, pre-existing): 29 `claude/*` branches and 4
   stray worktrees under `.claude/worktrees/`. Verified none of the 29 holds a unique
   commit. Also a file literally named `:` in the repo root, present at session start.

## Standing constraints

- **No money spent, no training run started** without explicit approval. B6 in particular
  is a hard stop.
- The **Pentobi ladder is the only trusted strength instrument.** Internal self-play Elo has
  disagreed with it five times.
- **Noise floors:** ~0.05 nats held-out policy CE; ~0.022 weighted ladder score; ~±7pp per
  ladder level at 50 games. Do not report smaller effects as findings.
- **Two ladder numbers are not comparable:** 0.539 (L1–5, 50 games, 400 sims) and 0.344
  (L1–9, 100 games). Never compare across them.
- **Committed configs are not a record of what ran** — use `config.resolved.json`.
- Box: `ssh gpu-linux`, RTX 3060 Ti 8 GB, 20 cores, checkout at `/home/henry/AlphaBlokus`.
  One GPU — serialise all GPU jobs. Ladder runs are CPU-bound (Pentobi); 16 workers uses
  ~5.0 of 8 GB.

## What does and does not survive a session change

- **Survives:** everything on disk, all pushed branches, PRs #69/#70, and the detached B1
  queue on the box (`setsid`/`nohup`, survives SSH disconnect — poll
  `temp/benchmarks/streamb1/*.log`).
- **Does not survive:** the four background agents. Stream B's agent in particular will need
  respawning to collect B1's results, compute per-level binomial CIs, and run B5.
- **Changes:** the auto-memory directory is keyed to the working directory, so a session
  started in AlphaBlokus gets a different memory store than one started in
  `dynamic-pricing-engine`. That is why this handoff lives in the repo rather than in memory.

## Operational notes for whoever picks this up

### Respawning the streams

Model and effort assignments used, with the rationale — effort should scale with the cost of
a *wrong confident answer*, not with how much code gets written:

| Stream | Model | Effort | Why |
|---|---|---|---|
| A — code changes | Opus | high | Silent-failure risk; statistics rework that every kill condition is read through |
| B — box runs | Sonnet | medium | Execution only. **Rule: report raw numbers, never conclusions** — misreading noise as signal has already cost this project three runs |
| C — bug sweep | Fable | xhigh | A false all-clear is worse than no answer. Highest-reasoning task in the set |
| D — value fixes | Opus | high | Surgery across torch + JAX bridge + ONNX with parity tests |

The Agent tool takes a model but **no effort parameter** — spawned agents inherit the
session's effort. To run one stream hotter, give it its own session.

### Two traps that already cost time

1. **Worktree isolation creates a worktree of the SESSION's repo**, not the repo named in the
   prompt. Launching from the wrong directory hands agents a worktree of the wrong project.
   Fixed by running the session in AlphaBlokus — but see (2).
2. **`temp/` is gitignored, so a fresh worktree has NO `temp/runs` or `temp/benchmarks`.**
   Any agent needing run data, checkpoints or eval sets must read it from the main checkout
   by absolute path, and treat it as read-only. Tell every worktree agent this explicitly.

Also: tell agents not to commit to `main`. The B agent committed its probe script there and
it had to be moved to a branch.

### B-stream specifics

- B1 queue script on the box: `/home/henry/AlphaBlokus/temp/run_b1_queue.sh`, driver log
  `temp/benchmarks/streamb1_driver.log`, per-level results
  `temp/benchmarks/streamb1/accepted_{32,36}_L{n}_g100_seed0.{log,html}`.
- Clean per-level timings at 16 workers, 100 games, 400 sims: **L1 332s, L2 318s.** L3 is
  confounded (concurrent CPU work) — discard it. Note L1 ≈ L2, which contradicts a
  linear-in-level model, so do not extrapolate the queue total from the low levels.
- **B3** — `scripts/capacity_probe.py` is the right tool (game-level holdout split, held-out
  CE, `--arms large-warm --lr --seed`). It needs a replay buffer, which is what decision 1
  above would produce.
- **B4** — `scripts/width_shadow_probe.py` on `feat/width-shadow-probe`, written and
  CPU-smoke-tested, never run for real. **Deliberately moved to after the pilot**: it informs
  a later run, is not on the critical path, and costs 15–30 box hours. Two design facts baked
  into it, both found empirically: jax's PRNG does **not** give matching Gumbel noise on
  shared actions between `top_k=64` and `128` once positions are batched (row 0 matches, later
  rows don't), so it runs one position at a time; and `search/mcts.py` treats `temp=0` as a
  special exact-argmax case rather than the smooth limit, so a small nonzero temp like 1e-3
  overflows (`34**1000`).
- **B5** — `scripts/validate_jax_search.py --dtype bfloat16|float32` is a ready-made harness,
  no new code needed. Short; run it right after B1.
- **B2** — needs "the best checkpoint", which B1 has now answered: gen-40 (gen-32 ties it).

### D-stream specifics

- **D1 (outcome-balanced value sampling) is the first item** — cheapest, strongest evidence,
  no architecture change. Critically: the 73% White-win skew lives entirely in
  `P(label=+1 | White to move) ≈ 0.73` while the marginal label split is ~50/50, so **the
  colour shortcut survives any uniform sampling.** D1 must weight *conditionally on colour*,
  not merely balance the marginal. Easy to get wrong.
- Threading `player` through `ProcessedExample` is D-stream work that also removes A7's
  piece-parity workaround in the colour-value diagnostic. Do it as part of D1.
- **D3 (λ-blend) cannot be built as-is** — self-play stores only (board, policy, value), so
  the search's root value must first be plumbed through `jax/harvest.py` and the buffer. The
  corpus-side version already exists (opening rows, count-shrunk blend, `--opening-value`).
- **D2 (WDL head) has never been implemented** — `13066c8` only registers the idea in
  `docs/IDEAS.md` as I8. No `wdl`/`win_draw_loss`/`draw_logit` anywhere in `src/`.
- Measured composition to design against: buffer reaches 60,000/60,000 games at gen 6 and
  starts **empty** on a warm start; lifetime reuse is 12 gradient contributions per stored
  example (24 per underlying position counting its transpose twin); a game's single outcome
  label is seen ~750 times by the value head over its life.

### Reproducing a bug sweep

Both audits wrote to the same JSONL schema, which is what made them diffable:
`{"id","auditor","item","status":"verified|suspicion|clean","severity":"corrupts-training|
degrades-silently|cosmetic","title","file","line","what","failure_scenario","how_to_verify",
"proposed_fix"}`. Emitting a `clean` record per checked item is what stops the next sweep
re-auditing settled ground. Give any future auditor the "already verified clean" list from
both reports plus the traps section above.
