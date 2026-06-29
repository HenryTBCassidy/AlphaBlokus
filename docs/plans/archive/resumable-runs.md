# Resumable runs + crash-safe report

## Context

`blokus_run2_bignet` was OOM-killed at gen-15 training (now fixed — see
[lazy-board-encoding.md](lazy-board-encoding.md)). Recovering it exposed two gaps that make
long overnight runs fragile:

1. **No real resume.** `_learn_loop` always iterates `range(1, num_generations + 1)`, and the
   `load_model` config path loads `best.pth.tar` but then calls
   `load_self_play_history(up_to_generation=0)` (empty) and **re-freezes the Elo baseline from
   whatever net is loaded** (`Coach.__init__` saves+reloads `elo_baseline.pth.tar` at
   construction). So the only "resume" available restarts generation numbering at 1, drops the
   replay buffer, and re-anchors the baseline — i.e. not a continuation. A crash (OOM, power,
   stray kill) means lost hours.
2. **End-of-run report can sink a finished run.** `create_html_report` runs in-process at the
   end of `main.py`; when it died for run1 the run looked failed even though training was
   complete.

**Goal:** a true `--resume` that continues a run *in place* from where it stopped — same
generation numbering, same replay window, **same frozen baseline** — plus make the end-of-run
report non-fatal. (Item 2 from the earlier discussion — surfacing the `max_queue_length`
silent drop — is intentionally deferred; Henry is reconsidering replay buffering separately.)

**Two facts that make this clean (verified in code):**
- `save_checkpoint` already persists `optimizer_state_dict` + `scheduler_state_dict`
  (`games/base_wrapper.py:420`), so resuming restores the optimizer too — a faithful
  continuation, not a cold restart. (R5: `load_checkpoint` already applies them.)
- The net to resume from is the one carried into the next generation. We can't rely on
  `best.pth.tar` — it doesn't exist until the *first* accept, so a run that crashes before any
  accept would have nothing to load. Instead the loop saves the carried net (`self.nnet` after
  the accept/reject step — accepted net, or reverted previous best) to **`latest.pth.tar`** at
  the end of every generation; it's always present and is exactly the continuation net.

## Design

### Item 1 — true in-place resume

- **Completion marker (single source of truth).** At the very end of each generation in
  `_learn_loop`, atomically write a small marker (e.g. `Logs/progress.json`:
  `{"last_completed_generation": g, "wandb_run_id": …}`). Resume reads `g` from this rather than
  inferring from partition dirs — which is fragile given the self-play parquet's 0-based naming
  (`self_play_<gen-1>.parquet`). Write-to-temp + `os.replace` so it's never half-written.
- **`start_generation` through the loop.** `_learn_loop(start_generation)` →
  `range(start_generation, num_generations + 1)`; `learn(start_generation=1)` default. Nothing
  else in the loop assumes gen 1.
- **Preserve the baseline on resume.** In `Coach.__init__`, when `elo_baseline.pth.tar` already
  exists, **load it** instead of re-saving from the current net. Keeps the original gen-0 random
  anchor so the resumed Elo curve stays comparable to the pre-crash portion and to other runs.
  (Guard on a `resume` flag and/or file existence.)
- **`main.py --resume`.** Detect `G` from the marker → load `latest.pth.tar` (net + optimizer +
  scheduler) → build `Coach(resume=True)` → `load_self_play_history_for_resume(G)` to refill the
  replay window (file index `G-1` for gen `G`) → `c.learn(start_generation=G + 1)`. All writes are
  generation-keyed
  (hive partitions, gen-suffixed nets), so continuing from `G+1` *appends* and never clobbers
  gens `1..G` — confirm this in validation.
- **W&B continuity.** Persist the W&B run id in the marker; on resume call
  `wandb.init(resume="allow", id=…)` so the live dashboard continues one run instead of starting a
  second. Lower priority — parquet data is the source of truth and the report is rebuilt from all
  gens regardless.
- **Determinism note (not a regression):** RNG state is not checkpointed, so a resumed run is not
  bit-identical to an uninterrupted one. That's expected for resume; the parallel/movegen
  determinism tests are not a guard here.

### Item 3 — crash-safe end-of-run report

- Wrap the final `create_html_report(args)` in `main.py` in try/except: on failure, log the error
  and a clear "training complete — regenerate the report with `--report-only`" message, and let
  the process exit 0. Training success no longer hinges on report rendering. (The replay-cap OOM
  that caused run1's failure is already fixed; this is belt-and-braces + covers any future report
  bug.) The explicit `--report-only` path stays un-wrapped so it surfaces errors directly.

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| R1 | Atomic per-generation completion marker (`last_completed_generation`, `wandb_run_id`) + carried-net `latest.pth.tar`, written at end of each `_learn_loop` iteration | ~1 hr | High | ✅ |
| R2 | `learn` / `_learn_loop` take `start_generation` (default 1); loop ranges from it | ~0.5 hr | High | ✅ |
| R3 | `Coach.__init__` preserves an existing `elo_baseline.pth.tar` on resume (load, don't overwrite) | ~0.5 hr | High | ✅ |
| R4 | `main.py --resume`: marker → `G`, load `latest.pth.tar`, `load_self_play_history_for_resume(G)`, `learn(start_generation=G+1)`; no clobber of gens `1..G` | ~1.5 hr | High | ✅ |
| R5 | `load_checkpoint` restores optimizer + scheduler — already does (`base_wrapper.py:439-441`); no change needed | ~0.5 hr | High | ✅ |
| R6 | W&B run continuity on resume (`resume="allow"`, id from marker) | ~1 hr | Medium | ✅ |
| R7 | Wrap end-of-run `create_html_report` in try/except with a clear regenerate message; run still exits clean | ~0.5 hr | High | ✅ |
| R8 | Validation: in-process resume test (2→4 gens) asserts continuation + baseline byte-identical + no clobber + marker advance; marker round-trip unit test. Both pass (3× no flake) | ~1.5 hr | High | ✅ |

## Validation

- **End-to-end (TTT, fast):** run ~6 generations, hard-kill after gen 3, `--resume`. Assert:
  generation numbering continues at 4 (not 1); `elo_baseline.pth.tar` is byte-identical before
  and after; gens 1–3 parquets/nets are untouched; gens 4–6 are appended; the final report spans
  1–6. Repeat once killing *during* a generation (partial gen) to confirm the marker resumes from
  the last *completed* gen, not a half-written one.
- **Unit:** marker round-trips atomically; `_learn_loop` honours `start_generation`; Coach loads
  an existing baseline rather than re-freezing.
- Existing `tests/test_core` + determinism tests still pass (the loop change is a no-op at the
  default `start_generation=1`).

## Outcome

Shipped in commit `e08a0a1`. All of R1–R8 done: `--resume` continues a run in place from the
last completed generation (marker + `latest.pth.tar`), reusing the frozen Elo baseline and W&B
run, and the end-of-run report is wrapped so it can't sink a finished run. The resume mechanics
are covered by an automated in-process test (2 gens → resume to 4: numbering continues, baseline
byte-identical, gens 1–2 untouched, marker advances) plus a marker round-trip unit test — both
green, 3× no flake. The only thing not yet exercised is `--resume` against a *real* multi-hour
run; that'll happen incidentally on the next big-config run (paused for now). Archived as
code-complete.

## Out of scope (deferred)

- **`max_queue_length` silent-drop warning** (earlier item 2) — Henry is rethinking replay
  buffering; revisit then.
- **Bit-reproducible resume** (checkpointing RNG state) — not needed; continuation, not replay.
- **Compact board storage** in the replay buffer — separate memory lever, only needed for runs
  well past run2's lookback/games-per-gen (see lazy-board-encoding.md "Out of scope").
