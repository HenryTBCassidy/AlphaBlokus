# Harden long training runs — crash fix, crash-safe reporting, W&B-online

**What this covers.** The first big cloud run (`blokus_cloud_60`, RTX 5090) **crashed at generation 59
of 60** with `RuntimeError: Pin memory thread exited unexpectedly` during the torch training step.
Root cause (diagnosed from the run log): **not memory** — RSS was flat at ~16 GB on a 107 GB box the
whole time — but the **training DataLoader forking a process that already has JAX loaded**. Every
training phase logged the warning:
> `os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.`

Self-play (JAX) and training (torch) run in the **same** process, so JAX's threads are live when the
DataLoader forks its workers. It survived 58 generations, then a worker/pin-memory thread died on gen
59 — a race, not a resource limit. This plan fixes that, and closes the two related gaps the incident
exposed: **a crash yields no report**, and **W&B was run offline** (nothing to analyse). All three are
"make a 13-hour run survivable and observable" fixes.

**Companion:** operational rules are in [`../guides/CLOUD-TRAINING.md`](../guides/CLOUD-TRAINING.md)
("Data-safety protocol", rules 8–9). This plan is the **code** that makes those automatic.

**Ground truth (verified on current `main`):**
- `src/alphablokus/games/base_wrapper.py:373-395` — the training DataLoader: `num_workers` (:383),
  `persistent_workers` (:384), `pin_memory` (:394), constructed at `loader = DataLoader(...)` (:390).
  It uses the default **fork** start method → forks the JAX-loaded parent. Crash surfaced iterating it
  (`base_wrapper.py:415`).
- `src/alphablokus/cli.py:126` — `c.learn(...)`; `:131` — `create_html_report(args)` runs **after**
  `learn()` returns, inside a try/except (`:129-134`) that only guards the post-run render. An exception
  *inside* `learn()` (our crash) skips the render entirely → no report. `--report-only` path at `:79-80`.
- `src/alphablokus/storage/metrics.py:167` `_init_wandb`, `:199` `mode=wandb_config.mode` — W&B mode
  comes straight from `RunConfig.wandb.mode`; nothing warns when it's `offline`.
- `RunConfig.num_parallel_workers` / `worker_cuda` and `net_config.perf.*` already exist (config.py).
- Precedent: self-play already uses **forkserver** on Linux; the parallel-benchmark work uses **spawn**
  for the same JAX-fork reason — reuse that pattern.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| H1 | Fix the crash: forkserver/spawn start method for the training DataLoader workers | 2 h | High | ✅ |
| H2 | Crash-safe reporting: render the report even when `learn()` raises | 1.5 h | High | ✅ |
| H3 | W&B online by default for real runs; loud warning + cloud config set to online | 1 h | High | ✅ |
| H4 | Validate: injected mid-run crash still yields a report; workers+JAX run is stable | 1 h | High | ✅ |

> **H4 note.** The code-side validation is done: full CI green (ruff, format, mypy, base + jax
> `pytest -m "not slow"`, plus the slow suite), the injected-crash report test passes, and Mac/CPU
> defaults (`dataloader_workers=0`, `wandb.mode="online"`) are unchanged. The end-to-end box re-run of
> the `blokus_cloud.json` recipe *past* the gen-59 failure point is pending the next box run (same
> "verify at scale on the box" tail as `archive/oom-hardening.md`).

---

## H1. Fix the crash — don't fork a JAX-loaded process for DataLoader workers

**Current state.** `base_wrapper.py:390` builds `DataLoader(..., num_workers=perf.dataloader_workers,
persistent_workers=..., pin_memory=...)` with the **default fork** start method. Because JAX is already
imported and multithreaded in the process, forking workers is unsafe (the logged warning), and it
eventually killed the pin-memory thread at gen 59.

**Fix.** Give the DataLoader a non-fork start method when `num_workers > 0`:
```python
import multiprocessing as mp
ctx = mp.get_context("forkserver")   # forkserver: cheap, workers not forked from the live JAX threads
loader = DataLoader(dataset, ..., num_workers=n, persistent_workers=..., pin_memory=...,
                    multiprocessing_context=ctx)
```
- Prefer **forkserver** (lighter than spawn; the forkserver process is started clean, before JAX
  proliferates threads). Fall back to **spawn** if forkserver misbehaves.
- **Picklability:** forkserver/spawn re-create workers without inheriting parent memory, so the
  `Dataset` (`_LazyPolicyDataset`) and its `encode_fn` must be **picklable**. Verify — the encode_fn is
  likely a bound method/closure; if it doesn't pickle, make it a module-level function or a small
  picklable callable. This is the main implementation risk; test it explicitly.
- Make it a config knob (e.g. `perf.dataloader_context: "forkserver" | "spawn" | "fork"`, default
  `"forkserver"` on Linux/CUDA, `"fork"`/current elsewhere) so Mac/CPU behaviour is unchanged and it's
  tunable. Keep `num_workers=0` fully supported (no context needed).
- Consider setting `pin_memory=False` as the safe fallback if a non-fork context still shows instability
  — pin_memory adds the very thread that died; it's a throughput nicety, not essential.

**Test:** a training run on CUDA (or a CPU stand-in) with `dataloader_workers>0` and JAX imported in the
process completes multiple epochs without the fork warning / pin-memory death. If CUDA isn't available
in CI, at least a unit test that the loader builds with the forkserver context and the dataset pickles.

**Effort:** 2 h.

---

## H2. Crash-safe reporting — render even when `learn()` raises

**Current state.** `cli.py:131` renders the report only *after* `c.learn()` (`:126`) returns; the
try/except at `:129-134` guards the render itself, not a crash *inside* learn. So a training crash →
`learn()` raises → report never renders (confirmed: our crashed run produced no `report.html`, only the
per-gen parquets, which we later rendered by hand with `--report-only`).

**Fix.** Ensure the report renders on the way out even if training crashed:
```python
try:
    c.learn(start_generation=start_generation)
finally:
    try:
        create_html_report(args)   # renders from whatever per-gen parquets exist
    except Exception:
        logger.exception("Report render failed; regenerate with --report-only.")
```
- Rendering from partial data must be safe — the report already reads per-generation parquets and we
  proved it renders a crashed run's data via `--report-only`, so a `finally` render should Just Work.
- **Optional (belt-and-braces):** render the report **periodically** inside the Coach loop (e.g. every
  N generations) so even a hard `kill -9` / pod reclaim leaves a recent report, not just an end-of-run
  one. Guard it best-effort (never let a report error kill training).

**Test:** inject an exception mid-`learn()` (e.g. monkeypatch the train step to raise on gen 2) and
assert `report.html` still gets written from the gen-1 data.

**Effort:** 1.5 h.

---

## H3. W&B online by default for a real run

**Current state.** `metrics.py:199` passes `mode=wandb_config.mode` with no guard; the first cloud run
was launched `offline` (to avoid putting the key on the pod), so **nothing synced** and the offline run
data sat on the container disk and was lost — leaving no way to analyse the run live.

**Fix.**
- Make **online** the default for a real run and **log a loud warning** when a multi-generation run
  starts in `offline` mode (`_init_wandb`), e.g. "W&B is OFFLINE — this run will not be observable and
  offline data is lost if the pod is terminated; set WANDB_API_KEY and mode=online."
- Set `run_configurations/blokus_cloud.json` `wandb.mode: "online"` and document passing
  `WANDB_API_KEY` as a pod env var (same mechanism as the SSH `PUBLIC_KEY`).
- Keep `offline` available for genuine local/throwaway tests (don't hard-fail), just make it a
  deliberate, warned choice — not the silent default for a 13-hour run.

**Test:** config with `mode="offline"` emits the warning; `mode="online"` without a key degrades
gracefully (doesn't crash the run).

**Effort:** 1 h.

---

## H4. Validate end-to-end

- Short CUDA run (few gens) with `dataloader_workers>0` + JAX self-play backend: completes without the
  fork warning escalating to a crash; report renders; W&B online shows metrics.
- Crash-injection test (H2) passes: report present after a mid-run exception.
- Re-run the `blokus_cloud.json` recipe (or a scaled-down version) far enough to clear the gen-59 point
  that failed before.
- Full CI green (ruff, format, mypy, tests).

**Effort:** 1 h.

---

## Notes for the executing agent

- **Style contract:** full type annotations (mypy `--strict`), ruff lint+format, frozen dataclasses,
  loguru (`{}`; no `print`), Google docstrings, `from __future__ import annotations`, real objects in
  tests. Keep CI green.
- **H1 is the real bug fix**; H2/H3 are the observability/robustness gaps the same incident exposed —
  do H1 first. Default behaviour on Mac/CPU (`cuda: false`, `num_workers=0`) must be unchanged.
- **The subtle risk in H1 is picklability** under forkserver/spawn — test the dataset + encode_fn
  pickle, and have `pin_memory=False` / `num_workers=0` as documented fallbacks.
- One commit per checklist row; tick Done as each lands. **Archive this file to
  `docs/plans/archive/harden-long-runs.md` (via `git mv`) once this branch merges** — the in-code
  references already point at the archive path.
