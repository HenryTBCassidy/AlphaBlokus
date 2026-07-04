# Cloud-Scale Training — single rented GPU, ~£100 budget

Get the repo to a state where **one run on one rented cloud GPU** produces the strongest net the
budget allows: containerized card-agnostic deploy, S3-compatible persistence + resume, a modernized
(opt-in) torch training loop, net size as a first-class knob, a cost-calibration tool, and Pentobi
ladder instrumentation. Companion recommendation:
[`docs/research/cloud-training-recommendation.md`](../research/cloud-training-recommendation.md) (C13).

**Reality check.** The current production net (64f×4b) loses ~75% of games vs Pentobi level 1; the
ladder goes to level 9. "Beat Pentobi at all levels" is the aspiration, **not** a promise for a
~£100 run. This plan designs for: (a) the strongest net ~£100 buys, (b) *measuring* where it lands
on the ladder, (c) resume-and-continue so a later spend extends the same run toward higher levels.
If the analysis says £100 is not enough, the recommendation doc says so and estimates the real cost.

**Hard constraints** (from the brief): card-agnostic (no per-card or AWS-only assumptions);
S3-compatible object storage (any endpoint); every new performance/scaling behaviour is opt-in via
config and **defaults to current behaviour** — Mac CPU (`cuda: false`) and existing box configs run
bit-identically unless a flag opts in; JAX/torch CUDA coexistence preserved; CI stays green.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| C1 | Parameterize the JAX XLA VRAM fraction (config + env, default 0.4) | 45 min | High | ✅ |
| C2 | `TrainingPerfConfig`: opt-in training-perf knobs on `NetConfig` (all default off) | 1 h | High | ✅ |
| C3 | Modernize `train()`: autocast bf16/fp16 + GradScaler, TF32, cudnn.benchmark, channels_last, DataLoader workers/pin_memory/prefetch, non_blocking copies, on-device loss accumulation | 3 h | High | ✅ |
| C4 | `torch.compile` on the net, guarded with fallback on failure | 1 h | High | ✅ |
| C5 | Net-size presets (named filters×blocks recipes) + param-count helper + scaling tests | 1.5 h | High | ✅ |
| C6 | S3-compatible object store client + `ObjectStoreConfig` (opt-in, local FS default) | 3 h | High | ✅ |
| C7 | Wire sync into the run loop (per-generation + final) and `--resume` from object storage | 2 h | High | ✅ |
| C8 | CUDA Dockerfile on uv (`jax-cuda` + torch CUDA), entrypoint, .dockerignore; verify build + `test_run.json` end-to-end | 2.5 h | High | |
| C9 | `docs/guides/CLOUD-TRAINING.md` runbook (rent box → pull image → mount → launch → resume → fetch) | 1.5 h | High | ✅ |
| C10 | Calibration tool `scripts/benchmarks/cloud_calibration.py` (net size → games/s, s/gen, £/gen, budget fit) | 3 h | High | ✅ |
| C11 | Pentobi ladder instrumentation: checkpoint-ladder mode + JSON results + ladder section in the HTML report | 2.5 h | Medium | ✅ |
| C12 | Cloud run-config family: `blokus_cloud.json` + `blokus_cloud_calibration.json` | 1 h | High | ✅ |
| C13 | Recommendation doc: card class, net size, full config for ~£100, staged ladder plan, cost beyond £100 | 2 h | High | ✅ |
| C14 | Verify defaults unchanged (Mac CPU test run), full CI gates, open PR | 1.5 h | High | |

> **Set aside (stretch, deliberately unnumbered):** a multi-GPU DDP/FSDP training path. Single-GPU
> is the primary target; DDP only makes sense after the single-GPU loop is measured (C10) and would
> otherwise risk regressing it. Logged in `docs/IDEAS.md` if it survives contact with the numbers.

---

## C1. Parameterize the JAX XLA VRAM fraction

**Current state:** `games/blokusduo/jax/__init__.py` hardcodes
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` — right for the 8 GB 3060 Ti sharing with torch, needlessly
tight on a 24–32 GB cloud card.

**Fix:** add `xla_mem_fraction: float = 0.4` to `JaxSelfPlayConfig`. The env var is consumed at
first `import jax`, which happens inside the backend workers *after* config load, so the backend
entry point (`backend.generate_self_play_games`) sets
`os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.jax_selfplay.xla_mem_fraction))`
before importing jax. The package `__init__` keeps a plain `setdefault` fallback for non-run entry
points (scripts). Precedence: explicit env var > config > 0.4 default. Torch/JAX coexistence
unchanged (`PREALLOCATE=false` stays).

## C2. `TrainingPerfConfig` — opt-in knobs

New frozen dataclass nested on `NetConfig` (`perf: TrainingPerfConfig = field(default_factory=...)`),
loaded from JSON like every other nested config. All fields default to "off" = today's behaviour:

- `autocast_dtype: Literal["off", "bf16", "fp16"] = "off"` — training-loop autocast; GradScaler
  used only for fp16 (bf16 needs none). Prefer bf16 on modern cards.
- `tf32: bool = False` — `torch.set_float32_matmul_precision("high")` + cuDNN allow_tf32.
- `cudnn_benchmark: bool = False` — fixed conv shapes, so benchmark mode is safe.
- `channels_last: bool = False` — net + batches to `memory_format=torch.channels_last`.
- `compile: bool = False` — see C4.
- `dataloader_workers: int = 0` — 0 keeps today's in-process loading.
- `pin_memory: bool = False`, `prefetch_factor: int = 2`, `persistent_workers: bool = False`.
- `log_every_batches: int = 1` — per-batch `.item()`/metrics cadence; 1 = today's behaviour.

All CUDA-only knobs are silently inert on CPU (guarded), so a Mac config with them set is still safe.

## C3. Modernize `train()`

`games/base_wrapper.py` `train()` is a plain fp32 single-threaded loop; the per-item dense encoding
of a 17,837-length policy runs on the main process, so on a fast card the GPU starves. Changes (all
gated on C2 knobs, defaults bit-identical):

- Apply TF32/cudnn.benchmark once in `__init__` when on CUDA and enabled.
- `channels_last` on the module + `.to(memory_format=...)` on batches.
- DataLoader: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor` from config
  (workers>0 moves the dense policy/board encoding into worker processes — the key GPU-feeding fix).
- Host→device copies `non_blocking=True` when pinned.
- Autocast context around forward+loss; `GradScaler` scale/step/update for fp16 only.
- Loss meters accumulate **on-device tensors**; `.item()` sync + `metrics.log_training` only every
  `log_every_batches` (aggregated), keeping the parquet schema unchanged.

Tests: CPU tests that (a) defaults produce the exact same loss sequence as before (seeded), (b) each
knob enabled on CPU is a no-op or works (workers>0 path runs), (c) fp16 GradScaler path is exercised
where possible.

## C4. Guarded `torch.compile`

`perf.compile: true` wraps the net in `torch.compile` at wrapper init, inside `try/except` — any
compile failure logs a warning and falls back to eager. Checkpoint save/load must keep working: save
`state_dict` from the *original* module (`_orig_mod` when compiled) so checkpoints stay
interchangeable between compiled and eager runs. Test: state-dict round-trip with compile on (CPU).

## C5. Net-size presets

`NET_PRESETS` in `config.py` mapping names → `(num_filters, num_residual_blocks)`:
`small` 64×4 (today's prod), `medium` 128×8 (run3), `large` 192×12, `xl` 256×16 (AGZ 20×256 scaled
to 14×14). JSON `net_config` accepts `"preset": "large"`; `load_args` merges the preset values in
(explicit `num_filters`/`num_residual_blocks` keys win). Add a `count_parameters` helper +
tests that presets construct, forward, and scale monotonically in params. VRAM is not the binding
constraint (44×14×14 activations are tiny); the preset ceiling is budget, not hardware.

## C6. Object-store client

New `storage/object_store.py`: `ObjectStoreConfig` frozen dataclass on `RunConfig`
(`object_store: ObjectStoreConfig | None = None` — None = today's local-FS behaviour):
`endpoint_url: str | None` (None = AWS; any S3-compatible endpoint otherwise), `bucket`,
`prefix` (default = run name), `region: str | None`. Credentials from the standard env vars
(`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`) — never in config JSON. `ObjectStore` class wraps a
boto3 client behind a small protocol: `upload_file`, `download_file`, `list_keys`,
`sync_up(local_dir, changed_since)` (mtime-based incremental), `sync_down(prefix, local_dir)`.
`boto3` ships in a new optional extra `s3`; importing the module without it raises a clear error
only when an object store is actually configured. Tests use an in-memory fake client (no network,
no moto dependency).

## C7. Sync in the run loop + `--resume` from object storage

- Coach: after `_write_progress_marker(generation)` (all of a generation's data is on disk at that
  point), call `object_store.sync_up(run_directory)` — incremental, so per-gen cost is the fresh
  parquet/checkpoint files only. Failures log a warning and never kill training (same policy as
  report rendering). Final sync after the report renders.
- CLI `--resume`: when an object store is configured and the local progress marker is missing or
  behind the remote one, `sync_down` the run directory first, then resume exactly as today. An
  interrupted/terminated instance therefore loses at most the in-flight generation.

## C8. CUDA Dockerfile

`Dockerfile` at repo root, `nvidia/cuda:12.x-cudnn-runtime-ubuntu24.04` base (card-agnostic — works
on any CUDA-12-capable card on RunPod/Lambda/Vast), uv installed, `uv sync --extra jax-cuda --extra
s3` + CUDA torch, package installed, `ENTRYPOINT ["alphablokus"]` with `CMD ["--config",
"run_configurations/test_run.json"]`. `.dockerignore` excludes `temp/`, `.git`, caches. Verify:
image builds locally and `docker run` executes `test_run.json` end-to-end on CPU (CUDA flag falls
back cleanly when no GPU is present — verify that fallback too, since it's the local-verification
path).

## C9. `docs/guides/CLOUD-TRAINING.md`

Runbook: choosing a card class on a neocloud (RunPod / Lambda / Vast — what matters: bf16
throughput + memory bandwidth, not VRAM), pulling/building the image, mounting a volume, setting S3
creds, launching `blokus_cloud.json` in tmux/detached, watching W&B, resuming after preemption
(`--resume` + object store), running the calibration tool first, fetching reports, running the
Pentobi ladder benchmark on the result, and extending the run (bump `num_generations`, `--resume`).

## C10. Calibration tool

`scripts/benchmarks/cloud_calibration.py`. On whatever GPU it runs on: for each net preset (or
explicit sizes), (1) time jax-Gumbel self-play games/s at the production search config on a short
burst, (2) time one training epoch over a synthetic buffer of realistic size → s/generation.
Inputs: `--rate-gbp-per-hour`, `--budget-gbp` (default 100), `--games-per-gen`. Output: a table
(markdown + JSON) — net size → games/s, min/gen, £/gen, and generations affordable in budget —
plus the recommended size×generations. Runs in miniature on CPU/Mac for tests (tiny presets, tiny
bursts); the money numbers only mean something on the rented card, which is the point: **rent the
card for ~15 min of calibration before committing the budget.**

## C11. Pentobi ladder instrumentation

- `scripts/pentobi_benchmark.py`: add `--checkpoint-ladder` conveniences — `--levels 1-5` subset
  sweeps, `--json <path>` machine-readable results, and a documented loop over saved
  `accepted_*.pth.tar` checkpoints; results land in the run's `PentobiLadder/` directory
  (new `RunConfig.pentobi_ladder_directory` property).
- Report: if `PentobiLadder/*.json` exists, render a ladder section (level × win-rate table +
  headline "Pentobi Level") in the HTML report. Absent → report unchanged.
- Resume-and-continue: verified in C14 — bump `num_generations` in the config and `--resume`
  continues training from `latest.pth.tar` (already supported; documented + tested as the
  "extend toward higher levels" path).

## C12. Cloud run-config family

- `run_configurations/blokus_cloud.json` — card-agnostic big-run defaults sized to ~£100 (numbers
  from C13; jax+Gumbel self-play, `large` preset, big buffer, perf knobs on, object store block
  present-but-commented via a `_comment` key or documented in C9).
- `run_configurations/blokus_cloud_calibration.json` — small config the calibration tool and image
  verification use.

## C13. Recommendation doc

`docs/research/cloud-training-recommendation.md`. From existing evidence (jax-pipeline-ab.md
throughput, deepmind-run-configs.md scaling, run3 results, current marketplace £/h): which card
class to rent, recommended net size + full run config for ~£100, expected outcome **with caveats**,
staged Pentobi-ladder plan (train → benchmark → extend), and an estimate of what pushing past £100
toward higher levels would cost. Every number tagged **[measured]**, **[extrapolated]**, or
**[calibrate]** (i.e. the calibration tool must confirm on the rented card before the big run).
Hard bound honoured: no empirical multi-net training sweeps — reasoning + the calibration tool.

## C14. Verification + PR

- `uv run pytest` (full, incl. slow), `uv run ruff check .`, `uv run ruff format --check src tests
  scripts`, `uv run mypy` — all green.
- Default-behaviour check: run `test_run.json` (Mac CPU) on this branch and confirm normal
  completion with defaults (no new flags set) and unchanged config surface for existing JSONs.
- Docker build + containerized `test_run.json` run (C8 evidence).
- Open PR: what changed, how to launch a single-GPU cloud run, how to benchmark vs the Pentobi
  ladder. Capture any extras in a "Scope additions" section here.
