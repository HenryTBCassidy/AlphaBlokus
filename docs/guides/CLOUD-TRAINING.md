# Cloud Training Runbook — single rented GPU

How to take `feat/cloud-scale-training`'s tooling from "rented a GPU" to "trained net + Pentobi
ladder result", losing nothing to preemption. Companion docs: the recommendation
([cloud-training-recommendation.md](../research/cloud-training-recommendation.md) — which card,
which net, staged plan) and the plan ([cloud-scale-training.md](../plans/archive/cloud-scale-training.md)).

Everything here is card- and cloud-agnostic: any CUDA-12 host with an NVIDIA driver and the
container runtime works (RunPod, Vast, Lambda, the home box).

---

## ⛔ Data-safety protocol — READ FIRST (non-negotiable)

The valuable output of a run is the trained net **and every per-generation checkpoint** (the pool-Elo
tournament needs all of them). Losing them throws away the entire run's money. The rules below are
non-negotiable; they exist because a run once completed cleanly and then its net was **stranded on a
stopped pod that lost its GPU** — because auto-shutdown ran *before* the results were pulled off, and
S3 sync had been skipped "to keep it simple." Do not repeat that.

1. **Every cloud run config MUST have an `object_store` block enabled** (see §0). Checkpoints + report
   mirror to a bucket *as the run progresses*, so the artifacts are safe no matter what happens to the
   pod. **Never launch a cloud run without it.** After generation 1, **verify the first sync actually
   landed in the bucket** before walking away — an unverified sync is not a backup.
2. **Data-safety is sequenced *before* cost-saving, never after.** Auto-stopping to avoid wasted GPU
   spend is correct, but it must happen **only after results are durably off the pod**. The one true
   end-of-run order is: **run finishes → results synced/pulled to durable storage → verify they exist →
   *then* stop/terminate.** Never stop or terminate a pod whose only copy of the results is its own disk.
3. **Treat a stopped pod as already gone.** On-demand / community pods can lose their GPU the instant
   they stop — **resume is not guaranteed** ("not enough free GPUs on the host"). So "stop now, resume
   later to grab the data" is not a plan. If the data isn't already off-box, do not stop.
4. **A shutdown watchdog must SYNC-then-stop, not just stop.** Any auto-shutdown must first confirm the
   final net + all checkpoints + report are in the bucket (or pulled locally), *then* stop. A watchdog
   that only stops is a data-loss trap — that is exactly what went wrong.
5. **Terminate wipes the container disk immediately** — there is no grace period or delayed flush. Only
   terminate once the results are confirmed durable somewhere else.
6. **Prefer terminate-safe storage.** Attach a **Network Volume** (survives terminate, remounts to a new
   pod) *and/or* use S3. With either, a GPU-reclaim or a terminate can never lose data and you can
   stop/terminate freely.
7. **Explicitly pull the final net + report at run end** as belt-and-braces even with S3 on, and confirm
   the files exist locally **before** killing the pod.
8. **Run W&B ONLINE, never offline, for a real run.** Pass `WANDB_API_KEY` as a pod env var so metrics
   stream to the dashboard live (and survive the pod). Offline mode writes to the pod's *container disk*
   (wiped on terminate) and gives you nothing to watch or analyse mid-run — a run you can't observe is
   half-useless. If you can SSH-inject a key for the pod, you can inject `WANDB_API_KEY` too; there's no
   excuse for offline.
9. **The report must render even on crash.** The HTML report only auto-renders after a *clean* run
   (the render call sits *after* `learn()` returns), so a crash *inside* training yields **no report** —
   that is exactly what happened here. Metrics are written per-generation, so the data survives; make the
   report survive too: render it in a `finally`/except path (and/or periodically every N gens), and know
   you can always regenerate post-hoc with `alphablokus --config <cfg> --report-only` from the synced
   parquets. Never let a crash leave you with nothing to look at.

**Pre-flight / post-flight checklist:**
- [ ] W&B online: `WANDB_API_KEY` set as a pod env var, first metrics visible on the dashboard.
- [ ] Reporting is crash-safe (renders on failure), or you know the `--report-only` recovery path.
- [ ] Before launch: `object_store` block set and the bucket is writable with the creds on the pod.
- [ ] After gen 1: the first checkpoint/report is visibly in the bucket.
- [ ] At run end: final net + all `accepted_*.pth.tar` + report confirmed in the bucket **and** pulled
      locally — *only then* stop/terminate the pod.

---

## 0. One-time setup

- **Object storage** (any S3-compatible bucket — Cloudflare R2, Backblaze B2, AWS S3, RunPod's
  MinIO...): create a bucket, e.g. `alphablokus-runs`, and an access key pair.
- **W&B** (required for a real run — see protocol rule 8): pass `WANDB_API_KEY` as a pod env var and keep
  `wandb.mode: "online"`. Offline is only for throwaway local tests — never for a run you care about
  (offline data lives on the container disk and is lost on terminate, and you can't watch it live).
- Add the object-store block to your run config:

```json
"object_store": {
  "bucket": "alphablokus-runs",
  "endpoint_url": "https://<account>.r2.cloudflarestorage.com",
  "region": null
}
```

  `prefix` defaults to `runs/<game-group>/<run_name>` — the same layout as `temp/` locally.
  Credentials are env-only (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`); never in JSON.

## 1. Rent the box

Per the recommendation: an **RTX 5090** (or 4090 / L40S) with ≥8 CPU cores, ≥32 GB RAM, ~40 GB
disk. VRAM does not matter for this workload — buy throughput. Pick an image with Docker + NVIDIA
runtime (every neocloud's default "CUDA" template has this), attach a persistent volume if the
provider offers one (nice-to-have; the object store is the real safety net).

## 2. Get the image on the box

Either pull a pushed image, or build from the repo (5–10 min, ~12 GB):

```bash
git clone https://github.com/HenryTBCassidy/AlphaBlokus && cd AlphaBlokus
docker build -t alphablokus .
```

(CPU-only variant for laptop verification: `--build-arg EXTRAS="--extra jax --extra s3"`.)

## 3. Calibrate before committing the budget (~£1)

```bash
docker run --gpus all --shm-size=2g -v alphablokus-runs:/app/temp \
  --entrypoint python alphablokus \
  -m scripts.benchmarks.cloud_calibration \
  --config run_configurations/blokus_cloud_calibration.json \
  --rate-gbp-per-hour 0.70 --budget-gbp 100 --json temp/calibration.json
```

This measures real self-play games/s and training throughput per net size **on this card** and
prints net size → min/gen → £/gen → generations-in-budget, with a recommendation. Sanity-check it
against the recommendation doc's §3 table; pick the preset accordingly (edit
`net_config.preset` in `blokus_cloud.json` if calibration disagrees with `large`).

## 4. Launch the run

```bash
docker run -d --name blokus --gpus all --shm-size=2g \
  -v alphablokus-runs:/app/temp \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e WANDB_API_KEY \
  alphablokus --config run_configurations/blokus_cloud.json

docker logs -f blokus     # watch; detach with ^C (container keeps running)
```

Every completed generation syncs checkpoints + parquet metrics + the resume marker to the bucket
(sync failures are logged and never kill training). Watch progress in W&B or via the logs.

## 5. Resume after preemption / interruption

On the same box, or a **brand-new** one (steps 1–2 again):

```bash
docker run -d --name blokus --gpus all --shm-size=2g \
  -v alphablokus-runs:/app/temp \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e WANDB_API_KEY \
  alphablokus --config run_configurations/blokus_cloud.json --resume
```

`--resume` compares the local progress marker with the bucket's; if local is missing or behind, it
force-restores the whole run directory first, then continues from the last completed generation
(same Elo baseline, same buffer, same W&B run). At most the in-flight generation is lost.

## 6. Extend the run (climb further)

Training is designed to continue past its original horizon: raise `num_generations` in the config
and `--resume` (exactly the mechanism in step 5 — resume doesn't care *why* it's continuing). Do
this while the Elo/ladder curves still climb; see the staged plan in the recommendation doc.

## 7. Benchmark against the Pentobi ladder

`pentobi-gtp` needs to be available (see `docs/plans/archive/pentobi-harness.md`; it's a
2-minute CMake build, no Qt). On the box or locally after fetching the run:

```bash
uv run python -m scripts.pentobi_benchmark \
  --config run_configurations/blokus_cloud.json \
  --net best.pth.tar --levels 1-5 --games 40
```

Each benchmark writes an HTML report *and* drops a JSON summary into the run's `PentobiLadder/`
directory; regenerate the training report (`alphablokus --config <cfg> --report-only`) and it
renders a "Pentobi Ladder" section (green = level beaten at >50%). Loop over
`accepted_<N>.pth.tar` checkpoints to see the climb over training time.

## 8. Fetch results

Everything mirrors to the bucket continuously, so from any machine with the creds:

```bash
uv run python - <<'EOF'
from alphablokus.config import load_args
from alphablokus.storage.object_store import create_object_store
config = load_args("run_configurations/blokus_cloud.json")
store = create_object_store(config)
print("downloaded:", store.sync_down(config.run_directory))
EOF
```

…then open `temp/runs/blokus/<run_name>/Reporting/report.html`. (Or use any S3 CLI —
`aws s3 sync --endpoint-url <url> s3://alphablokus-runs/runs/blokus/<run_name> <dest>`.)

## Troubleshooting

- **`object_store is configured but boto3 is not installed`** — the image installs the `s3` extra;
  outside the container run `uv sync --extra s3`.
- **XLA grabs too much / too little VRAM** — `jax_selfplay.xla_mem_fraction` (default 0.4 for an
  8 GB card; the cloud configs use 0.6). An explicit `XLA_PYTHON_CLIENT_MEM_FRACTION` env var wins.
- **torch.compile fails on an exotic image** — it logs a warning and falls back to eager;
  training continues. Set `"compile": false` in `net_config.perf` to silence it.
- **GPU idle during the training phase** — raise `net_config.perf.dataloader_workers` toward the
  box's core count and check `pin_memory: true`.
- **`unable to allocate shared memory (shm)` during training** — DataLoader workers pass batches
  through `/dev/shm`, and Docker's default is only 64 MB. Run with `--shm-size=2g` (all commands
  above include it) or set `dataloader_workers: 0`.
