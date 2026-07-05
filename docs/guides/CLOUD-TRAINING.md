# Cloud Training Runbook — single rented GPU

How to take `feat/cloud-scale-training`'s tooling from "rented a GPU" to "trained net + Pentobi
ladder result", losing nothing to preemption. Companion docs: the recommendation
([cloud-training-recommendation.md](../research/cloud-training-recommendation.md) — which card,
which net, staged plan) and the plan ([cloud-scale-training.md](../plans/archive/cloud-scale-training.md)).

Everything here is card- and cloud-agnostic: any CUDA-12 host with an NVIDIA driver and the
container runtime works (RunPod, Vast, Lambda, the home box).

---

## 0. One-time setup

- **Object storage** (any S3-compatible bucket — Cloudflare R2, Backblaze B2, AWS S3, RunPod's
  MinIO...): create a bucket, e.g. `alphablokus-runs`, and an access key pair.
- **W&B** (optional): have `WANDB_API_KEY` ready, or set `"mode": "offline"` in the config's
  `wandb` block.
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
