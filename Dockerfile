# AlphaBlokus training image — single-GPU cloud runs (docs/guides/CLOUD-TRAINING.md).
#
# Card- and cloud-agnostic by construction: the CUDA runtime comes entirely
# from the pip wheels (torch's cu12 build is the Linux default; jax[cuda12]
# bundles its own nvidia-*-cu12 libs), so any host with an NVIDIA driver and
# the nvidia container runtime (`--gpus all`) works — RunPod, Lambda, Vast,
# or the home box. No nvidia/cuda base image needed.
#
# Build (GPU image, the default):
#   docker build -t alphablokus .
# CPU-only variant (e.g. to verify the image on a laptop):
#   docker build -t alphablokus:cpu --build-arg EXTRAS="--extra jax --extra s3" .
# Run (mount a volume over /app/temp so run artefacts survive the container):
#   docker run --gpus all -v alphablokus-runs:/app/temp \
#     alphablokus --config run_configurations/blokus_cloud.json

FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

WORKDIR /app

# Lockfile + project definition first so dependency layers cache across
# source-only changes; src/ must be present for the editable project install.
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY run_configurations ./run_configurations
COPY scripts ./scripts

ARG EXTRAS="--extra jax-cuda --extra s3"
RUN uv sync --frozen --no-dev ${EXTRAS}

# W&B and tqdm behave better told they're headless.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["alphablokus"]
CMD ["--config", "run_configurations/test_run.json"]
