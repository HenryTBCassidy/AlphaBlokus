"""GPU-native Blokus Duo self-play backend: env, search, and driver as JAX array ops.

The environment (rules/step/net: ``kernels``, ``tables``, ``net``, ``search``,
``bridge``, ``checkpoint``) began as the de-risk spike
(``docs/plans/archive/jax-spike.md``); the batched driver (``actors``,
``harvest``, ``backend``) was added by the pipeline plan
(``docs/plans/archive/jax-selfplay-pipeline.md``). Everything is
``jit``/``vmap``-clean and parity-tested against the Python engine.

``backend.generate_self_play_games`` is the entry point the framework reaches
through ``selfplay/generate.py`` behind ``selfplay_backend: "jax"``. Modules
import jax lazily-by-placement, so python-backend runs never require the
``jax`` extra (``uv sync --extra jax`` CPU / ``--extra jax-cuda`` box).
"""

import os

# XLA preallocates 75% of VRAM by default, which starves the torch side of a
# mixed run (training step + CUDA eval workers -> cudaErrorMemoryAllocation,
# observed in the first A/B attempt). Allocate on demand instead; setdefault so
# an explicit env var still wins. This package is the import gateway for every
# jax entry point in the repo, so this runs before the first ``import jax``.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
