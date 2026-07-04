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

# Cap jax's on-demand growth so it can never fragment torch off a shared card
# (grow-on-demand alone still expands until the card is full). The default 0.4
# suits an 8 GB card sharing with torch's caching allocator (training step +
# CUDA eval workers); bigger cloud cards can raise it via
# ``jax_selfplay.xla_mem_fraction`` (the backend calls
# ``configure_xla_mem_fraction`` with the config value before its first
# ``import jax``) or the env var directly. Precedence: explicit env var >
# config > this default. The jax side's per-wave working set scales with
# ``batch_size × num_mcts_sims × top_k`` — not with ``num_eps`` — so raising
# games/gen stays safe under any given cap.
_MEM_FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"
_env_set_mem_fraction = _MEM_FRACTION_ENV in os.environ

DEFAULT_XLA_MEM_FRACTION = 0.4


def configure_xla_mem_fraction(fraction: float) -> None:
    """Set the XLA VRAM cap, unless an explicit env var already pinned it.

    Only effective if called before the first ``import jax`` in the process
    (XLA reads the env var once at backend init) — the jax self-play backend
    calls this with ``jax_selfplay.xla_mem_fraction`` at its entry point,
    which precedes every jax import in a training run.
    """
    if not _env_set_mem_fraction:
        os.environ[_MEM_FRACTION_ENV] = str(fraction)


configure_xla_mem_fraction(DEFAULT_XLA_MEM_FRACTION)
