"""GPU-native Blokus Duo environment: rules, step, and net as JAX array ops.

Born as the de-risk spike (``docs/plans/archive/jax-spike.md``, findings in
``docs/research/jax-spike-findings.md``), promoted here by the self-play
pipeline plan (``docs/plans/jax-selfplay-pipeline.md``). Everything is
``jit``/``vmap``-clean and parity-tested against the Python engine
(``tests/test_blokusduo/test_jaxenv_*.py``).

Requires the ``jax`` extra: ``uv sync --extra jax`` (CPU) or
``uv sync --extra jax-cuda`` (CUDA box). Modules import lazily so the Python
self-play path never needs jax installed.
"""

import os

# XLA preallocates 75% of VRAM by default, which starves the torch side of a
# mixed run (training step + CUDA eval workers -> cudaErrorMemoryAllocation,
# observed in the first A/B attempt). Allocate on demand instead; setdefault so
# an explicit env var still wins. This package is the import gateway for every
# jax entry point in the repo, so this runs before the first ``import jax``.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
