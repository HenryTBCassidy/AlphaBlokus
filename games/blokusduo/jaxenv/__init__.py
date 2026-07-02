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
