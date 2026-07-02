"""JAX de-risk spike: GPU-native Blokus Duo legality, step, and throughput ceiling.

Plan: ``docs/plans/jax-spike.md``. This package proves (or kills) the hypothesis
that Blokus Duo rules can run as fixed-shape JAX array ops, bit-identical to the
Python engine and fast enough batched to justify a full JAX/mctx self-play rewrite.

Spike-scoped, not throwaway-quality: the kernels here are the seed of the real
rewrite if the go/no-go decision lands on "go".

Requires the ``jax`` extra: ``uv sync --extra jax`` (Mac CPU) or
``uv sync --extra jax-cuda`` (CUDA box).
"""
