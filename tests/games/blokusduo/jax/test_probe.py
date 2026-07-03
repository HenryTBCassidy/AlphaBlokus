"""J1 probe: jax imports, runs a jitted op, and coexists with torch.

Guarded by ``importorskip`` so the suite stays green without the ``jax`` extra.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")


def test_jax_jit_basic() -> None:
    """A jitted op compiles and produces the expected value."""
    import jax.numpy as jnp

    @jax.jit
    def f(x: jnp.ndarray) -> jnp.ndarray:
        return (x * 2).sum()

    assert int(f(jnp.arange(10))) == 90


def test_jax_torch_coexistence() -> None:
    """Importing torch alongside jax in one process does not blow up.

    The spike benchmarks load the Python engine (torch-adjacent imports) and
    jax in the same process, so this must hold on both the Mac and the box.
    """
    import jax.numpy as jnp
    import torch

    assert torch.tensor([1.0]).item() == 1.0
    assert int(jnp.ones(3).sum()) == 3
