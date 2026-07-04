"""XLA VRAM-cap plumbing (config > default, explicit env var wins).

Deliberately jax-free: only the package ``__init__`` (which imports ``os``
alone) is under test, so this runs in the base CI job too.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import alphablokus.games.blokusduo.jax as jax_pkg
from alphablokus.config import JaxSelfPlayConfig

if TYPE_CHECKING:
    import pytest

MEM_FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"


def test_default_fraction_matches_package_default() -> None:
    assert JaxSelfPlayConfig().xla_mem_fraction == jax_pkg.DEFAULT_XLA_MEM_FRACTION


def test_config_value_overrides_package_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate "no explicit env var at import time" — the package's own 0.4
    # setdefault must not block a later config-driven override.
    monkeypatch.setattr(jax_pkg, "_env_set_mem_fraction", False)
    monkeypatch.setenv(MEM_FRACTION_ENV, str(jax_pkg.DEFAULT_XLA_MEM_FRACTION))
    jax_pkg.configure_xla_mem_fraction(0.9)
    assert os.environ[MEM_FRACTION_ENV] == "0.9"


def test_explicit_env_var_wins_over_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax_pkg, "_env_set_mem_fraction", True)
    monkeypatch.setenv(MEM_FRACTION_ENV, "0.25")
    jax_pkg.configure_xla_mem_fraction(0.9)
    assert os.environ[MEM_FRACTION_ENV] == "0.25"
