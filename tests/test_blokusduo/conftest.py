"""Shared fixtures for the JAX spike tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from games.blokusduo.game import BlokusDuoGame

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: The 5,000-position stratified parity cache (see tests/fixtures/blokus_positions.py).
DEV_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "blokus_duo_positions" / "dev_5000.npz"


@pytest.fixture(scope="module")
def blokus_game_module() -> BlokusDuoGame:
    """Module-scoped game instance — pieces.json parsing isn't free."""
    return BlokusDuoGame(pieces_config_path=REPO_ROOT / "games" / "blokusduo" / "pieces.json")
