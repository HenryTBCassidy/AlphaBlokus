"""Shared fixtures for the JAX spike tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pieces import default_pieces_path

# tests/games/blokusduo/conftest.py -> repo root is three levels above tests/
TESTS_ROOT = Path(__file__).resolve().parents[2]

#: The 5,000-position stratified parity cache (see tests/fixtures/blokus_positions.py).
DEV_CACHE_PATH = TESTS_ROOT / "fixtures" / "blokus_duo_positions" / "dev_5000.npz"


@pytest.fixture(scope="module")
def blokus_game_module() -> BlokusDuoGame:
    """Module-scoped game instance — pieces.json parsing isn't free."""
    return BlokusDuoGame(pieces_config_path=default_pieces_path())
