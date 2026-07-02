"""J4: scoring branches random replay can't reach (all-pieces-placed bonuses).

Random cache games never empty an inventory, so the +15 / +5-monomino-last
branches are pinned here on synthetic states, against the documented rules in
``BlokusDuoGame._calculate_score``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from games.blokusduo.game import BlokusDuoGame

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from experiments.jax_spike.kernels import make_kernels  # noqa: E402
from experiments.jax_spike.tables import build_jax_tables  # noqa: E402


@pytest.fixture(scope="module")
def kernels(blokus_game_module: BlokusDuoGame):
    return make_kernels(build_jax_tables(blokus_game_module))


def _with(state, **updates):
    return state._replace(**{key: jnp.asarray(value) for key, value in updates.items()})


def test_all_placed_bonus(kernels) -> None:
    state = kernels.initial_state()
    empty = np.zeros_like(np.asarray(state.remaining))
    state = _with(state, remaining=empty, last_piece=np.array([5, 5], dtype=np.int8))
    assert int(kernels.score(state, jnp.int32(0))) == 15


def test_monomino_last_bonus(kernels) -> None:
    state = kernels.initial_state()
    empty = np.zeros_like(np.asarray(state.remaining))
    state = _with(state, remaining=empty, last_piece=np.array([1, 5], dtype=np.int8))
    assert int(kernels.score(state, jnp.int32(0))) == 20
    assert int(kernels.score(state, jnp.int32(1))) == 15


def test_remaining_penalty(kernels, blokus_game_module: BlokusDuoGame) -> None:
    """Fresh inventory scores -(total squares) = -89, matching the engine."""
    board = blokus_game_module.initialise_board()
    expected = blokus_game_module._calculate_score(board, 1)
    assert expected == -89
    state = kernels.initial_state()
    assert int(kernels.score(state, jnp.int32(0))) == expected
    assert int(kernels.score(state, jnp.int32(1))) == expected


def test_monomino_last_wins_tiebreak(kernels) -> None:
    """All-placed vs all-placed: monomino-last (20) beats plain 15."""
    state = kernels.initial_state()
    empty = np.zeros_like(np.asarray(state.remaining))
    state = _with(state, remaining=empty, last_piece=np.array([1, 5], dtype=np.int8))
    assert float(kernels.game_result(state, jnp.int8(1))) == 1.0
    assert float(kernels.game_result(state, jnp.int8(-1))) == -1.0
