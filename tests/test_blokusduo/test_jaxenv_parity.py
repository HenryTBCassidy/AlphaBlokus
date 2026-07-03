"""J3: JAX legal-mask parity with the Python engine on the dev-5000 cache.

Three-way oracle, mirroring ``tests/test_blokusduo/test_movegen_equivalence.py``:

- all 5,000 positions vs the F2/numba generator (the production fast path),
- a stratified 500-position subsample vs the reference array generator (the
  ground-truth slow path).

Exact equality everywhere — one disagreeing action id anywhere is a failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_blokusduo.conftest import DEV_CACHE_PATH

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.game import BlokusDuoGame

jax = pytest.importorskip("jax")

from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board  # noqa: E402
from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels  # noqa: E402
from alphablokus.games.blokusduo.jax.tables import build_jax_tables  # noqa: E402

# Chunk size for the batched jax mask call — bounds the (chunk, 17837) int32
# intermediates without changing results.
CHUNK = 512

# Every Nth position also gets checked against the slow reference generator.
REFERENCE_SUBSAMPLE_STRIDE = 10


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_legal_mask_parity_dev_cache(blokus_game_module: BlokusDuoGame) -> None:
    from alphablokus.games.blokusduo.movegen.runtime import get_default_generator
    from tests.fixtures.blokus_positions import iter_cached_positions

    game = blokus_game_module
    f2 = get_default_generator()
    kernels = make_kernels(build_jax_tables(game))

    boards, players, f2_masks, reference_ids = [], [], [], {}
    for index, (board, player, _sequence) in enumerate(iter_cached_positions(game, DEV_CACHE_PATH)):
        boards.append(board)
        players.append(player)
        f2_masks.append(f2.valid_move_mask(game, board, player).astype(np.bool_))
        if index % REFERENCE_SUBSAMPLE_STRIDE == 0:
            reference_ids[index] = np.flatnonzero(game.valid_move_masking(board, player) > 0)

    rows = [numpy_state_from_board(board, player) for board, player in zip(boards, players, strict=True)]
    mismatches: list[int] = []
    jax_masks = np.empty((len(rows), kernels.action_size), dtype=np.bool_)
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start:start + CHUNK]
        batch = GameState(*(np.stack([row[field] for row in chunk]) for field in range(4)))
        jax_masks[start:start + len(chunk)] = np.asarray(kernels.legal_mask_batch(batch))

    for index in range(len(rows)):
        if not np.array_equal(jax_masks[index], f2_masks[index]):
            mismatches.append(index)

    assert not mismatches, (
        f"JAX mask disagrees with F2 on {len(mismatches)}/{len(rows)} positions; "
        f"first few: {mismatches[:5]} "
        f"(diff at {mismatches[0]}: jax-only="
        f"{np.flatnonzero(jax_masks[mismatches[0]] & ~f2_masks[mismatches[0]])[:10]}, f2-only="
        f"{np.flatnonzero(~jax_masks[mismatches[0]] & f2_masks[mismatches[0]])[:10]})"
        if mismatches else ""
    )

    for index, expected_ids in reference_ids.items():
        np.testing.assert_array_equal(
            np.flatnonzero(jax_masks[index]), expected_ids,
            err_msg=f"JAX mask disagrees with reference generator at position {index}",
        )
