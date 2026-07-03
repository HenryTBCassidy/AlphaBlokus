"""J4: JAX step / game-end / scoring parity by replaying the dev cache.

Every cached action sequence is replayed move-by-move through both engines.
After every ply the signed placement boards, inventories, last-piece records
and player-to-move must match bit-for-bit; at the final position of every
sequence the game-ended value must match for both player perspectives.

Per-ply ``get_game_ended`` comparison is restricted to a stride of sequences —
the Python side is the expensive part (two has-any-move sweeps per call), and
the terminal/endgame strata are where the semantics live.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.games.blokusduo.conftest import DEV_CACHE_PATH

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.game import BlokusDuoGame

jax = pytest.importorskip("jax")

from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board  # noqa: E402
from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels  # noqa: E402
from alphablokus.games.blokusduo.jax.tables import build_jax_tables  # noqa: E402

# Every Nth sequence also compares get_game_ended after every single ply.
PER_PLY_GAME_ENDED_STRIDE = 25


def _assert_states_match(
    state: GameState, board, player: int, sequence_index: int, ply: int,
) -> None:
    ppb, remaining, last_piece, current_player = numpy_state_from_board(board, player)
    np.testing.assert_array_equal(
        np.asarray(state.ppb), ppb,
        err_msg=f"ppb mismatch at sequence {sequence_index} ply {ply}",
    )
    np.testing.assert_array_equal(
        np.asarray(state.remaining), remaining,
        err_msg=f"inventory mismatch at sequence {sequence_index} ply {ply}",
    )
    np.testing.assert_array_equal(
        np.asarray(state.last_piece), last_piece,
        err_msg=f"last-piece mismatch at sequence {sequence_index} ply {ply}",
    )
    assert int(state.current_player) == player, f"player mismatch at sequence {sequence_index} ply {ply}"


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_step_and_game_ended_replay_parity(blokus_game_module: BlokusDuoGame) -> None:
    from alphablokus.testing.positions import PAD_ACTION, load_cache

    game = blokus_game_module
    game.enable_optimised_movegen()  # fast has-any-move path for get_game_ended
    kernels = make_kernels(build_jax_tables(game))

    actions_array, n_moves_array = load_cache(DEV_CACHE_PATH)
    for sequence_index in range(len(n_moves_array)):
        sequence = actions_array[sequence_index, : int(n_moves_array[sequence_index])]
        board = game.initialise_board()
        player = 1
        state = kernels.initial_state()
        per_ply_ended = sequence_index % PER_PLY_GAME_ENDED_STRIDE == 0
        for ply, action_id in enumerate(sequence):
            action_id = int(action_id)
            assert action_id != PAD_ACTION
            board, player = game.get_next_state(board, player, action_id)
            state = kernels.step(state, action_id)
            _assert_states_match(state, board, player, sequence_index, ply)
            if per_ply_ended:
                for perspective in (1, -1):
                    expected = float(game.get_game_ended(board, perspective))
                    actual = float(kernels.game_result(state, perspective))
                    assert actual == pytest.approx(expected), (
                        f"game_ended mismatch at sequence {sequence_index} ply {ply} "
                        f"perspective {perspective}: python={expected} jax={actual}"
                    )
        # Terminal-value check at the final position of every sequence.
        for perspective in (1, -1):
            expected = float(game.get_game_ended(board, perspective))
            actual = float(kernels.game_result(state, perspective))
            assert actual == pytest.approx(expected), (
                f"final game_ended mismatch at sequence {sequence_index} "
                f"perspective {perspective}: python={expected} jax={actual}"
            )
