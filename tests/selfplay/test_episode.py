"""What a self-play position records — in particular *whose turn it was*.

The board a self-play example stores is **canonical** (side-to-move perspective),
so the absolute colour is not in the bytes. It used to be recoverable only by
inference from piece-count parity, and that inference breaks the moment either
player passes — which happens in the endgame, exactly where the value signal is
strongest. These tests pin the explicit field instead
(``docs/plans/selfplay-data-and-loop.md`` D1).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.evaluation.colour_value import COLOUR_UNKNOWN, infer_mover_colour
from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
from alphablokus.search.mcts import MCTS
from alphablokus.selfplay.episode import ProcessedExample, play_self_play_episode
from alphablokus.storage.selfplay_store import SelfPlayStore
from alphablokus.storage.sparse_policy import sparsify

if TYPE_CHECKING:
    from alphablokus.config import MCTSConfig, RunConfig
    from alphablokus.games.tictactoe.game import TicTacToeGame


def _episode(ttt_game: TicTacToeGame, test_config: RunConfig, mcts_config: MCTSConfig) -> list[ProcessedExample]:
    """One real self-play episode driven by a tiny untrained net."""
    np.random.seed(0)
    nnet = NNetWrapper(ttt_game, test_config)
    mcts = MCTS(ttt_game, nnet, mcts_config)
    return play_self_play_episode(ttt_game, mcts, temp_threshold=test_config.sampling_temp_threshold)


def test_every_stored_position_records_the_side_that_was_to_move(
    ttt_game: TicTacToeGame,
    test_config: RunConfig,
    mcts_config: MCTSConfig,
) -> None:
    """Players alternate ply by ply, in symmetry-sized blocks — White first.

    TicTacToe has no passing, so the true mover sequence is fully determined: the
    episode visits plies 1, 2, 3, … with White (+1) to move on the odd ones, and
    stores one example per symmetry per ply. A field that was dropped, held
    constant, or offset by one ply fails here.
    """
    examples = _episode(ttt_game, test_config, mcts_config)
    board = ttt_game.initialise_board()
    per_ply = len(ttt_game.get_symmetries(board, np.full(ttt_game.get_action_size(), 1.0 / ttt_game.get_action_size())))

    assert examples
    assert len(examples) % per_ply == 0
    expected = [1 if (index // per_ply) % 2 == 0 else -1 for index in range(len(examples))]
    assert [example.player for example in examples] == expected


def test_the_stored_value_is_the_outcome_from_the_stored_player(
    ttt_game: TicTacToeGame,
    test_config: RunConfig,
    mcts_config: MCTSConfig,
) -> None:
    """``value`` is a function of ``player`` alone: one number per colour, negated.

    Every position in a game shares one outcome, signed by the side to move. So
    each colour must carry exactly one value and the two must be opposites — the
    invariant that ties the new field to the label it explains. A ``player`` that
    did not track the perspective the value was computed from breaks it.
    """
    examples = _episode(ttt_game, test_config, mcts_config)

    white = {example.value for example in examples if example.player == 1}
    black = {example.value for example in examples if example.player == -1}
    assert len(white) == 1
    assert len(black) == 1
    assert white.pop() == pytest.approx(-black.pop())


def test_the_side_to_move_survives_a_position_piece_parity_cannot_read(test_config: RunConfig) -> None:
    """A post-pass position round-trips its colour; the parity shortcut cannot.

    This is the reason D1 threads the real value rather than deriving it.
    ``infer_mover_colour`` reads the mover off the *difference* in piece counts,
    which only holds while both sides have moved on every turn. Here White is to
    move having placed two pieces against Black's one — the shape a single Black
    pass leaves behind — and the inference gives up. The stored field does not.
    """
    # Canonical compact board: positive ids belong to the side to move (White here).
    compact = np.zeros((14, 14), dtype=np.int8)
    compact[0, :2] = [1, 2]
    compact[13, 0] = -1
    assert infer_mover_colour(compact) == COLOUR_UNKNOWN

    policy = np.zeros(8, dtype=np.float32)
    policy[2] = 1.0
    store = SelfPlayStore(test_config.self_play_history_directory)
    store.save(
        deque([ProcessedExample(compact, sparsify(policy), -1.0, 1)]),
        generation=0,
        policy_size=8,
    )

    loaded = store.load(generation=0)
    assert loaded is not None
    assert loaded[0].player == 1


def test_a_file_written_without_the_side_to_move_is_refused(test_config: RunConfig) -> None:
    """A pre-D1 self-play file fails loudly instead of being loaded colour-blind.

    Guessing would be worse than refusing: the run would train and report
    colour-conditional diagnostics off invented labels.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = test_config.self_play_history_directory
    directory.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("board", pa.binary()),
            pa.field("policy_indices", pa.binary()),
            pa.field("policy_values", pa.binary()),
            pa.field("value", pa.float64()),
        ],
        metadata={
            b"board_kind": SelfPlayStore.BOARD_KIND.encode(),
            b"board_shape": b"3,3",
            b"board_dtype": b"int8",
            b"policy_kind": SelfPlayStore.POLICY_KIND.encode(),
            b"policy_size": b"10",
        },
    )
    table = pa.Table.from_pydict(
        {
            "board": [np.zeros((3, 3), dtype=np.int8).tobytes()],
            "policy_indices": [np.array([0], dtype=np.int32).tobytes()],
            "policy_values": [np.array([1.0], dtype=np.float32).tobytes()],
            "value": [1.0],
        },
        schema=schema,
    )
    pq.write_table(table, directory / "self_play_0.parquet")

    with pytest.raises(ValueError, match="player_kind"):
        SelfPlayStore(directory).load(generation=0)
