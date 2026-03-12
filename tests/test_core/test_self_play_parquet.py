from collections import deque

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.coach import Coach
from core.config import RunConfig
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper


@pytest.fixture
def coach(ttt_game: TicTacToeGame, test_config: RunConfig) -> Coach:
    """Coach with tiny config for self-play parquet tests."""
    nnet = NNetWrapper(ttt_game, test_config)
    return Coach(ttt_game, nnet, test_config)


def _make_dummy_examples(n: int = 5) -> deque:
    """Create dummy (board, policy, value) examples for testing."""
    examples: deque = deque()
    for _ in range(n):
        board = np.random.rand(3, 3).astype(np.float64)
        policy = np.random.dirichlet(np.ones(10)).astype(np.float64)
        value = np.random.choice([-1.0, 1.0])
        examples.append((board, policy, value))
    return examples


def test_save_creates_parquet_file(coach: Coach):
    """Saving self-play history should create a .parquet file."""
    coach.train_examples_history = [_make_dummy_examples()]
    coach.save_self_play_history(generation=0)

    filepath = coach.config.self_play_history_directory / "self_play_0.parquet"
    assert filepath.exists()


def test_save_load_roundtrip_values(coach: Coach):
    """Board arrays, policy vectors, and values should survive serialization."""
    original = _make_dummy_examples(3)
    coach.train_examples_history = [original]
    coach.save_self_play_history(generation=0)

    # Clear and reload
    coach.train_examples_history = []
    coach.load_self_play_history(up_to_generation=0)

    assert len(coach.train_examples_history) == 1
    loaded = coach.train_examples_history[0]
    assert len(loaded) == 3

    for (orig_b, orig_p, orig_v), (load_b, load_p, load_v) in zip(original, loaded):
        np.testing.assert_array_almost_equal(load_b, orig_b)
        np.testing.assert_array_almost_equal(load_p, orig_p)
        assert pytest.approx(load_v) == orig_v


def test_save_load_roundtrip_shapes(coach: Coach):
    """Reconstructed arrays should have correct shapes and dtypes."""
    coach.train_examples_history = [_make_dummy_examples()]
    coach.save_self_play_history(generation=0)

    coach.train_examples_history = []
    coach.load_self_play_history(up_to_generation=0)

    board, policy, value = coach.train_examples_history[0][0]
    assert board.shape == (3, 3)
    assert policy.shape == (10,)
    assert isinstance(value, float)


def test_metadata_contains_shapes(coach: Coach):
    """Parquet file metadata should contain board_shape, board_dtype, policy_size, policy_dtype."""
    coach.train_examples_history = [_make_dummy_examples()]
    coach.save_self_play_history(generation=0)

    filepath = coach.config.self_play_history_directory / "self_play_0.parquet"
    table = pq.read_table(filepath)
    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

    assert "board_shape" in metadata
    assert "board_dtype" in metadata
    assert "policy_size" in metadata
    assert "policy_dtype" in metadata

    assert metadata["board_shape"] == "3,3"
    assert metadata["policy_size"] == "10"


def test_load_respects_window(coach: Coach):
    """Loading with a window should only load recent generations."""
    # Save two generations
    coach.train_examples_history = [_make_dummy_examples(2)]
    coach.save_self_play_history(generation=0)

    coach.train_examples_history = [_make_dummy_examples(3)]
    coach.save_self_play_history(generation=1)

    # Load only gen 1 (window_size for gen 1 is 5, so both 0 and 1 are within window)
    coach.train_examples_history = []
    coach.load_self_play_history(up_to_generation=1)

    # Both generations should be loaded (window of 5 easily covers gen 0 and 1)
    total_examples = sum(len(e) for e in coach.train_examples_history)
    assert total_examples == 5  # 2 from gen 0 + 3 from gen 1


def test_load_missing_directory_does_not_crash(coach: Coach):
    """Loading from a non-existent directory should not crash and returns empty history."""
    coach.load_self_play_history(up_to_generation=0)
    assert coach.train_examples_history == []
