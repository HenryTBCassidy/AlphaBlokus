"""Tests for self-play parquet I/O via :class:`core.storage.SelfPlayStore`.

Most tests exercise the store directly (no Coach needed).
One integration test at the end verifies the Coach thin wrappers still work.
"""

from collections import deque

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.config import RunConfig
from core.storage import ProcessedExample, SelfPlayStore


def _make_dummy_examples(n: int = 5) -> deque[ProcessedExample]:
    """Create dummy (board, policy, value) examples for testing."""
    examples: deque[ProcessedExample] = deque()
    for _ in range(n):
        board = np.random.rand(3, 3).astype(np.float64)
        policy = np.random.dirichlet(np.ones(10)).astype(np.float64)
        value = np.random.choice([-1.0, 1.0])
        examples.append((board, policy, value))
    return examples


@pytest.fixture
def store(test_config: RunConfig) -> SelfPlayStore:
    """SelfPlayStore pointed at the test config's self-play directory."""
    return SelfPlayStore(test_config.self_play_history_directory)


# ---------------------------------------------------------------------------
# SelfPlayStore tests
# ---------------------------------------------------------------------------


def test_save_creates_parquet_file(store: SelfPlayStore, test_config: RunConfig):
    """save should create a .parquet file."""
    examples = _make_dummy_examples()
    store.save(examples, generation=0)

    filepath = test_config.self_play_history_directory / "self_play_0.parquet"
    assert filepath.exists()


def test_save_load_roundtrip_values(store: SelfPlayStore):
    """Board arrays, policy vectors, and values should survive serialisation."""
    original = _make_dummy_examples(3)
    store.save(original, generation=0)

    loaded = store.load(generation=0)
    assert loaded is not None
    assert len(loaded) == 3

    for (orig_b, orig_p, orig_v), (load_b, load_p, load_v) in zip(original, loaded):
        np.testing.assert_array_almost_equal(load_b, orig_b)
        np.testing.assert_array_almost_equal(load_p, orig_p)
        assert pytest.approx(load_v) == orig_v


def test_save_load_roundtrip_shapes(store: SelfPlayStore):
    """Reconstructed arrays should have correct shapes and dtypes."""
    store.save(_make_dummy_examples(), generation=0)

    loaded = store.load(generation=0)
    assert loaded is not None

    board, policy, value = loaded[0]
    assert board.shape == (3, 3)
    assert policy.shape == (10,)
    assert isinstance(value, float)


def test_metadata_contains_shapes(store: SelfPlayStore, test_config: RunConfig):
    """Parquet file metadata should contain board_shape, board_dtype, policy_size, policy_dtype."""
    store.save(_make_dummy_examples(), generation=0)

    filepath = test_config.self_play_history_directory / "self_play_0.parquet"
    table = pq.read_table(filepath)
    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

    assert "board_shape" in metadata
    assert "board_dtype" in metadata
    assert "policy_size" in metadata
    assert "policy_dtype" in metadata

    assert metadata["board_shape"] == "3,3"
    assert metadata["policy_size"] == "10"


def test_load_missing_file_returns_none(store: SelfPlayStore):
    """load should return None when the file doesn't exist."""
    result = store.load(generation=99)
    assert result is None


def test_load_window_missing_directory(tmp_path):
    """load_window should return [] when the directory doesn't exist."""
    store = SelfPlayStore(tmp_path / "does_not_exist")
    result = store.load_window(up_to_generation=5, window_size=5)
    assert result == []


def test_load_window(store: SelfPlayStore):
    """load_window should load the correct generations."""
    # Save four generations with different sizes
    store.save(_make_dummy_examples(2), generation=0)
    store.save(_make_dummy_examples(3), generation=1)
    store.save(_make_dummy_examples(4), generation=2)
    store.save(_make_dummy_examples(5), generation=3)

    # Window of 2 from generation 3 → start_gen = max(0, 3-2) = 1 → loads gens 1, 2, 3
    history = store.load_window(up_to_generation=3, window_size=2)
    assert len(history) == 3
    total_examples = sum(len(e) for e in history)
    assert total_examples == 12  # 3 from gen 1 + 4 from gen 2 + 5 from gen 3


def test_load_window_skips_missing(store: SelfPlayStore):
    """load_window should skip missing generations gracefully."""
    # Save gen 0 and gen 2, skip gen 1
    store.save(_make_dummy_examples(2), generation=0)
    store.save(_make_dummy_examples(4), generation=2)

    # Window of 5 from generation 2 → should find gens 0 and 2, skip missing 1
    history = store.load_window(up_to_generation=2, window_size=5)
    assert len(history) == 2
    total_examples = sum(len(e) for e in history)
    assert total_examples == 6  # 2 from gen 0 + 4 from gen 2


# ---------------------------------------------------------------------------
# Coach integration test (verifies thin wrappers still work end-to-end)
# ---------------------------------------------------------------------------


def test_coach_save_load_roundtrip(ttt_game, test_config: RunConfig):
    """Coach thin wrappers should delegate to SelfPlayStore correctly."""
    from core.coach import Coach
    from games.tictactoe.neuralnets.wrapper import NNetWrapper

    nnet = NNetWrapper(ttt_game, test_config)
    coach = Coach(ttt_game, nnet, test_config)

    original = _make_dummy_examples(3)
    coach.train_examples_history = [original]
    coach.save_self_play_history(generation=0)

    coach.train_examples_history = []
    coach.load_self_play_history(up_to_generation=0)

    assert len(coach.train_examples_history) == 1
    loaded = coach.train_examples_history[0]
    assert len(loaded) == 3

    for (orig_b, orig_p, orig_v), (load_b, load_p, load_v) in zip(original, loaded):
        np.testing.assert_array_almost_equal(load_b, orig_b)
        np.testing.assert_array_almost_equal(load_p, orig_p)
        assert pytest.approx(load_v) == orig_v
