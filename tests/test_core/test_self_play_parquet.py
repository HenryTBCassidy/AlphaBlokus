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


def test_save_writes_board_kind_marker(store: SelfPlayStore, test_config: RunConfig):
    """save should stamp the compact-board schema marker."""
    store.save(_make_dummy_examples(), generation=0)

    filepath = test_config.self_play_history_directory / "self_play_0.parquet"
    table = pq.read_table(filepath)
    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    assert metadata["board_kind"] == SelfPlayStore.BOARD_KIND


def test_load_refuses_legacy_dense_file(store: SelfPlayStore, test_config: RunConfig):
    """A file without the board_kind marker (legacy dense) must be refused."""
    import pandas as pd
    import pyarrow as pa

    store._directory.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"board": [b""], "policy": [b""], "value": [0.0]})
    table = pa.Table.from_pandas(df).replace_schema_metadata({
        b"board_shape": b"44,14,14", b"board_dtype": b"float32",
        b"policy_size": b"10", b"policy_dtype": b"float64",
    })
    pq.write_table(table, store._directory / "self_play_0.parquet")

    with pytest.raises(ValueError, match="board_kind"):
        store.load(generation=0)


def test_compact_board_roundtrip_reencodes(blokus_game, test_config: RunConfig):
    """A real compact Blokus board survives save→load and re-encodes exactly."""
    store = SelfPlayStore(test_config.self_play_history_directory)
    board = blokus_game.initialise_board()
    board, _ = blokus_game.get_next_state(
        board, 1, int(np.where(blokus_game.valid_move_masking(board, 1))[0][0]),
    )
    compact = board.to_compact()
    policy = np.zeros(blokus_game.get_action_size(), dtype=np.float64)
    policy[0] = 1.0
    store.save(deque([(compact, policy, 1.0)]), generation=0)

    loaded = store.load(generation=0)
    assert loaded is not None
    loaded_compact = loaded[0][0]
    assert np.array_equal(loaded_compact, compact)
    assert np.array_equal(
        blokus_game.encode_compact(loaded_compact), board.as_multi_channel(1),
    )


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


def test_load_games_splits_by_game_sizes(store: SelfPlayStore):
    """load_games should restore per-game boundaries from game_sizes metadata."""
    # One generation: 3 games of sizes 2, 1, 3 (flat in game order).
    flat = _make_dummy_examples(6)
    store.save(flat, generation=0, game_sizes=[2, 1, 3])

    games = store.load_games(generation=0)
    assert games is not None
    assert [len(g) for g in games] == [2, 1, 3]


def test_load_games_without_sizes_returns_one_game(store: SelfPlayStore):
    """A file saved without game_sizes is treated as a single game."""
    store.save(_make_dummy_examples(4), generation=0)
    games = store.load_games(generation=0)
    assert games is not None
    assert len(games) == 1
    assert len(games[0]) == 4


def test_load_recent_games_keeps_newest_n(store: SelfPlayStore):
    """load_recent_games should return the newest ``num_games`` games across files."""
    # Three generation files, 2 games each (sizes 1 each) → 6 games total.
    for gen in range(3):
        store.save(_make_dummy_examples(2), generation=gen, game_sizes=[1, 1])

    buffer = store.load_recent_games(last_file_index=2, num_games=3)
    assert len(buffer) == 3  # capped at num_games, newest kept


# ---------------------------------------------------------------------------
# Coach integration test (verifies thin wrappers still work end-to-end)
# ---------------------------------------------------------------------------


def test_coach_save_load_roundtrip(ttt_game, test_config: RunConfig):
    """Coach thin wrappers should delegate to SelfPlayStore correctly."""
    from core.coach import Coach
    from games.tictactoe.neuralnets.wrapper import NNetWrapper

    nnet = NNetWrapper(ttt_game, test_config)
    coach = Coach(ttt_game, nnet, test_config)

    # One generation's fresh games: a single game of 3 positions.
    original = list(_make_dummy_examples(3))
    coach._fresh_games_this_gen = [original]
    coach.save_self_play_history(file_index=0)

    coach.replay_buffer.clear()
    coach.load_self_play_history(up_to_generation=0)

    assert len(coach.replay_buffer) == 1  # one game
    loaded = coach.replay_buffer[0]
    assert len(loaded) == 3

    for (orig_b, orig_p, orig_v), (load_b, load_p, load_v) in zip(original, loaded):
        np.testing.assert_array_almost_equal(load_b, orig_b)
        np.testing.assert_array_almost_equal(load_p, orig_p)
        assert pytest.approx(load_v) == orig_v
