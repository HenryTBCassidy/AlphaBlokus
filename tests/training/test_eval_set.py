"""Tests for the held-out eval set: provenance, rebuild cadence, reproducibility.

The reproducibility test is a regression guard. The eval set used to be sampled
with a seeded numpy generator from a list that ``ReplayBuffer`` had shuffled with
Python's *global* ``random`` module — which the Coach never seeds. The indices
reproduced; the positions they indexed did not. That made every per-generation
diagnostic non-reproducible at a fixed seed while looking deterministic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from alphablokus.config import RunConfig
from alphablokus.storage.sparse_policy import sparsify
from alphablokus.training.eval_set import METADATA_FILENAME, build_or_load_eval_set, should_rebuild
from alphablokus.training.replay_buffer import ReplayBuffer


@pytest.fixture
def run_config_factory(mcts_config, net_config):
    """Build a tictactoe RunConfig rooted at an arbitrary directory."""

    def make(root, *, seed: int) -> RunConfig:
        return RunConfig(
            game="tictactoe",
            run_name="eval_set_test",
            num_generations=1,
            num_eps=2,
            temp_threshold=5,
            update_threshold=0.55,
            num_arena_matches=2,
            # Comfortably above N_FIXTURE_GAMES so the buffer never evicts and the
            # tests compare against every game they put in.
            replay_buffer_games=200,
            root_directory=root,
            load_model=False,
            mcts_config=mcts_config,
            net_config=net_config,
            seed=seed,
        )

    return make


#: Games in the synthetic fixture. Comfortably more than the eval set needs, so
#: ``MAX_EVAL_GAME_FRACTION`` does not bind in the tests that are not about it.
N_FIXTURE_GAMES = 60


@pytest.fixture
def games(ttt_game) -> list[list]:
    """Synthetic games of four positions each, distinguishable by board."""
    rng = np.random.default_rng(0)
    action_size = ttt_game.get_action_size()
    built = []
    for game_index in range(N_FIXTURE_GAMES):
        positions = []
        for ply in range(4):
            board = np.zeros((3, 3), dtype=np.int8)
            board[ply // 3, ply % 3] = game_index + 1
            policy = np.zeros(action_size, dtype=np.float32)
            policy[rng.integers(action_size)] = 1.0
            positions.append((board, sparsify(policy), float((-1) ** game_index)))
        built.append(positions)
    return built


# --- rebuild cadence ------------------------------------------------------


@pytest.mark.parametrize(
    ("generation", "every", "expected"),
    [
        (1, 0, False),  # disabled
        (7, 0, False),
        (1, 5, False),  # gen 1 builds via the "nothing on disk" path, not a rebuild
        (2, 5, False),
        (6, 5, True),
        (11, 5, True),
        (7, 5, False),
        (4, 3, True),
    ],
)
def test_should_rebuild_cadence(generation: int, every: int, expected: bool) -> None:
    assert should_rebuild(generation, every) is expected


# --- provenance -----------------------------------------------------------


def test_records_source_game_for_every_position(tmp_path, ttt_game, run_config_factory, games) -> None:
    config = run_config_factory(tmp_path, seed=7)

    eval_set = build_or_load_eval_set(config, ttt_game, None, games, size=20, generation=1)

    assert eval_set is not None
    assert eval_set.source_game_ids is not None
    assert len(eval_set.source_game_ids) == len(eval_set)
    # Ids must index the games actually passed in.
    assert set(eval_set.source_game_ids.tolist()) <= set(range(len(games)))
    assert eval_set.n_source_games == len(set(eval_set.source_game_ids.tolist()))
    assert eval_set.built_at_generation == 1


def test_writes_game_ids_and_metadata_to_disk(tmp_path, ttt_game, run_config_factory, games) -> None:
    config = run_config_factory(tmp_path, seed=3)

    build_or_load_eval_set(config, ttt_game, None, games, size=16, generation=4)

    eval_dir = config.eval_set_directory
    assert (eval_dir / "source_game_ids.npy").exists()
    metadata = json.loads((eval_dir / METADATA_FILENAME).read_text())
    assert metadata["built_at_generation"] == 4
    assert metadata["n_positions"] == 16
    assert metadata["n_source_games"] >= 1
    assert metadata["targets_kind"] == "selfplay_v1"


def test_reload_restores_game_ids_and_vintage(tmp_path, ttt_game, run_config_factory, games) -> None:
    config = run_config_factory(tmp_path, seed=3)
    first = build_or_load_eval_set(config, ttt_game, None, games, size=16, generation=4)

    # A later generation with the cadence disabled must reuse what is on disk.
    second = build_or_load_eval_set(config, ttt_game, None, games, size=16, generation=9)

    assert first is not None and second is not None
    assert second.built_at_generation == 4
    np.testing.assert_array_equal(first.source_game_ids, second.source_game_ids)
    np.testing.assert_array_equal(first.compact_boards, second.compact_boards)


def test_force_rebuild_resamples(tmp_path, ttt_game, run_config_factory, games) -> None:
    config = run_config_factory(tmp_path, seed=3)
    first = build_or_load_eval_set(config, ttt_game, None, games, size=16, generation=1)
    rebuilt = build_or_load_eval_set(config, ttt_game, None, games, size=16, generation=6, force_rebuild=True)

    assert first is not None and rebuilt is not None
    assert rebuilt.built_at_generation == 6
    # A different vintage draws a different sample (the seed is mixed with the
    # generation), so the set genuinely refreshes rather than re-picking the same rows.
    assert not np.array_equal(first.compact_boards, rebuilt.compact_boards)


def test_no_games_returns_none(tmp_path, ttt_game, run_config_factory) -> None:
    config = run_config_factory(tmp_path, seed=1)

    assert build_or_load_eval_set(config, ttt_game, None, [], size=8, generation=1) is None


# --- reproducibility (A11 regression) -------------------------------------


def test_same_seed_builds_a_byte_identical_eval_set(tmp_path, ttt_game, run_config_factory, games) -> None:
    """Two builds at the same seed must select the same positions.

    Regression guard: this failed before the unseeded ``random.shuffle`` was
    removed from the replay buffer, because the seeded index draw was applied to
    a differently-ordered list each time.
    """
    boards = []
    ids = []
    for run in range(2):
        config = run_config_factory(tmp_path / f"run{run}", seed=42)
        buffer = ReplayBuffer(config, ttt_game)
        buffer.add_generation([list(game) for game in games])
        eval_set = build_or_load_eval_set(config, ttt_game, None, buffer.games, size=20, generation=1)
        assert eval_set is not None
        boards.append(np.load(config.eval_set_directory / "compact_boards.npy"))
        ids.append(np.load(config.eval_set_directory / "source_game_ids.npy"))

    np.testing.assert_array_equal(boards[0], boards[1])
    np.testing.assert_array_equal(ids[0], ids[1])


# --- actually held out (A12 regression) -----------------------------------


def test_eval_positions_are_absent_from_the_training_flatten(tmp_path, ttt_game, run_config_factory, games) -> None:
    """No eval-set position may appear in what the trainer sees.

    Regression guard for the defect that made every "held-out" per-epoch
    diagnostic in-sample: the eval set was sampled from the training buffer and
    then trained on for as many generations as the positions stayed in the buffer.
    """
    config = run_config_factory(tmp_path, seed=5)
    buffer = ReplayBuffer(config, ttt_game)
    buffer.add_generation([list(game) for game in games])

    eval_set = build_or_load_eval_set(config, ttt_game, None, buffer.games, size=12, generation=1)
    assert eval_set is not None
    buffer.exclude_games(set(eval_set.source_fingerprints))

    training = {example[0].tobytes() for example in buffer.flat_examples()}
    for board in eval_set.compact_boards:
        assert board.tobytes() not in training, "an eval-set position is still in the training data"


def test_whole_source_games_are_withheld_not_just_the_sampled_positions(
    tmp_path, ttt_game, run_config_factory, games
) -> None:
    """Siblings of an eval position must be withheld too.

    Every position in a game carries the same outcome label and symmetry
    augmentation duplicates each one, so excluding only the sampled positions
    would leave the answer in the training set.
    """
    config = run_config_factory(tmp_path, seed=5)
    buffer = ReplayBuffer(config, ttt_game)
    buffer.add_generation([list(game) for game in games])

    eval_set = build_or_load_eval_set(config, ttt_game, None, buffer.games, size=12, generation=1)
    assert eval_set is not None
    buffer.exclude_games(set(eval_set.source_fingerprints))

    training = {example[0].tobytes() for example in buffer.flat_examples()}
    source_games = {int(gid) for gid in eval_set.source_game_ids}
    siblings = {example[0].tobytes() for gid in source_games for example in games[gid]}

    assert siblings, "expected the eval set to have source games with siblings"
    assert not (siblings & training), "a sibling position from an eval-set game leaked into training"
    # And the withholding is whole games, so the counts line up.
    assert buffer.held_out_game_count() == len(source_games)
    assert len(buffer.flat_examples()) == sum(len(g) for i, g in enumerate(games) if i not in source_games)


def test_positions_per_source_game_is_capped(tmp_path, ttt_game, run_config_factory, games) -> None:
    """Spreading the set over more games is what buys interval width."""
    from alphablokus.training.eval_set import MAX_EVAL_POSITIONS_PER_GAME

    config = run_config_factory(tmp_path, seed=11)
    eval_set = build_or_load_eval_set(config, ttt_game, None, games, size=12, generation=1)

    assert eval_set is not None
    counts = np.bincount(eval_set.source_game_ids)
    assert counts.max() <= MAX_EVAL_POSITIONS_PER_GAME
    assert eval_set.n_source_games >= len(eval_set) // MAX_EVAL_POSITIONS_PER_GAME


def test_eval_set_never_claims_the_whole_buffer(tmp_path, ttt_game, run_config_factory, games) -> None:
    """The holdout is capped, so training can never be starved by the eval set.

    Withholding whole games is what makes the set held out, but an uncapped
    request against a small buffer would withhold everything and leave nothing to
    train on.
    """
    from alphablokus.training.eval_set import MAX_EVAL_GAME_FRACTION

    config = run_config_factory(tmp_path, seed=5)
    buffer = ReplayBuffer(config, ttt_game)
    buffer.add_generation([list(game) for game in games])

    # Ask for far more positions than the cap allows.
    eval_set = build_or_load_eval_set(config, ttt_game, None, buffer.games, size=10_000, generation=1)

    assert eval_set is not None
    max_games = max(1, int(len(games) * MAX_EVAL_GAME_FRACTION))
    assert eval_set.n_source_games <= max_games
    buffer.exclude_games(set(eval_set.source_fingerprints))
    assert buffer.flat_examples(), "training data must remain after the eval-set holdout"


def test_tiny_buffer_still_leaves_training_data(tmp_path, ttt_game, run_config_factory, games) -> None:
    """The degenerate case: a two-game buffer must still train on something."""
    config = run_config_factory(tmp_path, seed=5)
    buffer = ReplayBuffer(config, ttt_game)
    buffer.add_generation([list(game) for game in games[:2]])

    eval_set = build_or_load_eval_set(config, ttt_game, None, buffer.games, size=200, generation=1)

    assert eval_set is not None
    buffer.exclude_games(set(eval_set.source_fingerprints))
    assert buffer.flat_examples()


def test_fingerprints_round_trip_through_disk(tmp_path, ttt_game, run_config_factory, games) -> None:
    """Exclusion must survive a resume, where buffer indices are meaningless."""
    config = run_config_factory(tmp_path, seed=5)
    first = build_or_load_eval_set(config, ttt_game, None, games, size=12, generation=1)
    assert first is not None

    reloaded = build_or_load_eval_set(config, ttt_game, None, games, size=12, generation=3)

    assert reloaded is not None
    assert reloaded.source_fingerprints == first.source_fingerprints
    assert reloaded.source_fingerprints != ()


def test_fingerprint_identifies_a_game_by_content(ttt_game, games) -> None:
    """Content-hashed, so it survives reordering and buffer rebuilds."""
    from alphablokus.training.replay_buffer import game_fingerprint

    assert game_fingerprint(games[0]) == game_fingerprint(list(games[0]))
    assert game_fingerprint(games[0]) != game_fingerprint(games[1])


def test_rebuild_releases_the_previous_holdout(tmp_path, ttt_game, run_config_factory, games) -> None:
    """A rebuilt eval set returns the old source games to training."""
    config = run_config_factory(tmp_path, seed=5)
    buffer = ReplayBuffer(config, ttt_game)
    buffer.add_generation([list(game) for game in games])

    first = build_or_load_eval_set(config, ttt_game, None, buffer.games, size=12, generation=1)
    assert first is not None
    buffer.exclude_games(set(first.source_fingerprints))
    rebuilt = build_or_load_eval_set(config, ttt_game, None, buffer.games, size=12, generation=6, force_rebuild=True)
    assert rebuilt is not None
    buffer.exclude_games(set(rebuilt.source_fingerprints))

    assert buffer.excluded_fingerprints == frozenset(rebuilt.source_fingerprints)
    released = set(first.source_fingerprints) - set(rebuilt.source_fingerprints)
    assert released, "expected a different draw on rebuild"
    assert not (released & buffer.excluded_fingerprints)


def test_coach_withholds_the_eval_games_from_training(tmp_path, ttt_game, run_config_factory, games) -> None:
    """End-to-end wiring: the Coach's own eval-set step must arm the exclusion.

    Guards the ordering as well as the mechanism — the eval set has to be built
    before the buffer is flattened, or the held-out positions land in training
    regardless of what the buffer supports.
    """
    from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
    from alphablokus.training.coach import Coach

    config = run_config_factory(tmp_path, seed=5)
    coach = Coach(ttt_game, NNetWrapper(ttt_game, config), config)
    coach.replay_buffer.add_generation([list(game) for game in games])

    coach._ensure_eval_set(1)

    eval_set = coach._eval_set
    assert eval_set is not None
    assert eval_set.source_fingerprints
    assert coach.replay_buffer.excluded_fingerprints == frozenset(eval_set.source_fingerprints)
    training = {example[0].tobytes() for example in coach.replay_buffer.flat_examples()}
    for board in eval_set.compact_boards:
        assert board.tobytes() not in training


def test_an_eval_set_without_fingerprints_is_rebuilt(tmp_path, ttt_game, run_config_factory, games) -> None:
    """A pre-fix eval set on disk cannot be held out, so it must not be trusted."""
    from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper
    from alphablokus.training.coach import Coach
    from alphablokus.training.eval_set import FINGERPRINTS_FILENAME

    config = run_config_factory(tmp_path, seed=5)
    build_or_load_eval_set(config, ttt_game, None, games, size=12, generation=1)
    # Simulate an eval set built before fingerprints existed.
    (config.eval_set_directory / FINGERPRINTS_FILENAME).unlink()

    coach = Coach(ttt_game, NNetWrapper(ttt_game, config), config)
    coach.replay_buffer.add_generation([list(game) for game in games])
    coach._ensure_eval_set(1)

    assert coach._eval_set is not None
    assert coach._eval_set.source_fingerprints, "a fingerprint-less eval set must be rebuilt, not reused"
    assert coach.replay_buffer.excluded_fingerprints


def test_explicit_seed_zero_is_not_treated_as_unseeded(tmp_path, ttt_game, run_config_factory, games) -> None:
    """A13: ``config.seed or 0`` conflated seed=0 with seed=None.

    A three-seed sweep including 0 would silently duplicate an arm, so seed 0 must
    behave like any other seed and differ from a different seed.
    """
    zero = build_or_load_eval_set(
        run_config_factory(tmp_path / "zero", seed=0), ttt_game, None, games, size=12, generation=1
    )
    one = build_or_load_eval_set(
        run_config_factory(tmp_path / "one", seed=1), ttt_game, None, games, size=12, generation=1
    )

    assert zero is not None and one is not None
    assert not np.array_equal(zero.source_game_ids, one.source_game_ids)


def test_no_exclusion_keeps_the_whole_buffer(tmp_path, ttt_game, run_config_factory, games) -> None:
    config = run_config_factory(tmp_path, seed=5)
    buffer = ReplayBuffer(config, ttt_game)
    buffer.add_generation([list(game) for game in games])

    assert len(buffer.flat_examples()) == sum(len(g) for g in games)
    assert buffer.held_out_game_count() == 0


def test_replay_buffer_flattening_is_deterministic(tmp_path, ttt_game, run_config_factory, games) -> None:
    """The flattened buffer must be in a stable order, not a globally-random one."""
    config = run_config_factory(tmp_path, seed=42)
    first = ReplayBuffer(config, ttt_game)
    first.add_generation([list(game) for game in games])
    second = ReplayBuffer(config, ttt_game)
    second.add_generation([list(game) for game in games])

    left = [example[0].tobytes() for example in first.flat_examples()]
    right = [example[0].tobytes() for example in second.flat_examples()]

    assert left == right
    # And it is genuinely the concatenation of the games, in buffer order.
    expected = [example[0].tobytes() for game in games for example in game]
    assert left == expected
