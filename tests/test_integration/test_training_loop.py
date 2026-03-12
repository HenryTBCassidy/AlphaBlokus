import pytest

from core.coach import Coach
from core.config import RunConfig
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper


@pytest.mark.slow
def test_one_generation_tictactoe(ttt_game: TicTacToeGame, test_config: RunConfig):
    """Full 1-generation training loop: self-play → train → arena → metrics flush.

    Verifies:
    - The loop completes without exceptions
    - Parquet metric files are created (training, arena, timings)
    - Neural network checkpoint files are created
    """
    nnet = NNetWrapper(ttt_game, test_config)
    coach = Coach(ttt_game, nnet, test_config)
    coach.learn()

    # Check training data parquet exists
    training_dir = test_config.training_data_directory
    assert training_dir.exists(), "TrainingData directory should exist after learn()"
    assert any(training_dir.rglob("*.parquet")), "Should have training parquet files"

    # Check arena data parquet exists
    arena_dir = test_config.arena_data_directory
    assert arena_dir.exists(), "ArenaData directory should exist after learn()"
    assert any(arena_dir.rglob("*.parquet")), "Should have arena parquet files"

    # Check timings parquet exists
    timings_dir = test_config.timings_directory
    assert timings_dir.exists(), "Timings directory should exist after learn()"
    assert any(timings_dir.rglob("*.parquet")), "Should have timings parquet files"

    # Check that at least one neural network checkpoint exists
    net_dir = test_config.net_directory
    assert net_dir.exists(), "Nets directory should exist after learn()"
    checkpoints = list(net_dir.glob("*.pth.tar"))
    assert len(checkpoints) > 0, "Should have at least one checkpoint file"
