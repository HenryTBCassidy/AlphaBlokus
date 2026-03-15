from pathlib import Path

import pytest

from core.config import MCTSConfig, NetConfig, RunConfig
from games.blokusduo.board import BlokusDuoBoard
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pieces import PieceManager, pieces_loader
from games.tictactoe.game import TicTacToeGame


@pytest.fixture(scope="session")
def ttt_game() -> TicTacToeGame:
    """A reusable TicTacToe game instance."""
    return TicTacToeGame(3)


@pytest.fixture(scope="session")
def mcts_config() -> MCTSConfig:
    """Tiny MCTS config for fast tests."""
    return MCTSConfig(num_mcts_sims=2, cpuct=1.0)


@pytest.fixture(scope="session")
def net_config() -> NetConfig:
    """Tiny neural network config for fast tests (CPU only)."""
    return NetConfig(
        learning_rate=0.001,
        dropout=0.3,
        epochs=1,
        batch_size=4,
        cuda=False,
        num_filters=32,
        num_residual_blocks=1,
    )


@pytest.fixture
def test_config(tmp_path: Path, mcts_config: MCTSConfig, net_config: NetConfig) -> RunConfig:
    """Full RunConfig with tiny values and tmp_path-based directories.

    Function-scoped because tmp_path is per-test.
    """
    return RunConfig(
        run_name="test_run",
        num_generations=1,
        num_eps=2,
        temp_threshold=5,
        update_threshold=0.55,
        max_queue_length=10,
        num_arena_matches=2,
        max_generations_lookback=1,
        root_directory=tmp_path,
        load_model=False,
        mcts_config=mcts_config,
        net_config=net_config,
    )


@pytest.fixture(scope="session")
def pieces_path() -> Path:
    """Path to the BlokusDuo pieces.json config file."""
    return Path(__file__).resolve().parent.parent / "games" / "blokusduo" / "pieces.json"


@pytest.fixture(scope="session")
def piece_manager(pieces_path: Path) -> PieceManager:
    """PieceManager loaded from the real pieces.json."""
    return pieces_loader(pieces_path)


@pytest.fixture(scope="session")
def blokus_game(pieces_path: Path) -> BlokusDuoGame:
    """A reusable BlokusDuoGame instance."""
    return BlokusDuoGame(pieces_config_path=pieces_path)


@pytest.fixture
def blokus_board(blokus_game: BlokusDuoGame) -> BlokusDuoBoard:
    """A fresh BlokusDuoBoard for each test."""
    return blokus_game.initialise_board()
