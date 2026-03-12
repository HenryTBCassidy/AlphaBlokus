import numpy as np
import pytest

from core.config import MCTSConfig, RunConfig
from core.mcts import MCTS, MCTSEpisodeStats
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper


@pytest.fixture
def mcts_instance(
    ttt_game: TicTacToeGame, test_config: RunConfig, mcts_config: MCTSConfig,
) -> MCTS:
    """MCTS with a real (tiny, untrained) TicTacToe neural network."""
    nnet = NNetWrapper(ttt_game, test_config)
    return MCTS(ttt_game, nnet, mcts_config)


def test_mcts_episode_stats_populated(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """After get_action_prob(), all profiling stats should be non-zero."""
    board = ttt_game.initialise_board()
    mcts_instance.get_action_prob(board, temp=1)

    stats = mcts_instance.get_episode_stats()
    assert stats.num_moves == 1
    assert stats.total_sims > 0
    assert stats.total_search_time_s > 0
    assert stats.total_inference_time_s > 0
    assert stats.num_leaf_expansions > 0
    assert stats.tree_size > 0


def test_mcts_episode_stats_accumulates(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """Two get_action_prob() calls should produce num_moves == 2."""
    board = ttt_game.initialise_board()
    mcts_instance.get_action_prob(board, temp=1)

    # Play a move and get probs from the next state
    board, next_player = ttt_game.get_next_state(board, 1, 4)
    canonical = ttt_game.get_canonical_form(board, next_player)
    mcts_instance.get_action_prob(canonical, temp=1)

    stats = mcts_instance.get_episode_stats()
    assert stats.num_moves == 2
    assert stats.total_sims == 2 * mcts_instance.config.num_mcts_sims


def test_mcts_episode_stats_frozen(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """MCTSEpisodeStats should be immutable (frozen dataclass)."""
    board = ttt_game.initialise_board()
    mcts_instance.get_action_prob(board, temp=1)

    stats = mcts_instance.get_episode_stats()
    with pytest.raises(AttributeError):
        stats.num_moves = 999  # type: ignore[misc]


def test_mcts_inference_time_less_than_search(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """Inference time should be a fraction of total search time."""
    board = ttt_game.initialise_board()
    mcts_instance.get_action_prob(board, temp=1)

    stats = mcts_instance.get_episode_stats()
    assert stats.total_inference_time_s < stats.total_search_time_s


def test_mcts_tree_size_matches_state_visits(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """tree_size should equal the number of unique states in state_visits."""
    board = ttt_game.initialise_board()
    mcts_instance.get_action_prob(board, temp=1)

    stats = mcts_instance.get_episode_stats()
    assert stats.tree_size == len(mcts_instance.state_visits)
