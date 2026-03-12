import numpy as np
import pytest

from core.config import MCTSConfig, RunConfig
from core.mcts import MCTS
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper


@pytest.fixture
def mcts_instance(
    ttt_game: TicTacToeGame, test_config: RunConfig, mcts_config: MCTSConfig,
) -> MCTS:
    """MCTS with a real (tiny, untrained) TicTacToe neural network."""
    nnet = NNetWrapper(ttt_game, test_config)
    return MCTS(ttt_game, nnet, mcts_config)


def test_action_probs_valid_moves_only(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """Non-zero probabilities should only appear on legal moves."""
    board = ttt_game.initialise_board()
    # Place a piece to create some invalid moves
    board, _ = ttt_game.get_next_state(board, 1, 0)  # Player 1 at (0,0)
    canonical = ttt_game.get_canonical_form(board, -1)

    probs = mcts_instance.get_action_prob(canonical, temp=1)

    # Action 0 is occupied — its probability should be 0
    assert probs[0] == 0

    # Every non-zero prob should correspond to a valid move
    valids = ttt_game.valid_move_masking(canonical, 1)
    for i, p in enumerate(probs):
        if p > 0:
            assert valids[i] > 0, f"Action {i} has prob {p} but is not valid"


def test_action_probs_sum_to_one(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """Probability vector from get_action_prob should sum to ~1.0."""
    board = ttt_game.initialise_board()
    probs = mcts_instance.get_action_prob(board, temp=1)
    assert pytest.approx(sum(probs), abs=1e-6) == 1.0


def test_action_probs_deterministic_temp0(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """temp=0 should return a one-hot vector (exactly one action selected)."""
    board = ttt_game.initialise_board()
    probs = mcts_instance.get_action_prob(board, temp=0)
    assert sum(1 for p in probs if p > 0) == 1
    assert pytest.approx(sum(probs), abs=1e-6) == 1.0


def test_search_returns_value(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """search() should return a numeric value in [-1, 1]."""
    board = ttt_game.initialise_board()
    value = mcts_instance.search(board)
    # Value may be a numpy scalar or 1-element array from the NN
    v = np.asarray(value).item()
    assert -1 <= v <= 1


def test_mcts_tree_grows(
    ttt_game: TicTacToeGame, mcts_instance: MCTS,
):
    """After simulations, state_visits dict should be non-empty."""
    board = ttt_game.initialise_board()
    # Run several simulations
    for _ in range(mcts_instance.config.num_mcts_sims):
        mcts_instance.search(board)

    assert len(mcts_instance.state_visits) > 0
    assert len(mcts_instance.policy_priors) > 0
