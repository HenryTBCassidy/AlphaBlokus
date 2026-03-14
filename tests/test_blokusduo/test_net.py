import numpy as np
import torch
import pytest

from core.config import NetConfig
from games.blokusduo.board import BlokusDuoBoard
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.neuralnets.net import AlphaBlokusDuo


@pytest.fixture
def blokus_net(net_config: NetConfig) -> AlphaBlokusDuo:
    """A tiny BlokusDuo network for shape tests."""
    return AlphaBlokusDuo(net_config)


def test_forward_pass_empty_board(
    blokus_board: BlokusDuoBoard,
    blokus_net: AlphaBlokusDuo,
):
    """Feed an empty board through the net and check output shapes."""
    tensor = blokus_board.as_multi_channel(1)
    assert tensor.shape == (44, 14, 14)
    assert tensor.dtype == np.float32

    x = torch.FloatTensor(tensor).unsqueeze(0)  # (1, 44, 14, 14)
    blokus_net.eval()
    with torch.no_grad():
        log_pi, v = blokus_net(x)

    assert log_pi.shape == (1, 17837)
    assert v.shape == (1, 1)

    # Value should be in [-1, 1] (tanh output)
    assert -1 <= v.item() <= 1

    # Log-softmax should produce valid log-probabilities
    pi = torch.exp(log_pi)
    assert pytest.approx(pi.sum().item(), abs=1e-4) == 1.0


def test_forward_pass_after_placement(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
    blokus_net: AlphaBlokusDuo,
):
    """Feed a board with pieces placed through the net."""
    white_action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(white_action, player_side=1)
    black_action = blokus_game.initial_actions[-1][0]
    board = board.with_piece(black_action, player_side=-1)

    tensor = board.as_multi_channel(1)
    assert tensor.shape == (44, 14, 14)
    assert np.sum(tensor) > 0  # Non-empty board should have non-zero channels

    x = torch.FloatTensor(tensor).unsqueeze(0)
    blokus_net.eval()
    with torch.no_grad():
        log_pi, v = blokus_net(x)

    assert log_pi.shape == (1, 17837)
    assert v.shape == (1, 1)


def test_forward_pass_canonical_neg1(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
    blokus_net: AlphaBlokusDuo,
):
    """Canonical form for player -1 should also produce valid net output."""
    white_action = blokus_game.initial_actions[1][0]
    board = blokus_board.with_piece(white_action, player_side=1)

    canonical = board.canonical(-1)
    tensor = canonical.as_multi_channel(1)  # Always player 1 after canonical
    assert tensor.shape == (44, 14, 14)

    x = torch.FloatTensor(tensor).unsqueeze(0)
    blokus_net.eval()
    with torch.no_grad():
        log_pi, v = blokus_net(x)

    assert log_pi.shape == (1, 17837)
    assert v.shape == (1, 1)


def test_batch_forward_pass(
    blokus_board: BlokusDuoBoard,
    blokus_game: BlokusDuoGame,
    blokus_net: AlphaBlokusDuo,
):
    """Batched forward pass should produce correct batch dimension."""
    boards = [blokus_board.as_multi_channel(1)]

    white_action = blokus_game.initial_actions[1][0]
    board_with_piece = blokus_board.with_piece(white_action, player_side=1)
    boards.append(board_with_piece.as_multi_channel(1))

    batch = torch.FloatTensor(np.array(boards))  # (2, 44, 14, 14)
    assert batch.shape == (2, 44, 14, 14)

    blokus_net.eval()
    with torch.no_grad():
        log_pi, v = blokus_net(batch)

    assert log_pi.shape == (2, 17837)
    assert v.shape == (2, 1)
