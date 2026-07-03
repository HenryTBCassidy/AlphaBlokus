from __future__ import annotations

from typing import TYPE_CHECKING

from alphablokus.games.base_wrapper import BaseNNetWrapper
from alphablokus.games.tictactoe.nn.net import AlphaTicTacToe

if TYPE_CHECKING:
    import torch.nn as nn

    from alphablokus.config import RunConfig
    from alphablokus.interfaces import IGame


class NNetWrapper(BaseNNetWrapper):
    """Neural Network wrapper for TicTacToe."""

    def __init__(self, game: IGame, config: RunConfig) -> None:
        super().__init__(game, config)

    def _create_network(self) -> nn.Module:
        board = self.game.initialise_board()
        rows, cols = self.game.get_board_size()
        return AlphaTicTacToe(
            board_rows=rows,
            board_cols=cols,
            action_size=self.game.get_action_size(),
            num_input_channels=board.num_channels,
            config=self.net_config,
        )
