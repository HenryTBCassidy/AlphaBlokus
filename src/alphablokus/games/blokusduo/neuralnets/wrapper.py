import torch.nn as nn

from alphablokus.core.config import RunConfig
from alphablokus.core.interfaces import IGame
from alphablokus.games.base_wrapper import BaseNNetWrapper
from alphablokus.games.blokusduo.neuralnets.net import AlphaBlokusDuo


class NNetWrapper(BaseNNetWrapper):
    """Neural Network wrapper for BlokusDuo."""

    def __init__(self, game: IGame, config: RunConfig) -> None:
        super().__init__(game, config)

    def _create_network(self) -> nn.Module:
        board = self.game.initialise_board()
        rows, cols = self.game.get_board_size()
        return AlphaBlokusDuo(
            board_rows=rows,
            board_cols=cols,
            action_size=self.game.get_action_size(),
            num_input_channels=board.num_channels,
            config=self.net_config,
        )
