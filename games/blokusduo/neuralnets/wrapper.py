import torch.nn as nn

from core.config import RunConfig
from core.interfaces import IGame
from games.base_wrapper import BaseNNetWrapper
from games.blokusduo.neuralnets.net import AlphaBlokusDuo


class NNetWrapper(BaseNNetWrapper):
    """Neural Network wrapper for BlokusDuo."""

    def __init__(self, game: IGame, args: RunConfig) -> None:
        super().__init__(game, args)

    def _create_network(self) -> nn.Module:
        return AlphaBlokusDuo(self.config)
