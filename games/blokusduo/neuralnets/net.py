import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.config import NetConfig
from core.interfaces import IGame


def calc_conv2d_output(y_x, kernel_size=3, stride=1, pad=1, dilation=1):
    """
    Calculate output dimensions after a 2D convolution.

    Takes and returns a (rows, cols) tuple.
    """

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    y = math.floor(((y_x[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    x = math.floor(((y_x[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return y, x


class ResNetBlock(nn.Module):
    """Basic residual block."""

    def __init__(
            self,
            num_filters: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class AlphaBlokusDuo(nn.Module):
    def __init__(self, game: IGame, config: NetConfig):
        """Initialise the Blokus Duo ResNet.

        The neural net receives a 44-channel representation produced by
        ``BlokusDuoBoard.as_multi_channel()``:

            Channels  0-20:  Current player's 21 per-piece binary planes
            Channels 21-41:  Opponent's 21 per-piece binary planes
            Channel  42:     Aggregate current player occupancy
            Channel  43:     Aggregate opponent occupancy

        Net input shape: batch_size x 44 x 14 x 14
        """
        super().__init__()
        self.board_rows, self.board_cols = game.get_board_size()
        self.action_size = game.get_action_size()
        self.num_input_channels = 44
        self.config = config

        conv_out_y_x = calc_conv2d_output((self.board_rows, self.board_cols), 3, 1, 1)
        conv_out = conv_out_y_x[0] * conv_out_y_x[1]

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_input_channels, out_channels=config.num_filters, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=config.num_filters),
            nn.ReLU()
        )

        # Residual blocks
        residual_blocks = []
        for _ in range(config.num_residual_blocks):
            residual_blocks.append(ResNetBlock(config.num_filters))

        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=config.num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, config.num_filters),
            nn.ReLU(),
            nn.Linear(config.num_filters, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=config.num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, self.action_size),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Multi-channel board representation.
               Shape: batch_size x 44 x 14 x 14

        Returns:
            pi: Log-softmax policy over all actions.   Shape: batch_size x 17837
            v:  Value estimate for the current player.  Shape: batch_size x 1
        """

        x = x.view(-1, self.num_input_channels, self.board_rows, self.board_cols)
        conv_block_out = self.conv_block(x)  # batch_size * num_channels * board_rows_conv * board_cols_conv
        features = self.residual_blocks(conv_block_out)  # batch_size * num_channels * board_rows_conv * board_cols_conv

        # Predict raw logits distributions wrt policy
        pi_logits = self.policy_head(features)  # batch_size * 17837 (which is 14 x 14 x 91 + 1)

        # Predict evaluated value from current player's perspective.
        value = self.value_head(features)  # batch_size

        return F.log_softmax(pi_logits, dim=1), value
