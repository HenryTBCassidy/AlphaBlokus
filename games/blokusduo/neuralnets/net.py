import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.config import NetConfig


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
    def __init__(self, config: NetConfig):
        super().__init__()
        """
        The physical Blokus Duo board is a 14×14 square grid. The neural net's input representation
        extends this to 14×18 by concatenating piece-tracking columns on each side:

            [Black pieces (14×2)] [Board state (14×14)] [White pieces (14×2)]

        The 2-column regions on each side encode which of the 21 pieces that player has played,
        since knowing remaining pieces is critical for evaluating board positions.

        Net input shape: batch_size × 1 × 14 × 18  (i.e. board_rows × (physical_board_cols + 4))
        """
        self.board_rows, self.board_cols = 14, 18
        # Actions: place any of 91 piece-orientations on any of the 14×14 physical board squares, plus pass
        self.action_size = (14 * 14 * 91) + 1  # TODO: Get this from the game object (maybe)
        self.config = config

        conv_out_y_x = calc_conv2d_output((self.board_rows, self.board_cols), 3, 1, 1)
        conv_out = conv_out_y_x[0] * conv_out_y_x[1]

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=config.num_filters, kernel_size=3, stride=1, padding=1, bias=False
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
            x: Flattened net representation of the board.
               Shape: batch_size × 252  (14 rows × 18 cols, flattened)
               This is the 14×14 physical board with 2 piece-encoding columns
               concatenated on each side (see __init__ docstring for layout).

        Returns:
            pi: Log-softmax policy over all actions.   Shape: batch_size × 17837
            v:  Value estimate for the current player.  Shape: batch_size × 1
        """

        x = x.view(-1, 1, self.board_rows, self.board_cols)  # batch_size * 1 * board_rows * board_cols
        conv_block_out = self.conv_block(x)  # batch_size * num_channels * board_rows_conv * board_cols_conv
        features = self.residual_blocks(conv_block_out)  # batch_size * num_channels * board_rows_conv * board_cols_conv

        # Predict raw logits distributions wrt policy
        pi_logits = self.policy_head(features)  # batch_size * 17837 (which is 14 x 14 x 91 + 1)

        # Predict evaluated value from current player's perspective.
        value = self.value_head(features)  # batch_size

        return F.log_softmax(pi_logits, dim=1), value
