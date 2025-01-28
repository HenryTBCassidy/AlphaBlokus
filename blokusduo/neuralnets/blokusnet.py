import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.config import NetConfig


def calc_conv2d_output(y_x, kernel_size=3, stride=1, pad=1, dilation=1):
    """
    Helper for calculating the dimensions of the layers after applying a convolution
    takes a tuple of (y, x) and returns a tuple of (y, x)
    """

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    y = math.floor(((y_x[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    x = math.floor(((y_x[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return y, x


class ResNetBlock(nn.Module):
    """Basic redisual block."""

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
        There are 14 x 14 squares on the board and 21 pieces. The net's board representation needs to encode both where
        and which pieces have been played. We extend the board columns by 4 to have dimensions 14 x 18.
        batch_size * board_y * (board_x + 4) = batch_size * 14 * 18
        """
        self.board_y, self.board_x = 14, 18
        # Stack of 14 * 14 planes for each of the 94 "pieces" + 1 for do nothing move.
        # Piece planes are obtained by treating each unique rotation or flip of a piece as an entirely new plane
        self.action_size = 14, 14, 95
        self.config = config

        conv_out_y_x = calc_conv2d_output((self.board_y, self.board_x), 3, 1, 1)
        conv_out = conv_out_y_x[0] * conv_out_y_x[1]

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=config.num_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=config.num_channels),
            nn.ReLU()
        )

        # Residual blocks
        residual_blocks = []
        for _ in range(config.num_residual_blocks):
            residual_blocks.append(ResNetBlock(config.num_channels))

        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=config.num_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, config.num_channels),
            nn.ReLU(),
            nn.Linear(config.num_channels, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=config.num_channels,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, math.prod(self.action_size)),
        )

    def forward(self, x):
        """
        This is where the net assembly happens
        :param x:
            Inputs: There are 14 x 14 squares on the board and 21 pieces. The net's board representation needs to encode
                    where pieces have been played and which pieces have been played. We extend the board columns by 2 in
                    each direction to have dimensions 14 x 18.
                    batch_size * board_y * (board_x + 4) = batch_size * 14 * 18

        :return: Policy tensor pi, probability of next move to take:   batch_size * 18620
        :return: Value vector v of expected result of player:          batch_size * 1
        """

        x = x.view(-1, 1, self.board_x, self.board_y)       # batch_size * 1 * board_x * board_y
        conv_block_out = self.conv_block(x)                 # batch_size * num_channels * board_x_conv * board_y_conv
        features = self.residual_blocks(conv_block_out)     # batch_size * num_channels * board_x_conv * board_y_conv

        # Predict raw logits distributions wrt policy
        pi_logits = self.policy_head(features)              # batch_size * 18620 (which is 14 x 14 x 95)

        # Predict evaluated value from current player's perspective.
        value = self.value_head(features)                   # batch_size

        return F.log_softmax(pi_logits, dim=1), value
