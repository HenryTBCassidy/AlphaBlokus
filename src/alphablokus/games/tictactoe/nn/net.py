from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from alphablokus.config import NetConfig


class AlphaTicTacToe(nn.Module):
    def __init__(self, board_rows: int, board_cols: int, action_size: int, num_input_channels: int, config: NetConfig):
        super().__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.action_size = action_size
        self.num_input_channels = num_input_channels
        self.config = config

        self.conv1 = nn.Conv2d(num_input_channels, config.num_filters, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(config.num_filters, config.num_filters, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(config.num_filters, config.num_filters, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(config.num_filters, config.num_filters, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(config.num_filters)
        self.bn2 = nn.BatchNorm2d(config.num_filters)
        self.bn3 = nn.BatchNorm2d(config.num_filters)
        self.bn4 = nn.BatchNorm2d(config.num_filters)

        self.fc1 = nn.Linear(config.num_filters * (self.board_rows - 4) * (self.board_cols - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Multi-channel board representation, shape
                ``(batch, channels, board_rows, board_cols)``.

        Returns:
            Tuple of ``(pi, v)`` — log-softmax policy over all actions,
            shape ``(batch, action_size)``, and tanh value estimate,
            shape ``(batch, 1)``.
        """
        x = x.view(-1, self.num_input_channels, self.board_rows, self.board_cols)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # conv3/conv4 are unpadded, so each shaves a border cell per side.
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(-1, self.config.num_filters * (self.board_rows - 4) * (self.board_cols - 4))

        x = F.dropout(
            F.relu(self.fc_bn1(self.fc1(x))),
            p=self.config.dropout,
            training=self.training,
        )
        x = F.dropout(
            F.relu(self.fc_bn2(self.fc2(x))),
            p=self.config.dropout,
            training=self.training,
        )

        pi = self.fc3(x)
        v = self.fc4(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
