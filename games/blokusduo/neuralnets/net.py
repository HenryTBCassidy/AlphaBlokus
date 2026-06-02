import math

import numpy as np
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


def build_action_permutation(board_rows: int, board_cols: int, num_orientations: int) -> np.ndarray:
    """Gather index from a flat conv-policy output into ``ActionCodec`` order.

    The conv policy head emits a ``(num_orientations, board_rows, board_cols)``
    tensor; flattened channel-major its element order is
    ``o · (rows·cols) + row · cols + col`` (array coords, top-left origin).

    ``ActionCodec`` orders actions as ``index = y · (cols·O) + x · O + o`` in
    board coords (bottom-left origin), with ``CoordinateIndexDecoder.to_idx``
    giving ``row = N-1-y``, ``col = x``.

    Returns ``perm`` such that ``conv_flat[perm]`` is in ActionCodec order:
    ``perm[action_index]`` = the conv-flat position holding that action's logit.
    Pure arithmetic replicating ``ActionCodec``/``CoordinateIndexDecoder``; the
    C4 one-hot probe test pins it against the real ``ActionCodec.encode``.
    """
    board_size = board_cols  # Blokus Duo is square; ActionCodec uses one board_size
    cells = board_rows * board_cols
    perm = np.empty(cells * num_orientations, dtype=np.int64)
    for action_index in range(perm.size):
        o = action_index % num_orientations
        remaining = action_index // num_orientations
        x = remaining % board_size
        y = remaining // board_size
        row = board_rows - 1 - y  # CoordinateIndexDecoder.to_idx: length_idx
        col = x                   # width_idx
        perm[action_index] = o * cells + row * board_cols + col
    return perm


class ConvPolicyHead(nn.Module):
    """Fully-convolutional policy head (F4).

    A 1×1 convolution maps the trunk's per-cell features to ``num_orientations``
    logit planes (one per piece-orientation), reordered into ``ActionCodec``
    action order. Pass is a single logit from a small global head rather than a
    wasted full plane. Output: ``(B, action_size)`` raw logits, matching the FC
    head's interface so the surrounding net/forward is unchanged.
    """

    def __init__(self, num_filters: int, num_orientations: int,
                 board_rows: int, board_cols: int) -> None:
        super().__init__()
        self.move_conv = nn.Conv2d(
            in_channels=num_filters, out_channels=num_orientations,
            kernel_size=1, stride=1, bias=True,
        )
        # Pass action: one scalar logit from globally-pooled features.
        self.pass_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_filters, 1),
        )
        perm = build_action_permutation(board_rows, board_cols, num_orientations)
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        moves = self.move_conv(features)               # (B, O, rows, cols)
        moves = moves.reshape(moves.size(0), -1)        # (B, O·rows·cols), channel-major
        moves = moves[:, self.perm]                     # -> ActionCodec order
        pass_logit = self.pass_head(features)           # (B, 1)
        return torch.cat([moves, pass_logit], dim=1)    # (B, action_size)


class AlphaBlokusDuo(nn.Module):
    def __init__(self, board_rows: int, board_cols: int, action_size: int,
                 num_input_channels: int, config: NetConfig):
        """Initialise the Blokus Duo ResNet.

        The neural net receives a multi-channel representation produced by
        ``BlokusDuoBoard.as_multi_channel()``. Channel layout and counts are
        determined by the board class and passed in as ``num_input_channels``.

        Args:
            board_rows: Board height (e.g. 14 for Blokus Duo).
            board_cols: Board width (e.g. 14 for Blokus Duo).
            action_size: Total actions including pass (e.g. 17,837 = 14² × 91 + 1).
            num_input_channels: Input channels (e.g. 44 = 21 per player + 2 aggregate).
            config: Network hyperparameters.
        """
        super().__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.action_size = action_size
        self.num_input_channels = num_input_channels
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

        if config.policy_head == "conv":
            # Action space = board_rows · board_cols · num_orientations + 1 (pass).
            cells = self.board_rows * self.board_cols
            num_orientations, remainder = divmod(self.action_size - 1, cells)
            if remainder != 0:
                raise ValueError(
                    f"action_size {self.action_size} is not cells·O+1 for a "
                    f"{self.board_rows}×{self.board_cols} board; conv head needs "
                    "an (orientation, cell) action space.")
            self.policy_head = ConvPolicyHead(
                num_filters=config.num_filters,
                num_orientations=num_orientations,
                board_rows=self.board_rows,
                board_cols=self.board_cols,
            )
        else:
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
