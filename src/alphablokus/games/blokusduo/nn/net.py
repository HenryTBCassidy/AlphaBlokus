from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from alphablokus.aux_heads import AUX_HEAD_NAMES

if TYPE_CHECKING:
    from alphablokus.config import NetConfig

# Ownership classes per cell, in the order the head's channels emit them:
# 0 = the opponent of the side to move holds it at the end of the game,
# 1 = nobody holds it, 2 = the side to move holds it. The class index is
# ``ownership + 1`` for an ownership map in ``{-1, 0, +1}``, so the label is
# monotone in "how good for me" and the frame is the position's own canonical
# frame — the same frame the input planes are in.
OWNERSHIP_CLASSES = 3


def calc_conv2d_output(
    y_x: tuple[int, int],
    kernel_size: int | tuple[int, int] = 3,
    stride: int = 1,
    pad: int = 1,
    dilation: int = 1,
) -> tuple[int, int]:
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
    The one-hot probe test pins it against the real ``ActionCodec.encode``.
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
        col = x  # width_idx
        perm[action_index] = o * cells + row * board_cols + col
    return perm


class ConvPolicyHead(nn.Module):
    """Fully-convolutional policy head.

    A 1×1 convolution maps the trunk's per-cell features to ``num_orientations``
    logit planes (one per piece-orientation), reordered into ``ActionCodec``
    action order. Pass is a single logit from a small global head rather than a
    wasted full plane. Output: ``(B, action_size)`` raw logits, matching the FC
    head's interface so the surrounding net/forward is unchanged.
    """

    def __init__(self, num_filters: int, num_orientations: int, board_rows: int, board_cols: int) -> None:
        super().__init__()
        self.move_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_orientations,
            kernel_size=1,
            stride=1,
            bias=True,
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
        moves = self.move_conv(features)  # (B, O, rows, cols)
        moves = moves.reshape(moves.size(0), -1)  # (B, O·rows·cols), channel-major
        moves = moves[:, self.perm]  # -> ActionCodec order
        pass_logit = self.pass_head(features)  # (B, 1)
        return torch.cat([moves, pass_logit], dim=1)  # (B, action_size)


class AlphaBlokusDuo(nn.Module):
    def __init__(
        self, board_rows: int, board_cols: int, action_size: int, num_input_channels: int, config: NetConfig
    ) -> None:
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
                in_channels=self.num_input_channels,
                out_channels=config.num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=config.num_filters),
            nn.ReLU(),
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

        self.policy_head: nn.Module = self._build_policy_head(config, conv_out)

        # ------------------------------------------------------------------ #
        # Auxiliary heads
        # ------------------------------------------------------------------ #
        # Every one of these is **off by default** and **never consulted when
        # choosing a move**: ``forward`` appends the built ones after ``(log_pi,
        # value)`` and ``predict``/``predict_batch`` drop them all.
        #
        # They are constructed **after every primary head, in the fixed
        # ``AUX_HEAD_NAMES`` order, appending only**, so that at a fixed seed the
        # trunk, value head, policy head and every *earlier* auxiliary head
        # initialise identically whether or not a later one exists. That is what
        # makes a one-head-at-a-time A/B measure the head rather than a shifted
        # RNG stream (docs/plans/supervised-network-improvements.md N1).
        #
        # The arity of ``forward`` varies rather than absent heads returning
        # ``None``: a ``None`` in a module's output makes it untraceable, which
        # would break the web ONNX export even with every head switched off.

        # Score head — a near-copy of the value head predicting the bounded
        # score-margin target ``tanh(margin / score_scale)`` (see
        # ``alphablokus.training.score_target``).
        self.score_head: nn.Module | None = None
        if config.score_head:
            self.score_head = nn.Sequential(
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

        # Ownership head — a bare 1×1 convolution to three logit planes over the
        # board: cell held at the end of the game by the side to move, by their
        # opponent, or by neither. Deliberately a single convolution and no
        # normalisation: the target is per-cell and spatially local to the
        # trunk's own features, so anything deeper would be learning capacity
        # spent in the *head* rather than pressure applied to the *trunk*, which
        # is the whole point of an auxiliary target.
        self.ownership_head: nn.Module | None = None
        if config.ownership_head:
            self.ownership_head = nn.Conv2d(
                in_channels=config.num_filters,
                out_channels=OWNERSHIP_CLASSES,
                kernel_size=1,
                stride=1,
                bias=True,
            )

        # Opponent-reply head — a second policy-shaped head over the same action
        # space, predicting the opponent's reply to this position. Built by the
        # same factory as the main policy head so the two cannot drift apart in
        # architecture (and so ``policy_head: "fc"`` configs get an fc reply head).
        self.reply_head: nn.Module | None = None
        if config.reply_head:
            self.reply_head = self._build_policy_head(config, conv_out)

        # Which auxiliary heads this net actually built, in ``forward``'s output
        # order. The single source of truth for unpacking a forward output —
        # positional indexing is wrong the moment one head is on and an earlier
        # one is off. Read by ``BaseNNetWrapper._split_net_outputs``.
        self.aux_head_names: tuple[str, ...] = tuple(
            name for name in AUX_HEAD_NAMES if getattr(self, name, None) is not None
        )

    def _build_policy_head(self, config: NetConfig, conv_out: int) -> nn.Module:
        """Construct a policy-shaped head: ``(B, action_size)`` raw logits.

        Shared by the main policy head and the auxiliary opponent-reply head, which
        must have the same shape by definition — the reply is a move like any other.

        Raises:
            ValueError: If ``policy_head="conv"`` but the action space is not
                ``cells·O + 1`` for this board, which the conv head requires.
        """
        if config.policy_head == "conv":
            # Action space = board_rows · board_cols · num_orientations + 1 (pass).
            cells = self.board_rows * self.board_cols
            num_orientations, remainder = divmod(self.action_size - 1, cells)
            if remainder != 0:
                raise ValueError(
                    f"action_size {self.action_size} is not cells·O+1 for a "
                    f"{self.board_rows}×{self.board_cols} board; conv head needs "
                    "an (orientation, cell) action space."
                )
            return ConvPolicyHead(
                num_filters=config.num_filters,
                num_orientations=num_orientations,
                board_rows=self.board_rows,
                board_cols=self.board_cols,
            )
        return nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass through the network.

        Args:
            x: Multi-channel board representation.
               Shape: batch_size x 44 x 14 x 14

        Returns:
            ``(pi, v)`` — log-softmax policy over all actions (batch_size x 17837)
            and the value estimate for the current player (batch_size x 1) — followed
            by one element per **built** auxiliary head, in ``AUX_HEAD_NAMES`` order:

            - ``score`` (batch_size x 1), ``NetConfig.score_head``;
            - ``ownership`` (batch_size x 3 x rows x cols) raw logits,
              ``NetConfig.ownership_head``;
            - ``reply`` (batch_size x action_size) **log-softmax**,
              ``NetConfig.reply_head``.

            The arity varies rather than absent heads being ``None``: a ``None`` in a
            module's output makes it untraceable ("Only tensors, lists, tuples of
            tensors ... can be output from traced functions"), which would break the
            web ONNX export even with every head switched off. With all heads off the
            output is byte-for-byte the pre-auxiliary-head 2-tuple.

            Because the arity varies, callers must unpack via
            ``BaseNNetWrapper._split_net_outputs`` (which consults
            ``aux_head_names``) and never by position — with the score head off and
            the ownership head on, element 2 is the *ownership* map. Every auxiliary
            output is dropped by ``predict``/``predict_batch``, so **no code path
            consults one when choosing a move**.
        """

        x = x.view(-1, self.num_input_channels, self.board_rows, self.board_cols)
        conv_block_out = self.conv_block(x)
        features = self.residual_blocks(conv_block_out)

        pi_logits = self.policy_head(features)

        value = self.value_head(features)

        log_pi = F.log_softmax(pi_logits, dim=1)
        if not self.aux_head_names:
            return log_pi, value
        outputs = [log_pi, value]
        if self.score_head is not None:
            outputs.append(self.score_head(features))
        if self.ownership_head is not None:
            outputs.append(self.ownership_head(features))
        if self.reply_head is not None:
            outputs.append(F.log_softmax(self.reply_head(features), dim=1))
        return tuple(outputs)
