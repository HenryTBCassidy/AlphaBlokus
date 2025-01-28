import torch
import torch.nn as nn
from torch.nn import functional as F

from core.config import NetConfig


class AlphaTicTacToe(nn.Module):
    def __init__(self, game, config: NetConfig):
        super().__init__()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.config = config

        self.conv1 = nn.Conv2d(1, config.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(config.num_channels, config.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(config.num_channels, config.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(config.num_channels, config.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(config.num_channels)
        self.bn2 = nn.BatchNorm2d(config.num_channels)
        self.bn3 = nn.BatchNorm2d(config.num_channels)
        self.bn4 = nn.BatchNorm2d(config.num_channels)

        self.fc1 = nn.Linear(config.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        """
        This is where the net assembly happens
        :param x: Inputs:                                              batch_size * board_x * board_y
        :return: Policy vector pi, probability of next move to take:   batch_size * action_size
        :return: Value vector v of expected result of players:         batch_size * num_players
        """
        x = x.view(-1, 1, self.board_x, self.board_y)  # batch_size * 1 * board_x * board_y
        x = F.relu(self.bn1(self.conv1(x)))  # batch_size * num_channels * board_x * board_y
        x = F.relu(self.bn2(self.conv2(x)))  # batch_size * num_channels * board_x * board_y
        x = F.relu(self.bn3(self.conv3(x)))  # batch_size * num_channels * (board_x-2) * (board_y-2)
        x = F.relu(self.bn4(self.conv4(x)))  # batch_size * num_channels * (board_x-4) * (board_y-4)

        x = x.view(-1, self.config.num_channels * (self.board_x - 4) * (self.board_y - 4))

        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=self.config.dropout,  # batch_size * 1024
                      training=self.training)
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=self.config.dropout,  # batch_size * 512
                      training=self.training)

        pi = self.fc3(x)  # batch_size * action_size
        v = self.fc4(x)  # batch_size

        return F.log_softmax(pi, dim=1), torch.tanh(v)
