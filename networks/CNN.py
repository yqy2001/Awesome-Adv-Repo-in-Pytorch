"""
@Author:        禹棋赢
@StartTime:     2021/8/2 20:01
@Filename:      CNN.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(torch.nn.Module):

    def __init__(self, in_channels=1):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 64, 8, 1  # in_channels, out_channels(kernel nums), kernel_size, stride
        )  # (batch_size, 3, 28, 28) --> (batch_size, 64, 21, 21)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)  # (batch_size, 64, 21, 21) --> (batch_size, 128, 8, 8)
        self.conv3 = nn.Conv2d(128, 128, 5, 1)  # (batch_size, 128, 8, 8) --> (batch_size, 128, 4, 4)
        self.lr1 = nn.Linear(128 * 4 * 4, 128)  # (batch_size, 128*4*4) --> (batch_size, 128)
        self.lr2 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 128 * 4 * 4)
        x = self.lr2(self.lr1(x))
        return x
