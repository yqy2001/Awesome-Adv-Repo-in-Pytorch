"""
@Author:        禹棋赢
@StartTime:     2021/8/2 20:01
@Filename:      CNN.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_reverse(torch.nn.Module):

    def __init__(self, in_channels=1):
        super(CNN_reverse, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 64, 8, 1  # in_channels, out_channels(kernel nums), kernel_size, stride
        )  # (batch_size, 3, 28, 28) --> (batch_size, 64, 21, 21)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)  # (batch_size, 64, 21, 21) --> (batch_size, 128, 8, 8)
        self.conv3 = nn.Conv2d(128, 128, 5, 1)  # (batch_size, 128, 8, 8) --> (batch_size, 128, 4, 4)
        self.lr1 = nn.Linear(128 * 4 * 4, 128)  # (batch_size, 128*4*4) --> (batch_size, 128)
        self.lr2 = nn.Linear(128, 10)  # 10 classes

        self.reverse = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels= in_channels,
                      kernel_size= 1, stride= 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x,  turns=4):
        for i in range(turns):
            if i == turns-1:
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
            else:
                x_ = x
                x_ = F.relu(self.conv1(x_))
                x_ = F.relu(self.conv2(x_))
                x_ = F.relu(self.conv3(x_))
                x_ = self.reverse(x_)
                x = x - x_.repeat(1,1,7,7)
        x = x.reshape(-1, 128 * 4 * 4)
        x = self.lr2(self.lr1(x))
        return x
