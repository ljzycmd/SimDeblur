# CMD

import torch
import torch.nn as nn 
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)

        return out + x