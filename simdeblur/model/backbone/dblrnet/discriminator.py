# CMD
# the discriminator for the DBLRNet


import torch
import torch.nn as nn 
import torch.nn.functional as F 


class DBLRNetD(nn.Module):
    def __init__(self, in_channels):
        super(DBLRNetD, self).__init__()
        self.in_channels = in_channels
        self.bn_eps = 1e-5
        
        self.L1_2 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64, self.bn_eps),
            nn.ReLU(inplace=True),
        )

        self.L3_5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, self.bn_eps),
            nn.ReLU(inplace=True),
        )

        self.L6_9 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, self.bn_eps),
            nn.ReLU(inplace=True),
        )

        self.L10_14 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, self.bn_eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, self.bn_eps),
            nn.ReLU(inplace=True),
        )

        self.L15_17 = nn.Sequential(
            nn.Linear(4096, 4096), 
            nn.Linear(4096, 4096), 
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        assert x.dim() == 4, "Input tensor should be in 4 dims!"
        out = self.L1_2(x)
        out = self.L3_5(out)
        out = self.L6_9(out)
        out = self.L10_14(out)
        out = out.flatten()
        out = self.L15_17(out)

        return out