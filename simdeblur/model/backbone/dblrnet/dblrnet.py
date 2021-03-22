"""
Adversarial Spatio-Temporal Learning for Video Deblurring
The DBLRNet adopts 3D convolution for spatio-temporal modeling, which serves as a generator for adversarial training.
"""


import torch
import torch.nn as nn

from ...build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
class DBLRNet(nn.Module):
    def __init__(self, num_frames, in_channels, inner_channels):
        super(DBLRNet, self).__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.inner_channels = inner_channels

        self.layer_counts = 15

        self.L_in = nn.Sequential(
            nn.Conv3d(self.in_channels, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, self.inner_channels, kernel_size=(3,3,3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )

        Ln = []
        for i in range(self.layer_counts):
            Ln.append(
                ResBlock(self.inner_channels)
            )
        self.Ln = nn.Sequential(
            *Ln
        )

        self.L_out = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.inner_channels * 4, self.inner_channels * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.inner_channels * 4, 3, 3, 1, 1)
        )
    
    def forward(self, x):
        assert x.dim() == 5, "Input tensor should be in 5 dims!"
        b, n, c, h, w = x.shape

        x = x.transpose(1, 2)
        l2 = self.L_in(x)

        ln = self.Ln(l2.view(b, -1, h, w))

        out = self.L_out(ln + l2.view(b, -1, h, w))

        return out



class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels

        self.convs = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            nn.BatchNorm2d(num_features=self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            nn.BatchNorm2d(num_features=self.in_channels)
        )
    
    def forward(self, x):
        
        return self.convs(x) + x