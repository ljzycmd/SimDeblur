"""
A pytorch implementation for cvpr17 paper "Deep Video Deblurring for Hand-held Cameras"
Author: Mingdeng Cao
"""


import torch
import torch.nn as nn 
import torch.nn.functional as F 

from ...build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
class DBN(nn.Module):
    def __init__(self, num_frames, in_channels, inner_channels):
        super(DBN, self).__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.inner_channels = inner_channels

        self.eps = 1e-3

        # define the model blocks
        # F0
        self.F0 = nn.Sequential(
            nn.Conv2d(self.num_frames * self.in_channels, self.inner_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.inner_channels, self.eps),
        )

        # down-sampling stage1
        self.D1 = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels, self.eps)
        )
        self.F1_1 = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*2, self.eps),
        )
        self.F1_2 = nn.Sequential(
            nn.Conv2d(self.inner_channels*2, self.inner_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*2, self.eps),
        )
        
        # down-sampling stage2
        self.D2 = nn.Sequential(
            nn.Conv2d(self.inner_channels*2, self.inner_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels*4, self.eps),
        )
        self.F2_1 = nn.Sequential(
            nn.Conv2d(self.inner_channels*4, self.inner_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*4, self.eps),
        )
        self.F2_2 = nn.Sequential(
            nn.Conv2d(self.inner_channels*4, self.inner_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*4, self.eps),
        )
        self.F2_3 = nn.Sequential(
            nn.Conv2d(self.inner_channels*4, self.inner_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*4, self.eps),
        )

        # down-sampling stage3
        self.D3 = nn.Sequential(
            nn.Conv2d(self.inner_channels*4, self.inner_channels*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels*8, self.eps),
        )
        self.F3_1 = nn.Sequential(
            nn.Conv2d(self.inner_channels*8, self.inner_channels*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*8, self.eps),
        )
        self.F3_2 = nn.Sequential(
            nn.Conv2d(self.inner_channels*8, self.inner_channels*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*8, self.eps),
        )
        self.F3_3 = nn.Sequential(
            nn.Conv2d(self.inner_channels*8, self.inner_channels*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*8, self.eps),
        )

        # up-sampling stage1
        self.U1 = nn.Sequential(
            nn.ConvTranspose2d(self.inner_channels*8, self.inner_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels*4, self.eps)
        )
        self.F4_1 = nn.Sequential(
            nn.Conv2d(self.inner_channels*4, self.inner_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*4, self.eps),
        )
        self.F4_2 = nn.Sequential(
            nn.Conv2d(self.inner_channels*4, self.inner_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*4, self.eps)
        )
        self.F4_3 = nn.Sequential(
            nn.Conv2d(self.inner_channels*4, self.inner_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*4, self.eps)
        )

        # up-sampling stage2
        self.U2 = nn.Sequential(
            nn.ConvTranspose2d(self.inner_channels*4, self.inner_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels*2, self.eps)
        )
        self.F5_1 = nn.Sequential(
            nn.Conv2d(self.inner_channels*2, self.inner_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels*2, self.eps)
        )
        self.F5_2 = nn.Sequential(
            nn.Conv2d(self.inner_channels*2, self.inner_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inner_channels, self.eps)
        )

        # up-sampling stage3
        self.U3 = nn.Sequential(
            nn.ConvTranspose2d(self.inner_channels, self.inner_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels, self.eps)
        )
        self.F6_1 = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.in_channels*self.num_frames, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels*self.num_frames, self.eps)
        )
        self.F6_2 = nn.Sequential(
            nn.Conv2d(self.num_frames*self.in_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3, self.eps)
        )

    
    def init_parameters(self):
        pass


    def forward(self, x: torch.Tensor):
        assert x.dim() == 5, "input frames' num should be 5"
        b, n, c, h, w = x.shape
        central_frame = x[:, self.num_frames // 2]
        x = x.reshape(b, -1, h, w)
        
        assert n == self.num_frames and c == self.in_channels, "inputs should be {} frames and {} channels".format(self.num_frames, self.in_channels)

        f0 = F.relu(self.F0(x))
        
        f1 = F.relu(self.D1(f0))
        f1 = F.relu(self.F1_1(f1))
        f1 = F.relu(self.F1_2(f1))

        f2 = F.relu(self.D2(f1))
        f2 = F.relu(self.F2_1(f2))
        f2 = F.relu(self.F2_2(f2))
        f2 = F.relu(self.F2_3(f2))

        f3 = F.relu(self.D3(f2))
        f3 = F.relu(self.F3_1(f3))
        f3 = F.relu(self.F3_2(f3))
        f3 = F.relu(self.F3_3(f3))

        f4 = F.relu(self.U1(f3))
        f4 = F.relu(f4 + f2)
        f4 = F.relu(self.F4_1(f4))
        f4 = F.relu(self.F4_2(f4))
        f4 = F.relu(self.F4_3(f4))

        f5 = F.relu(self.U2(f4))
        f5 = F.relu(f5 + f1)
        f5 = F.relu(self.F5_1(f5))
        f5 = F.relu(self.F5_2(f5))

        f6 = F.relu(self.U3(f5))
        f6 = F.relu(f6 + f0)
        f6 = F.relu(self.F6_1(f6))
        f6 = self.F6_2(f6) + central_frame

        return f6