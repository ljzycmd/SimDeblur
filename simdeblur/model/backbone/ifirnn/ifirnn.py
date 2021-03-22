"""
# _*_ coding: utf-8 _*_
# @Time    :   2021/02/08 02:19:23
# @FileName:   ifirnn.py
# @Author  :   Minton Cao
# @Software:   VSCode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...build import BACKBONE_REGISTRY


class BasicBlock(nn.Module):
    """
    Basic block without batch normalization.
    """
    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)

        return out + x


class RNNCell(nn.Module):
    def __init__(self, in_channels, inner_channels, dual_cell=True):
        super(RNNCell, self).__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.dual_cell = dual_cell

        # F_B: feature extraction with 4x down-sampling
        self.F_B = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inner_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.inner_channels, self.inner_channels*2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.inner_channels*2, self.inner_channels*4, kernel_size=3, stride=2, padding=1),
        )
        # F_R: residual blocks
        res_blocks = []
        for i in range(6):
            res_blocks.append(BasicBlock(self.inner_channels*4*2))
        self.F_R = nn.Sequential(*res_blocks)
        
        if not dual_cell:
            # F_L: reconstruct
            self.F_L = nn.Sequential(
                nn.ConvTranspose2d(self.inner_channels*8, self.inner_channels*4, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose2d(self.inner_channels*4, self.inner_channels, kernel_size=4, stride=2, padding=1),
                nn.Conv2d(self.inner_channels, 3, kernel_size=3, stride=1, padding=1)
            )
        
        # F_H: hidden state
        self.F_H = nn.Sequential(
            nn.Conv2d(self.inner_channels*8, self.inner_channels*4, kernel_size=3, stride=1, padding=1),
            BasicBlock(self.inner_channels*4),
            nn.Conv2d(self.inner_channels*4, self.inner_channels*4, 3, 1, 1)
        )

    def forward(self, x, h_s=None, last_iter=False):
        # x structure: (batch_size, channel, height, width)
        feats = self.F_B(x)
        if h_s is None:
            # feats = self.F_R(torch.cat([feats, torch.zeros_like(feats)], dim=1))
            feats = self.F_R(torch.cat([feats, feats], dim=1))
        else:
            feats = self.F_R(torch.cat([feats, h_s], dim=1))
        if (not self.dual_cell) and last_iter:
            out = self.F_L(feats)
        else:
            out = None
        h_s = self.F_H(feats)
        return out, h_s


@BACKBONE_REGISTRY.register()
class IFIRNN(nn.Module):
    """
    Recurrent Neural Networks with Intra-Frame Iterations for Video Deblurring (IFIRNN, CVPR2019)
    """

    def __init__(self, in_channels, inner_channels, dual_cell=True, intra_iters=3):
        super(IFIRNN, self).__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.dual_cell = dual_cell
        # C2H3: 2 cells, 3 intra iterations
        self.intra_iters = intra_iters
        self.rnncells = nn.ModuleList()
        if self.dual_cell:
            self.rnncells.append(RNNCell(self.in_channels, self.inner_channels, dual_cell=True))
        self.rnncells.append(RNNCell(self.in_channels, self.inner_channels, dual_cell=False))

    def forward(self, x):
        """
        x shape: (b, n, c, h, w) = (16, 12, 3, 720, 1024)
        h_s: hidden state
        """
        assert x.dim() == 5, f"Input x should be in 5 dims, but got {x.dim()} now!"
        b, n, c, h, w = x.shape

        outputs = []
        h_s = None
        for frame_idx in range(n):
            # outputs: (b, n, c, h, w) = (16, 3, 720, 1024)
            out, h_s = self.rnncells[0](x[:, frame_idx], h_s)
            assert out == None
            for j in range(self.intra_iters):
                if j == self.intra_iters - 1:
                    # output the latent frame in the last iteration.
                    if self.dual_cell:
                        out, h_s = self.rnncells[1](x[:, frame_idx], h_s, last_iter=True)
                    else:
                        out, h_s = self.rnncells[0](x[:, frame_idx], h_s, last_iter=True)
                else:
                    if self.dual_cell:
                        out, h_s = self.rnncells[1](x[:, frame_idx], h_s)
                    else:
                        out, h_s = self.rnncells[0](x[:, frame_idx], h_s)
                    assert out == None
            outputs.append(out)

        return torch.stack(outputs, dim=1)