""" ************************************************
* fileName: srn.py
* desc: The pytorch implementation for Scale-recurrent Network for Deep Image Deblurring.
* author: mingdeng_cao
* date: 2021/04/06 17:05
* last revised: None
************************************************ """

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from simdeblur.model.build import BACKBONE_REGISTRY


class ConvLSTMCell(nn.Module):
    def __init__(self, shape, ):
        """
        TODO
        """
        super().__init__()
        pass
    def forward(self, x):
        pass


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size//2)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return out + x


class InBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, kernel_size=5, num_resblocks=3):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, inner_channels, kernel_size, 1, kernel_size//2)
        self.resblocks = nn.Sequential(
            *[ResBlock(inner_channels, kernel_size) for _ in range(num_resblocks)]
        )
    
    def forward(self, x):
        out = F.relu(self.in_conv(x))
        out = self.resblocks(out)
        return out


class EBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=5, num_resblocks=3):
        super().__init__()
        self.down_conv = nn.Conv2d(in_channels, 2*in_channels, kernel_size, 2, kernel_size//2)

        self.resblocks = nn.Sequential(
            *[ResBlock(in_channels*2, kernel_size) for _ in range(num_resblocks)]
        )
    
    def forward(self, x):
        out = F.relu(self.down_conv(x))
        out = self.resblocks(out)
        return out


class DBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=5, num_resblocks=3):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResBlock(in_channels, kernel_size) for _ in range(num_resblocks)]
        )
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels//2, 4, 2, 1)

    def forward(self, x):
        out = self.resblocks(x)
        out = F.relu(self.up_conv(out))
        return out


class OutBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=5, num_resblocks=3):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResBlock(in_channels, kernel_size) for _ in range(num_resblocks)]
        )
        self.out_conv = nn.Conv2d(in_channels, 3, kernel_size, 1, kernel_size//2)

    def forward(self, x):
        out = self.resblocks(x)
        out = self.out_conv(out)
        return out


@BACKBONE_REGISTRY.register()
class SRN(nn.Module):
    def __init__(self, in_channels=6, inner_channels=32, num_levels=3, with_lstm=False):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.num_levels = num_levels

        self.in_block = InBlock(self.in_channels, self.inner_channels, kernel_size=5, num_resblocks=3)
        self.eblock1 = EBlock(self.inner_channels, kernel_size=5, num_resblocks=3)
        self.eblock2 = EBlock(self.inner_channels*2, kernel_size=5, num_resblocks=3)

        if with_lstm:
            """
            TODO later.
            """
        
        self.dblock1 = DBlock(self.inner_channels*4, kernel_size=5, num_resblocks=3)
        self.dblock2 = DBlock(self.inner_channels*2, kernel_size=5, num_resblocks=3)
        self.out_block = OutBlock(self.inner_channels, kernel_size=5, num_resblocks=3)
    
    def forward(self, x: list):
        """
        Args:
            x: input pyramid list [(b, c, h, w), (b, c, h//2, w//2), (b, c, h//4, w//4)]
        """
        out_imgs = [None] * self.num_levels
        for i in range(self.num_levels-1, -1, -1):
            # from coarse (smaller spatial size) to fine (larger spatial size)
            if i == self.num_levels - 1:
                input_imgs = torch.cat([x[-1], x[-1]], dim=1)
            else:
                upsampled_img = F.interpolate(out_imgs[i+1], scale_factor=2)
                input_imgs = torch.cat([x[i], upsampled_img], dim=1)
            in_block_out = self.in_block(input_imgs)
            eblock1_out = self.eblock1(in_block_out)
            eblock2_out = self.eblock2(eblock1_out)

            dblock1_out = self.dblock1(eblock2_out)
            dblock2_out = self.dblock2(dblock1_out + eblock1_out)
            out = self.out_block(dblock2_out + in_block_out)
            out_imgs[i] = out

        return out_imgs
        
