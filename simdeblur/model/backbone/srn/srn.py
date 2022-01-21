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


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()

        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2  # in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state  # hidden and c are images with several channels
        combined = torch.cat((input, hidden), 1)  # oncatenate in the channels
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, x):
        batch_size = x.shape[0]
        shape = x.shape[-2:]
        return (torch.zeros(batch_size, self.num_features, shape[0]//4, shape[1]//4).to(x),
                torch.zeros(batch_size, self.num_features, shape[0]//4, shape[1]//4).to(x))


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
    def __init__(self, in_channels=6, inner_channels=32, num_levels=3, with_lstm=True):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.num_levels = num_levels

        self.in_block = InBlock(self.in_channels, self.inner_channels, kernel_size=5, num_resblocks=3)
        self.eblock1 = EBlock(self.inner_channels, kernel_size=5, num_resblocks=3)
        self.eblock2 = EBlock(self.inner_channels*2, kernel_size=5, num_resblocks=3)

        self.conv_lstm = CLSTM_cell(inner_channels*4, inner_channels*4, 5)

        self.dblock1 = DBlock(self.inner_channels*4, kernel_size=5, num_resblocks=3)
        self.dblock2 = DBlock(self.inner_channels*2, kernel_size=5, num_resblocks=3)
        self.out_block = OutBlock(self.inner_channels, kernel_size=5, num_resblocks=3)

    def forward(self, x: list):
        """
        Args:
            x: input pyramid list [(b, c, h, w), (b, c, h//2, w//2), (b, c, h//4, w//4)]
        """
        out_imgs = [None] * self.num_levels
        h, c = self.conv_lstm.init_hidden(x[-1])
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
            h, c = self.conv_lstm(eblock2_out, (h, c))
            dblock1_out = self.dblock1(c)
            dblock2_out = self.dblock2(dblock1_out + eblock1_out)
            out = self.out_block(dblock2_out + in_block_out)
            out_imgs[i] = out
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            c = F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)

        return out_imgs
