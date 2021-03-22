"""
# _*_ coding: utf-8 _*_
# @Time    :   2021/02/09 00:47:19
# @FileName:   strcnn.py
# @Author  :   Minton Cao
# @Software:   VSCode
"""


import torch
import torch.nn as nn
import torch.nn.functional as F 

from ...build import BACKBONE_REGISTRY

class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            nn.BatchNorm2d(self.in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            nn.BatchNorm2d(self.in_channels)
        )
    
    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))

        return F.relu(self.conv2(out) + x)


class Encoder(nn.Module):
    def __init__(self, in_channels, inner_channels, blocks=4):
        """
        Args
        num_frames: 5
        in_channels: 3
        inner_channels: 64
        blocks: 4, residual blocks number
        """
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.blocks = blocks

        self.feature_extract = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inner_channels, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels, self.inner_channels // 2, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.feature_fuse = nn.Sequential(
            *[BasicBlock(self.inner_channels) for i in range(self.blocks)],
            nn.Conv2d(self.inner_channels, self.inner_channels, 3, 1, 1)
        )
    
    def forward(self, b_n, f_n_1=None):
        assert b_n.dim() == 4, "Input blurry frame should be in 4 dims!"

        feats = self.feature_extract(b_n)
        if f_n_1 is None:
            # f_n_1 = f_n
            f_n_1 = torch.zeros_like(feats)
        feats = torch.cat([feats, f_n_1], dim=1)
        
        h_n = self.feature_fuse(feats)

        return h_n
    

class DTB(nn.Module):
    """
    Dynamic temporal blending network
    """
    def __init__(self, in_channels, inner_channels):
        super(DTB, self).__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels

        self.conv = nn.Conv2d(self.in_channels, self.inner_channels, 5, 1, 2)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    
    def forward(self, h_n, h_n_1_b):
        """
        h_n_1_b: the blended feature map h_{n-1}
        """
        if h_n_1_b is None:
            h_n_1_b = torch.zeros_like(h_n)
        w = self.conv(torch.cat([h_n, h_n_1_b], dim=1))
        w = torch.abs(torch.tanh(w)) + self.beta
        w = torch.clamp(w, 0.0, 1.0)

        h_n_b = w * h_n + (1 - w) * h_n_1_b

        return h_n_b


class Decoder(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels=3, blocks=4):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.out_channels = out_channels
        self.blocks = blocks

        self.feature_fuse = nn.Sequential(
            *[BasicBlock(self.in_channels) for i in range(self.blocks)],
            nn.Conv2d(self.in_channels, self.inner_channels, 3, 1, 1)
        )

        self.feature_extract = nn.Sequential(
            nn.Conv2d(self.inner_channels, self.inner_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(self.inner_channels, self.inner_channels, 4, 2, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels, self.out_channels, 3, 1, 1)
        )
    
    def forward(self, h_n_b):
        feats = self.feature_fuse(h_n_b)
        f_n = self.feature_extract(feats)
        l_n = self.reconstruct(feats)

        return l_n, f_n


class STRCNNCell(nn.Module):
    def __init__(self, num_frames_encoded, in_channels, inner_channels, encoder_blocks, decoder_blocks, is_dtb=True):
        """
        num_frames_encoded = 2m + 1
        """
        super(STRCNNCell, self).__init__()
        self.num_frames_encoded = num_frames_encoded
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.is_dtb = is_dtb

        self.encoder = Encoder(self.num_frames_encoded*self.in_channels, self.inner_channels, blocks=self.encoder_blocks)
        self.dtb = DTB(self.inner_channels * 2, self.inner_channels) if self.is_dtb else None
        self.decoder = Decoder(self.inner_channels, self.inner_channels, out_channels=3, blocks=self.decoder_blocks)

    def forward(self, b_ns, f_n_1=None, h_n_1_b=None):
        assert b_ns.dim() == 5, "The blurry frames to be encoded should be in 5 dims!"
        # h_n
        feats = self.encoder(b_ns.flatten(1, 2), f_n_1)
        
        # temporal feature map blending module
        if self.is_dtb:
            # h_n_b
            h_n_b = self.dtb(feats, h_n_1_b)
            feats = h_n_b
        # latent frame l_n, spatial-temporal feature map f_n
        l_n, f_n = self.decoder(feats)

        return l_n, f_n, h_n_b


@BACKBONE_REGISTRY.register()
class STRCNN(nn.Module):
    def __init__(self, num_frames_encoded, in_channels, inner_channels, encoder_blocks, decoder_blocks, is_dtb=True):
        super(STRCNN, self).__init__()
        self.num_frames_encoded = num_frames_encoded
        # future and post frames number for reference frames b_n
        self.future_frames = self.past_frames = num_frames_encoded // 2
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.is_dtb = is_dtb

        self.strcnn_cell = STRCNNCell(num_frames_encoded, in_channels, inner_channels, encoder_blocks, decoder_blocks, is_dtb=True)
    
    def forward(self, x):
        assert x.dim() == 5, "The input tensor should be in 5 dims!"

        b, n, c, h, w = x.shape
        assert n >= self.num_frames_encoded, "Input frames should be larger than encoded frame nums."
        
        outputs = []
        f_n = h_n_b = None
        for i in range(n - self.num_frames_encoded + 1):
            l_n, f_n, h_n_b = self.strcnn_cell(x[:, i:i+self.num_frames_encoded], f_n, h_n_b)
            outputs.append(l_n)
        
        return torch.stack(outputs, dim=1) # 5 dims output tensor