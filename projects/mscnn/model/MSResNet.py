import torch
import torch.nn as nn

from . import common
from .ResNet import ResNet

from simdeblur.model.build import BACKBONE_REGISTRY

class conv_end(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, ratio=2):
        super(conv_end, self).__init__()

        modules = [
            common.default_conv(in_channels, out_channels, kernel_size),
            nn.PixelShuffle(ratio)
        ]

        self.uppath = nn.Sequential(*modules)

    def forward(self, x):
        return self.uppath(x)


@BACKBONE_REGISTRY.register()
class MSResNet(nn.Module):
    def __init__(self, in_channels, num_feats=64, out_channels=3, num_resblocks=19, kernel_size=5, num_scales=3):
        super(MSResNet, self).__init__()
        # the max value of all pixels
        self.rgb_range = 1.0 
        self.mean = self.rgb_range / 2

        self.num_resblocks = num_resblocks
        self.num_feats = num_feats
        self.kernel_size = kernel_size

        self.num_scales = num_scales

        self.body_models = nn.ModuleList([
            ResNet(3, 3, self.num_feats, self.kernel_size, self.num_resblocks, mean_shift=False),
        ])
        for _ in range(1, self.num_scales):
            self.body_models.insert(0, ResNet(6, 3, self.num_feats, self.kernel_size, self.num_resblocks, mean_shift=False))

        self.conv_end_models = nn.ModuleList([None])
        for _ in range(1, self.num_scales):
            self.conv_end_models += [conv_end(3, 12)]

    def forward(self, input_pyramid):
        scales = range(self.num_scales-1, -1, -1)    # 0: fine, 2: coarse

        for s in scales:
            input_pyramid[s] = input_pyramid[s] - self.mean

        output_pyramid = [None] * self.num_scales

        input_s = input_pyramid[-1]
        for s in scales:    # [2, 1, 0]
            output_pyramid[s] = self.body_models[s](input_s)
            if s > 0:
                up_feat = self.conv_end_models[s](output_pyramid[s])
                input_s = torch.cat((input_pyramid[s-1], up_feat), 1)

        for s in scales:
            output_pyramid[s] = output_pyramid[s] + self.mean

        return output_pyramid
