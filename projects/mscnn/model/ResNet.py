import torch.nn as nn

from . import common


class ResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_feats=None, kernel_size=None, num_resblocks=None, mean_shift=True):
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_feats = num_feats
        self.kernel_size = kernel_size
        self.num_resblocks = num_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = 1.0
        self.mean = self.rgb_range / 2

        modules = []
        modules.append(nn.Conv2d(self.in_channels, self.num_feats, self.kernel_size, stride=1, padding=self.kernel_size//2))
        for _ in range(self.num_resblocks):
            modules.append(common.ResBlock(self.num_feats, self.kernel_size))
        modules.append(common.default_conv(self.num_feats, self.out_channels, self.kernel_size))

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        if self.mean_shift:
            x = x - self.mean

        output = self.body(x)

        if self.mean_shift:
            output = output + self.mean

        return output

