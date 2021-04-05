import math

import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)

def default_norm(n_feats):
    return nn.BatchNorm2d(n_feats)

def default_act():
    return nn.ReLU(True)

def empty_h(x, n_feats):
    '''
        create an empty hidden state

        input
            x:      B x T x 3 x H x W

        output
            h:      B x C x H/4 x W/4
    '''
    b = x.size(0)
    h, w = x.size()[-2:]
    return x.new_zeros((b, n_feats, h//4, w//4))

class Normalization(nn.Conv2d):
    """Normalize input tensor value with convolutional layer"""
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        super(Normalization, self).__init__(3, 3, kernel_size=1)
        tensor_mean = torch.Tensor(mean)
        tensor_inv_std = torch.Tensor(std).reciprocal()

        self.weight.data = torch.eye(3).mul(tensor_inv_std).view(3, 3, 1, 1)
        self.bias.data = torch.Tensor(-tensor_mean.mul(tensor_inv_std))

        for params in self.parameters():
            params.requires_grad = False

class BasicBlock(nn.Sequential):
    """Convolution layer + Activation layer"""
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act):

        modules = []
        modules.append(
            conv(in_channels, out_channels, kernel_size, bias=bias))
        if norm: modules.append(norm(out_channels))
        if act: modules.append(act())

        super(BasicBlock, self).__init__(*modules)

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act):

        super(ResBlock, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm: modules.append(norm(n_feats))
            if act and i == 0: modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class ResBlock_mobile(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act, dropout=False):

        super(ResBlock_mobile, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=False, groups=n_feats))
            modules.append(conv(n_feats, n_feats, 1, bias=False))
            if dropout and i == 0: modules.append(nn.Dropout2d(dropout))
            if norm: modules.append(norm(n_feats))
            if act and i == 0: modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(
        self, scale, n_feats, bias=True,
        conv=default_conv, norm=False, act=False):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feats, 4 * n_feats, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if norm: modules.append(norm(n_feats))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feats, 9 * n_feats, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if norm: modules.append(norm(n_feats))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)

# Only support 1 / 2
class PixelSort(nn.Module):
    """The inverse operation of PixelShuffle
    Reduces the spatial resolution, increasing the number of channels.
    Currently, scale 0.5 is supported only.
    Later, torch.nn.functional.pixel_sort may be implemented.
    Reference:
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/modules/pixelshuffle.html#PixelShuffle
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/functional.html#pixel_shuffle
    """
    def __init__(self, upscale_factor=0.5):
        super(PixelSort, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, 2, 2, h // 2, w // 2)
        x = x.permute(0, 1, 5, 3, 2, 4).contiguous()
        x = x.view(b, 4 * c, h // 2, w // 2)

        return x

class Downsampler(nn.Sequential):
    def __init__(
        self, scale, n_feats, bias=True,
        conv=default_conv, norm=False, act=False):

        modules = []
        if scale == 0.5:
            modules.append(PixelSort())
            modules.append(conv(4 * n_feats, n_feats, 3, bias))
            if norm: modules.append(norm(n_feats))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Downsampler, self).__init__(*modules)

