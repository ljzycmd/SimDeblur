import torch.nn as nn

from .common import ResBlock, default_conv

def encoder(in_channels, n_feats):
    """RGB / IR feature encoder
    """

    # in_channels == 1 or 3 or 4 or ....
    # After 1st conv, B x n_feats x H x W
    # After 2nd conv, B x 2n_feats x H/2 x W/2
    # After 3rd conv, B x 3n_feats x H/4 x W/4
    return nn.Sequential(
        nn.Conv2d(in_channels, 1 * n_feats, 5, stride=1, padding=2),
        nn.Conv2d(1 * n_feats, 2 * n_feats, 5, stride=2, padding=2),
        nn.Conv2d(2 * n_feats, 3 * n_feats, 5, stride=2, padding=2),
    )

def decoder(out_channels, n_feats):
    """RGB / IR / Depth decoder
    """
    # After 1st deconv, B x 2n_feats x H/2 x W/2
    # After 2nd deconv, B x n_feats x H x W
    # After 3rd conv, B x out_channels x H x W
    deconv_kargs = {'stride': 2, 'padding': 1, 'output_padding': 1}

    return nn.Sequential(
        nn.ConvTranspose2d(3 * n_feats, 2 * n_feats, 3, **deconv_kargs),
        nn.ConvTranspose2d(2 * n_feats, 1 * n_feats, 3, **deconv_kargs),
        nn.Conv2d(n_feats, out_channels, 5, stride=1, padding=2),
    )

# def ResNet(n_feats, in_channels=None, out_channels=None):
def ResNet(n_feats, kernel_size, n_blocks, in_channels=None, out_channels=None):
    """sequential ResNet
    """

    # if in_channels is None:
    #     in_channels = n_feats
    # if out_channels is None:
    #     out_channels = n_feats
    # # currently not implemented

    m = []

    if in_channels is not None:
        m += [default_conv(in_channels, n_feats, kernel_size)]

    m += [ResBlock(n_feats, 3)] * n_blocks

    if out_channels is not None:
        m += [default_conv(n_feats, out_channels, kernel_size)]


    return nn.Sequential(*m)

