import torch
import torch.nn as nn
# from detectron2.layers import ModulatedDeformConv
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


def deconv4x4(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)


def deconv5x5(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, output_padding=1)


def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


def make_blocks(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_chs, activation='relu', batch_norm=False):
        super(ResBlock, self).__init__()
        op = []
        for i in range(2):
            op.append(conv3x3(in_chs, in_chs))
            if batch_norm:
                op.append(nn.BatchNorm2d(in_chs))
            if i == 0:
                op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        return out


class DenseLayer(nn.Module):
    """
    Dense layer for residual dense block
    """

    def __init__(self, in_chs, growth_rate, activation='relu'):
        super(DenseLayer, self).__init__()
        self.conv = conv3x3(in_chs, growth_rate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class ResDenseBlock(nn.Module):
    """
    Residual Dense Block
    """

    def __init__(self, in_chs, growth_rate, num_layer, activation='relu'):
        super(ResDenseBlock, self).__init__()
        in_chs_acc = in_chs
        op = []
        for i in range(num_layer):
            op.append(DenseLayer(in_chs_acc, growth_rate, activation))
            in_chs_acc += growth_rate
        self.dense_layers = nn.Sequential(*op)
        self.conv1x1 = conv1x1(in_chs_acc, in_chs)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


class RDNet(nn.Module):
    """
    Middle network of residual dense blocks
    """

    def __init__(self, in_chs, growth_rate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(ResDenseBlock(in_chs, growth_rate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_chs, in_chs)
        self.conv3x3 = conv3x3(in_chs, in_chs)
        self.act = actFunc(activation)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.act(self.conv1x1(out))
        out = self.act(self.conv3x3(out))
        return out


class SpaceToDepth(nn.Module):
    """
    Pixel Unshuffle
    """

    def __init__(self, block_size=4):
        super().__init__()
        assert block_size in {2, 4}, "Space2Depth only supports blocks size = 4 or 2"
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.block_size
        x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * S * S, H // S, W // S)  # (N, C*bs^2, H//bs, W//bs)
        return x

    def extra_repr(self):
        return f"block_size={self.block_size}"


# based on https://github.com/rogertrullo/pytorch_convlstm
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

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())

# class ModulatedDeformLayer(nn.Module):
#     """
#     Modulated Deformable Convolution (v2)
#     """
#
#     def __init__(self, in_chs, out_chs, kernel_size=3, padding=1, stride=1, deformable_groups=1, activation='relu'):
#         super(ModulatedDeformLayer, self).__init__()
#         assert isinstance(kernel_size, (int, list, tuple))
#         self.deform_offset = conv3x3(in_chs, (3 * kernel_size ** 2) * deformable_groups)
#         # self.act = actFunc(activation)
#         self.deform = ModulatedDeformConv(
#             in_chs,
#             out_chs,
#             kernel_size,
#             stride=padding,
#             padding=stride,
#             deformable_groups=deformable_groups
#         )
#
#     def forward(self, x, feat):
#         offset_mask = self.deform_offset(feat)
#         offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
#         offset = torch.cat((offset_x, offset_y), dim=1)
#         mask = mask.sigmoid()
#         out = self.deform(x, offset, mask)
#         # out = self.act(out)
#         return out
#
#
# @torch.no_grad()
# def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
#     """Initialize network weights.
#
#     Args:
#         module_list (list[nn.Module] | nn.Module): Modules to be initialized.
#         scale (float): Scale initialized weights, especially for residual
#             blocks. Default: 1.
#         bias_fill (float): The value to fill bias. Default: 0
#         kwargs (dict): Other arguments for initialization function.
#     """
#     if not isinstance(module_list, list):
#         module_list = [module_list]
#     for module in module_list:
#         for m in module.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, **kwargs)
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.fill_(bias_fill)
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight, **kwargs)
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.fill_(bias_fill)
#             elif isinstance(m, _BatchNorm):
#                 init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     m.bias.data.fill_(bias_fill)
#
#
# class ResidualBlockNoBN(nn.Module):
#     """Residual block without BN.
#
#     It has a style of:
#         ---Conv-ReLU-Conv-+-
#          |________________|
#
#     Args:
#         num_feat (int): Channel number of intermediate features.
#             Default: 64.
#         res_scale (float): Residual scale. Default: 1.
#         pytorch_init (bool): If set to True, use pytorch default init,
#             otherwise, use default_init_weights. Default: False.
#     """
#
#     def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
#         super(ResidualBlockNoBN, self).__init__()
#         self.res_scale = res_scale
#         self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
#         self.relu = nn.ReLU(inplace=True)
#
#         if not pytorch_init:
#             default_init_weights([self.conv1, self.conv2], 0.1)
#
#     def forward(self, x):
#         identity = x
#         out = self.conv2(self.relu(self.conv1(x)))
#         return identity + out * self.res_scale
