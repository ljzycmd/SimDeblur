import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        # self.args = args
        n_feats = args.n_feats
        kernel_size = args.kernel_size

        def conv(kernel_size, in_channel, n_feats, stride, pad=None):
            if pad is None:
                pad = (kernel_size-1)//2

            return nn.Conv2d(in_channel, n_feats, kernel_size, stride=stride, padding=pad, bias=False)

        self.conv_layers = nn.ModuleList([
            conv(kernel_size, 3,         n_feats//2, 1),    # 256
            conv(kernel_size, n_feats//2, n_feats//2, 2),   # 128
            conv(kernel_size, n_feats//2, n_feats,   1),
            conv(kernel_size, n_feats,   n_feats,   2),     # 64
            conv(kernel_size, n_feats,   n_feats*2, 1),
            conv(kernel_size, n_feats*2, n_feats*2, 4),     # 16
            conv(kernel_size, n_feats*2, n_feats*4, 1),
            conv(kernel_size, n_feats*4, n_feats*4, 4),     # 4
            conv(kernel_size, n_feats*4, n_feats*8, 1),
            conv(4,           n_feats*8, n_feats*8, 4, 0),  # 1
        ])

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dense = nn.Conv2d(n_feats*8, 1, 1, bias=False)

    def forward(self, x):

        for layer in self.conv_layers:
            x = self.act(layer(x))

        x = self.dense(x)

        return x

