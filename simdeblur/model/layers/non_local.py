import torch
import torch.nn as nn
import torch.nn.functional as F 


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inner_channels=None, sub_sample=False):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inner_channels = inner_channels

        self.sub_sample = sub_sample

        if self.inner_channels is None:
            self.inner_channels = self.in_channels
        
        # define the linear transformation using 1 \times 1 convolution.
        self.q_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inner_channels, 1, 1, 0),
        )

        self.k_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inner_channels, 1, 1, 0),
        )

        self.v_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inner_channels, 1, 1, 0),
        )

        self.out_conv = nn.Conv2d(self.inner_channels, self.in_channels, 1, 1, 0)

        if self.sub_sample:
            self.k_conv.add_module("down_sample", nn.MaxPool2d(2, 2))
            self.v_conv.add_module("down_sample", nn.MaxPool2d(2, 2))

    def forward(self, x):
        """
        x: (b, c, h, w)
        """
        assert x.dim() == 4, "Input tensor should be in 4 dims!"
        b, c, h, w = x.shape

        q = self.q_conv(x).flatten(2) # (b, inner_channels, h*w)
        k = self.k_conv(x).flatten(2)
        v = self.v_conv(x).flatten(2)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = q @ k
        attn = F.softmax(attn, -1)

        outputs = attn @ v
        outputs = outputs.transpose(1, 2).reshape(b, self.inner_channels, h, w)

        outputs = self.out_conv(outputs) # change the output channels to the input channels
        # residual
        outputs = outputs + x
        return outputs
        

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # if self.store_last_batch_nl_map:
        #     self.nl_map = f_div_C

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NonLocalBlockMH(nn.Module):
    def __init__(self, in_channels, nhead):
        super(NonLocalBlockMH, self).__init__()
        self.mhsa = nn.MultiheadAttention(in_channels, nhead, dropout=0)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(in_channels)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1) # (h*w, b, c)
        out = self.mhsa(x, x, x)[0]
        out += x
        out2 = self.ffn(out)
        out = self.norm(out2 + out)
        out = out.reshape(h, w, b, c).permute(2, 3, 0, 1)
        return out

if __name__ == "__main__":
    non_local = NonLocalBlock(512, 512)
    a = torch.ones(2, 512, 4, 4)
    print(non_local(a).shape)
