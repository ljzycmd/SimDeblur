import torch
import torch.nn as nn
import collections
import numpy as np

class PVDNetLargeBackbone(nn.Module):
    def __init__(self, PV_input_dim):
        super(PVDNetLargeBackbone, self).__init__()
        PV_dec_ch = 64
        self.PV_dec = nn.Sequential(
            nn.Conv2d(PV_input_dim, PV_dec_ch, 3, stride=1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(PV_dec_ch, PV_dec_ch, 3, stride=1, padding = 1),
            nn.ReLU(),
        )
        img_dec_ch = 32
        self.img_dec = nn.Sequential(
            nn.Conv2d(9, img_dec_ch, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(img_dec_ch, img_dec_ch, 3, stride = 1, padding = 1),
            nn.ReLU(),
        )

        ch = 64
        res_ch = 192
        self.d0 = nn.Conv2d(PV_dec_ch + img_dec_ch, ch, 5, stride = 1, padding = 2)
        self.d1 = nn.Sequential(
            nn.Conv2d(ch, ch, 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 5, stride = 1, padding = 2),
        )
        self.temp = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(ch * 2, res_ch, 5, stride = 1, padding = 2),
            nn.ReLU(),
        )

        self.RB_num = 24
        self.RBs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(res_ch, res_ch, 3, stride = 1, padding = 1),
                nn.ReLU(),
                nn.Conv2d(res_ch, res_ch, 3, stride = 1, padding = 1),
                nn.ReLU()) for i in range(self.RB_num)
            ])

        self.RB_end = nn.Sequential(
            nn.Conv2d(res_ch, res_ch, 3, stride = 1, padding = 1),
            nn.ReLU(),
        )

        self.dconv1 = nn.ConvTranspose2d(res_ch, ch, 4, stride = 2, padding = 1)
        self.dconv1_end = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch, ch, 5, stride = 1, padding = 2),
            nn.ReLU(),
        )

        self.dconv2 = nn.ConvTranspose2d(ch, ch, 4, stride = 2, padding = 1)
        self.dconv2_end = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch, ch, 5, stride = 1, padding = 2),
            nn.ReLU(),
        )
        self.end = nn.Conv2d(ch, 3, 5, stride = 1, padding = 2)

    def forward(self, PV, I_prev, I_curr, I_next):
        n_dec = self.PV_dec(PV)
        n_img = self.img_dec(torch.cat((I_curr, I_prev, I_next), axis = 1))
        n = torch.cat((n_dec, n_img), axis = 1)

        d0 = self.d0(n)
        d1 = self.d1(d0)
        temp = self.temp(d1)

        n = temp.clone()
        for i in range(self.RB_num):
            nn = self.RBs[i](n)
            n = n + nn

        n = self.RB_end(n)
        n = n + temp

        n = self.dconv1_end((self.dconv1(n) + d1))
        n = self.dconv2_end((self.dconv2(n) + d0))
        n = self.end(n)
        n = I_curr + n 

        return n


