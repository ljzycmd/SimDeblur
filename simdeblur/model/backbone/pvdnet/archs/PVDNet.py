import torch
import torch.nn as nn
import collections
import numpy as np

class PVDNetBackbone(nn.Module):
    def __init__(self, PV_input_dim):
        super(PVDNetBackbone, self).__init__()
        self.PV_dec = nn.Sequential(
            nn.Conv2d(PV_input_dim, 64, 3, stride=1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding = 1),
            nn.ReLU(),
        )
        self.img_dec = nn.Sequential(
            nn.Conv2d(9, 32, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride = 1, padding = 1),
            nn.ReLU(),
        )

        self.d0 = nn.Conv2d(96, 64, 5, stride = 1, padding = 2)
        self.d1 = nn.Sequential(
            nn.Conv2d(64, 64, 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride = 1, padding = 2),
        )
        self.temp = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, stride = 1, padding = 2),
            nn.ReLU(),
        )

        # for i in np.arange(12):
            # exec('self.RB{} = nn.Sequential(\
            #     nn.Conv2d(128, 128, 3, stride = 1, padding = 1),\
            #     nn.ReLU(),\
            #     nn.Conv2d(128, 128, 3, stride = 1, padding = 1),\
            #     nn.ReLU(),\
            # )'.format(i))
        self.RBs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
                nn.ReLU()) for i in range(12)
            ])

        self.RB_end = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
            nn.ReLU(),
        )

        self.dconv1 = nn.ConvTranspose2d(128, 64, 4, stride = 2, padding = 1)
        self.dconv1_end = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride = 1, padding = 2),
            nn.ReLU(),
        )

        self.dconv2 = nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1)
        self.dconv2_end = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride = 1, padding = 2),
            nn.ReLU(),
        )
        self.end = nn.Conv2d(64, 3, 5, stride = 1, padding = 2)

    def forward(self, PV, I_prev, I_curr, I_next):
        n_dec = self.PV_dec(PV)
        n_img = self.img_dec(torch.cat((I_curr, I_prev, I_next), axis = 1))
        n = torch.cat((n_dec, n_img), axis = 1)

        d0 = self.d0(n)
        d1 = self.d1(d0)
        temp = self.temp(d1)

        n = temp.clone()
        for i in range(12):
            nn = self.RBs[i](n)
            n = n + nn

        n = self.RB_end(n)
        n = n + temp

        n = self.dconv1_end((self.dconv1(n) + d1))
        n = self.dconv2_end((self.dconv2(n) + d0))
        n = self.end(n)
        n = I_curr + n 

        return n


