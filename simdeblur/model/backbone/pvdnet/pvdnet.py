import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

from .archs.liteFlowNet import LiteFlowNet
from .archs.PVDNet import PVDNetBackbone
from .archs.PVDNet_large import PVDNetLargeBackbone
from .pixel_volume import get_pixel_volume
from .utils import *

from simdeblur.model.build import BACKBONE_REGISTRY

def norm(inp):
    return (inp + 1.) / 2.


@BACKBONE_REGISTRY.register()
class PVDNet(nn.Module):
    def __init__(self,
        PV_ksize=5,
        fix_BIMNet=True,
        wi=1.0,
        large_model = False,
        bimnet_ckpt_path = None
    ):
        super(PVDNet, self).__init__()
        self.fix_BIMNet = fix_BIMNet
        self.wi=wi
        self.bimnet_ckpt_path = bimnet_ckpt_path
        # BIMNet
        self.BIMNet = LiteFlowNet()

        # PVDNet backbone
        if large_model:
            self.PVDNet = PVDNetLargeBackbone(PV_ksize ** 2)
        else:
            self.PVDNet = PVDNetBackbone(PV_ksize ** 2)

        self.init()

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = self.wi)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)

    def init(self):
        if self.fix_BIMNet:
            ckpt = torch.load(self.bimnet_ckpt_path)
            print('\t\tBIMNet loaded: ', self.BIMNet.load_state_dict(ckpt))
            for param in self.BIMNet.parameters():
                param.requires_grad_(False)
        else:
            self.BIMNet.apply(self.weights_init)

        self.PVDNet.apply(self.weights_init)


    def forward(self, input_imgs, prev_deblurred):
        _, _, h, w = input_imgs[:, 1].size()

        refine_h = h - h % 32 # 32:mod crop for liteflownet
        refine_w = w - w % 32
        I_curr_refined = input_imgs[:, 1, :, 0 : refine_h, 0 : refine_w]
        I_prev_refined = input_imgs[:, 1, :, 0 : refine_h, 0 : refine_w]

        outs = collections.OrderedDict()
        ## BIMNet
        w_bb = upsample(self.BIMNet(norm(I_curr_refined ), norm(I_prev_refined )), refine_h, refine_w)
        if refine_h != h or refine_w != w:
            w_bb = F.pad(w_bb,(0, w - refine_w, 0, h - refine_h, 0, 0, 0, 0))

        ## PVDNet
        outs['PV'] = get_pixel_volume(prev_deblurred, w_bb, input_imgs[:, 1], h, w) # C: 5 X 5 (gray) X 3 (color)
        outs['result'] = self.PVDNet(outs['PV'], input_imgs[:, 0], input_imgs[:, 1], input_imgs[:, 2])

        return outs["result"]
