# CMD


import torch
import torch.nn as nn

from ..build import build_backbone, build_loss 
from ..build import DEBLUR_ARCHS_REGISTRY

import logging


@DEBLUR_ARCHS_REGISTRY.register()
class RNNVideoDeblurArch(nn.Module):
    def __init__(self, cfg):
        super(RNNVideoDeblurArch, self).__init__()
        self.cfg = cfg
    
    def _construct_arch(self):
        logging.info("Building the video deblurring architecture ...")
        self.backbone = build_backbone(self.cfg.model)
        self.criterion = build_loss(self.loss)

        logging.info(slef.backbone)

        logging.info(f"The {self.__name__} is builded!")
    
    def forward(self, input_frames, gt_frames):
        outputs = self.backbone(input_frames)
        loss = self.criterion(gt_frames, outputs)

        return {
            "outputs" : outputs,
            "loss" : loss,
            }
