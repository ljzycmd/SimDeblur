# CMD


import torch
import torch.nn as nn
import torch.nn.functional as F 

from ..build import build_backbone, build_loss


class DeblurMetaArch:
    def __init__(self, cfg):
        # construct all modules
        self.model = build_backbone(cfg.model)
        self.criterion = build_loss(cfg.loss)
    
    def preprocess(self, batch_data):
        pass

    def postprocess(self, **kwargs):
        pass