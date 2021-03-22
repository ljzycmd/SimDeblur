# CMD

import torch
import torch.nn as nn
import torch.nn.functional as F 

from ..build import build_backbone, build_loss
from ..build import META_ARCH_REGISTRY


META_ARCH_REGISTRY.register()
class SingleScalePlainCNN(nn.Module):
    def __init__(self, cfg):
        # construct all modules
        self.model = build_backbone(cfg.model)
        self.criterion = build_loss(cfg.loss)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def preprocess(self, batch_data):
        # transform for input
        pass

    def postprocess(self, batch_data, model_outputs):
        # 
        pass
    
    def forward(self, batch_data):
        """
        save the outputs and calculate the loss for backward.
        
        Return
        a dict contains model outputs and the calculated loss.
        """
        input_frames = self.batch_data["input_frames"].to(self.device)
        gt_frames = self.batch_data["gt_frames"].to(self.device).flatten(0, 1)

        model_outputs = self.model(input_frames)
        if model_outputs.dim() == 5:
            model_outputs = model_outputs.flatten(0, 1)
        loss = self.criterion(gt_frames, model_outputs)
        
        return {"model_outputs": model_outputs,
                "loss": loss,
                }
