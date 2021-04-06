""" ************************************************
* fileName: srn_trainer.py
* desc: 
* author: mingdeng_cao
* date: 2021/04/07 00:30
* last revised: None
************************************************ """


import torch
import torch.nn as nn 
import torch.nn.functional as F 

# Register the model
from srn import SRN

from simdeblur.model import build_backbone
from simdeblur.engine.trainer import Trainer


# reconstruct the trainer because of that there are 3 scales images for input.
class SRNTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_scales = cfg.model.num_levels

    def preprocess(self, batch_data):
        input_frames = batch_data["input_frames"].to(self.device).flatten(0, 1)
        input_pyramid = [input_frames]
        for s in range(1, self.num_scales):
            input_pyramid.append(F.interpolate(input_frames, scale_factor=1/(2**s)))
        return input_pyramid

    def calculate_loss(self, batch_data, model_outputs):
        gt_frames = batch_data["gt_frames"].to(self.device).flatten(0, 1)
        gt_pyramid = [gt_frames]
        for s in range(1, self.num_scales):
            gt_pyramid.append(F.interpolate(gt_frames, scale_factor=1/2**s))
        
        loss = 0
        for i, (gt_i, outputs_i) in enumerate(zip(gt_pyramid, model_outputs)):
            loss += self.criterion(gt_i, outputs_i)
        
        # only save the finer output for calculating psnr and ssim metric.
        self.outputs = self.outputs[0]
        
        return loss