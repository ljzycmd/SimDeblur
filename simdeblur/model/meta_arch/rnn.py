""" ************************************************
* fileName: rnn.py
* desc: The recurrent video processing architecture
* author: mingdeng_cao
* date: 2021/12/10 14:13
* last revised: None
************************************************ """


import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plain_cnn import SingleScalePlainCNN

from ..build import META_ARCH_REGISTRY

import logging
logger = logging.getLogger("simdeblur")


@META_ARCH_REGISTRY.register()
class PVDNetArch(SingleScalePlainCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def preprocess(self, batch_data):
        """
        prepare for input, different model archs needs different inputs.
        """
        batch_data["gt_frames"] = batch_data["gt_frames"][:, 1:-1]
        batch_data["gt_names"] = batch_data["gt_names"][1:-1]
        return batch_data["input_frames"].to(self.device) * 2 - 1
    
    def postprocess(self, outputs):
        outputs = (outputs + 1) / 2.
        if outputs.dim() == 5:
            return outputs.flatten(0, 1)
        return outputs
    
    def inference(self, input_data):
        B, N, C, H, W = input_data.shape
        # recurrent strtegy
        outputs_list = []
        prev_deblurred = input_data[:, 0]
        for i in range(1, N-1):
            model_outputs = self.model(input_data[:, i-1:i+2], prev_deblurred)  # (B, C, H, W)
            prev_deblurred = model_outputs.detach()
            outputs_list.append(prev_deblurred)

        return torch.stack(outputs_list, dim=1)

    def update_params(self, batch_data, optimizer):
        input_data = self.preprocess(batch_data)
        gt_frames = batch_data["gt_frames"].to(self.device)
        B, N, C, H, W = input_data.shape
        # recurrent strtegy
        outputs_list = []
        loss_total = 0.

        prev_deblurred = input_data[:, 0]
        for i in range(1, N-1):
            model_outputs = self.model(input_data[:, i-1:i+2], prev_deblurred)  # (B, C, H, W)
            loss = self.calculate_loss(
            self.criterion, self.criterion_weights, gt_frames[:, i-1], model_outputs)

            # 2 optimize model parameters: a) zero_grad, b) backward, c) update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss
            outputs_list.append(model_outputs.detach())

            # update next inputs
            prev_deblurred = model_outputs.detach()

        return {
            "results" : torch.stack(outputs_list, dim=1),
            "loss" : {"loss": loss_total},
        }


@META_ARCH_REGISTRY.register()
class STFANRNNArch(SingleScalePlainCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def inference(self, input_data):
        B, N, C, H, W = input_data.shape
        # recurrent strtegy
        outputs_list = []
        out_img_last = input_data[:, 0]
        blur_img_last = input_data[:, 0]
        hidden_feats_last = None
        for i in range(N):
            out_img, hidden_feats = self.model(input_data[:, i], blur_img_last, out_img_last, hidden_feats_last)  # (B, C, H, W)
            assert out_img is not None, "Out img is None!"

            outputs_list.append(out_img)

            # update next inputs
            out_img_last = out_img
            blur_img_last = input_data[:, i]
            hidden_feats_last = hidden_feats

        return torch.stack(outputs_list, dim=1)

    def update_params(self, batch_data, optimizer):
        input_data = self.preprocess(batch_data)
        gt_frames = batch_data["gt_frames"].to(self.device)
        B, N, C, H, W = input_data.shape
        # recurrent strtegy
        outputs_list = []
        loss_total = 0.
        out_img_last = input_data[:, 0]
        blur_img_last = input_data[:, 0]
        hidden_feats_last = None
        for i in range(N):
            out_img, hidden_feats = self.model(input_data[:, i], blur_img_last, out_img_last, hidden_feats_last)  # (B, C, H, W)
            assert out_img is not None, "Out img is None!"
            loss = self.calculate_loss(
                self.criterion, self.criterion_weights, gt_frames[:, i], out_img)
            loss_total += loss.detach()
            outputs_list.append(out_img)
            # 2 optimize model parameters: a) zero_grad, b) backward, c) update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update next inputs
            out_img_last = out_img.detach()
            blur_img_last = input_data[:, i]
            hidden_feats_last = hidden_feats.detach()

        return {
            "results" : torch.stack(outputs_list, dim=1),
            "loss" : {"loss": loss_total},
        }