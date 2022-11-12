""" ************************************************
* fileName: plain_cnn.py
* desc: SingleScalePlainCNN based module
* author: minton_cao
* last revised: None
* TODO: 1 replace the cfg.args.rank by the dist_utils functions.
************************************************ """

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_arch import BaseArch

from ..build import META_ARCH_REGISTRY

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim = (-d))
        r = torch.stack((t.real, t.imag), -1)
        return r
    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:,:,0], x[:,:,1]), dim = (-d))
        return t.real


@META_ARCH_REGISTRY.register()
class SingleScalePlainCNN(BaseArch):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.build_model(cfg).to(self.device)
        self.criterion, self.criterion_weights = self.build_losses(cfg.loss)
        self.criterion = {k: v.to(self.device) for k, v in self.criterion.items()}

    def preprocess(self, batch_data):
        """
        prepare for input, different model archs needs different inputs.
        """
        return batch_data["input_frames"].to(self.device)

    def postprocess(self, outputs):
        """
        transfer the outputs with 5 dims into 4 dims by flatten the batch and number frames.
        """
        if outputs.dim() == 5:
            return outputs.flatten(0, 1)
        return outputs

    def update_params(self, batch_data, optimizer):
        # forward to generate model results
        model_outputs = self.model(self.preprocess(batch_data))
        # 1 calculate losses
        gt_frames = batch_data["gt_frames"].to(self.device).flatten(0, 1)
        if model_outputs.dim() == 5:
            model_outputs = model_outputs.flatten(0, 1)  # (b*n, c, h, w)
        loss = self.calculate_loss(
            self.criterion, self.criterion_weights, gt_frames, model_outputs)

        # 2 optimize model parameters: a) zero_grad, b) backward, c) update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "results": model_outputs,
            "loss": {"loss": loss}
        }

    @classmethod
    def calculate_loss(cls, criterion, weights, gt_data, output_data):
        loss = 0.
        for key, cri in criterion.items():
            loss += cri(gt_data, output_data) * weights[key]
        return loss


@META_ARCH_REGISTRY.register()
class MultiScalePlainCNN(SingleScalePlainCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_scales = cfg.model.get("num_levels")
        print(cfg.model.get("num_scales"))
        if self.num_scales is None:
            self.num_scales = cfg.model.get("num_scales")
        assert self.num_scales, "Model config must contains a \'num_scales\' or \'num_levels\'" + \
            " property when using MultiScalePlainCNN architecture. "

    def preprocess(self, batch_data):
        input_frames = batch_data["input_frames"].to(self.device).flatten(0, 1)
        input_pyramid = [input_frames]
        # construct multi-scale inputs
        for s in range(1, self.num_scales):
            input_pyramid.append(F.interpolate(
                input_frames, scale_factor=1/(2**s)))
        return input_pyramid

    def postprocess(self, outputs):
        # only adopts the finest level output to calculate evaluation metrics
        return outputs[0]

    def update_params(self, batch_data, optimizer):
        # forward to generate model results
        model_outputs = self.model(self.preprocess(batch_data))

        # 1 calculate losses
        gt_frames = batch_data["gt_frames"].to(self.device).flatten(0, 1)
        gt_pyramid = [gt_frames]
        for s in range(1, self.num_scales):
            gt_pyramid.append(F.interpolate(gt_frames, scale_factor=1/2**s))

        loss = 0
        for i, (gt_i, outputs_i) in enumerate(zip(gt_pyramid, model_outputs)):
            loss += self.calculate_loss(
                self.criterion, self.criterion_weights, gt_i, outputs_i)

        # 2 optimize model parameters: a) zero_grad, b) backward, c) update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "results": model_outputs,
            "loss": {"loss": loss}
        }


@META_ARCH_REGISTRY.register()
class ESTRNNArch(SingleScalePlainCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

    def preprocess(self, batch_data):
        """
        prepare for input, different model archs needs different inputs.
        """
        batch_data["input_frames"] = batch_data["input_frames"] - 0.5
        return batch_data["input_frames"].to(self.device)

    def postprocess(self, outputs):
        """
        transfer the outputs with 5 dims into 4 dims by flatten the batch and number frames.
        """
        if outputs.dim() == 5:
            outputs = outputs.flatten(0, 1)
        outputs = (outputs + 0.5).clamp(0, 1)
        return outputs


@META_ARCH_REGISTRY.register()
class MIMOUnetArch(SingleScalePlainCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def postprocess(self, outputs):
        # only adopts the finest level output to calculate evaluation metrics
        return outputs[-1]

    def update_params(self, batch_data, optimizer):
        # forward to generate model results
        model_outputs = self.model(self.preprocess(batch_data))
        # 1 calculate losses
        gt_frames = batch_data["gt_frames"].to(self.device)
        if gt_frames.dim() == 5:
            gt_frames = gt_frames.flatten(0, 1)
        gt_frames_2 = F.interpolate(gt_frames, scale_factor=0.5, mode="bilinear")
        gt_frames_4 = F.interpolate(gt_frames, scale_factor=0.25, mode="bilinear")
        outputs_4, outputs_2, outputs_1 = model_outputs  # (B, C, H, W) multi-scale outpust

        loss = 0
        for i, (gt_i, output_i) in enumerate(
            zip((gt_frames, gt_frames_2, gt_frames_4), (outputs_1, outputs_2, outputs_4))):
            gt_fft = rfft(gt_i, 2)
            output_fft = rfft(output_i, 2)
            loss += self.calculate_loss(
                self.criterion, self.criterion_weights, gt_i, output_i)
            loss += self.calculate_loss(
                self.criterion, self.criterion_weights, gt_fft, output_fft)

        # 2 optimize model parameters: a) zero_grad, b) backward, c) update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "results": model_outputs,
            "loss": {"loss": loss}
        }