import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_arch import BaseArch

from ..build import META_ARCH_REGISTRY
from ..build import build_backbone
from ...scheduler.build import build_optimizer
from ...scheduler.build import build_lr_scheduler
from ...utils import dist_utils

logger = logging.getLogger("simdeblur")

@META_ARCH_REGISTRY.register()
class DeblurGANArch(BaseArch):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.build_generator(cfg).to(self.device)
        self.discriminator = self.build_discriminator(cfg).to(self.device)

        # Gan loss
        self.criterion_GAN = nn.MSELoss().to(self.device)
        self.criterion_content = nn.L1Loss().to(self.device)

    def preprocess(self, batch_data):
        """
        prepare for input, different model archs needs different inputs.
        """
        return batch_data["input_frames"].to(self.device).flatten(0, 1)

    def get_output_shape_D(self, input_shape=(256, 256), down_times=3):
        return [i // 2**down_times for i in input_shape]

    def update_params(self, batch_data, model_outputs, optimizer):
        gt_frames = batch_data["gt_frames"].to(self.device).flatten(0, 1)
        if model_outputs.dim() == 5:
            model_outputs = model_outputs.flatten(0, 1)  # (b*n, c, h, w)
        valid = torch.ones(
            gt_frames.shape[0], 1, *self.get_output_shape_D(gt_frames.shape[-2:])).to(self.device)
        fake = torch.zeros(
            gt_frames.shape[0], 1, *self.get_output_shape_D(gt_frames.shape[-2:])).to(self.device)

        # update D
        for i in range(self.cfg.schedule.update_d_period):
            optimizer["optimizer_d"].zero_grad()
            loss_d_real = self.criterion_GAN(
                self.discriminator(gt_frames), valid) * 0.5
            loss_d_real.backward()

            loss_d_fake = self.criterion_GAN(
                self.discriminator(model_outputs.data), fake) * 0.5
            loss_d_fake.backward()

            optimizer["optimizer_d"].step()

        # Update G
        optimizer["optimizer_g"].zero_grad()
        loss_GAN = self.criterion_GAN(self.discriminator(model_outputs), valid)
        loss_content = self.criterion_content(gt_frames, model_outputs)
        loss_G = loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optimizer["optimizer_g"].step()

        return {"loss_G": loss_G, "loss_D": loss_d_real + loss_d_fake}

    def build_scheduler(self):
        optimizer_g = build_optimizer(
            self.cfg.schedule.optimizer_g, self.model)
        optimizer_d = build_optimizer(
            self.cfg.schedule.optimizer_d, self.discriminator)

        lr_scheduler_g = build_lr_scheduler(
            self.cfg.schedule.lr_scheduler_g, optimizer_g)
        lr_scheduler_d = build_lr_scheduler(
            self.cfg.schedule.lr_scheduler_d, optimizer_d)

        return ({"optimizer_g": optimizer_g,
                "optimizer_d": optimizer_d},
                {"lr_scheduler_g": lr_scheduler_g,
                "lr_scheduler_d": lr_scheduler_d})

    @classmethod
    def build_generator(cls, cfg):
        """
        build a backbone generator for GAN architecture
        """
        model = build_backbone(cfg.model.cfg_g)
        if cfg.args.gpus > 1:
            rank = cfg.args.local_rank
            model = nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[rank], output_device=rank)
        if cfg.args.local_rank == 0:
            logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_discriminator(cls, cfg):
        """
        build a backbone generator for GAN architecture
        """
        model = build_backbone(cfg.model.cfg_d)
        if cfg.args.gpus > 1:
            rank = cfg.args.local_rank
            model = nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[rank], output_device=rank)
        if cfg.args.local_rank == 0:
            logger.info("Model:\n{}".format(model))
        return model

    def load_ckpt(self, ckpt, **kwargs):
        if "model" in ckpt.keys():
            model_ckpt = ckpt["model"]
            new_model_ckpt = OrderedDict()
            # strip `module.` prefix
            for k, v in model_ckpt.items():
                name = k[7:] if k.startswith("module.") else k
                new_model_ckpt[name] = v
            model_ckpt = new_model_ckpt

            if dist_utils.get_world_size() > 1:
                self.model.module.load_state_dict(model_ckpt, **kwargs)
            else:
                self.model.load_state_dict(model_ckpt, **kwargs)
            logger.info("Checkponit loaded to model successfully!")

        if "discriminator" in ckpt.keys():
            model_ckpt = ckpt["model_d"]
            new_model_ckpt = OrderedDict()
            # strip `module.` prefix
            for k, v in model_ckpt.items():
                name = k[7:] if k.startswith("module.") else k
                new_model_ckpt[name] = v
            model_ckpt = new_model_ckpt

            if dist_utils.get_world_size() > 1:
                self.model.module.load_state_dict(model_ckpt, **kwargs)
            else:
                self.model.load_state_dict(model_ckpt, **kwargs)
            logger.info("Checkponit loaded to Discriminator successfully!")

    def generate_ckpt(self):
        """
        generate a dict containing model's parameters to be saved
        """
        if dist_utils.get_world_size() > 1:
            model_ckpt = self.model.module.state_dict()
            model_d_ckpt = self.discriminator.module.state_dict()
        else:
            model_ckpt = self.model.state_dict()
            model_d_ckpt = self.discriminator.state_dict()

        return {"model": model_ckpt, "model_d": model_d_ckpt}
