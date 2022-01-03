import os
import logging
import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from ...utils import dist_utils

from ..build import build_backbone, build_loss

import logging
from ..utils import print_model_params

logger = logging.getLogger("simdeblur")

class BaseArch():
    """
    The base architecture of different model architectures.
    The classes inherit this base class are used to adapt different inputs, losses, etc.
    """
    def __init__(self) -> None:
        self.model = None
        self.criterion = None

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    def update_params(self):
        """
        This is a the key training loop of SimDeblur, indicating the training strategy of each model.
        """
        raise NotImplementedError

    def __call__(self, *args):
        return self.model(*args)

    def load_ckpt(self, ckpt, **kwargs):
        """
        Args:
            ckpt: a parameter dict
        """
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

    def generate_ckpt(self):
        """
        generate a dict containing model's parameters to be saved
        """
        if dist_utils.get_world_size() > 1:
            model_ckpt = self.model.module.state_dict()
        else:
            model_ckpt = self.model.state_dict()
        return {"model": model_ckpt}

    def build_model(self, cfg):
        """
        build a backbone model
        TODO: re-write these by dist_utils.py
        """
        model = build_backbone(cfg.model)
        if cfg.args.gpus > 1:
            rank = cfg.args.local_rank
            model = nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[rank], output_device=rank)
        if cfg.args.local_rank == 0:
            logger.info("Model:\n{}".format(model))
            print_model_params(model)

        return model

    @classmethod
    def build_losses(cls, loss_cfg):
        """
        build all losses and reture a loss dict
        """
        criterion_cfg = loss_cfg.get("criterion")
        weights_cfg = loss_cfg.get("weights")

        criterions = OrderedDict()
        weights = OrderedDict()
        if isinstance(criterion_cfg, list):
            assert len(criterion_cfg) == len(
                weights_cfg), "The length of criterions and weights in config file should be same!"
            for loss_item, loss_weight in zip(criterion_cfg, weights_cfg):
                criterions[loss_item.name] = build_loss(loss_item)
                weights[loss_item.name] = loss_weight
        else:
            criterions[criterion_cfg.name] = build_loss(criterion_cfg)
            weights[criterion_cfg.name] = 1.0
        if dist_utils.get_local_rank() == 0:
            logger.info("Loss items: ")
            for k in criterions.keys():
                logger.info(f"    {k}, weight: {weights[k]}")

        return criterions, weights

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
