# CMD

import torch

from utils import Registry


OPTIMIZER_REGISTRY = Registry("optimizer")
LR_SCHEDULER_REGISTRY = Registry("lr_scheduler")


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    optimizer = OPTIMIZER_REGISTRY.get(cfg.schedule.optimizer.pop("name"))(model.parameters(), **cfg.schedule.optimizer)
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    lr_scheduler = LR_SCHEDULER_REGISTRY.get(cfg.schedule.lr_scheduler.pop("name"))(optimizer, **cfg.schedule.lr_scheduler)
    return lr_scheduler