import torch

from ..utils import Registry


OPTIMIZER_REGISTRY = Registry("optimizer")
LR_SCHEDULER_REGISTRY = Registry("lr_scheduler")


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    optimizer = OPTIMIZER_REGISTRY.get(cfg.pop("name"))(model.parameters(), **cfg)
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    lr_scheduler = LR_SCHEDULER_REGISTRY.get(cfg.pop("name"))(optimizer, **cfg)
    return lr_scheduler