""" ************************************************
* fileName: lr_scheduler.py
* desc: The learning rate adjusting scheduler.
* author: mingdeng_cao
* date: 2021/03/02 19:17
* last revised:
    2021.07.20, add the warmup multi step lr_scheduler
************************************************ """


import math

import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR, ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler

from .build import LR_SCHEDULER_REGISTRY


# Register the lr_scheduler
LR_SCHEDULER_REGISTRY.register(MultiStepLR)
LR_SCHEDULER_REGISTRY.register(CosineAnnealingLR)
LR_SCHEDULER_REGISTRY.register(LambdaLR)
LR_SCHEDULER_REGISTRY.register(ExponentialLR)


# def warm_up_with_multistep_lr(
#     epoch): return epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.1**len([m for m in args.milestones if m <= epoch])
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=warm_up_with_multistep_lr)


@LR_SCHEDULER_REGISTRY.register()
class WarmupMultiStepLR(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 milestones: list,
                 warmup_epochs=30,
                 warmup_init_lr=1e-6,
                 gamma=0.1,
                 last_epoch=-1
                 ):
        assert warmup_epochs > 1, "the warmup epochs should be larger than 1. "
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_init_lr = 1e-8 if warmup_init_lr is None else warmup_init_lr
        self.warmup_steplength = [
            (group["lr"] - warmup_init_lr) / (warmup_epochs - 1) for group in optimizer.param_groups]
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_init_lr + self.last_epoch * step_length for step_length in self.warmup_steplength]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


@LR_SCHEDULER_REGISTRY.register()
class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self,
                 optimizer,
                 T_period,
                 restarts=None,
                 weights=None,
                 eta_min=0.00000001,
                 last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


@LR_SCHEDULER_REGISTRY.register()
class WarmupCosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self,
                 optimizer,
                 T_period,
                 restarts=None,
                 weights=None,
                 warmup_epochs=10,
                 warmup_init_lr=1e-6,
                 eta_min=0.00000001,
                 last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_init_lr = warmup_init_lr
        self.warmup_steplength = [
            (group["lr"] - warmup_init_lr) / (warmup_epochs - 1) for group in optimizer.param_groups]
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(WarmupCosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_init_lr + self.last_epoch * step_length for step_length in self.warmup_steplength]
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
