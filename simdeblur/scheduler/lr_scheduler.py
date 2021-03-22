"""
************************************************
* fileName: lr_scheduler.py
* desc: 
* author: cmd
* date: 2021/03/02 17:01
************************************************
"""


from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR

from .build import LR_SCHEDULER_REGISTRY


# Register the lr_scheduler
LR_SCHEDULER_REGISTRY.register(MultiStepLR)
LR_SCHEDULER_REGISTRY.register(CosineAnnealingLR)
# LR_SCHEDULER_REGISTRY.register(LambdaLR)