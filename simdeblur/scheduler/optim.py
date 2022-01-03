""" ************************************************
* fileName: optim.py
* desc: optimizers
* author: mingdeng_cao
* date: 2021/12/06 15:24
* last revised: None
************************************************ """


from torch.optim import Adam, SGD, AdamW

from .build import OPTIMIZER_REGISTRY

# Register the optimizer
OPTIMIZER_REGISTRY.register(Adam)
OPTIMIZER_REGISTRY.register(SGD)
OPTIMIZER_REGISTRY.register(AdamW)
