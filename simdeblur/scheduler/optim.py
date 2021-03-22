"""
************************************************
* fileName: optim.py
* desc: 
* author: cmd
* date: 2021/03/02 16:59
************************************************
"""


from torch.optim import Adam, SGD

from .build import OPTIMIZER_REGISTRY

# Register the optimizer
OPTIMIZER_REGISTRY.register(Adam)
OPTIMIZER_REGISTRY.register(SGD)