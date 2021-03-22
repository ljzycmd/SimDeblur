from torch.nn import MSELoss, L1Loss
from .loss import CharbonnierLoss

from ..build import LOSS_REGISTRY

# Register the MSELoss L1Loss
LOSS_REGISTRY.register(MSELoss)
LOSS_REGISTRY.register(L1Loss)


__all__ = [k for k in globals().keys() if not k.startswith("_")]