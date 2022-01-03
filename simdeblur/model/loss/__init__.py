from torch.nn import MSELoss, L1Loss
from .charbonnier_loss import CharbonnierLoss
from .perceptual_loss import PerceptualLossVGG19
from .hem_loss import HEM

from ..build import LOSS_REGISTRY

# Register the MSELoss L1Loss
LOSS_REGISTRY.register(MSELoss)
LOSS_REGISTRY.register(L1Loss)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
