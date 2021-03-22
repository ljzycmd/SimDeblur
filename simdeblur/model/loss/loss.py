# reference it on EDVR

import torch
import torch.nn as nn 
from ..build import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-6):
        """
        the original eps is 1e-12
        """
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return torch.sum(torch.sqrt((pred - target)**2 + self.eps)) / target.shape[0]