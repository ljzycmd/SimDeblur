# CMD
# perceptual loss using vggnet with conv1_2, conv2_2, conv3_3 feature, before relu layer


import torch
import torch.nn as nn 
import torch.nn.functional as F 

from ..layers.vgg import VGG19

from ...build import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights = [1, 0.2, 0.04]):
        self.layer_weights = layer_weights
        self.vggnet = VGG19()
        
    def forward(self, img1, img2):
        features1 = self.vggnet(img1)
        features2 = self.vggnet(img2)

        loss = 0
        for feat1, feat2 in zip(features1, features2):
            assert feat1.shape == feat2.shape, "The input tensor should be in same shape!"
            loss += F.mse_loss(feat1, feat2)
        
        return loss