""" ************************************************
* fileName: perceptual_loss.py
* desc: Perceptual loss using vggnet with conv1_2, conv2_2, conv3_3 feature,
        before relu layer. 
* author: mingdeng_cao
* date: 2021/07/09 11:08
* last revised: None
************************************************ """


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg19, vgg16

from ..build import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PerceptualLossVGG19(nn.Module):
    def __init__(self, layer_idx=[2, 7, 14], layer_weights=[1, 0.2, 0.04], reduction="sum"):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_weights = layer_weights
        self.vggnet_feats_layers = vgg19(pretrained=True).features

        self.reduction = reduction

    def vgg_forward(self, img):
        selected_feats = []
        out = img
        self.vggnet_feats_layers = self.vggnet_feats_layers.to(img)
        for i, layer in enumerate(self.vggnet_feats_layers):
            out = layer(out)
            if i in self.layer_idx:
                selected_feats.append(out)
            if i == self.layer_idx[-1]:
                break
        assert len(selected_feats) == len(self.layer_idx)
        return selected_feats

    def forward(self, img1, img2):
        selected_feats1 = self.vgg_forward(img1)
        selected_feats2 = self.vgg_forward(img2)

        loss = 0
        for i, (feat1, feat2) in enumerate(zip(selected_feats1, selected_feats2)):
            assert feat1.shape == feat2.shape, "The input tensor should be in same shape!"
            loss += F.mse_loss(feat1, feat2, reduction=self.reduction) * self.layer_weights[i]

        return loss
