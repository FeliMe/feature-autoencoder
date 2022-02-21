from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from torch import Tensor


RESNETLAYERS = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']


def _set_requires_grad_false(layer):
    for param in layer.parameters():
        param.requires_grad = False


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, layer_names: List[str] = RESNETLAYERS):
        """
        Returns features on multiple levels from a ResNet18.
        Available layers: 'layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
        Args:
            resnet (nn.Module): Type of resnet used
            layer_names (list): List of string of layer names where to return
                                the features. Must be ordered
            pretrained (bool): Whether to load pretrained weights
        Returns:
            out (dict): Dictionary containing the extracted features as
                        torch.tensors
        """
        super().__init__()

        _set_requires_grad_false(resnet)

        # [b, 3, 256, 256]
        self.layer0 = nn.Sequential(
            *list(resnet.children())[:4])  # [b, 64, 64, 64]
        self.layer1 = resnet.layer1  # [b, 64, 64, 64]
        self.layer2 = resnet.layer2  # [b, 128, 32, 32]
        self.layer3 = resnet.layer3  # [b, 256, 16, 16]
        self.layer4 = resnet.layer4  # [b, 512, 8, 8]
        self.avgpool = resnet.avgpool  # [b, 512, 1, 1]

        self.layer_names = layer_names

    def forward(self, inp: Tensor) -> Dict[str, Tensor]:
        if inp.shape[1] == 1:
            inp = inp.repeat(1, 3, 1, 1)

        out = {}
        for name, module in self._modules.items():
            inp = module(inp)
            if name in self.layer_names:
                out[name] = inp
            if name == self.layer_names[-1]:
                break
        return out


class ResNet18FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self, layer_names: List[str] = RESNETLAYERS,
                 pretrained: bool = True):
        super().__init__(tv_models.resnet18(pretrained=pretrained), layer_names)


class Extractor(nn.Module):
    """
    Muti-scale regional feature based on VGG-feature maps.
    """

    def __init__(
        self,
        cnn_layers=['layer1', 'layer2'],
        upsample='bilinear',
        inp_size=128,
        keep_feature_prop=1.0,
        pretrained=True,
    ):
        super().__init__()

        self.backbone = ResNet18FeatureExtractor(layer_names=cnn_layers,
                                                 pretrained=pretrained)
        self.inp_size = inp_size
        self.featmap_size = inp_size // 4
        self.upsample = upsample
        self.align_corners = True if upsample == "bilinear" else None

        # Find out how many channels we got from the backbone
        c_feats = self.get_out_channels()

        # Create mask to drop random features_channels
        self.register_buffer('feature_mask', torch.Tensor(
            c_feats).uniform_() < keep_feature_prop)
        self.c_feats = self.feature_mask.sum().item()

    def get_out_channels(self):
        device = next(self.backbone.parameters()).device
        inp = torch.randn((2, 1, self.inp_size, self.inp_size), device=device)
        return sum([feat_map.shape[1] for feat_map in self.backbone(inp).values()])

    def forward(self, inp: Tensor):
        # Center input
        inp = (inp - 0.5) * 2

        # Extract feature maps
        feat_maps = self.backbone(inp)

        features = []
        for feat_map in feat_maps.values():
            # Resizing
            feat_map = F.interpolate(feat_map, size=self.featmap_size,
                                     mode=self.upsample,
                                     align_corners=self.align_corners)
            features.append(feat_map)

        # Concatenate to tensor
        features = torch.cat(features, dim=1)

        # Drop out feature maps
        features = features[:, self.feature_mask]

        return features


if __name__ == '__main__':
    from argparse import Namespace
    config = Namespace()
    config.inp_size = 128
    config.keep_feature_prop = 1.
    config.cnn_layers = ['layer1', 'layer2']
    device = "cpu"

    extractor = Extractor(**vars(config)).to(device)
    x = torch.randn((8, 1, config.inp_size, config.inp_size), device=device)
    y = extractor(x)
    # print(y.shape)
