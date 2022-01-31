import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch import Tensor
from torchvision import models as tv_models


VGG16MAPPING = {
    '1': 'relu1_1',
    '3': 'relu1_2',  # [b, 64, 256, 256]
    '6': 'relu2_1',
    '8': 'relu2_2',  # [b, 128, 128, 128]
    '11': 'relu3_1',
    '13': 'relu3_2',
    '15': 'relu3_3',  # [b, 256, 64, 64]
    '18': 'relu4_1',
    '20': 'relu4_2',
    '22': 'relu4_3',  # [b, 512, 32, 32]
    '25': 'relu5_1',
    '27': 'relu5_2',
    '29': 'relu5_3',  # [b, 512, 16, 16]
}

VGG16LAYERS = list(VGG16MAPPING.values())

VGG19MAPPING = {
    '1': 'relu1_1',
    '3': 'relu1_2',  # [b, 64, 256, 256]
    '6': 'relu2_1',
    '8': 'relu2_2',  # [b, 128, 128, 128]
    '11': 'relu3_1',
    '13': 'relu3_2',
    '15': 'relu3_3',
    '17': 'relu3_4',  # [b, 256, 64, 64]
    '20': 'relu4_1',
    '22': 'relu4_2',
    '24': 'relu4_3',
    '26': 'relu4_4',  # [b, 512, 32, 32]
    '29': 'relu5_1',
    '31': 'relu5_2',
    '33': 'relu5_3',
    '35': 'relu5_4',  # [b, 512, 16, 16]
}

VGG19LAYERS = list(VGG19MAPPING.values())


def _set_requires_grad_false(layer):
    for param in layer.parameters():
        param.requires_grad = False


class VGG16FeatureExtractor(nn.Module):
    def __init__(
        self,
        layer_names: List[str] = VGG16LAYERS
    ):
        """
        Returns features on multiple levels from a VGG16.

        Args:
            layer_names (list): List of string of layer names where to return
                                the features. Must be ordered
        """
        super().__init__()

        vgg = tv_models.vgg16(pretrained=True)
        _set_requires_grad_false(vgg)

        self.features = nn.Sequential(
            *(list(vgg.features.children()) + [vgg.avgpool]))

        self.layer_name_mapping = VGG16MAPPING

        self.layer_names = layer_names

        # self.pad = nn.ReflectionPad2d(padding=1)

    def forward(self, inp: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            inp (Tensor): Input tensor of shape [b, 1, h, w] or [b, 3, h, w]
        Returns:
            out (dict): Dictionary containing the extracted features as Tensors
        """
        if inp.shape[1] == 1:
            inp = inp.repeat(1, 3, 1, 1)
        out = {}
        # inp = self.pad(inp)
        for name, module in self.features._modules.items():
            inp = module(inp)
            # inp = self.pad(inp)
            if name in self.layer_name_mapping.keys():
                if self.layer_name_mapping[name] in self.layer_names:
                    out[self.layer_name_mapping[name]] = inp
                if self.layer_name_mapping[name] == self.layer_names[-1]:
                    break
        return out


class VGG19FeatureExtractor(nn.Module):
    def __init__(
        self,
        layer_names: List[str] = VGG19LAYERS
    ):
        """
        Returns features on multiple levels from a VGG19.

        Args:
            layer_names (list): List of string of layer names where to return
                                the features. Must be ordered
        """
        super().__init__()

        vgg = tv_models.vgg19(pretrained=True)
        _set_requires_grad_false(vgg)

        self.features = nn.Sequential(
            *(list(vgg.features.children()) + [vgg.avgpool]))

        self.layer_name_mapping = VGG19MAPPING

        self.layer_names = layer_names

        # self.pad = nn.ReflectionPad2d(padding=1)

    def forward(self, inp: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            inp (Tensor): Input tensor of shape [b, 1, h, w] or [b, 3, h, w]
        Returns:
            out (dict): Dictionary containing the extracted features as Tensors
        """
        if inp.shape[1] == 1:
            inp = inp.repeat(1, 3, 1, 1)
        out = {}
        # inp = self.pad(inp)
        for name, module in self.features._modules.items():
            inp = module(inp)
            # inp = self.pad(inp)
            if name in self.layer_name_mapping.keys():
                if self.layer_name_mapping[name] in self.layer_names:
                    out[self.layer_name_mapping[name]] = inp
                if self.layer_name_mapping[name] == self.layer_names[-1]:
                    break
        return out


backbone_nets = {
    'vgg16': VGG16FeatureExtractor,
    'vgg19': VGG19FeatureExtractor,
}


class AvgFeatAGG2d(nn.Module):
    """
    Convolution operation with mean kernel
    """
    def __init__(self, kernel_size: int, output_size: int = None,
                 dilation: int = 1, stride: int = 1):
        super(AvgFeatAGG2d, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.fold = nn.Fold(output_size=output_size, kernel_size=1, dilation=1, stride=1)
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        N, C, _, _ = input.shape
        output = self.unfold(input)  # (b, cxkxk, h*w)
        output = torch.reshape(output, (N, C, int(self.kernel_size**2), int(self.output_size**2)))
        output = torch.mean(output, dim=2)
        output = self.fold(output)
        return output


class Extractor(nn.Module):
    """
    Muti-scale regional feature based on VGG-feature maps.
    """
    def __init__(
        self,
        cnn_layers: List[str] = VGG19LAYERS[:12],
        backbone: str = 'vgg19',
        upsample: str = 'bilinear',
        kernel_size: int = 4,
        stride: int = 4,
        featmap_size: int = 256,  # Before convolving
        is_agg: bool = True,
    ):
        super().__init__()

        self.backbone = backbone_nets[backbone](layer_names=cnn_layers)
        self.featmap_size = featmap_size
        self.upsample = upsample
        self.is_agg = is_agg
        self.align_corners = True if upsample == "bilinear" else None

        # Calculate padding
        padding = (kernel_size - stride) // 2
        self.replicationpad = nn.ReplicationPad2d([padding] * 4)

        # Calculate output size
        self.out_size = int((featmap_size + 2 * padding -
                            ((kernel_size - 1) + 1)) / stride + 1)

        if not self.is_agg:
            self.featmap_size = self.out_size

        # Find out how many channels we got from the backbone
        self.c_out = self.get_out_channels()

        self.feat_agg = AvgFeatAGG2d(kernel_size=kernel_size,
                                     output_size=self.out_size, stride=stride)

    def get_out_channels(self):
        device = next(self.backbone.parameters()).device
        inp = torch.randn((2, 1, 224, 224), device=device)
        feat_maps = self.backbone(inp)
        channels = 0
        for feat_map in feat_maps.values():
            channels += feat_map.shape[1]
        return channels

    def forward(self, inp: Tensor) -> Tensor:
        feat_maps = self.backbone(inp)

        features = []
        for _, feat_map in feat_maps.items():
            # Resizing
            feat_map = F.interpolate(feat_map, size=self.featmap_size,
                                     mode=self.upsample,
                                     align_corners=self.align_corners)

            if self.is_agg:
                feat_map = self.replicationpad(feat_map)
                feat_map = self.feat_agg(feat_map)

            features.append(feat_map)

        # Concatenate to tensor
        features = torch.cat(features, dim=1)

        return features


class FeatureAE(nn.Module):
    def __init__(self, img_size: int, latent_channels: int,
                 use_batchnorm: bool = True):
        super().__init__()

        self.extractor = Extractor(featmap_size=img_size)
        in_channels = self.extractor.c_out

        ks = 1
        pad = ks // 2

        hidden_channels = [
            in_channels,
            (in_channels + 2 * latent_channels) // 2,
            2 * latent_channels,
            latent_channels,
        ]

        # Encoder
        encoder = nn.Sequential()
        for i, (c_in, c_out) in enumerate(zip(hidden_channels[:-1], hidden_channels[1:])):
            encoder.add_module(f"encoder_conv_{i}",
                               nn.Conv2d(c_in, c_out, ks, padding=pad, bias=False))
            if use_batchnorm:
                encoder.add_module(f"encoder_batchnorm_{i}",
                                   nn.BatchNorm2d(c_out))
            encoder.add_module(f"encoder_relu_{i}", nn.ReLU())

        # Decoder
        decoder = nn.Sequential()
        for i, (c_in, c_out) in enumerate(zip(hidden_channels[-1::-1], hidden_channels[-2::-1])):
            decoder.add_module(f"decoder_conv_{i}",
                               nn.Conv2d(c_in, c_out, ks, padding=pad, bias=False))
            if use_batchnorm:
                decoder.add_module(f"decoder_batchnorm_{i}",
                                   nn.BatchNorm2d(c_out))
            decoder.add_module(f"decoder_relu_{i}", nn.ReLU())

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feats = self.extractor(x)
        z = self.encoder(feats)
        rec = self.decoder(z)
        return feats, rec

    def loss(self, x: Tensor):
        feats, rec = self(x)
        loss = torch.mean((feats - rec) ** 2)
        return loss

    def predict_anomaly(self, x: Tensor) -> Tensor:
        feats, rec = self.forward(x)
        # Compute anomaly map
        map_small = torch.mean((feats - rec) ** 2, dim=1, keepdim=True)
        anomaly_map = F.interpolate(map_small, x.shape[-2:], mode='bilinear',
                                    align_corners=True)
        # Anomaly score only where object is detected (i.e. not background)
        anomaly_score = torch.tensor([m[x_ > 0].mean() for m, x_ in zip(anomaly_map, x)])
        # Loss from map small
        loss = map_small.mean()
        return anomaly_map, anomaly_score, loss

    def load(self, path: str):
        """
        Load model from W&B
        :param path: Path to the model <entity>/<project>/<run_id>/<model_name>
        """
        name = os.path.basename(path)
        run_path = os.path.dirname(path)
        weights = wandb.restore(name, run_path=run_path)
        self.load_state_dict(torch.load(weights.name))

    def save(self, name: str):
        torch.save(self.state_dict(), os.path.join(wandb.run.dir, name))


if __name__ == '__main__':
    from argparse import Namespace
    torch.manual_seed(0)

    # Config
    config = Namespace()
    config.device = "cuda"
    config.img_size = 256
    config.latent_channels = 150
    config.backbone = 'vgg19'
    config.cnn_layers = VGG19LAYERS[:12]

    # Create model
    ae = FeatureAE(
        img_size=config.img_size,
        latent_channels=config.latent_channels,
        use_batchnorm=True
    ).to(config.device)

    # Sample input
    x = torch.randn((2, 1, config.img_size, config.img_size), device=config.device)

    # Forward
    anomaly_map, anomaly_score = ae.predict_anomaly(x)
    print(x.shape, anomaly_map.shape, anomaly_score)
