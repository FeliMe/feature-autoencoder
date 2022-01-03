from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from fae.models.feature_extractor import Extractor
from fae.utils.pytorch_ssim import SSIMLoss


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def predict_anomaly(self, x):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


def vanilla_feature_encoder(in_channels: int, hidden_dims: List[int], use_batchnorm: bool = True):
    """
    Vanilla feature encoder.
    Args:
        in_channels (int): Number of input channels
        hidden_dims (List[int]): List of hidden channel dimensions
    Returns:
        encoder (nn.Module): The encoder
    """
    ks = 3  # Kernel size
    pad = 1  # Padding

    # Build encoder
    enc = nn.Sequential()
    for i, hidden_dim in enumerate(hidden_dims):
        # Convolution
        enc.add_module(f"encoder_conv_{i}",
                       nn.Conv2d(in_channels, hidden_dims[i], ks, padding=pad, bias=False))

        # Batch normalization
        if use_batchnorm:
            enc.add_module(f"encoder_batchnorm_{i}",
                           nn.BatchNorm2d(hidden_dims[i]))

        # LeakyReLU
        enc.add_module(f"encoder_relu_{i}", nn.LeakyReLU())
        in_channels = hidden_dim
    return enc


def vanilla_feature_decoder(out_channels: int, hidden_dims: List[int], use_batchnorm: bool = True):
    """
    Vanilla feature decoder.
    Args:
        out_channels (int): Number of output channels
        hidden_dims (List[int]): List of hidden channel dimensions
    Returns:
        decoder (nn.Module): The decoder
    """
    ks = 3  # Kernel size
    pad = 1  # Padding
    out_pad = 0  # Output padding

    # Build decoder
    dec = nn.Sequential()
    for i in range(len(hidden_dims) - 1, 0, -1):
        # Transposed convolution
        dec.add_module(f"decoder_tconv_{i}",
                       nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1],
                                          ks, padding=pad, output_padding=out_pad,
                                          bias=False))

        # Batch normalization
        if use_batchnorm:
            dec.add_module(f"decoder_batchnorm_{i}",
                           nn.BatchNorm2d(hidden_dims[i - 1]))

        # LeakyReLU
        dec.add_module(f"decoder_relu_{i}", nn.LeakyReLU())

    # Final layer
    dec.add_module("decoder_conv_final",
                   nn.Conv2d(hidden_dims[0], out_channels, 1, bias=False))

    return dec


class FeatureAutoencoder(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.extractor = Extractor(inp_size=config.image_size,
                                   keep_feature_prop=config.keep_feature_prop)
        c_in = self.extractor.c_out
        hidden_dims = [
            (c_in + 2 * config.c_latent) // 2,
            4 * config.c_latent,
            2 * config.c_latent,
            config.c_latent
        ]
        self.enc = vanilla_feature_encoder(self.extractor.c_out, hidden_dims)
        self.dec = vanilla_feature_decoder(self.extractor.c_out, hidden_dims)

    def forward(self, x):
        feats = self.extractor(x)
        z = self.enc(feats)
        y = self.dec(z)
        return feats, y

    def loss(self, x):
        return self.predict_anomaly_(x).mean()

    def predict_anomaly_(self, x):
        feats, y = self(x)
        return F.l1_loss(y, feats, reduction='none').sum(1, keepdim=True)

    def predict_anomaly(self, x):
        return F.interpolate(self.predict_anomaly_(x), x.shape[-2:],
                             mode='bilinear', align_corners=True)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)


if __name__ == '__main__':
    from argparse import Namespace
    config = Namespace()
    config.image_size = 224
    config.c_latent = 128
    config.keep_feature_prop = 1.0

    x = torch.randn(1, 1, *[config.image_size] * 2)
    fae = FeatureAutoencoder(config)
    anomaly_map = fae.predict_anomaly(x)
    import IPython; IPython.embed(); exit(1)
