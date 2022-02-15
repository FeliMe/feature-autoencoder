import os
from argparse import Namespace
from re import I
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import wandb

from fae.utils.pytorch_ssim import SSIMLoss


class Reshape(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.size)


def build_encoder(in_channels: int, hidden_dims: List[int],
                  use_batchnorm: bool = True, dropout: float = 0.0) -> nn.Module:
    encoder = nn.Sequential()
    for i, h_dim in enumerate(hidden_dims):
        layer = nn.Sequential()

        # Convolution
        layer.add_module(f"encoder_conv_{i}",
                         nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2,
                                   padding=1, bias=False))

        # Batch normalization
        if use_batchnorm:
            layer.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm2d(h_dim))

        # LeakyReLU
        layer.add_module(f"encoder_leakyrelu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"encoder_dropout_{i}", nn.Dropout(p=dropout))

        # Add layer to encoder
        encoder.add_module(f"encoder_layer_{i}", layer)

        in_channels = h_dim

    return encoder


def build_decoder(out_channels: int, hidden_dims: List[int],
                  use_batchnorm: bool = True, dropout: float = 0.0) -> nn.Module:
    h_dims = [out_channels] + hidden_dims

    dec = nn.Sequential()
    for i in range(len(h_dims) - 1, 0, -1):
        # Add a new layer
        layer = nn.Sequential()

        # Upsample
        layer.add_module(f"decoder_upsample_{i}", nn.Upsample(scale_factor=2))

        # Convolution
        layer.add_module(f"decoder_conv_{i}",
                         nn.Conv2d(h_dims[i], h_dims[i - 1], kernel_size=3,
                                   padding=1, bias=False))

        # Batch normalization
        if use_batchnorm:
            layer.add_module(f"decoder_batchnorm_{i}",
                             nn.BatchNorm2d(h_dims[i - 1]))

        # LeakyReLU
        layer.add_module(f"decoder_leakyrelu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"decoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the decoder
        dec.add_module(f"decoder_layer_{i}", layer)

    # Final layer
    dec.add_module("decoder_conv_final",
                   nn.Conv2d(h_dims[0], out_channels, 1, bias=False))

    return dec


class AE(nn.Module):
    """
    A n-layer variational autoencoder
    adapted from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    """

    def __init__(self, config):
        super().__init__()

        # Unpack config
        image_size = config.image_size
        latent_dim = config.latent_dim
        hidden_dims = config.hidden_dims
        use_batchnorm = config.use_batchnorm if "use_batchnorm" in config else True
        dropout = config.dropout if "dropout" in config else 0.0

        intermediate_res = image_size // 2 ** len(hidden_dims)
        intermediate_feats = intermediate_res * \
            intermediate_res * hidden_dims[-1]

        # Build encoder
        self.encoder = build_encoder(1, hidden_dims, use_batchnorm, dropout)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(intermediate_feats, latent_dim, bias=False),
        )
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, intermediate_feats, bias=False),
            Reshape((-1, hidden_dims[-1], intermediate_res, intermediate_res)),
        )

        # Build decoder
        self.decoder = build_decoder(1, hidden_dims, use_batchnorm, dropout)

        # Loss function
        if config.loss_fn == "l1":
            self.loss_fn = nn.L1Loss(reduction='none')
        elif config.loss_fn == "mse":
            self.loss_fn = nn.MSELoss(reduction='none')
        elif config.loss_fn == "ssim":
            self.loss_fn = SSIMLoss(size_average=False)
        else:
            raise ValueError("Unknown loss function")

    def loss(self, inp: Tensor, rec: Tensor) -> Tensor:
        """
        :param inp: Input image
        :param rec: Reconstructed image
        """
        return {'rec_loss': self.loss_fn(inp, rec).mean()}

    def predict_anomaly(self, inp: Tensor, rec: Tensor):
        """
        Return anomaly map and anomaly score
        """
        # Anomaly map
        anomaly_map = self.loss_fn(inp, rec).mean(1, keepdim=True)

        # Anomaly score only where object in the image, i.e. at x > 0
        anomaly_score = []
        for i in range(inp.shape[0]):
            roi = anomaly_map[i][inp[i] > 0]
            anomaly_score.append(roi.mean())
        anomaly_score = torch.stack(anomaly_score)

        return anomaly_map, anomaly_score

    def forward(self, inp: Tensor) -> Tensor:
        # Encode
        res = self.encoder(inp)
        # Bottleneck
        z = self.bottleneck(res)
        decoder_inp = self.decoder_input(z)
        # Decode
        rec = self.decoder(decoder_inp)
        return rec

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
    # Test AE
    config = Namespace()
    config.image_size = 128
    config.latent_dim = 128
    config.hidden_dims = [32, 64, 128, 256]
    config.loss_fn = "l1"
    net = AE(config)
    print(net)
    x = torch.randn(2, 1, *[config.image_size] * 2)
    with torch.no_grad():
        rec = net(x)
    print(rec.shape)
