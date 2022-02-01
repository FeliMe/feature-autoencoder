import os
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import wandb


DIM = 64
LATENT_DIM = 128


""" Helper functions """


def initialize(net: nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):  # Also includes custom Conv2d
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


""" Modules """


class Conv2d(nn.Conv2d):
    """nn.Conv2d with same padding"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, bias: bool = True) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=kernel_size // 2, bias=bias)


class UpSampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 bias: bool = True):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, inp: Tensor) -> Tensor:
        out = inp
        out = out.repeat(1, 4, 1, 1)
        out = F.pixel_shuffle(out, 2)  # Equiv. to tensorflow.nn.depth_to_space
        out = self.conv(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 resample: str, hw: int = None,
                 norm_layer: str = "batchnorm"):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolving kernel
        :param resample: 'down', or 'up'
        :param hw: height and width of the image (one integer value)
        :param norm_layer: layer that will be used for normalization
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resample = resample

        if resample == 'down':
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2),
                Conv2d(in_channels, out_channels, kernel_size=1),
            )
            self.conv1 = Conv2d(in_channels, in_channels, kernel_size,
                                bias=False)
            self.conv2 = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1),
                nn.AvgPool2d(2),
            )
            bn1_channels = in_channels
            bn2_channels = in_channels

        elif resample == 'up':
            self.shortcut = UpSampleConv(
                in_channels, out_channels, kernel_size=1)
            self.conv1 = UpSampleConv(
                in_channels, out_channels, kernel_size, bias=False)
            self.conv2 = Conv2d(out_channels, out_channels, kernel_size)
            bn1_channels = in_channels
            bn2_channels = out_channels

        else:
            raise RuntimeError('resample must be either "down" or "up"')

        if norm_layer == "batchnorm":
            self.bn1 = nn.BatchNorm2d(bn1_channels)
            self.bn2 = nn.BatchNorm2d(bn2_channels)
        elif norm_layer == "layernorm":
            self.bn1 = nn.LayerNorm([bn1_channels, hw, hw])
            self.bn2 = nn.LayerNorm([bn2_channels, hw, hw])
        else:
            raise RuntimeError(
                'norm_layer must be either "batchnorm" or "layernorm"')

    def forward(self, inp: Tensor) -> Tensor:
        shortcut = self.shortcut(inp)

        # Layer 1
        out = self.bn1(inp)
        out = torch.relu(out)
        out = self.conv1(out)

        # Layer 2
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)

        return shortcut + out


""" Models """


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Unpack config
        dims = config.generator_hidden_dims
        latent_dim = config.latent_dim if 'latent_dim' in config else LATENT_DIM
        self.latent_dim = latent_dim

        # Define layers
        self.ln1 = nn.Linear(latent_dim, 4 * 4 * dims[0])
        resblock = partial(ResidualBlock, resample='up',
                           norm_layer="batchnorm")
        self.res_blocks = nn.Sequential(
            *[resblock(d_in, d_out, kernel_size=3) for d_in, d_out in zip(dims[:-1], dims[1:])])
        self.bn = nn.BatchNorm2d(dims[-1])
        self.conv1 = Conv2d(dims[-1], config.feat_channels, kernel_size=3)

        # Initialize weights
        initialize(self)

    def sample_latent(self, batch_size: int) -> Tensor:
        device = next(self.parameters()).device
        return torch.randn(batch_size, self.latent_dim, device=device)

    def forward(self, z: Tensor = None, batch_size: int = None) -> Tensor:
        # Sample latent
        if z is None:
            z = self.sample_latent(batch_size)
        out = z

        # First linear layer
        out = self.ln1(out)
        out = out.view(out.shape[0], -1, 4, 4)

        # Residual blocks
        out = self.res_blocks(out)

        # Last conv layer
        out = torch.relu(self.bn(out))
        out = torch.tanh(self.conv1(out))
        return out


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Unpack config
        dims = config.discriminator_hidden_dims
        img_size = config.image_size // 4

        # Define layers
        self.conv1 = Conv2d(config.feat_channels, dims[0], kernel_size=3)
        resblock = partial(ResidualBlock, resample='down',
                           norm_layer="layernorm")
        self.res_blocks = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.res_blocks.append(resblock(d_in, d_out, kernel_size=3,
                                            hw=img_size))
            img_size //= 2
        self.res_blocks = nn.Sequential(*self.res_blocks)
        self.ln1 = nn.Linear(4 * 4 * dims[-1], 1)

        # Initialize weights
        initialize(self)

    def extract_feature(self, inp: Tensor) -> Tensor:
        out = inp
        out = self.conv1(out)
        out = self.res_blocks(out)
        out = out.view(out.shape[0], -1)
        return out

    def forward(self, inp: Tensor) -> Tensor:
        feats = self.extract_feature(inp)
        out = self.ln1(feats)
        out = out.view(-1)
        return out, feats


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Unpack config
        dims = config.discriminator_hidden_dims
        latent_dim = config.latent_dim if 'latent_dim' in config else LATENT_DIM
        dropout = config.dropout if 'dropout' in config else 0.

        # Define layers
        self.dropout = nn.Dropout(dropout)
        self.conv_in = nn.Conv2d(config.feat_channels,
                                 dims[0], 3, 1, padding=1)

        resblock = partial(ResidualBlock, resample='down',
                           norm_layer="batchnorm")
        # For 64x64
        self.res_blocks = nn.Sequential(
            *[resblock(d_in, d_out, kernel_size=3) for d_in, d_out in zip(dims[:-1], dims[1:])])

        self.fc = nn.Linear(4 * 4 * dims[-1], latent_dim)

        # Initialize weights
        initialize(self)

    def forward(self, inp: Tensor) -> Tensor:
        out = inp
        out = self.dropout(out)
        out = self.conv_in(out)
        out = self.res_blocks(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return torch.tanh(out)


class fAnoGAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.G = Generator(config)
        self.D = Discriminator(config)
        self.E = Encoder(config)

    def forward(self, x: Tensor, feat_weight: float = 1.) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return the anomaly map, anomaly score, and reconstruction for the
        given input.
        :param x: Input image
        :param feat_weight: Weight of the feature difference in the anoamly score
        """
        x_rec = self.G(self.E(x))
        f_x, f_x_rec = self.D.extract_feature(
            torch.cat((x, x_rec), dim=0)).chunk(2, 0)

        # Anomaly map is the residual of the input and the reconstructed image
        anomaly_map = (x - x_rec).abs().mean(1, keepdim=True)

        # Anomaly score
        img_diff = (x - x_rec).pow(2).mean((1, 2, 3))
        feat_diff = (f_x - f_x_rec).pow(2).mean((1))
        anomaly_score = img_diff + feat_weight * feat_diff

        return anomaly_map, anomaly_score, x_rec

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
    # Config
    from argparse import Namespace
    config = Namespace()
    config.image_size = 128
    config.feat_size = config.image_size // 4
    config.feat_channels = 192
    config.generator_hidden_dims = [300, 200, 150, 100]
    config.discriminator_hidden_dims = [100, 150, 200, 300]

    # Models
    model = fAnoGAN(config)
    print("Generator:", model.G, "\n")
    print("Discriminator:", model.D, "\n")
    print("Encoder:", model.E, "\n")
    print(config)

    # Data
    x = torch.randn(2, config.feat_channels, *[config.feat_size] * 2)

    # Forward
    z_enc = model.E(x)
    x_gen = model.G(z_enc)
    d_out, d_feats = model.D(x_gen)
    print(x_gen.shape, d_out.shape, d_feats.shape, z_enc.shape)
