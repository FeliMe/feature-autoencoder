import os

import torch
import torch.nn as nn
from torch import Tensor
import wandb


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)


class WideResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 dropout: int = 0.0, direction: str = 'down'):
        """
        Creates a wide residual block.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: stride of the convolutional layers
        :param dropout: dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Main path
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride if direction == 'down' else 1)
        self.up = nn.Upsample(scale_factor=stride) if direction == 'up' else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv3x3(out_channels, out_channels)

        # Shortcut connection
        skip = nn.Sequential()
        if in_channels != out_channels:
            skip.add_module("skip_conv",
                            conv1x1(in_channels, out_channels,
                                    1 if direction == 'up' else stride))
            if direction == 'up':
                skip.add_module("skip_upsample", nn.Upsample(scale_factor=stride))
        else:
            if stride == 1:
                skip.add_module("skip_none", nn.Identity())
            elif direction == 'up':
                skip.add_module("skip_upsample", nn.Upsample(scale_factor=stride))
            else:
                skip.add_module("skip_downsample", nn.AvgPool2d(stride))
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        if self.in_channels != self.out_channels:
            x = self.bn1(x)
            x = self.relu(x)
            out = x
        else:
            out = self.bn1(x)
            out = self.relu(out)
        out = self.conv1(out)
        out = self.up(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        # Shortcut
        shortcut = self.skip(x)

        return out + shortcut


def layer_(in_channels: int, out_channels: int, n_blocks: int,
           stride: int, **kwargs):
    layer = nn.Sequential()
    layer.add_module("layer_0", WideResNetBlock(in_channels, out_channels, stride, **kwargs))
    for i in range(2, int(n_blocks + 1)):
        layer.add_module(f'layer_{i}', WideResNetBlock(out_channels, out_channels, 1, **kwargs))
    return layer


class WideResNetAE(nn.Module):
    """For image size of 128"""
    def __init__(self, config):
        super().__init__()

        # Unpack config or set default values
        in_channels = config.in_channels if "in_channels" in config else 1
        out_channels = config.out_channels if "out_channels" in config else 1
        depth = config.depth if "depth" in config else 16
        k = config.k if "depth" in config else 4
        dropout = config.dropout if "dropout" in config else 0.0

        assert((depth - 6) % 10 == 0), 'depth should be 10n+6'
        n = (depth - 6) // 10

        n_stages = [16, 16 * k, 32 * k, 64 * k, 64 * k, 64 * k]

        # Encoder
        self.encoder = nn.Sequential(
            conv3x3(in_channels, n_stages[0], stride=1),
            layer_(n_stages[0], n_stages[1], n, 1, direction='down',
                   dropout=dropout),
            layer_(n_stages[1], n_stages[2], n, 2, direction='down',
                   dropout=dropout),
            layer_(n_stages[2], n_stages[3], n, 2, direction='down',
                   dropout=dropout),
            layer_(n_stages[3], n_stages[4], n, 2, direction='down',
                   dropout=dropout),
        )

        # Decoder
        self.decoder = nn.Sequential(
            layer_(n_stages[4], n_stages[1], 1, 2, direction='up',
                   dropout=dropout),
            layer_(n_stages[1], n_stages[0], 1, 2, direction='up',
                   dropout=dropout),
            layer_(n_stages[0], out_channels, 1, 2, direction='up',
                   dropout=dropout),
        )

    def forward(self, inp: Tensor) -> Tensor:
        latent = self.encoder(inp)
        pred = self.decoder(latent)
        pred = torch.sigmoid(pred)
        return pred

    def load(self, path: str):
        """
        Load model from W&B
        :param path: Path to the model <entity>/<project>/<run_id>/<model_name>
        """
        name = os.path.basename(path)
        run_path = os.path.dirname(path)
        weights = wandb.restore(name, run_path=run_path, root="/tmp", replace=True)
        self.load_state_dict(torch.load(weights.name))
        os.remove(weights.name)

    def save(self, name: str):
        torch.save(self.state_dict(), os.path.join(wandb.run.dir, name))


if __name__ == '__main__':
    model = WideResNetAE(1, 1)
    print(model)
    inp = torch.randn(2, 1, 128, 128)
    pred = model(inp)
    print(pred.shape)
