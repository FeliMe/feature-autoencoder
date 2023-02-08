"""Adapted from https://github.com/jusiro/constrained_anomaly_segmentation/blob/main/code/models/models.py"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models

import wandb


class CAVGA_Ru(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

    def forward(self, x: Tensor):
        z, mu, logvar, allF = self.encoder(x)
        y = self.decoder(z)
        return y, mu, logvar, allF

    def loss(
        self,
        x: Tensor,
        w_vae: float = 1.0,
        w_adv: float = 1.0,
        w_ae: float = 0.01
    ) -> Dict[str, Tensor]:
        with torch.enable_grad():
            y, mu, logvar, allF = self(x)
            gcam = grad_cam(allF[-1], mu.sum())  # (N, 1, h, w)

        # VAE loss (Eq. 1)
        rec_loss = torch.mean((x - y) ** 2)
        kl_loss = torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))
        vae_loss = rec_loss + kl_loss

        # Adversarial loss (Eq. 2)
        real_loss = -self.discriminator(x).log().mean()
        fake_loss = -(1 - self.discriminator(y)).log().mean()
        adv_loss = real_loss + fake_loss

        # Attention expansion loss (Eq. 3)
        ae_loss = (1 - gcam).mean()

        # Combine
        loss = w_vae * vae_loss + w_adv * adv_loss + w_ae * ae_loss

        return {
            'loss': loss,
            'vae_loss': vae_loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
            'adv_loss': adv_loss,
            'ae_loss': ae_loss
        }

    def predict_anomaly(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self.zero_grad()
        size = x.shape[-2:]
        with torch.enable_grad():
            y, mu, _, allF = self(x)
            gcam = 1 - grad_cam(allF[-1], mu.sum())
            anomaly_maps = F.interpolate(gcam, size, mode='bilinear', align_corners=True)  # (N, 1, H, W)
        anomaly_scores = (x - y).abs().mean((1, 2, 3))  # (N)
        return anomaly_maps.detach(), anomaly_scores.detach()

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


class Encoder(nn.Module):
    def __init__(self, fin: int = 1, zdim: int = 128, n_blocks: int = 4) -> None:
        super(Encoder, self).__init__()
        self.fin = fin
        self.zdim = zdim
        self.n_blocks = n_blocks

        self.backbone = Resnet(in_channels=self.fin, n_blocks=self.n_blocks)
        self.mu = nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
        self.log_var = nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))

    def reparameterize(self, mu: Tensor, log_var) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        sample = mu + (eps * std)
        return sample

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        x, allF = self.backbone(x)

        z_mu = self.mu(x)  # (N, zdim, h, w)
        z_logvar = self.log_var(x)  # (N, zdim, h, w)
        z = self.reparameterize(z_mu, z_logvar)  # (N, zdim, h, w)

        return z, z_mu, z_logvar, allF


class Decoder(nn.Module):
    def __init__(
        self,
        nf0: int = 128,
        n_channels: int = 1,
        n_blocks: int = 4,
    ) -> None:
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks
        self.nf0 = nf0

        # Set number of input and output channels
        n_filters_in = [nf0 // 2**(i) for i in range(0, self.n_blocks + 1)]
        n_filters_out = [nf0 // 2**(i + 1) for i in range(0, self.n_blocks)] + [n_channels]

        self.blocks = nn.ModuleList()
        for i in np.arange(0, self.n_blocks):
            self.blocks.append(nn.Sequential(BasicBlock(n_filters_in[i], n_filters_out[i], upsample=True),
                                             BasicBlock(n_filters_out[i], n_filters_out[i])))
        self.out = nn.Conv2d(n_filters_in[-1], n_filters_out[-1], kernel_size=(3, 3), padding=(1, 1))

        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        for i in np.arange(0, self.n_blocks):
            x = self.blocks[i](x)
        y = self.out(x)
        return y


class Discriminator(nn.Module):
    def __init__(
        self,
        fin: int = 32,
        n_channels: int = 1,
        n_blocks: int = 4,
    ) -> None:
        super(Discriminator, self).__init__()

        # Number of feature extractor blocks
        self.n_blocks = n_blocks
        # Set number of input and output channels
        n_filters_in = [n_channels] + [fin * (2**i) for i in range(self.n_blocks)]
        n_filters_out = [fin * (2**i) for i in range(self.n_blocks + 1)]
        # Number of output features
        nFeats = n_filters_out[-1]
        # Prepare blocks:
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks + 1):
            self.blocks.append(ConvBlock(n_filters_in[i], n_filters_out[i]))
        # Output for binary clasification
        self.out = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(),
                                 nn.Linear(nFeats, 1),
                                 nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.n_blocks + 1):
            x = self.blocks[i](x)
        y = self.out(x)
        return y


class Resnet(nn.Module):
    def __init__(self, in_channels: int, n_blocks: int = 4):
        super(Resnet, self).__init__()
        self.n_blocks = n_blocks
        self.nfeats = 512 // (2**(4 - n_blocks))

        resnet = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.resnet = nn.Sequential(*(list(resnet.children())[i + 4] for i in range(0, self.n_blocks)))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        F = []
        for iBlock in range(0, self.n_blocks):
            x = list(self.resnet.children())[iBlock](x)
            F.append(x)

        return x, F


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int = 32,
        planes: int = 64,
        stride: int = 1,
        upsample: bool = False,
        bn: bool = True
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        if self.upsample:
            self.upsample_layer_conv = nn.Conv2d(inplanes, planes, kernel_size=(1, 1))
            self.upsample_layer = nn.Upsample(scale_factor=(2, 2))
            self.upsample_layer_bn = norm_layer(planes)
        self.stride = stride
        self.bn = bn

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self.upsample:
            out = self.upsample_layer(out)
        if self.bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.upsample:
            identity = self.upsample_layer_conv(identity)
            identity = self.upsample_layer(identity)
            if self.bn:
                identity = self.upsample_layer_bn(identity)

        out += identity
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, fin: int, fout: int):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(fin, fout, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(fout)
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.bn(self.conv(x)))
        return out


def grad_cam(activations: Tensor, output: Tensor) -> Tensor:
    # Get gradients
    gradients = torch.autograd.grad(output, activations, grad_outputs=None,
                                    retain_graph=True, create_graph=True,
                                    only_inputs=True, allow_unused=True)[0]  # (N, d, h, w)

    # Global average pooling of the gradients
    gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)  # (N, d, 1, 1)

    # Compute grad-CAM
    gcam = (gradients * activations).sum(1, keepdim=True)  # (N, 1, h, w)
    # Apply ReLU
    gcam = torch.relu(gcam)
    # Normalize
    gcam = gcam / gcam.max()

    return gcam  # (N, 1, h, w)


if __name__ == '__main__':
    model = CAVGA_Ru()
    x = torch.randn(2, 1, 128, 128)

    loss = model.loss(x)
    anomaly_maps, anomaly_scores = model.predict_anomaly(x)
    print(anomaly_maps.shape, anomaly_scores.shape)
    print(loss)
