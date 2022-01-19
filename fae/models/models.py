from collections import defaultdict
from typing import List


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fae.models.feature_extractor import Extractor
from fae.utils.pytorch_ssim import SSIMLoss
from fae.utils.utils import mahalanobis_distance_image


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


def vanilla_feature_encoder(in_channels: int, hidden_dims: List[int],
                            use_batchnorm: bool = False, dropout: float = 0.0,
                            bias: bool = False):
    """
    Vanilla feature encoder.
    Args:
        in_channels (int): Number of input channels
        hidden_dims (List[int]): List of hidden channel dimensions
    Returns:
        encoder (nn.Module): The encoder
    """
    ks = 5  # Kernel size
    pad = ks // 2  # Padding

    # Build encoder
    enc = nn.Sequential()
    for i, hidden_dim in enumerate(hidden_dims):
        # Add a new layer
        layer = nn.Sequential()

        # Convolution
        layer.add_module(f"encoder_conv_{i}",
                         nn.Conv2d(in_channels, hidden_dims[i], ks, stride=2,
                                   padding=pad, bias=bias))

        # Batch normalization
        if use_batchnorm:
            layer.add_module(f"encoder_batchnorm_{i}",
                             nn.BatchNorm2d(hidden_dims[i]))

        # LeakyReLU
        layer.add_module(f"encoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"encoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the encoder
        enc.add_module(f"encoder_layer_{i}", layer)

        in_channels = hidden_dim
    return enc


def vanilla_feature_decoder(out_channels: int, hidden_dims: List[int],
                            use_batchnorm: bool = False, dropout: float = 0.0,
                            bias: bool = False):
    """
    Vanilla feature decoder.
    Args:
        out_channels (int): Number of output channels
        hidden_dims (List[int]): List of hidden channel dimensions
    Returns:
        decoder (nn.Module): The decoder
    """
    ks = 6  # Kernel size
    pad = 2  # Padding

    hidden_dims = [out_channels] + hidden_dims

    # Build decoder
    dec = nn.Sequential()
    for i in range(len(hidden_dims) - 1, 0, -1):
        # Add a new layer
        layer = nn.Sequential()

        # Transposed convolution
        layer.add_module(f"decoder_tconv_{i}",
                         nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1],
                                            ks, stride=2, padding=pad,
                                            bias=bias))

        # Batch normalization
        if use_batchnorm:
            layer.add_module(f"decoder_batchnorm_{i}",
                             nn.BatchNorm2d(hidden_dims[i - 1]))

        # LeakyReLU
        layer.add_module(f"decoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"decoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the decoder
        dec.add_module(f"decoder_layer_{i}", layer)

    # Final layer
    dec.add_module("decoder_conv_final",
                   nn.Conv2d(hidden_dims[0], out_channels, 1, bias=False))

    return dec


class FeatureAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.enc = vanilla_feature_encoder(config.in_channels,
                                           config.hidden_dims,
                                           use_batchnorm=True,
                                           dropout=config.dropout,
                                           bias=False)
        self.dec = vanilla_feature_decoder(config.in_channels,
                                           config.hidden_dims,
                                           use_batchnorm=True,
                                           dropout=config.dropout,
                                           bias=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat

class FeatureReconstructor(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.extractor = Extractor(inp_size=config.image_size,
                                   keep_feature_prop=config.keep_feature_prop)

        config.in_channels = self.extractor.c_feats
        self.ae = FeatureAutoencoder(config)

    def forward(self, x: Tensor):
        with torch.no_grad():
            feats = self.extractor(x)
        return feats, self.ae(feats)

    def loss(self, x: Tensor):
        feats, rec = self(x)
        loss = SSIMLoss(size_average=True)(rec, feats).mean()
        # loss = F.l1_loss(rec, feats, reduction='mean')
        # loss = F.mse_loss(rec, feats, reduction='mean')
        return {'loss': loss}

    def predict_anomaly(self, x: Tensor):
        """Returns per image anomaly maps and anomaly scores"""
        # Extract features
        feats, rec = self(x)

        # Compute anomaly map
        anomaly_map = SSIMLoss(size_average=False)(rec, feats).mean(1, keepdim=True)
        # anomaly_map = F.l1_loss(rec, feats, reduction='none').mean(1, keepdim=True)
        # anomaly_map = F.mse_loss(rec, feats, reduction='none').mean(1, keepdim=True)
        anomaly_map = F.interpolate(anomaly_map, x.shape[-2:], mode='bilinear',
                                    align_corners=True)

        # Anomaly score only where object in the image, i.e. at x > 0
        anomaly_score = []
        for i in range(x.shape[0]):
            roi = anomaly_map[i][x[i] > 0]
            roi = roi[roi > torch.quantile(roi, 0.9)]
            anomaly_score.append(roi.mean())
        anomaly_score = torch.stack(anomaly_score)
        return anomaly_map, anomaly_score

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.state_dict(), path)


class FeatureVAE(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.extractor = Extractor(inp_size=config.image_size,
                                   keep_feature_prop=config.keep_feature_prop)

        config.c_feats = self.extractor.c_feats
        hidden_dims = config.hidden_dims
        enc_hidden_dims = hidden_dims[:-1] + [hidden_dims[-1] * 2]
        self.enc = vanilla_feature_encoder(self.extractor.c_feats, enc_hidden_dims,
                                           use_batchnorm=True,
                                           dropout=config.dropout, bias=False)
        # self.mu = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 1)
        # self.logvar = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 1)
        self.dec = vanilla_feature_decoder(self.extractor.c_feats, hidden_dims,
                                           use_batchnorm=True,
                                           dropout=config.dropout, bias=False)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor):
        with torch.no_grad():
            feats = self.extractor(x)
        enc_out = self.enc(feats)
        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        # mu = self.mu(enc_out)
        # logvar = self.logvar(enc_out)
        z = self.reparameterize(mu, logvar)
        y = self.dec(z)
        return feats, y, mu, logvar

    def loss(self, x: Tensor, kl_weight: float = 1.0):
        """Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        Args:
            x (Tensor): Input tensor
            kl_weight (float): Weight for the KL loss
        """
        feats, y, mu, logvar = self(x)
        rec_loss = SSIMLoss(size_average=True)(y, feats)
        kl_loss = torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))
        loss = rec_loss + kl_weight * kl_loss
        return {
            'loss': loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
        }

    def predict_anomaly(self, x):
        feats, mu, _, _ = self(x)
        map_small = SSIMLoss(size_average=False)(mu, feats).mean(1, keepdim=True)
        # map_small = F.l1_loss(y, feats, reduction='none').sum(1, keepdim=True)
        anomaly_map = F.interpolate(map_small, x.shape[-2:], mode='bilinear',
                                    align_corners=True)
        anomaly_score = torch.tensor([m[x_ > 0].mean() for m, x_ in zip(anomaly_map, x)])
        return anomaly_map, anomaly_score


class FeatureDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = vanilla_feature_encoder(
            config.c_feats,
            hidden_dims=config.discriminator_hidden_dims,
            use_batchnorm=False,
            dropout=0.0, bias=True
        )
        print(self.encoder)

    def forward(self, x):
        res = [x]
        for name, module in self.encoder.named_modules():
            if len(name) > 0 and '.' not in name:
                res.append(module(res[-1]))
        return res[1:]


class EnsembleFAE(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.n_ensemble = config.n_ensemble
        self.extractor = Extractor(inp_size=config.image_size,
                                   keep_feature_prop=config.keep_feature_prop)
        config.in_channels = self.extractor.c_feats
        self.faes = nn.ModuleList([FeatureAutoencoder(config) for _ in range(self.n_ensemble)])

    def forward(self, x: Tensor):
        feats = self.extractor(x)
        recs = []
        for fae in self.faes:
            rec = fae(feats)
            recs.append(rec)
        return feats, torch.stack(recs, dim=1)  # (B, n_ensemble, C, H, W)

    def loss(self, x: Tensor):
        feats, recs = self(x)
        loss = torch.stack([
            SSIMLoss(size_average=True)(recs[:, i], feats).mean() for i in range(self.n_ensemble)
            # F.l1_loss(recs[:, i], feats, reduction='mean') for i in range(self.n_ensemble)
        ]).mean()
        return {'loss': loss}

    def predict_anomaly(self, x: Tensor):
        """Returns an anomaly map and an anomaly score."""
        feats, recs = self(x)

        # Compute pixel-wise mahalanobis distance
        anomaly_map = torch.stack([
            mahalanobis_distance_image(feat, rec) for feat, rec in zip(feats, recs)
        ])[:, None]

        # Resize to original size
        anomaly_map = F.interpolate(anomaly_map, x.shape[-2:], mode='bilinear',
                                    align_corners=True)

        # Compute anomaly score
        anomaly_score = torch.tensor([m[x_ > 0].mean() for m, x_ in zip(anomaly_map, x)])

        return anomaly_map, anomaly_score

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.state_dict(), path)


if __name__ == '__main__':
    from argparse import Namespace
    config = Namespace()
    config.image_size = 128
    config.n_ensemble = 5
    config.hidden_dims = [400, 450, 500, 600]
    config.dropout = 0.2
    config.discriminator_hidden_dims = [400, 450, 500, 1]
    config.keep_feature_prop = 1.0
    device = "cuda"

    x = torch.randn(32, 1, *[config.image_size] * 2).to(device)
    fae = FeatureReconstructor(config).to(device)
    from time import perf_counter
    t_start = perf_counter()
    anomaly_map, anomaly_score = fae.predict_anomaly(x)
    # loss = fae.loss(x)
    print(f"{perf_counter() - t_start:.2f}s")
    import IPython; IPython.embed(); exit(1)
