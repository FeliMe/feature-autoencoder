import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def mahalanobis_distance_image(x: Tensor, y: Tensor) -> Tensor:
    """Compute the pixel-wise mahalanobis distance between an image x and a set of images y

    Args:
        x: Single image, shape = (c, h, w)
        y: Set of images, shape = (n, c, h, w)
    where c is the feature length of each pixel, h and w are the height and width
    of the image, and b is the number of images in y.
    """

    n, c, h, w = y.shape
    d = h * w

    # Reshape
    x = x.view(c, d).T  # (h*w, c)
    y = y.view(n, c, d)
    y = y.permute(2, 1, 0)  # (h*w, c, n)

    # Compute mean of y
    mu = y.mean(dim=2)  # (h*w, c)

    # Compute the inverse covariance matrix of y
    y_center = y - mu[..., None]  # (h*w, c, n)
    cov = torch.bmm(y_center, y_center.permute(0, 2, 1)) / (n - 1)  # (h*w, c, c)
    # Add a small number to the diagonal to avoid numerical instability when inverting
    cov += torch.eye(c, c, device=cov.device)[None].repeat(d, 1, 1) * 0.01  # 1e-5
    cov_inv = torch.inverse(cov)  # (h*w, c, c)

    # Compute the mahalanobis distance
    delta = x - mu  # (h*w, c)
    dist = torch.einsum('bc,bck,bk->b', delta, cov_inv, delta)  # (h*w)
    # Remove overflow numbers to avoid NaN
    dist[dist < 0] = torch.finfo(torch.float32).max
    dist = torch.sqrt(dist)
    # dist2 = torch.stack([torch.sqrt((delt.T @ c_inv @ delt)) for delt, c_inv in zip(delta, cov_inv)])

    # Reshape to image
    dist = dist.view(h, w)  # (h, w)

    return dist


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        from torchvision.models import resnet18
        resnet = resnet18(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False

        self.model = resnet.layer4
        self.c_in = 256  # Layer 4 expects 256 channels as input

    def select_channels(self, n_channels_inp: int):
        c_select = torch.cat(
            [torch.ones(self.c_in), torch.zeros(n_channels_inp - self.c_in)]
        ).bool()
        perm = torch.randperm(c_select.shape[0])
        c_select = c_select[perm]
        return c_select

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c_select = self.select_channels(pred.shape[1])
        feats_pred = self.model(pred[:, c_select])
        feats_target = self.model(target[:, c_select])
        return F.mse_loss(feats_pred, feats_target, reduction='none')


if __name__ == '__main__':
    pred = torch.randn(64, 448, 32, 32)
    target = torch.randn(64, 448, 32, 32)
    loss_fn = PerceptualLoss()
    print(loss_fn(pred, target))
