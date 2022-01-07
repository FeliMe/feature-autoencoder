import os
import random
import numpy as np
import torch

import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
