import math

import torch
import torch.nn.functional as F
from torch import Tensor


def smooth_tensor(tensor: Tensor, kernel_size: int = 7, sigma: int = 3,
                  channels: int = 1) -> Tensor:
    if kernel_size % 2 == 0:
        raise RuntimeError("Kernel size must be odd.")

    # Make 1D gaussian kernel
    kernel = Tensor(
        [math.exp(-(x - kernel_size // 2)**2 / float(2 * sigma**2)) for x in range(kernel_size)])

    # To 2D
    kernel = kernel.unsqueeze(1)
    kernel = kernel @ kernel.T

    # Normalize
    kernel /= kernel.sum()

    # Reshape and repeat
    kernel = kernel[None, None].repeat(channels, 1, 1, 1).to(tensor.device)

    # Apply smoothing kernel convolution
    return F.conv2d(tensor, kernel, padding=kernel_size // 2)


def normalize(tensor: Tensor) -> Tensor:
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor


if __name__ == '__main__':
    t = torch.randn(1, 1, 32, 32)
    print(t.min(), t.max())
    t = smooth_tensor(t)
    print(t.min(), t.max())
