import torch
import torch.nn as nn
from torch import autograd, Tensor


def calc_gradient_penalty(D: nn.Module, x_real: Tensor, x_fake: Tensor) -> Tensor:
    """
    Calculate the gradient penalty loss for WGAN-GP.
    Gradient norm for an interpolated version of x_real and x_fake.
    See https://arxiv.org/abs/1704.00028

    :param D: Discriminator
    :param x_real: Real images
    :param x_fake: Fake images
    """

    # Useful variables
    device = x_real.device
    b = x_real.size(0)

    # Interpolate images
    alpha = torch.rand(b, 1, 1, 1, device=device)
    interp = alpha * x_real.detach() + ((1 - alpha) * x_fake.detach())
    interp.requires_grad = True

    # Forward discrminator
    d_out, _ = D(interp)

    # Calculate gradients
    grads = autograd.grad(outputs=d_out, inputs=interp,
                          grad_outputs=torch.ones_like(d_out),
                          create_graph=True)[0]
    grads = grads.view(b, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()

    return gp


if __name__ == '__main__':
    device = "cuda"
    torch.manual_seed(0)
    x_real = torch.randn(2, 1, 64, 64).to(device)
    x_fake = torch.randn(2, 1, 64, 64).to(device)
    D = nn.Linear(64 * 64, 1).to(device)
    gp = calc_gradient_penalty(D, x_real, x_fake)
    print(gp)
