"""Utility functions for PyTorch Lightning wrappers.

This module provides utility functions for computing gradient penalties
and other training-related operations.
"""

import torch
import torch.autograd as autograd


def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Calculate the gradient penalty loss for WGAN-GP.

    Computes gradient penalty to enforce Lipschitz constraint on the critic
    for Wasserstein GAN with Gradient Penalty training.

    Args:
        critic: Critic/discriminator network.
        real_samples: Real data samples tensor of shape (B, C, W, H).
        fake_samples: Generated/fake data samples of same shape as real_samples.

    Returns:
        torch.Tensor: Scalar gradient penalty loss value.
    """
    batch_size, C, W, H = real_samples.shape
    epsilon = torch.rand(batch_size, 1, 1, 1).repeat(1, C, W, H).type_as(real_samples)
    interpolated_images = epsilon * real_samples + ((1 - epsilon) * fake_samples)
    interpolated_scores = critic(interpolated_images)

    gradients = autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = torch.mean(
        (1.0 - torch.sqrt(1e-8 + torch.sum(gradients**2, dim=1))) ** 2
    )
    return gradient_penalty
