"""Tests for qumet.plt_wrapper.utils module."""
import pytest
import torch
from qumet.plt_wrapper.utils import compute_gradient_penalty


class TestComputeGradientPenalty:
    """Tests for compute_gradient_penalty."""

    @staticmethod
    def _make_critic(in_channels=3, img_size=32):
        """Create a simple CNN critic that accepts 4D image tensors (B,C,H,W)."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 4, 3, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 1),
        )

    def test_returns_scalar_tensor(self):
        """Gradient penalty should return a scalar tensor."""
        critic = self._make_critic()
        real = torch.randn(4, 3, 32, 32)
        fake = torch.randn(4, 3, 32, 32)
        gp = compute_gradient_penalty(critic, real, fake)
        assert torch.is_tensor(gp)
        assert gp.dim() == 0  # scalar

    def test_requires_grad(self):
        """Gradient penalty should support backpropagation."""
        critic = self._make_critic()
        real = torch.randn(4, 3, 32, 32)
        fake = torch.randn(4, 3, 32, 32)
        gp = compute_gradient_penalty(critic, real, fake)
        # Should be able to call backward
        gp.backward()

    def test_different_real_fake_gives_nonzero(self):
        """Different real and fake images should give non-zero penalty."""
        critic = self._make_critic()
        real = torch.ones(4, 3, 32, 32)
        fake = torch.zeros(4, 3, 32, 32)
        gp = compute_gradient_penalty(critic, real, fake)
        assert gp.item() > 0

    def test_identical_real_fake_gives_one(self):
        """Identical real and fake: gradients are zero, so (||grad|| - 1)^2 ≈ 1."""
        critic = self._make_critic()
        real = torch.randn(4, 3, 32, 32)
        fake = real.clone()
        gp = compute_gradient_penalty(critic, real, fake)
        # When real == fake, critic output is constant w.r.t. input,
        # so gradients are ~0, and penalty = (0 - 1)^2 ≈ 1
        assert abs(gp.item() - 1.0) < 0.1

    def test_handles_different_batch_sizes(self):
        critic = self._make_critic()
        for bs in [1, 2, 8]:
            real = torch.randn(bs, 3, 32, 32)
            fake = torch.randn(bs, 3, 32, 32)
            gp = compute_gradient_penalty(critic, real, fake)
            assert gp.dim() == 0
