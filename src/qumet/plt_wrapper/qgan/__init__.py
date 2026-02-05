"""QGAN PyTorch Lightning wrapper modules.

This module provides wrapper classes for training various Quantum Generative
Adversarial Network architectures using PyTorch Lightning.
"""

from .discrete_gen import QGANDiscreteGenModelWrapper
from .generation import QGANGenerationModelWrapper
from .image_generation import (
    APQGANWrapper,
    GANWrapper,
    MosaiQGANWrapper,
    PatchGANWrapper,
    PQWGANWrapper,
    ProbsQGANWrapper,
    QINRWrapper,
)
from .probs_gen import QGANProbsGenModelWrapper


def denorm(x):
    """Denormalize tensor values from [-1, 1] to [0, 1] range.
    
    Args:
        x: Input tensor with values in [-1, 1] range.
    
    Returns:
        torch.Tensor: Denormalized tensor clamped to [0, 1].
    """
    out = (x + 1) / 2
    return out.clamp(0, 1)
