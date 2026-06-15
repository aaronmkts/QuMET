"""Neural network utilities for QuMET models.

This module provides utilities for feature extraction and forward hook registration.
"""

from torch import nn


class FeatureExtractor:
    """Extract intermediate layer features from neural networks.

    Registers forward hooks to capture outputs from specified layers during
    forward passes.

    Attributes:
        features: List storing captured feature outputs.
    """

    def __init__(self) -> None:
        """Initialize the feature extractor."""
        super().__init__()
        self.features = []

    def __call__(self, module: nn.Module):
        """Register a forward hook on the module.

        Args:
            module: PyTorch module to extract features from.

        Returns:
            nn.Module: The module with registered hook.
        """
        module.register_forward_hook(self.forward_hook())
        return module

    def forward_hook(self):
        """Create a forward hook function.

        Returns:
            Callable: Hook function that captures module outputs.
        """

        def fn(module, input, output):
            self.features.append(output)

        return fn

    def clean(self):
        """Clear the stored features."""
        self.features = []
