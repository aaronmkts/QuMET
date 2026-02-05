"""Base classes for PyTorch Lightning model wrappers.

This module provides the base wrapper class and result dataclass used by
all model-specific wrappers in QuMET.
"""

import io
from dataclasses import dataclass, field

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image


class WrapperBase(pl.LightningModule):
    """Base class for all QuMET PyTorch Lightning wrappers.

    Provides common functionality for wrapping models with PyTorch Lightning,
    including learning rate management, optimizer configuration, and visualization.

    Attributes:
        model: Wrapped model instance.
        learning_rate: Learning rate for optimization.
        weight_decay: Weight decay for regularization.
        loss_fn: Loss function (default: CrossEntropyLoss).
        epochs: Number of training epochs.
        optimizer: Optimizer name.
        freeze_modules: Modules to freeze during training.
    """

    def __init__(
        self,
        model,
        learning_rate=5e-4,
        weight_decay=0.0,
        epochs=1,
        optimizer=None,
        freeze_modules=None,
    ):
        """Initialize the wrapper base.

        Args:
            model: Model to wrap.
            learning_rate: Learning rate for optimization. Defaults to 5e-4.
            weight_decay: Weight decay for regularization. Defaults to 0.0.
            epochs: Number of training epochs. Defaults to 1.
            optimizer: Optimizer name. Defaults to None.
            freeze_modules: Modules to freeze during training. Defaults to None.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optimizer
        self.freeze_modules = freeze_modules

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def training_step(self, batch):
        """Execute one training step.

        Args:
            batch: Batch of training data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def configure_optimizers(self):
        """Configure optimizers for training.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def plot_to_image(self, figure):
        """Convert a matplotlib figure to a PyTorch tensor image.

        Args:
            figure: Matplotlib figure to convert.

        Returns:
            torch.Tensor: Image tensor in CHW format.
        """
        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        plt.close(figure)
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        return im


@dataclass
class ValidationResult:
    """Container for validation step results.

    Stores various outputs from a validation step including images, distributions,
    and latent representations.

    Attributes:
        others: Dictionary for additional custom results.
        real_image: Real images from the dataset.
        fake_image: Generated/fake images from the model.
        recon_image: Reconstructed images (for autoencoders).
        label: Ground truth labels.
        encode_latent: Encoded latent representations.
        real_distribution: Probability distribution of real data.
        fake_distribution: Probability distribution of generated data.
    """

    others: dict = field(default_factory=dict)
    real_image: torch.Tensor = None
    fake_image: torch.Tensor = None
    recon_image: torch.Tensor = None
    label: torch.Tensor = None
    encode_latent: torch.Tensor = None
    real_distribution: torch.Tensor = None
    fake_distribution: torch.Tensor = None
