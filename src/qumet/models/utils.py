"""Model utility classes and types for QuMET.

This module defines core enums and dataclasses for model metadata,
including model types, task types, and model information structures.
"""

from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Enumeration of supported model architectures.

    Attributes:
        QGAN: Quantum Generative Adversarial Network.
        GAN: Classical Generative Adversarial Network.
        QCBM: Quantum Circuit Born Machine.
        VAE: Variational Autoencoder.
    """

    QGAN = "qgan"
    GAN = "gan"
    QCBM = "qcbm"
    VAE = "vae"


class ModelTaskType(Enum):
    """Enumeration of model task types.

    Attributes:
        GENERATION: Unsupervised learning task to generate new data samples.
    """

    GENERATION = "generation"


@dataclass
class QumetModelInfo:
    """Model metadata and configuration for QuMET models.

    Attributes:
        name: Model identifier name.
        model_type: Type of model architecture (QGAN, QCBM, VAE, etc.).
        task_type: Type of task the model performs (GENERATION, etc.).
        bitstring_sampling: Whether the model supports bitstring sampling.
        observable_sampling: Whether the model supports observable sampling.
    """

    name: str
    model_type: ModelType
    task_type: ModelTaskType
    bitstring_sampling: bool = False
    observable_sampling: bool = False

    def __post_init__(self):
        """Validate and convert model configuration after initialization."""
        self.model_type = (
            ModelType(self.model_type)
            if isinstance(self.model_type, str)
            else self.model_type
        )
        self.task_type = (
            ModelTaskType(self.task_type)
            if isinstance(self.task_type, str)
            else self.task_type
        )

        if self.task_type == ModelTaskType.GENERATION:
            assert (
                self.bitstring_sampling + self.observable_sampling >= 1
            ), "Must be a generative model"

    @property
    def is_generation_model(self):
        """Check if the model is a generation model.

        Returns:
            bool: True if the model performs generation tasks.
        """
        return self.task_type == ModelTaskType.GENERATION
