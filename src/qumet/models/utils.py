from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """
    The type of the model, must be one of the following:
    - QGAN: Quantum Generative Adversarial Network
    """

    QGAN = "qgan"
    VAE = "vae"


class ModelTaskType(Enum):
    """
    The task type of the model, must be one of the following:
    - GENERATION: Unsupervised learning to generate new data
    """

    GENERATION = "generation"


@dataclass
class QumetModelInfo:
    """
    The model info for QuMET.
    """

    # model name
    name: str

    model_type: ModelType
    task_type: ModelTaskType

    # Generation models
    bitstring_sampling: bool = False
    observable_sampling: bool = False
    

    def __post_init__(self):
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

        # Vision models
        if self.task_type == ModelTaskType.GENERATION:
            assert self.bitstring_sampling + self.observable_sampling >= 1, "Must be a generative model"

    @property
    def is_generation_model(self):
        return self.task_type == ModelTaskType.GENERATION
