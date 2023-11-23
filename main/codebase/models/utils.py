from dataclasses import dataclass
from enum import Enum

class ModelSource(Enum):
    """
    The source of the model, must be one of the following:
    - MANUAL: manually implemented
    - TOY: toy model for testing and debugging
    - PHYSICAL: model that perform classification using physical data point vectors
    """

    MANUAL = "manual"

class ModelTaskType(Enum):
    """
    The task type of the model, must be one of the following:
    - VISION: computer vision
    - PHYSICAL: categorize data points into predefined classes based on their features or attributes
    """

    VISION = "vision"
    PHYSICAL = "physical"

@dataclass
class QumetModelInfo:
    """
    The model info for QuMET.
    """

    # model name
    name: str

    model_source: ModelSource
    task_type: ModelTaskType

    # Vision models
    image_classification: bool = False

    # Physical models
    physical_data_point_classification: bool = False

    def __post_init__(self):
        self.model_source = (
            ModelSource(self.model_source)
            if isinstance(self.model_source, str)
            else self.model_source
        )
        self.task_type = (
            ModelTaskType(self.task_type)
            if isinstance(self.task_type, str)
            else self.task_type
        )

        # Vision models
        if self.task_type == ModelTaskType.VISION:
            assert self.image_classification, "Must be an image classification model"

        # Classification models
        if self.task_type == ModelTaskType.PHYSICAL:
            assert (
                self.physical_data_point_classification
            ), "Must be an physical data point classification model"


