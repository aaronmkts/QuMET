"""Dataset utility classes and types for QuMET.

This module defines enums and dataclasses for dataset metadata, including
dataset sources, splits, and comprehensive dataset information structures.
"""

from dataclasses import dataclass
from enum import Enum


class DatasetSource(Enum):
    """Enumeration of dataset sources.

    Attributes:
        MANUAL: Manual dataset from QuMET.
        HF_DATASETS: Dataset from HuggingFace datasets library.
        TORCHVISION: Dataset from torchvision.
        OTHERS: Other dataset sources.
    """

    MANUAL = "manual"
    HF_DATASETS = "hf_datasets"
    TORCHVISION = "torchvision"
    OTHERS = "others"


class DatasetSplit(Enum):
    """Enumeration of dataset splits.

    Attributes:
        TRAIN: Training split.
        VALIDATION: Validation split.
        TEST: Test split.
        PRED: Prediction split.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PRED = "pred"


@dataclass
class QuMETDatasetInfo:
    """Comprehensive dataset metadata for QuMET datasets.

    Attributes:
        name: Dataset name.
        dataset_source: Source of the dataset (MANUAL, HF_DATASETS, etc.).
        available_splits: Tuple of available dataset splits.
        requires_preprocessing: Whether the dataset requires preprocessing.
        preprocess_one_split_for_all: Whether preprocessing one split applies to all.
        probs_generation: Whether the dataset is for probability generation.
        bitsring_generation: Whether the dataset is for bitstring generation.
        image_generation: Whether the dataset is for image generation.
        continuous_generation: Whether the dataset is for continuous generation.
        num_classes: Number of classes for classification tasks.
        image_size: Image dimensions for vision datasets.
        num_features: Number of features in the dataset.
    """

    name: str
    dataset_source: DatasetSource
    available_splits: tuple[DatasetSplit]
    requires_preprocessing: bool = False
    preprocess_one_split_for_all: bool = True
    probs_generation: bool = False
    bitsring_generation: bool = False
    image_generation: bool = False
    continuous_generation: bool = False
    num_classes: int = None
    image_size: tuple[int] = None
    num_features: int = None

    def __post_init__(self):
        """Validate and convert dataset configuration after initialization."""
        self.dataset_source = (
            DatasetSource(self.dataset_source)
            if isinstance(self.dataset_source, str)
            else self.dataset_source
        )
        self.available_splits = tuple(
            DatasetSplit(split) if isinstance(split, str) else split
            for split in self.available_splits
        )
        self._entries = {
            "name",
            "dataset_source",
            "available_splits",
            "probs_generation",
            "bitsring_generation",
            "continuous_generation",
            "image_generation",
            "num_classes",
            "image_size",
            "num_features",
        }

    @property
    def train_split_available(self):
        """Check if training split is available.

        Returns:
            bool: True if training split is available.
        """
        return DatasetSplit.TRAIN in self.available_splits

    @property
    def validation_split_available(self):
        """Check if validation split is available.

        Returns:
            bool: True if validation split is available.
        """
        return DatasetSplit.VALIDATION in self.available_splits

    @property
    def test_split_available(self):
        """Check if test split is available.

        Returns:
            bool: True if test split is available.
        """
        return DatasetSplit.TEST in self.available_splits

    @property
    def pred_split_available(self):
        """Check if prediction split is available.

        Returns:
            bool: True if prediction split is available.
        """
        return DatasetSplit.PRED in self.available_splits

    def __getitem__(self, key: str):
        """Get dataset info attribute by key.

        Args:
            key: Attribute key to retrieve.

        Returns:
            Attribute value.

        Raises:
            KeyError: If key is not found in dataset info.
        """
        if key in self._entries:
            return getattr(self, key)
        else:
            raise KeyError(f"key {key} not found in QuMetDatasetInfo")


def add_dataset_info(
    name: str,
    dataset_source: DatasetSource,
    available_splits: tuple[DatasetSplit],
    bitsring_generation: bool = False,
    continuous_generation: bool = False,
    probs_generation: bool = False,
    image_generation: bool = False,
    num_classes: int = None,
    image_size: tuple[int] = None,
    num_features: int = None,
):
    """Decorator factory for adding dataset info to a dataset class.

    This decorator attaches a QuMETDatasetInfo instance to a dataset class,
    providing metadata about the dataset's capabilities and structure.

    Args:
        name: Dataset name.
        dataset_source: Source of the dataset (DatasetSource enum).
        available_splits: Tuple of available splits (DatasetSplit enums).
        bitsring_generation: Whether dataset is for bitstring generation.
        continuous_generation: Whether dataset is for continuous generation.
        probs_generation: Whether dataset is for probability generation.
        image_generation: Whether dataset is for image generation.
        num_classes: Number of classes for classification tasks.
        image_size: Image dimensions for vision datasets.
        num_features: Number of features in the dataset.

    Returns:
        Callable: Decorator function that adds dataset info to a class.
    """

    def _add_dataset_info_to_cls(cls: type):
        """Add dataset info to the provided class.

        Args:
            cls: Dataset class to decorate.

        Returns:
            type: Decorated dataset class with info attribute.
        """
        cls.info = QuMETDatasetInfo(
            name=name,
            dataset_source=dataset_source,
            available_splits=available_splits,
            probs_generation=probs_generation,
            bitsring_generation=bitsring_generation,
            continuous_generation=continuous_generation,
            image_generation=image_generation,
            num_classes=num_classes,
            image_size=image_size,
            num_features=num_features,
        )

        return cls

    return _add_dataset_info_to_cls
