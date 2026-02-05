"""Dataset loading and management for QuMET.

This module provides a unified interface for loading and managing datasets,
including vision datasets (MNIST, CIFAR) and manual quantum datasets.
"""

import os

import lightning.pytorch as pl
import torch
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader

from ..tools.registry import MAIN_CACHE_DIR
from .manual import MANUAL_DATASET_MAPPING, get_manual_dataset, get_manual_dataset_cls
from .vision import VISION_DATASET_MAPPING, get_vision_dataset, get_vision_dataset_cls
from .vision.transforms.transformations import get_transform

DATASET_CACHE_DIR = MAIN_CACHE_DIR / "dataset"


def get_dataset_info(name: str):
    """Get metadata information for a dataset.
    
    Args:
        name: Name of the dataset.
    
    Returns:
        dict: Dataset information including keys like 'num_classes' and 'image_size'
            for vision datasets, or other relevant metadata for manual datasets.
    
    Raises:
        ValueError: If the dataset is not supported.
    """
    name = name.lower()
    if name in VISION_DATASET_MAPPING:
        return get_vision_dataset_cls(name).info
    elif name in MANUAL_DATASET_MAPPING:
        return get_manual_dataset_cls(name).info
    else:
        raise ValueError(f"Dataset {name} is not supported")


def get_dataset(
    name: str,
    split: bool,
    transform,
    discretise: bool,
    n_samples: int,
    n_qubits: int,
    num_workers: int = os.cpu_count(),
    model_name: str = None,
):
    """Load a dataset with specified configuration.
    
    Args:
        name: Name of the dataset.
        split: Dataset split ('train', 'validation', 'test', or 'pred').
        transform: Transform to apply to the data.
        discretise: Whether to discretise the data.
        n_samples: Number of samples to load.
        n_qubits: Number of qubits for quantum datasets.
        num_workers: Number of workers for data loading. Defaults to CPU count.
        model_name: Optional model name for model-dependent transforms.
    
    Returns:
        torch.utils.data.Dataset: Configured dataset instance with transforms.
    
    Raises:
        AssertionError: If split is not one of 'train', 'validation', 'test', 'pred'.
        ValueError: If the dataset is not supported.
    """
    global DATASET_CACHE_DIR
    MAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of train, validation, test, pred"

    name = name.lower()
    if name in MANUAL_DATASET_MAPPING:

        dataset = get_manual_dataset(
            name=name,
            split=split,
            transform=transform,
            discretise=discretise,
            n_samples=n_samples,
            n_qubits=n_qubits,
            model_name=model_name,
        )

    elif name in VISION_DATASET_MAPPING:
        path = DATASET_CACHE_DIR / name
        dataset = get_vision_dataset(
            name, path, split, n_samples, model_name, transform
        )
    else:
        raise ValueError(f"Dataset {name} is not supported")
    return dataset


AVAILABLE_DATASETS = list(VISION_DATASET_MAPPING.keys()) + list(
    MANUAL_DATASET_MAPPING.keys()
)


class QuMETDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for unified dataset loading in QuMET.
    
    This class provides a standardized interface for loading and managing datasets
    across different splits (train, validation, test, predict) with support for
    various transforms and preprocessing options.
    
    Note:
        QuMETDataModule requires .prepare_data() and .setup() to be called before
        accessing dataloaders if not passed to a PyTorch Lightning Trainer.
    
    Attributes:
        name: Dataset name.
        num_workers: Number of data loading workers.
        n_samples: Number of samples to load.
        n_qubits: Number of qubits for quantum datasets.
        transform: Transform type to apply.
        discretise: Whether to discretise the data.
        model_name: Optional model name for model-specific transforms.
        batch_size: Batch size for data loaders.
        dataset_info: Metadata about the dataset.
        transform_instance: Fitted transform instance (for PCA, VAE, etc.).
    """

    def __init__(
        self,
        name: str,
        batch_size: int,
        transform: str,
        discretise: bool,
        n_samples: int,
        n_qubits: int,
        num_workers: int,
        model_name: str = None,
    ) -> None:
        """Initialize the QuMET DataModule.
        
        Args:
            name: Dataset name.
            batch_size: Batch size for dataloaders.
            transform: Transform type ('minmax', 'pit', 'pca', 'vae').
            discretise: Whether to discretise the data.
            n_samples: Number of samples to load.
            n_qubits: Number of qubits for quantum datasets.
            num_workers: Number of data loading workers.
            model_name: Optional model name for model-specific transforms.
        """
        super().__init__()

        self.name = name
        self.num_workers = num_workers
        self.n_samples = n_samples
        self.n_qubits = n_qubits
        self.transform = transform
        self.discretise = discretise
        self.model_name = model_name

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pred_dataset = None
        self.dataset_info = get_dataset_info(name)
        self.transform_instance = None

        self.batch_size = 1 if self.dataset_info.probs_generation else batch_size

    def prepare_data(self) -> None:
        """Download and prepare datasets.
        
        This method is called only on a single GPU/process to handle dataset
        downloads and preprocessing that should not be duplicated across workers.
        """

        if self.transform in ["pca", "vae"]:
            match self.transform:
                case "pca":
                    self.transform_instance = get_transform(
                        self.transform, n_components=40
                    )
                case "vae":
                    pass
                case _:
                    raise ValueError(f"Unknown transform {self.transform}")

            train_dataset = get_dataset(
                self.name,
                split="train",
                num_workers=self.num_workers,
                n_samples=self.n_samples,
                n_qubits=self.n_qubits,
                transform=self.transform,
                discretise=self.discretise,
                model_name=self.model_name,
            )

            all_data = []
            for img, _ in train_dataset:
                all_data.append(img)
            all_data = torch.stack(all_data)
            self.transform_instance.fit(all_data)

        else:
            train_dataset = get_dataset(
                self.name,
                split="train",
                num_workers=self.num_workers,
                n_samples=self.n_samples,
                n_qubits=self.n_qubits,
                transform=self.transform,
                discretise=self.discretise,
                model_name=self.model_name,
            )

        val_dataset = get_dataset(
            self.name,
            split="validation",
            num_workers=self.num_workers,
            n_samples=self.n_samples,
            n_qubits=self.n_qubits,
            transform=self.transform,
            discretise=self.discretise,
            model_name=self.model_name,
        )
        test_dataset = get_dataset(
            self.name,
            split="test",
            num_workers=self.num_workers,
            n_samples=self.n_samples,
            n_qubits=self.n_qubits,
            transform=self.transform,
            discretise=self.discretise,
            model_name=self.model_name,
        )
        pred_dataset = get_dataset(
            self.name,
            split="pred",
            num_workers=self.num_workers,
            n_samples=self.n_samples,
            n_qubits=self.n_qubits,
            transform=self.transform,
            discretise=self.discretise,
            model_name=self.model_name,
        )

        train_dataset.prepare_data()
        if not self.dataset_info.preprocess_one_split_for_all:
            val_dataset.prepare_data()
            if test_dataset is not None:
                test_dataset.prepare_data()
            if pred_dataset is not None:
                pred_dataset.prepare_data()

    def setup(self, stage: str = None) -> None:
        """Set up datasets for the specified stage.
        
        Args:
            stage: Training stage ('fit', 'validate', 'test', 'predict', or None).
                If None, sets up datasets for all stages.
        """
        if self.transform_instance is not None:
            transform = self.transform_instance
        else:
            transform = self.transform

        if stage in ["fit", None]:
            self.train_dataset = get_dataset(
                self.name,
                split="train",
                num_workers=self.num_workers,
                n_samples=self.n_samples,
                n_qubits=self.n_qubits,
                transform=transform,
                discretise=self.discretise,
                model_name=self.model_name,
            )
            if self.train_dataset is not None:
                self.train_dataset.setup()
        if stage in ["fit", "validate", None]:
            self.val_dataset = get_dataset(
                self.name,
                split="validation",
                num_workers=self.num_workers,
                n_samples=self.n_samples,
                n_qubits=self.n_qubits,
                transform=transform,
                discretise=self.discretise,
                model_name=self.model_name,
            )
            if self.val_dataset is not None:
                self.val_dataset.setup()
        if stage in ["test", None]:
            self.test_dataset = get_dataset(
                self.name,
                split="test",
                num_workers=self.num_workers,
                n_samples=self.n_samples,
                n_qubits=self.n_qubits,
                transform=transform,
                discretise=self.discretise,
                model_name=self.model_name,
            )
            if self.test_dataset is not None:
                self.test_dataset.setup()
        if stage in ["predict", None]:
            self.pred_dataset = get_dataset(
                self.name,
                split="pred",
                num_workers=self.num_workers,
                n_samples=self.n_samples,
                n_qubits=self.n_qubits,
                transform=transform,
                discretise=self.discretise,
                model_name=self.model_name,
            )
            if self.pred_dataset is not None:
                self.pred_dataset.setup()

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.
        
        Returns:
            DataLoader: Training data loader with shuffling enabled.
        
        Raises:
            RuntimeError: If training dataset is not available.
        """
        if self.train_dataset is None:
            raise RuntimeError(
                "The train dataset is not available"
                "probably because the train set does not have ground truth labels, "
                "or the train dataset does not exist."
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.
        
        Returns:
            DataLoader: Validation data loader without shuffling.
        
        Raises:
            RuntimeError: If validation dataset is not available.
        """
        if self.val_dataset is None:
            raise RuntimeError(
                "The validation dataset is not available"
                "probably because the validation set does not have ground truth labels, "
                "or the validation dataset does not exist."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader.
        
        Returns:
            DataLoader: Test data loader without shuffling.
        
        Raises:
            RuntimeError: If test dataset is not available.
        """
        if self.test_dataset is None:
            raise RuntimeError(
                "The test dataset is not available"
                "probably because the test set does not have ground truth labels, "
                "or the test dataset does not exist. For the former case, try predict_dataloader"
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def pred_dataloader(self) -> DataLoader:
        """Get the prediction dataloader.
        
        Returns:
            DataLoader: Prediction data loader without shuffling.
        
        Raises:
            RuntimeError: If prediction dataset is not available.
        """
        if self.pred_dataset is None:
            raise RuntimeError("The pred dataset is not available.")
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
