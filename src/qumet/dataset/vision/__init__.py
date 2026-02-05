"""Vision datasets for QuMET.

This module provides access to computer vision datasets including MNIST,
Fashion-MNIST, and CIFAR-10 with QuMET-specific preprocessing and transforms.
"""

import os
from pathlib import Path

from .cifar import Cifar10QuMET, get_cifar_dataset
from .fashion_mnist import Fashion_MNISTQuMET, get_fashion_mnist_dataset
from .mnist import MNISTQuMET, get_mnist_dataset
from .transforms import get_vision_dataset_transform


def get_vision_dataset(
    name: str,
    path: os.PathLike,
    split: str,
    n_samples: int,
    model_name: str,
    transform: str,
):
    """Load a vision dataset with specified configuration.

    Args:
        name: Name of the vision dataset ('mnist', 'fashion_mnist', 'cifar10').
        path: Path to store/load the dataset.
        split: Dataset split ('train', 'validation', 'test', or 'pred').
        n_samples: Number of samples to load.
        model_name: Optional model name for model-dependent transforms.
        transform: Transform type to apply ('minmax', 'pit', etc.).

    Returns:
        torch.utils.data.Dataset: Configured vision dataset with transforms.

    Raises:
        AssertionError: If split is not one of the valid options.
    """
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of train, validation, test, pred"

    preprocessing = transform
    train = split == "train"
    transform = get_vision_dataset_transform(
        name, train, model_name, preprocessing
    )  # Torchvision transforms

    match name:
        case "mnist":
            dataset = get_mnist_dataset(
                name, path, train, n_samples, transform, model_name
            )
        case "fashion_mnist":
            dataset = get_fashion_mnist_dataset(
                name, path, train, n_samples, transform, model_name
            )
        case "cifar10":
            dataset = get_cifar_dataset(
                name, path, train, transform, n_samples, model_name
            )

    return dataset


VISION_DATASET_MAPPING = {
    "mnist": MNISTQuMET,
    "fashion_mnist": Fashion_MNISTQuMET,
    "cifar10": Cifar10QuMET,
}


def get_vision_dataset_cls(name: str):
    """Get the dataset class for a vision dataset.

    Args:
        name: Name of the vision dataset.

    Returns:
        type: Dataset class for the specified dataset.

    Raises:
        AssertionError: If dataset name is not recognized.
    """
    assert name in VISION_DATASET_MAPPING, f"Unknown dataset {name}"
    return VISION_DATASET_MAPPING[name.lower()]
