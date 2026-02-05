"""Manual quantum datasets for QuMET.

This module provides access to manually constructed quantum datasets including
bars and stripes patterns, 2D Gaussian distributions, and their bitstring variants.
"""

import os
from pathlib import Path

from .bars_and_stripes import BarsAndStripesDataset
from .bitstrings import (
    TwoDGaussianDatasetB,
    TwoDGridGaussianDatasetB,
    TwoDRingGaussianDatasetB,
)
from .transforms import get_manual_dataset_transform
from .two_d_gaussian import TwoDGaussianDataset
from .two_d_grid_gaussian import TwoDGridGaussianDataset
from .two_d_ring_gaussian import TwoDRingGaussianDataset


def get_manual_dataset(
    name: str,
    split: str,
    transform: str,
    discretise: bool,
    model_name: str,
    n_qubits: int,
    n_samples: int,
):
    """Load a manual quantum dataset with specified configuration.
    
    Args:
        name: Name of the manual dataset.
        split: Dataset split ('train', 'validation', 'test', or 'pred').
        transform: Transform type to apply.
        discretise: Whether to discretise the data.
        model_name: Optional model name for model-dependent transforms.
        n_qubits: Number of qubits for quantum circuits.
        n_samples: Number of samples to generate.
    
    Returns:
        Dataset: Configured manual dataset with transforms, or None if split
            is not available for the dataset.
    
    Raises:
        AssertionError: If split is not one of the valid options.
        ValueError: If dataset name is not recognized.
    """

    ori_split = split
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of train, validation, test, pred"

    normaliser, discretisation = get_manual_dataset_transform(
        name, transform, discretise, model_name
    )

    match name:
        case "bars_and_stripes":
            dataset_cls = BarsAndStripesDataset
        case "2d_gaussian":
            dataset_cls = TwoDGaussianDataset
        case "2d_grid_gaussian":
            dataset_cls = TwoDGridGaussianDataset
        case "2d_ring_gaussian":
            dataset_cls = TwoDRingGaussianDataset
        case "2d_gaussian_b":
            dataset_cls = TwoDGaussianDatasetB
        case "2d_grid_gaussian_b":
            dataset_cls = TwoDGridGaussianDatasetB
        case "2d_ring_gaussian_b":
            dataset_cls = TwoDRingGaussianDatasetB
        case _:
            raise ValueError(f"Unknown dataset {name}")

    if ori_split == "train" and not dataset_cls.info.train_split_available:
        return None

    if ori_split == "validation" and not dataset_cls.info.validation_split_available:
        return None

    if ori_split == "test" and not dataset_cls.info.test_split_available:
        return None

    if ori_split == "pred" and not dataset_cls.info.pred_split_available:
        return None

    if ori_split == "pred" and dataset_cls.info.pred_split_available:
        split = "test"

    dataset = dataset_cls(split, normaliser, discretisation, n_qubits, n_samples)
    return dataset


MANUAL_DATASET_MAPPING = {
    "bars_and_stripes": BarsAndStripesDataset,
    "2d_gaussian": TwoDGaussianDataset,
    "2d_grid_gaussian": TwoDGridGaussianDataset,
    "2d_ring_gaussian": TwoDRingGaussianDataset,
    "2d_gaussian_b": TwoDGaussianDatasetB,
    "2d_grid_gaussian_b": TwoDGridGaussianDatasetB,
    "2d_ring_gaussian_b": TwoDRingGaussianDatasetB,
}


def get_manual_dataset_cls(name: str):
    """Get the dataset class for a manual dataset.
    
    Args:
        name: Name of the manual dataset.
    
    Returns:
        type: Dataset class for the specified dataset.
    
    Raises:
        AssertionError: If dataset name is not recognized.
    """
    assert name in MANUAL_DATASET_MAPPING, f"Unknown dataset {name}"
    return MANUAL_DATASET_MAPPING[name.lower()]
