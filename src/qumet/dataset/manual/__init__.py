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
    """
    Args:
        name (str): name of the dataset
        model_name (Optional[str, None]): name of the model. Some pretrained models have
        model-dependent transforms for training and evaluation.
    Returns:
        dataset (torch.utils.data.Dataset): dataset (with transforms)
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
    assert name in MANUAL_DATASET_MAPPING, f"Unknown dataset {name}"
    return MANUAL_DATASET_MAPPING[name.lower()]
