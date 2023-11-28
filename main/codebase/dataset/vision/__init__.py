import os
from pathlib import Path

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

from .mnist import get_mnist_dataset, MNISTQuMET
from .transforms import get_vision_dataset_transform

def get_vision_dataset(name: str, path: os.PathLike, split: str, model_name: str):
    """
    Args:
        name (str): name of the dataset
        path (str): path to the dataset
        train (bool): whether the dataset is used for training
        model_name (Optional[str, None]): name of the model. Some pretrained models have
        model-dependent transforms for training and evaluation.
    Returns:
        dataset (torch.utils.data.Dataset): dataset (with transforms)
    """
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of train, validation, test, pred"

    train = split == "train"
    transform = get_vision_dataset_transform(name, train, model_name)

    match name:
        case "mnist":
            dataset = get_mnist_dataset(name, path, train, transform)
    return dataset


VISION_DATASET_MAPPING = {
    "mnist": MNISTQuMET,
}


def get_vision_dataset_cls(name: str):
    assert name in VISION_DATASET_MAPPING, f"Unknown dataset {name}"
    return VISION_DATASET_MAPPING[name.lower()]