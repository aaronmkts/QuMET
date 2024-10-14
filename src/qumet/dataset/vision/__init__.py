import os
from pathlib import Path

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

from .mnist import get_mnist_dataset, MNISTQuMET
from .fashion_mnist import get_fashion_mnist_dataset, Fashion_MNISTQuMET
from .cifar import get_cifar_dataset, Cifar10QuMET
from .transforms import get_vision_dataset_transform


def get_vision_dataset(name: str, path: os.PathLike, split: str, n_samples:int, model_name: str, transform: str):
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
    
    preprocessing = transform
    train = split == "train"
    transform = get_vision_dataset_transform(name, train, model_name) #Torchvision transforms

    match name:
        case "mnist":
            dataset = get_mnist_dataset(name, path, train, n_samples, transform, model_name)
        case "fashion_mnist":
            dataset = get_fashion_mnist_dataset(name, path, train, n_samples, transform, model_name)
        case "cifar":
            dataset = get_cifar_dataset(name, path, train, transform, n_samples, model_name)

    return dataset


VISION_DATASET_MAPPING = {
    "mnist": MNISTQuMET,
    "fashion_mnist": Fashion_MNISTQuMET,
    "cifar10": Cifar10QuMET,
}


def get_vision_dataset_cls(name: str):
    assert name in VISION_DATASET_MAPPING, f"Unknown dataset {name}"
    return VISION_DATASET_MAPPING[name.lower()]
