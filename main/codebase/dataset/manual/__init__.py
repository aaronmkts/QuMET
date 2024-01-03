import os
from pathlib import Path


from .gaussian import GaussianDataset
from .two_d_gaussian import TwoDGaussianDataset

def get_manual_dataset(name: str, split: str):
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
    ori_split = split
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of train, validation, test, pred"

    match name:
        case "gaussian":
            dataset_cls = GaussianDataset
        case "two_d_gaussian":
            dataset_cls = TwoDGaussianDataset
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

    dataset = dataset_cls(split,)
    return dataset


MANUAL_DATASET_MAPPING = {
    "gaussian": GaussianDataset,
    "two_d_gaussian": TwoDGaussianDataset,
}


def get_manual_dataset_cls(name: str):
    assert name in MANUAL_DATASET_MAPPING, f"Unknown dataset {name}"
    return MANUAL_DATASET_MAPPING[name.lower()]
