from torch.utils.data import Dataset
from torchvision import datasets
import os
from ..utils import add_dataset_info
from .transforms import DataTransformationMixin
import torch
from PIL import Image

@add_dataset_info(
    name="mnist",
    dataset_source="torchvision",
    available_splits=("train", "test"),
    image_generation=True,
    num_classes=1,
    image_size=(1, 8, 8),
)
class MNISTQuMET(datasets.MNIST, DataTransformationMixin):
    def __init__(
        self, root: os.PathLike, train: bool, n_samples:int, transform: callable, download: bool, model_name: str
    ) -> None:
        super().__init__(root, train=train, transform=transform, download=download)

        self.model_name = model_name
        filter_labels = [0]

        if filter_labels:
            self.filter_labels(filter_labels)
            self.n_samples_(n_samples)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_mnist_dataset(
    name: str, path: os.PathLike, train: bool, n_samples:int, transform: callable, model_name: str
) -> Dataset:
    match name.lower():
        case "mnist":
            dataset = MNISTQuMET(path, train=train, transform=transform, n_samples = n_samples, model_name = model_name, download=True)

        case _:
            raise ValueError(f"Unknown dataset {name}")
    return dataset
