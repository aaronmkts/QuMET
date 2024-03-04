from torch.utils.data import Dataset
from torchvision import datasets
import os
from ..utils import add_dataset_info


@add_dataset_info(
    name="fashion_mnist",
    dataset_source="torchvision",
    available_splits=("train", "test"),
    image_generation=True,
    num_classes=10,
    image_size=(1, 28, 28),
)
class Fashion_MNISTQuMET(datasets.FashionMNIST):
    def __init__(
        self, root: os.PathLike, train: bool, transform: callable, download: bool
    ) -> None:
        super().__init__(root, train=train, transform=transform, download=download)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass


def get_fashion_mnist_dataset(
    name: str, path: os.PathLike, train: bool, transform: callable
) -> Dataset:
    match name.lower():
        case "fashion_mnist":
            dataset = Fashion_MNISTQuMET(
                path, train=train, transform=transform, download=True
            )
        case _:
            raise ValueError(f"Unknown dataset {name}")
    return dataset
