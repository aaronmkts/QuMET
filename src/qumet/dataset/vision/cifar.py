from torch.utils.data import Dataset
from torchvision import datasets
import os
from ..utils import add_dataset_info
from .transforms import DataTransformationMixin

@add_dataset_info(
    name="cifar10",
    dataset_source="torchvision",
    available_splits=("train", "test"),
    image_generation=True,
    num_classes=10,
    image_size=(3, 32, 32),
)
class Cifar10QuMET(datasets.CIFAR10, DataTransformationMixin):
    def __init__(
        self, root: os.PathLike, train: bool, n_samples:int, transform: callable, download: bool, model_name:str
    ) -> None:
        super().__init__(root, train=train, transform=transform, download=download)

        self.model_name = model_name

        filter_labels = [0, 1]
        if filter_labels:
            self.filter_labels(filter_labels)
            self.n_samples_(n_samples)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass


def get_cifar_dataset(
    name: str, path: os.PathLike, train: bool, transform: callable, n_samples: int, model_name: str 
) -> Dataset:
    match name.lower():
        case "cifar10":
            dataset = Cifar10QuMET(
                path, train=train, transform=transform, download=True, n_samples=n_samples, model_name=model_name
            )
        case _:
            raise ValueError(f"Unknown dataset {name}")
    return dataset
