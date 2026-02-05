import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

from ..utils import add_dataset_info
from .transforms import DataTransformationMixin


@add_dataset_info(
    name="fashion_mnist",
    dataset_source="torchvision",
    available_splits=("train", "test"),
    image_generation=True,
    num_classes=2,
    image_size=(1, 28, 28),
)
class Fashion_MNISTQuMET(datasets.FashionMNIST, DataTransformationMixin):
    """
    Labels:
    0 - T-shirt/top
    1 - Trouser
    2 - Pullover
    3 - Dress
    4 - Coat
    5 - Sandal
    6 - Shirt
    7 - Sneaker
    8 - Bag
    9 - Ankle boot
    """

    def __init__(
        self,
        root: os.PathLike,
        train: bool,
        n_samples: int,
        transform: callable,
        download: bool,
        model_name: str,
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
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.model_name == "mosaiq":
            img = self.apply_pca(img)

        return img, target


def get_fashion_mnist_dataset(
    name: str,
    path: os.PathLike,
    train: bool,
    n_samples: int,
    transform: callable,
    model_name: str,
) -> Dataset:
    match name.lower():
        case "fashion_mnist":
            dataset = Fashion_MNISTQuMET(
                path,
                train=train,
                transform=transform,
                n_samples=n_samples,
                model_name=model_name,
                download=True,
            )
        case _:
            raise ValueError(f"Unknown dataset {name}")
    return dataset
