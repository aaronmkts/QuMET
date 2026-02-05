import numpy as np
import torch

from .cifar import get_cifar10_transform
from .fashion_mnist import get_fashion_mnist_transform
from .mnist import get_mnist_transform
from .utils import PCA, filter_by_labels, n_samples, scale_data


class DataTransformationMixin:

    def filter_labels(self, labels: list):
        """
        Filters the dataset to only include specified labels.
        """
        return filter_by_labels(self, labels)

    def n_samples_(self, samples: int):

        return n_samples(self, samples)

    def fit_pca(self, n_components):

        self.pca_model = PCA(n_components)

        # Convert dataset to numpy array and flatten
        X_flat_data = self.data.numpy().reshape(self.data.shape[0], -1)
        # Scale data if necessary
        X_data = self.scale_data(X_flat_data, scale=[0, 1])
        # Fit PCA
        self.pca_model.fit(torch.tensor(X_data, dtype=torch.float32))

    def apply_pca(self, img):

        transformed_img = self.pca_model.transform(img)

        return transformed_img

    @staticmethod
    def scale_data(data, scale=None, dtype=np.float32):
        return scale_data(data, scale=scale, dtype=np.float32)


def get_vision_dataset_transform(
    name: str, train: bool, model_name: str, transform: None
):
    """
    Args:
        name (str): name of the dataset
        train (bool): whether the dataset is used for training
        model_name (Optional[str, None]): name of the model. Some pretrained models have model-dependent transforms.
    Returns:
        transform (callable): transform function
    """
    if callable(transform):
        # If transform is already a callable (e.g., PCA_Transform instance), return it directly
        return transform
    else:
        match name.lower():
            case "mnist":
                return get_mnist_transform(train, model_name)
            case "fashion_mnist":
                return get_fashion_mnist_transform(train, model_name)
            case "cifar10":
                return get_cifar10_transform(train, model_name)
            case _:
                raise ValueError(f"Unknown dataset {name}")
