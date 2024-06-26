from .mnist import get_mnist_transform
from .fashion_mnist import get_fashion_mnist_transform
from .cifar import get_cifar10_transform
from .utils import filter_by_labels, apply_pca

class DataTransformationMixin:
    def filter_labels(self, labels: list):
        """
        Filters the dataset to only include specified labels.
        """
        return filter_by_labels(self, labels)

    def apply_pca(self, n_features: int):
        """
        Applies PCA to the dataset to reduce the number of features.
        """
        return apply_pca(self, n_features)

    
def get_vision_dataset_transform(name: str, train: bool, model_name: str):
    """
    Args:
        name (str): name of the dataset
        train (bool): whether the dataset is used for training
        model_name (Optional[str, None]): name of the model. Some pretrained models have model-dependent transforms.
    Returns:
        transform (callable): transform function
    """
    match name.lower():
        case "mnist":
            return get_mnist_transform(train, model_name)
        case "fashion_mnist":
            return get_fashion_mnist_transform(train, model_name)
        case "cifar10":
            return get_cifar10_transform(train, model_name)
        case _:
            raise ValueError(f"Unknown dataset {name}")
